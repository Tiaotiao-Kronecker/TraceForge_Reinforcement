from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from utils.traj_filter_utils import (
    QUERY_DEPTH_EDGE_PATCH_STD_THRESHOLD,
    compute_query_depth_edge_risk_mask,
    compute_query_depth_quality_mask,
)


QUERY_SAMPLER_MODE_AUTO = "auto"
QUERY_SAMPLER_MODE_GRID = "grid"
QUERY_SAMPLER_MODE_RELEVANCE_V1 = "relevance_first_v1"

QUERY_SOURCE_HAND = np.uint16(1 << 0)
QUERY_SOURCE_MANIPULATOR = np.uint16(1 << 1)
QUERY_SOURCE_INTERACTION_OBJECT = np.uint16(1 << 2)
QUERY_SOURCE_CONTEXT = np.uint16(1 << 3)
QUERY_SOURCE_FALLBACK_GRID = np.uint16(1 << 4)

QUERY_RISK_LOW_TEXTURE = np.uint16(1 << 0)
QUERY_RISK_SPECULAR = np.uint16(1 << 1)
QUERY_RISK_DEPTH_EDGE = np.uint16(1 << 2)
QUERY_RISK_IMAGE_BORDER = np.uint16(1 << 3)

DEFAULT_QUERY_CANDIDATE_GRID_FACTOR = 2.0
DEFAULT_QUERY_ACTIVITY_FRAMES = 4
DEFAULT_BORDER_RISK_MARGIN_RATIO = 0.03
DEFAULT_LOW_TEXTURE_THRESHOLD = 0.72
DEFAULT_SPECULAR_THRESHOLD = 0.60
DEFAULT_DEPTH_EDGE_THRESHOLD = 0.50


@dataclass(frozen=True)
class QueryCueResult:
    hand_score: np.ndarray
    manipulator_score: np.ndarray
    object_score: np.ndarray
    context_score: np.ndarray
    valid_mask: np.ndarray
    provider_name: str
    provider_meta: dict[str, float | int | str]


class RelevanceCueProvider(ABC):
    @abstractmethod
    def propose(
        self,
        *,
        query_rgb: np.ndarray,
        query_depth: np.ndarray,
        temporal_rgbs: np.ndarray,
        candidate_keypoints: np.ndarray,
        min_depth: float,
        max_depth: float,
    ) -> QueryCueResult:
        raise NotImplementedError


class ExternalModelCueProvider(RelevanceCueProvider):
    """Adapter contract for future third-party cue providers."""

    def propose(
        self,
        *,
        query_rgb: np.ndarray,
        query_depth: np.ndarray,
        temporal_rgbs: np.ndarray,
        candidate_keypoints: np.ndarray,
        min_depth: float,
        max_depth: float,
    ) -> QueryCueResult:
        raise NotImplementedError("ExternalModelCueProvider is an interface-only adapter in v1.")


def resolve_query_sampler_mode(*, mode: str | None, traj_filter_profile: str | None) -> str:
    mode = str(mode or QUERY_SAMPLER_MODE_AUTO)
    if mode == QUERY_SAMPLER_MODE_AUTO:
        if str(traj_filter_profile or "") == "egocentric_object_interaction_v1":
            return QUERY_SAMPLER_MODE_RELEVANCE_V1
        return QUERY_SAMPLER_MODE_GRID
    if mode not in {
        QUERY_SAMPLER_MODE_GRID,
        QUERY_SAMPLER_MODE_RELEVANCE_V1,
    }:
        raise ValueError(f"Unsupported query_sampler_mode: {mode}")
    return mode


def resolve_candidate_grid_size(grid_size: int, *, factor: float = DEFAULT_QUERY_CANDIDATE_GRID_FACTOR) -> int:
    grid_size = int(grid_size)
    factor = float(max(1.0, factor))
    if grid_size <= 0:
        return 0
    return max(grid_size, int(np.ceil(grid_size * factor)))


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H,W,3), got {image.shape}")
    return np.clip(
        0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2],
        0.0,
        1.0,
    ).astype(np.float32)


def _normalize_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float32)
    if not np.any(finite):
        return out
    lo = float(np.min(values[finite]))
    hi = float(np.max(values[finite]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
        out[finite] = 1.0
        return out
    out[finite] = (values[finite] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _sample_2d_map(map_2d: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    map_2d = np.asarray(map_2d, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if map_2d.ndim != 2:
        raise ValueError(f"Expected map_2d shape (H,W), got {map_2d.shape}")
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")
    h, w = map_2d.shape
    xs = np.clip(np.round(keypoints[:, 0]).astype(np.int32), 0, max(w - 1, 0))
    ys = np.clip(np.round(keypoints[:, 1]).astype(np.int32), 0, max(h - 1, 0))
    return map_2d[ys, xs].astype(np.float32, copy=False)


def _compute_border_margin_px(height: int, width: int) -> int:
    return max(8, int(round(DEFAULT_BORDER_RISK_MARGIN_RATIO * float(min(height, width)))))


def _compute_border_distances_px(
    keypoints: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    x = keypoints[:, 0].astype(np.float32, copy=False)
    y = keypoints[:, 1].astype(np.float32, copy=False)
    border_dist = np.minimum.reduce(
        [
            x,
            y,
            np.maximum(float(width - 1) - x, 0.0),
            np.maximum(float(height - 1) - y, 0.0),
        ]
    )
    return np.clip(np.round(border_dist), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def _compute_gradient_map(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(np.maximum(grad_x * grad_x + grad_y * grad_y, 0.0))
    return _normalize_finite(grad_mag)


def _compute_temporal_activity_map(temporal_rgbs: np.ndarray) -> np.ndarray:
    temporal_rgbs = np.asarray(temporal_rgbs, dtype=np.float32)
    if temporal_rgbs.ndim != 4 or temporal_rgbs.shape[-1] != 3:
        raise ValueError(f"Expected temporal_rgbs shape (T,H,W,3), got {temporal_rgbs.shape}")
    gray = np.stack([_to_gray_float(frame) for frame in temporal_rgbs], axis=0)
    if gray.shape[0] <= 1:
        return np.zeros(gray.shape[1:], dtype=np.float32)
    diffs = np.abs(gray[1:] - gray[:1]).astype(np.float32)
    activity = np.mean(diffs, axis=0)
    activity = cv2.GaussianBlur(activity, (0, 0), sigmaX=2.0, sigmaY=2.0)
    return _normalize_finite(activity)


def _compute_specular_score_map(query_rgb: np.ndarray) -> np.ndarray:
    query_rgb = np.asarray(query_rgb, dtype=np.float32)
    max_rgb = np.max(query_rgb, axis=-1)
    min_rgb = np.min(query_rgb, axis=-1)
    saturation = np.where(max_rgb > 1e-6, (max_rgb - min_rgb) / (max_rgb + 1e-6), 0.0)
    brightness = _normalize_finite(max_rgb)
    specular = brightness * (1.0 - np.clip(saturation, 0.0, 1.0))
    return np.clip(specular, 0.0, 1.0).astype(np.float32)


def _compute_candidate_depth_rank_score(
    keypoints: np.ndarray,
    query_depth: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    query_depth = np.asarray(query_depth, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    depths = _sample_2d_map(query_depth, keypoints)
    valid = np.isfinite(depths) & (depths > float(min_depth)) & (depths < float(max_depth))
    rank = np.full(depths.shape, np.nan, dtype=np.float32)
    if np.any(valid):
        valid_depths = depths[valid]
        order = np.argsort(valid_depths, kind="stable")
        ranked = np.empty_like(order, dtype=np.float32)
        if order.size == 1:
            ranked[0] = 0.0
        else:
            ranked[order] = np.linspace(0.0, 1.0, num=order.size, dtype=np.float32)
        rank[valid] = ranked
    near_score = 1.0 - np.nan_to_num(rank, nan=1.0, posinf=1.0, neginf=1.0)
    return rank.astype(np.float32), np.clip(near_score, 0.0, 1.0).astype(np.float32)


class HeuristicCueProvider(RelevanceCueProvider):
    def propose(
        self,
        *,
        query_rgb: np.ndarray,
        query_depth: np.ndarray,
        temporal_rgbs: np.ndarray,
        candidate_keypoints: np.ndarray,
        min_depth: float,
        max_depth: float,
    ) -> QueryCueResult:
        query_rgb = np.asarray(query_rgb, dtype=np.float32)
        query_depth = np.asarray(query_depth, dtype=np.float32)
        temporal_rgbs = np.asarray(temporal_rgbs, dtype=np.float32)
        candidate_keypoints = np.asarray(candidate_keypoints, dtype=np.float32)

        gray = _to_gray_float(query_rgb)
        gradient_map = _compute_gradient_map(gray)
        activity_map = _compute_temporal_activity_map(temporal_rgbs)
        texture_score = _sample_2d_map(gradient_map, candidate_keypoints)
        activity_score = _sample_2d_map(activity_map, candidate_keypoints)
        depth_rank, near_depth_score = _compute_candidate_depth_rank_score(
            candidate_keypoints,
            query_depth,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        depth_valid_mask = compute_query_depth_quality_mask(
            candidate_keypoints,
            query_depth,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        h, w = query_depth.shape
        x = candidate_keypoints[:, 0].astype(np.float32, copy=False)
        y = candidate_keypoints[:, 1].astype(np.float32, copy=False)
        center_x = 0.5 * max(float(w - 1), 1.0)
        center_bias = 1.0 - np.clip(np.abs(x - center_x) / max(center_x, 1.0), 0.0, 1.0)
        lower_bias = np.clip(y / max(float(h - 1), 1.0), 0.0, 1.0)
        lower_center_bias = np.clip(0.6 * lower_bias + 0.4 * center_bias, 0.0, 1.0)

        manipulator_score = np.clip(
            0.45 * near_depth_score
            + 0.35 * activity_score
            + 0.20 * lower_center_bias,
            0.0,
            1.0,
        ).astype(np.float32)
        hand_score = np.clip(
            0.35 * near_depth_score
            + 0.40 * activity_score
            + 0.25 * center_bias,
            0.0,
            1.0,
        ).astype(np.float32)

        hand_seed_mask = depth_valid_mask & (hand_score >= np.nanpercentile(hand_score[depth_valid_mask], 75.0)) if np.any(depth_valid_mask) else np.zeros_like(depth_valid_mask)
        if np.any(hand_seed_mask):
            hand_seed_points = candidate_keypoints[hand_seed_mask]
            diff = candidate_keypoints[:, None, :] - hand_seed_points[None, :, :]
            dist_px = np.sqrt(np.min(np.sum(diff * diff, axis=-1), axis=1))
            object_proximity = np.exp(
                -np.square(dist_px / max(0.18 * min(float(h), float(w)), 1.0))
            ).astype(np.float32)
        else:
            object_proximity = np.zeros(candidate_keypoints.shape[0], dtype=np.float32)

        medium_depth_score = np.clip(1.0 - np.abs(np.nan_to_num(depth_rank, nan=0.5) - 0.35) / 0.35, 0.0, 1.0)
        object_score = np.clip(
            0.35 * object_proximity
            + 0.35 * activity_score
            + 0.20 * texture_score
            + 0.10 * medium_depth_score,
            0.0,
            1.0,
        ).astype(np.float32)
        object_score *= depth_valid_mask.astype(np.float32)
        object_score *= (0.45 + 0.55 * (1.0 - np.clip(hand_score, 0.0, 1.0)))

        context_score = np.clip(
            0.55 * texture_score
            + 0.25 * (1.0 - np.maximum(hand_score, object_score))
            + 0.20 * np.clip(1.0 - np.abs(np.nan_to_num(depth_rank, nan=0.5) - 0.55) / 0.55, 0.0, 1.0),
            0.0,
            1.0,
        ).astype(np.float32)
        context_score *= depth_valid_mask.astype(np.float32)

        return QueryCueResult(
            hand_score=hand_score.astype(np.float32, copy=False),
            manipulator_score=manipulator_score.astype(np.float32, copy=False),
            object_score=object_score.astype(np.float32, copy=False),
            context_score=context_score.astype(np.float32, copy=False),
            valid_mask=depth_valid_mask.astype(bool, copy=False),
            provider_name="heuristic_v1",
            provider_meta={
                "valid_candidate_count": int(np.count_nonzero(depth_valid_mask)),
                "activity_frames": int(temporal_rgbs.shape[0]),
            },
        )


def _pick_top_indices(
    scores: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    taken_mask: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    taken_mask = np.asarray(taken_mask, dtype=bool)
    eligible = candidate_mask & (~taken_mask) & np.isfinite(scores)
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        return np.zeros(0, dtype=np.int32)
    order = np.argsort(scores[eligible_indices], kind="stable")[::-1]
    return eligible_indices[order[:count]].astype(np.int32, copy=False)


def _compute_query_risk_result(
    *,
    keypoints: np.ndarray,
    query_rgb: np.ndarray,
    query_depth: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> dict[str, np.ndarray]:
    gray = _to_gray_float(query_rgb)
    gradient_map = _compute_gradient_map(gray)
    texture_score = _sample_2d_map(gradient_map, keypoints)
    low_texture_score = np.clip(1.0 - texture_score, 0.0, 1.0).astype(np.float32)
    specular_score = _sample_2d_map(_compute_specular_score_map(query_rgb), keypoints)

    edge_result = compute_query_depth_edge_risk_mask(
        keypoints,
        query_depth,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    depth_edge_score = np.zeros(keypoints.shape[0], dtype=np.float32)
    patch_std = np.asarray(edge_result["patch_std"]).astype(np.float32, copy=False)
    finite_std = np.isfinite(patch_std)
    if np.any(finite_std):
        depth_edge_score[finite_std] = np.clip(
            patch_std[finite_std] / max(QUERY_DEPTH_EDGE_PATCH_STD_THRESHOLD * 4.0, 1e-6),
            0.0,
            1.0,
        )
    query_edge_mask = np.asarray(edge_result["query_edge_mask"]).astype(bool, copy=False)
    depth_edge_score *= query_edge_mask.astype(np.float32)

    h, w = query_depth.shape
    border_dist = _compute_border_distances_px(keypoints, height=h, width=w)
    border_margin = _compute_border_margin_px(h, w)

    risk_bits = np.zeros(keypoints.shape[0], dtype=np.uint16)
    risk_bits[low_texture_score >= DEFAULT_LOW_TEXTURE_THRESHOLD] |= QUERY_RISK_LOW_TEXTURE
    risk_bits[specular_score >= DEFAULT_SPECULAR_THRESHOLD] |= QUERY_RISK_SPECULAR
    risk_bits[depth_edge_score >= DEFAULT_DEPTH_EDGE_THRESHOLD] |= QUERY_RISK_DEPTH_EDGE
    risk_bits[border_dist <= np.uint16(border_margin)] |= QUERY_RISK_IMAGE_BORDER
    return {
        "query_risk_bits": risk_bits.astype(np.uint16, copy=False),
        "query_low_texture_score": low_texture_score.astype(np.float32, copy=False),
        "query_specular_score": specular_score.astype(np.float32, copy=False),
        "query_depth_edge_score": depth_edge_score.astype(np.float32, copy=False),
        "query_border_dist_px": border_dist.astype(np.uint16, copy=False),
        "query_depth_edge_mask": query_edge_mask.astype(bool, copy=False),
        "query_depth_edge_risk_mask": np.asarray(edge_result["mask"]).astype(bool, copy=False),
        "query_depth_patch_valid_ratio": np.asarray(edge_result["patch_valid_ratio"]).astype(np.float32, copy=False),
        "query_depth_patch_std": patch_std.astype(np.float32, copy=False),
    }


def build_relevance_first_query_sampler_result(
    *,
    candidate_keypoints: np.ndarray,
    query_rgb: np.ndarray,
    query_depth: np.ndarray,
    temporal_rgbs: np.ndarray,
    target_query_count: int,
    min_depth: float,
    max_depth: float,
    cue_provider: RelevanceCueProvider | None = None,
) -> dict[str, np.ndarray | str | dict[str, float | int | str]]:
    candidate_keypoints = np.asarray(candidate_keypoints, dtype=np.float32)
    target_query_count = int(target_query_count)
    if cue_provider is None:
        cue_provider = HeuristicCueProvider()
    cue_result = cue_provider.propose(
        query_rgb=query_rgb,
        query_depth=query_depth,
        temporal_rgbs=temporal_rgbs,
        candidate_keypoints=candidate_keypoints,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    valid_mask = np.asarray(cue_result.valid_mask).astype(bool, copy=False)
    num_candidates = int(candidate_keypoints.shape[0])
    if target_query_count <= 0 or num_candidates == 0:
        empty_uint16 = np.zeros(0, dtype=np.uint16)
        empty_float = np.zeros(0, dtype=np.float32)
        empty_bool = np.zeros(0, dtype=bool)
        return {
            "keypoints": candidate_keypoints[:0].astype(np.float32),
            "tracked_query_indices": np.zeros(0, dtype=np.int32),
            "prefilter_mask": empty_bool,
            "reason_bits": np.zeros(0, dtype=np.uint16),
            "query_source_bits": empty_uint16,
            "query_sampler_score": empty_float,
            "query_risk_bits": empty_uint16,
            "query_low_texture_score": empty_float,
            "query_specular_score": empty_float,
            "query_depth_edge_score": empty_float,
            "query_border_dist_px": empty_uint16,
            "query_depth_edge_mask": empty_bool,
            "query_depth_edge_risk_mask": empty_bool,
            "query_depth_patch_valid_ratio": empty_float,
            "query_depth_patch_std": empty_float,
            "provider_name": cue_result.provider_name,
            "provider_meta": cue_result.provider_meta,
        }

    hand_score = np.asarray(cue_result.hand_score).astype(np.float32, copy=False)
    manipulator_score = np.asarray(cue_result.manipulator_score).astype(np.float32, copy=False)
    object_score = np.asarray(cue_result.object_score).astype(np.float32, copy=False)
    context_score = np.asarray(cue_result.context_score).astype(np.float32, copy=False)
    combined_hand_score = np.clip(np.maximum(hand_score, manipulator_score), 0.0, 1.0).astype(np.float32)
    total_budget = min(target_query_count, int(np.count_nonzero(valid_mask)))
    if total_budget <= 0:
        valid_mask = np.isfinite(candidate_keypoints).all(axis=1)
        total_budget = min(target_query_count, int(np.count_nonzero(valid_mask)))
    if total_budget <= 0:
        total_budget = min(target_query_count, num_candidates)
        valid_mask = np.ones(num_candidates, dtype=bool)

    border_dist = _compute_border_distances_px(
        candidate_keypoints,
        height=int(query_depth.shape[0]),
        width=int(query_depth.shape[1]),
    )
    border_margin = _compute_border_margin_px(int(query_depth.shape[0]), int(query_depth.shape[1]))
    non_border_valid_mask = valid_mask & (border_dist > np.uint16(border_margin))
    primary_valid_mask = (
        non_border_valid_mask
        if int(np.count_nonzero(non_border_valid_mask)) >= total_budget
        else valid_mask
    )

    hand_budget = max(1, int(round(total_budget * 0.40)))
    object_budget = max(1, int(round(total_budget * 0.35)))
    context_budget = max(1, int(round(total_budget * 0.15)))
    fallback_budget = max(0, total_budget - hand_budget - object_budget - context_budget)

    taken_mask = np.zeros(num_candidates, dtype=bool)
    selected_chunks: list[np.ndarray] = []
    hand_indices = _pick_top_indices(
        combined_hand_score,
        primary_valid_mask,
        taken_mask=taken_mask,
        count=hand_budget,
    )
    selected_chunks.append(hand_indices)
    taken_mask[hand_indices] = True

    object_candidate_mask = primary_valid_mask & (object_score > 0.0)
    object_indices = _pick_top_indices(
        object_score,
        object_candidate_mask,
        taken_mask=taken_mask,
        count=object_budget,
    )
    selected_chunks.append(object_indices)
    taken_mask[object_indices] = True

    context_indices = _pick_top_indices(
        context_score,
        primary_valid_mask,
        taken_mask=taken_mask,
        count=context_budget,
    )
    selected_chunks.append(context_indices)
    taken_mask[context_indices] = True

    fallback_score = np.clip(0.5 * context_score + 0.5 * np.maximum(combined_hand_score, object_score), 0.0, 1.0)
    fallback_indices = _pick_top_indices(
        fallback_score,
        primary_valid_mask,
        taken_mask=taken_mask,
        count=fallback_budget,
    )
    selected_chunks.append(fallback_indices)
    taken_mask[fallback_indices] = True

    selected_indices = (
        np.concatenate([chunk for chunk in selected_chunks if chunk.size > 0], axis=0)
        if any(chunk.size > 0 for chunk in selected_chunks)
        else np.zeros(0, dtype=np.int32)
    )
    if selected_indices.shape[0] < total_budget:
        extra_indices = _pick_top_indices(
            np.maximum.reduce([combined_hand_score, object_score, context_score]),
            primary_valid_mask,
            taken_mask=taken_mask,
            count=total_budget - selected_indices.shape[0],
        )
        if extra_indices.size > 0:
            selected_indices = np.concatenate([selected_indices, extra_indices], axis=0)
            taken_mask[extra_indices] = True

    if selected_indices.shape[0] < target_query_count:
        pad_indices = _pick_top_indices(
            np.maximum.reduce([combined_hand_score, object_score, context_score]),
            np.ones(num_candidates, dtype=bool),
            taken_mask=taken_mask,
            count=target_query_count - selected_indices.shape[0],
        )
        if pad_indices.size > 0:
            selected_indices = np.concatenate([selected_indices, pad_indices], axis=0)
            taken_mask[pad_indices] = True

    selected_indices = selected_indices[:target_query_count].astype(np.int32, copy=False)
    selected_keypoints = candidate_keypoints[selected_indices].astype(np.float32, copy=False)

    source_bits = np.zeros(selected_indices.shape[0], dtype=np.uint16)
    selected_hand_score = hand_score[selected_indices]
    selected_manip_score = manipulator_score[selected_indices]
    selected_object_score = object_score[selected_indices]
    selected_context_score = context_score[selected_indices]
    if selected_indices.size > 0:
        hand_threshold = float(np.percentile(selected_hand_score, 55.0))
        manip_threshold = float(np.percentile(selected_manip_score, 55.0))
        object_threshold = float(np.percentile(selected_object_score, 55.0))
        context_threshold = float(np.percentile(selected_context_score, 50.0))
        source_bits[selected_hand_score >= hand_threshold] |= QUERY_SOURCE_HAND
        source_bits[selected_manip_score >= manip_threshold] |= QUERY_SOURCE_MANIPULATOR
        source_bits[selected_object_score >= object_threshold] |= QUERY_SOURCE_INTERACTION_OBJECT
        source_bits[selected_context_score >= context_threshold] |= QUERY_SOURCE_CONTEXT
        source_bits[np.isin(selected_indices, fallback_indices)] |= QUERY_SOURCE_FALLBACK_GRID
        source_bits[source_bits == 0] |= QUERY_SOURCE_FALLBACK_GRID

    sampler_score = np.clip(
        np.maximum.reduce(
            [
                combined_hand_score[selected_indices],
                object_score[selected_indices],
                context_score[selected_indices],
            ]
        ),
        0.0,
        1.0,
    ).astype(np.float32)

    risk_result = _compute_query_risk_result(
        keypoints=selected_keypoints,
        query_rgb=query_rgb,
        query_depth=query_depth,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    return {
        "keypoints": selected_keypoints.astype(np.float32, copy=False),
        "tracked_query_indices": np.arange(selected_keypoints.shape[0], dtype=np.int32),
        "prefilter_mask": np.ones(selected_keypoints.shape[0], dtype=bool),
        "reason_bits": np.zeros(selected_keypoints.shape[0], dtype=np.uint16),
        "query_source_bits": source_bits.astype(np.uint16, copy=False),
        "query_sampler_score": sampler_score.astype(np.float32, copy=False),
        "query_risk_bits": np.asarray(risk_result["query_risk_bits"]).astype(np.uint16, copy=False),
        "query_low_texture_score": np.asarray(risk_result["query_low_texture_score"]).astype(np.float32, copy=False),
        "query_specular_score": np.asarray(risk_result["query_specular_score"]).astype(np.float32, copy=False),
        "query_depth_edge_score": np.asarray(risk_result["query_depth_edge_score"]).astype(np.float32, copy=False),
        "query_border_dist_px": np.asarray(risk_result["query_border_dist_px"]).astype(np.uint16, copy=False),
        "query_depth_edge_mask": np.asarray(risk_result["query_depth_edge_mask"]).astype(bool, copy=False),
        "query_depth_edge_risk_mask": np.asarray(risk_result["query_depth_edge_risk_mask"]).astype(bool, copy=False),
        "query_depth_patch_valid_ratio": np.asarray(risk_result["query_depth_patch_valid_ratio"]).astype(
            np.float32,
            copy=False,
        ),
        "query_depth_patch_std": np.asarray(risk_result["query_depth_patch_std"]).astype(np.float32, copy=False),
        "provider_name": cue_result.provider_name,
        "provider_meta": cue_result.provider_meta,
    }
