from __future__ import annotations

from typing import Any

import numpy as np

from utils.external_wobble_diagnostics import (
    project_world_points,
    sample_image_at_keypoints,
    unproject_keypoints_to_world,
)


DEFAULT_FIXED_VIEW_DEPTH_GATE_UV_THRESHOLD_PX = 1.0
DEFAULT_FIXED_VIEW_DEPTH_GATE_DEPTH_THRESHOLD_M = 0.10


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {"finite_count": 0, "median": None, "p95": None, "max": None}
    valid = arr[finite].astype(np.float64)
    return {
        "finite_count": int(valid.size),
        "median": float(np.median(valid)),
        "p95": float(np.percentile(valid, 95)),
        "max": float(np.max(valid)),
    }


def _masked_nanmin(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"Expected values/mask to share shape, got {values.shape} vs {mask.shape}")
    result = np.full(values.shape[0], np.nan, dtype=np.float32)
    finite_rows = np.any(mask & np.isfinite(values), axis=1)
    if np.any(finite_rows):
        result[finite_rows] = np.nanmin(np.where(mask[finite_rows], values[finite_rows], np.nan), axis=1).astype(
            np.float32
        )
    return result


def _masked_nanmax(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"Expected values/mask to share shape, got {values.shape} vs {mask.shape}")
    result = np.full(values.shape[0], np.nan, dtype=np.float32)
    finite_rows = np.any(mask & np.isfinite(values), axis=1)
    if np.any(finite_rows):
        result[finite_rows] = np.nanmax(np.where(mask[finite_rows], values[finite_rows], np.nan), axis=1).astype(
            np.float32
        )
    return result


def _build_first_trigger_step(
    *,
    query_world_valid_mask: np.ndarray,
    trigger_mask: np.ndarray,
) -> np.ndarray:
    query_world_valid_mask = np.asarray(query_world_valid_mask, dtype=bool).reshape(-1)
    trigger_mask = np.asarray(trigger_mask, dtype=bool)
    if trigger_mask.ndim != 2:
        raise ValueError(f"Expected trigger_mask shape (N,T), got {trigger_mask.shape}")
    if trigger_mask.shape[0] != query_world_valid_mask.shape[0]:
        raise ValueError(
            "Expected trigger_mask and query_world_valid_mask to share track count, "
            f"got {trigger_mask.shape[0]} vs {query_world_valid_mask.shape[0]}"
        )

    first_trigger_step = np.full(query_world_valid_mask.shape[0], -1, dtype=np.int32)
    first_trigger_step[~query_world_valid_mask] = 0
    if trigger_mask.shape[1] == 0:
        return first_trigger_step

    has_trigger = np.any(trigger_mask, axis=1) & query_world_valid_mask
    if np.any(has_trigger):
        first_trigger_step[has_trigger] = np.argmax(trigger_mask[has_trigger], axis=1).astype(np.int32)
    return first_trigger_step.astype(np.int32, copy=False)


def compute_query_fixed_view_depth_gate(
    depth_frames: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    keypoints: np.ndarray,
    query_frame: int = 0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    uv_threshold_px: float = DEFAULT_FIXED_VIEW_DEPTH_GATE_UV_THRESHOLD_PX,
    depth_threshold_m: float = DEFAULT_FIXED_VIEW_DEPTH_GATE_DEPTH_THRESHOLD_M,
) -> dict[str, Any]:
    depth_frames = np.asarray(depth_frames, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if depth_frames.ndim != 3:
        raise ValueError(f"Expected depth_frames shape (T,H,W), got {depth_frames.shape}")
    if intrinsics.shape != (depth_frames.shape[0], 3, 3):
        raise ValueError(f"Expected intrinsics shape {(depth_frames.shape[0], 3, 3)}, got {intrinsics.shape}")
    if extrinsics_w2c.shape != (depth_frames.shape[0], 4, 4):
        raise ValueError(
            f"Expected extrinsics_w2c shape {(depth_frames.shape[0], 4, 4)}, got {extrinsics_w2c.shape}"
        )
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")

    frame_count, height, width = depth_frames.shape
    query_frame = int(query_frame)
    if query_frame < 0 or query_frame >= frame_count:
        raise IndexError(f"query_frame={query_frame} exceeds frame_count={frame_count}")
    if float(uv_threshold_px) < 0.0:
        raise ValueError(f"Expected uv_threshold_px >= 0, got {uv_threshold_px}")
    if float(depth_threshold_m) < 0.0:
        raise ValueError(f"Expected depth_threshold_m >= 0, got {depth_threshold_m}")

    query_depth_values, query_depth_sample_valid = sample_image_at_keypoints(depth_frames[query_frame], keypoints)
    world_points, query_world_valid = unproject_keypoints_to_world(
        keypoints,
        query_depth_values,
        intrinsics=intrinsics[query_frame],
        w2c=extrinsics_w2c[query_frame],
        min_depth=min_depth,
        max_depth=max_depth,
    )
    query_world_valid = query_depth_sample_valid & query_world_valid
    world_points[~query_world_valid] = np.nan

    projected_uvz = np.full((keypoints.shape[0], frame_count, 3), np.nan, dtype=np.float32)
    projected_valid_mask = np.zeros((keypoints.shape[0], frame_count), dtype=bool)
    query_reprojected_uvz = np.full((keypoints.shape[0], frame_count, 3), np.nan, dtype=np.float32)
    query_reprojected_valid_mask = np.zeros((keypoints.shape[0], frame_count), dtype=bool)
    compare_mask = np.zeros((keypoints.shape[0], frame_count), dtype=bool)
    uv_delta_px = np.full((keypoints.shape[0], frame_count), np.nan, dtype=np.float32)
    depth_delta_m = np.full((keypoints.shape[0], frame_count), np.nan, dtype=np.float32)

    for frame_idx in range(frame_count):
        frame_projected_uvz, projected_valid = project_world_points(
            world_points,
            intrinsics=intrinsics[frame_idx],
            w2c=extrinsics_w2c[frame_idx],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        projected_uvz[:, frame_idx] = frame_projected_uvz.astype(np.float32, copy=False)
        projected_valid_mask[:, frame_idx] = projected_valid.astype(bool, copy=False)

        observed_depth, observed_valid = sample_image_at_keypoints(depth_frames[frame_idx], frame_projected_uvz[:, :2])
        observed_world, observed_world_valid = unproject_keypoints_to_world(
            frame_projected_uvz[:, :2],
            observed_depth,
            intrinsics=intrinsics[frame_idx],
            w2c=extrinsics_w2c[frame_idx],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        frame_query_reprojected_uvz, frame_query_reprojected_valid = project_world_points(
            observed_world,
            intrinsics=intrinsics[query_frame],
            w2c=extrinsics_w2c[query_frame],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        query_reprojected_uvz[:, frame_idx] = frame_query_reprojected_uvz.astype(np.float32, copy=False)
        query_reprojected_valid_mask[:, frame_idx] = frame_query_reprojected_valid.astype(bool, copy=False)

        frame_compare_mask = (
            query_world_valid
            & projected_valid
            & observed_valid
            & observed_world_valid
            & frame_query_reprojected_valid
        )
        compare_mask[:, frame_idx] = frame_compare_mask.astype(bool, copy=False)
        if not np.any(frame_compare_mask):
            continue

        frame_uv_delta = np.linalg.norm(frame_query_reprojected_uvz[:, :2] - keypoints, axis=1).astype(np.float32)
        frame_depth_delta = np.abs(frame_query_reprojected_uvz[:, 2] - query_depth_values).astype(np.float32)
        uv_delta_px[frame_compare_mask, frame_idx] = frame_uv_delta[frame_compare_mask]
        depth_delta_m[frame_compare_mask, frame_idx] = frame_depth_delta[frame_compare_mask]

    uv_stable_mask = compare_mask & np.isfinite(uv_delta_px) & (uv_delta_px < float(uv_threshold_px))
    depth_jump_mask = compare_mask & np.isfinite(depth_delta_m) & (depth_delta_m > float(depth_threshold_m))
    depth_anomaly_mask = uv_stable_mask & depth_jump_mask

    compare_frame_count = compare_mask.sum(axis=1).astype(np.uint16)
    uv_stable_hit_count = uv_stable_mask.sum(axis=1).astype(np.uint16)
    depth_jump_hit_count = depth_jump_mask.sum(axis=1).astype(np.uint16)
    depth_anomaly_hit_count = depth_anomaly_mask.sum(axis=1).astype(np.uint16)
    reliable_track_mask = query_world_valid & (depth_anomaly_hit_count == 0)
    first_anomaly_step = _build_first_trigger_step(
        query_world_valid_mask=query_world_valid,
        trigger_mask=depth_anomaly_mask,
    )
    min_uv_delta_px = _masked_nanmin(uv_delta_px, compare_mask)
    max_depth_delta_m = _masked_nanmax(depth_delta_m, compare_mask)

    summary = {
        "track_count": int(keypoints.shape[0]),
        "query_world_valid_count": int(np.count_nonzero(query_world_valid)),
        "reliable_track_count": int(np.count_nonzero(reliable_track_mask)),
        "depth_anomaly_track_count": int(np.count_nonzero(query_world_valid & (~reliable_track_mask))),
        "compare_frame_count_summary": _finite_summary(compare_frame_count.astype(np.float32)),
        "uv_stable_hit_count_summary": _finite_summary(uv_stable_hit_count.astype(np.float32)),
        "depth_jump_hit_count_summary": _finite_summary(depth_jump_hit_count.astype(np.float32)),
        "depth_anomaly_hit_count_summary": _finite_summary(depth_anomaly_hit_count.astype(np.float32)),
        "min_uv_delta_px_summary": _finite_summary(min_uv_delta_px),
        "max_depth_delta_m_summary": _finite_summary(max_depth_delta_m),
    }

    return {
        "query_frame": int(query_frame),
        "image_height": int(height),
        "image_width": int(width),
        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "uv_threshold_px": float(uv_threshold_px),
        "depth_threshold_m": float(depth_threshold_m),
        "keypoints": keypoints.astype(np.float32, copy=False),
        "query_depth_values": query_depth_values.astype(np.float32, copy=False),
        "query_depth_sample_valid_mask": query_depth_sample_valid.astype(bool, copy=False),
        "query_world_valid_mask": query_world_valid.astype(bool, copy=False),
        "world_points": world_points.astype(np.float32, copy=False),
        "projected_uvz": projected_uvz.astype(np.float32, copy=False),
        "projected_valid_mask": projected_valid_mask.astype(bool, copy=False),
        "query_reprojected_uvz": query_reprojected_uvz.astype(np.float32, copy=False),
        "query_reprojected_valid_mask": query_reprojected_valid_mask.astype(bool, copy=False),
        "compare_mask": compare_mask.astype(bool, copy=False),
        "uv_delta_px": uv_delta_px.astype(np.float32, copy=False),
        "depth_delta_m": depth_delta_m.astype(np.float32, copy=False),
        "uv_stable_mask": uv_stable_mask.astype(bool, copy=False),
        "depth_jump_mask": depth_jump_mask.astype(bool, copy=False),
        "depth_anomaly_mask": depth_anomaly_mask.astype(bool, copy=False),
        "compare_frame_count": compare_frame_count.astype(np.uint16, copy=False),
        "uv_stable_hit_count": uv_stable_hit_count.astype(np.uint16, copy=False),
        "depth_jump_hit_count": depth_jump_hit_count.astype(np.uint16, copy=False),
        "depth_anomaly_hit_count": depth_anomaly_hit_count.astype(np.uint16, copy=False),
        "reliable_track_mask": reliable_track_mask.astype(bool, copy=False),
        "first_anomaly_step": first_anomaly_step.astype(np.int32, copy=False),
        "min_uv_delta_px": min_uv_delta_px.astype(np.float32, copy=False),
        "max_depth_delta_m": max_depth_delta_m.astype(np.float32, copy=False),
        "summary": summary,
    }


def summarize_query_fixed_view_depth_gate(
    *,
    gate_result: dict[str, Any],
    tracked_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    reliable_track_mask = np.asarray(gate_result["reliable_track_mask"], dtype=bool)
    query_world_valid_mask = np.asarray(gate_result["query_world_valid_mask"], dtype=bool)
    depth_anomaly_hit_count = np.asarray(gate_result["depth_anomaly_hit_count"], dtype=np.uint16)
    first_anomaly_step = np.asarray(gate_result["first_anomaly_step"], dtype=np.int32)
    max_depth_delta_m = np.asarray(gate_result["max_depth_delta_m"], dtype=np.float32)

    if tracked_mask is None:
        effective_tracked_mask = np.ones(reliable_track_mask.shape[0], dtype=bool)
    else:
        effective_tracked_mask = np.asarray(tracked_mask, dtype=bool).reshape(-1)
        if effective_tracked_mask.shape != reliable_track_mask.shape:
            raise ValueError(
                f"Expected tracked_mask shape {reliable_track_mask.shape}, got {effective_tracked_mask.shape}"
            )

    removed_tracked_mask = effective_tracked_mask & (~reliable_track_mask)
    removed_tracked_first_anomaly = first_anomaly_step[removed_tracked_mask]
    if removed_tracked_first_anomaly.size > 0:
        unique_steps, counts = np.unique(removed_tracked_first_anomaly, return_counts=True)
        first_anomaly_histogram = {
            str(int(step)): int(count) for step, count in zip(unique_steps.tolist(), counts.tolist(), strict=False)
        }
    else:
        first_anomaly_histogram = {}

    return {
        "track_count": int(reliable_track_mask.shape[0]),
        "tracked_count": int(np.count_nonzero(effective_tracked_mask)),
        "query_world_valid_count": int(np.count_nonzero(query_world_valid_mask)),
        "reliable_track_count": int(np.count_nonzero(reliable_track_mask)),
        "removed_track_count": int(np.count_nonzero(~reliable_track_mask)),
        "removed_tracked_count": int(np.count_nonzero(removed_tracked_mask)),
        "depth_anomaly_hit_count_summary": _finite_summary(depth_anomaly_hit_count.astype(np.float32)),
        "removed_track_max_depth_delta_m_summary": _finite_summary(max_depth_delta_m[~reliable_track_mask]),
        "kept_track_max_depth_delta_m_summary": _finite_summary(max_depth_delta_m[reliable_track_mask]),
        "removed_tracked_first_anomaly_histogram": first_anomaly_histogram,
    }
