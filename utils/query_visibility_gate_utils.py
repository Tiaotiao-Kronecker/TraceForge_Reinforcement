from __future__ import annotations

from typing import Any

import numpy as np

from utils.external_wobble_diagnostics import (
    compute_border_distance_px,
    project_world_points,
    sample_image_at_keypoints,
    unproject_keypoints_to_world,
)

DEFAULT_QUERY_VISIBILITY_GATE_NEAR_DEPTH_EXEMPT_THRESHOLD_M = 0.0


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


def _build_first_invalid_step(
    *,
    query_world_valid_mask: np.ndarray,
    future_visible_mask: np.ndarray,
    future_step_indices: np.ndarray,
) -> np.ndarray:
    query_world_valid_mask = np.asarray(query_world_valid_mask, dtype=bool).reshape(-1)
    future_visible_mask = np.asarray(future_visible_mask, dtype=bool)
    future_step_indices = np.asarray(future_step_indices, dtype=np.int32).reshape(-1)
    if future_visible_mask.ndim != 2:
        raise ValueError(f"Expected future_visible_mask shape (N,T), got {future_visible_mask.shape}")
    if future_visible_mask.shape[0] != query_world_valid_mask.shape[0]:
        raise ValueError(
            "Expected future_visible_mask and query_world_valid_mask to share track count, "
            f"got {future_visible_mask.shape[0]} vs {query_world_valid_mask.shape[0]}"
        )
    if future_visible_mask.shape[1] != future_step_indices.shape[0]:
        raise ValueError(
            "Expected future_visible_mask and future_step_indices to share future-frame count, "
            f"got {future_visible_mask.shape[1]} vs {future_step_indices.shape[0]}"
        )

    first_invalid_step = np.full(query_world_valid_mask.shape[0], -1, dtype=np.int32)
    first_invalid_step[~query_world_valid_mask] = 0
    if future_step_indices.size == 0:
        return first_invalid_step

    invalid_future = (~future_visible_mask) & query_world_valid_mask[:, None]
    has_invalid_future = np.any(invalid_future, axis=1)
    if np.any(has_invalid_future):
        first_future_offset = np.argmax(invalid_future[has_invalid_future], axis=1)
        first_invalid_step[has_invalid_future] = future_step_indices[first_future_offset]
    return first_invalid_step.astype(np.int32, copy=False)


def compute_query_visibility_gate(
    depth_frames: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    keypoints: np.ndarray,
    query_frame: int = 0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    min_border_dist_px: float = 0.0,
    near_depth_exempt_threshold_m: float = DEFAULT_QUERY_VISIBILITY_GATE_NEAR_DEPTH_EXEMPT_THRESHOLD_M,
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
    if float(near_depth_exempt_threshold_m) < 0.0:
        raise ValueError(
            "Expected near_depth_exempt_threshold_m >= 0, "
            f"got {near_depth_exempt_threshold_m}"
        )

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
    projected_in_bounds_mask = np.zeros((keypoints.shape[0], frame_count), dtype=bool)
    projected_visible_mask = np.zeros((keypoints.shape[0], frame_count), dtype=bool)
    projected_border_dist_px = np.full((keypoints.shape[0], frame_count), np.nan, dtype=np.float32)

    for frame_idx in range(frame_count):
        uvz, projected_valid = project_world_points(
            world_points,
            intrinsics=intrinsics[frame_idx],
            w2c=extrinsics_w2c[frame_idx],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        projected_uvz[:, frame_idx] = uvz.astype(np.float32, copy=False)
        projected_valid_mask[:, frame_idx] = projected_valid.astype(bool, copy=False)

        u = uvz[:, 0]
        v = uvz[:, 1]
        in_bounds = (
            projected_valid
            & np.isfinite(u)
            & np.isfinite(v)
            & (u >= 0.0)
            & (u <= float(width - 1))
            & (v >= 0.0)
            & (v <= float(height - 1))
        )
        border_dist = compute_border_distance_px(
            uvz[:, :2],
            height=int(height),
            width=int(width),
        )
        visible = in_bounds & np.isfinite(border_dist) & (border_dist >= float(min_border_dist_px))

        projected_in_bounds_mask[:, frame_idx] = in_bounds.astype(bool, copy=False)
        projected_visible_mask[:, frame_idx] = visible.astype(bool, copy=False)
        projected_border_dist_px[:, frame_idx] = border_dist.astype(np.float32, copy=False)

    future_step_indices = np.arange(query_frame + 1, frame_count, dtype=np.int32)
    future_visible_mask = projected_visible_mask[:, future_step_indices] if future_step_indices.size > 0 else np.ones(
        (keypoints.shape[0], 0), dtype=bool
    )
    all_future_visible_mask = query_world_valid.copy()
    if future_step_indices.size > 0:
        all_future_visible_mask &= np.all(future_visible_mask, axis=1)

    near_depth_exempt_mask = np.zeros(keypoints.shape[0], dtype=bool)
    if float(near_depth_exempt_threshold_m) > 0.0:
        near_depth_exempt_mask = (
            query_world_valid
            & np.isfinite(query_depth_values)
            & (query_depth_values < float(near_depth_exempt_threshold_m))
        )
    reliable_track_mask = all_future_visible_mask | near_depth_exempt_mask

    future_visible_ratio = np.ones(keypoints.shape[0], dtype=np.float32)
    if future_step_indices.size > 0:
        future_visible_ratio = np.mean(future_visible_mask.astype(np.float32), axis=1).astype(np.float32, copy=False)
    future_visible_ratio[~query_world_valid] = 0.0

    first_invalid_step = _build_first_invalid_step(
        query_world_valid_mask=query_world_valid,
        future_visible_mask=future_visible_mask,
        future_step_indices=future_step_indices,
    )

    ever_out_of_view_mask = query_world_valid & (~all_future_visible_mask)
    query_border_dist_px = compute_border_distance_px(
        keypoints,
        height=int(height),
        width=int(width),
    )
    summary = {
        "track_count": int(keypoints.shape[0]),
        "query_depth_valid_count": int(np.count_nonzero(query_depth_sample_valid)),
        "query_world_valid_count": int(np.count_nonzero(query_world_valid)),
        "reliable_track_count": int(np.count_nonzero(reliable_track_mask)),
        "unreliable_track_count": int(np.count_nonzero(~reliable_track_mask)),
        "ever_out_of_view_count": int(np.count_nonzero(ever_out_of_view_mask)),
        "near_depth_exempt_count": int(np.count_nonzero(near_depth_exempt_mask)),
        "future_frame_count": int(future_step_indices.shape[0]),
        "future_visible_ratio_summary": _finite_summary(future_visible_ratio),
        "query_border_dist_px_summary": _finite_summary(query_border_dist_px),
    }

    return {
        "query_frame": int(query_frame),
        "image_height": int(height),
        "image_width": int(width),
        "min_depth": float(min_depth),
        "max_depth": float(max_depth),
        "min_border_dist_px": float(min_border_dist_px),
        "near_depth_exempt_threshold_m": float(near_depth_exempt_threshold_m),
        "keypoints": keypoints.astype(np.float32, copy=False),
        "query_depth_values": query_depth_values.astype(np.float32, copy=False),
        "query_depth_sample_valid_mask": query_depth_sample_valid.astype(bool, copy=False),
        "query_world_valid_mask": query_world_valid.astype(bool, copy=False),
        "query_border_dist_px": query_border_dist_px.astype(np.float32, copy=False),
        "world_points": world_points.astype(np.float32, copy=False),
        "projected_uvz": projected_uvz.astype(np.float32, copy=False),
        "projected_valid_mask": projected_valid_mask.astype(bool, copy=False),
        "projected_in_bounds_mask": projected_in_bounds_mask.astype(bool, copy=False),
        "projected_visible_mask": projected_visible_mask.astype(bool, copy=False),
        "projected_border_dist_px": projected_border_dist_px.astype(np.float32, copy=False),
        "future_step_indices": future_step_indices.astype(np.int32, copy=False),
        "future_visible_mask": future_visible_mask.astype(bool, copy=False),
        "future_visible_ratio": future_visible_ratio.astype(np.float32, copy=False),
        "near_depth_exempt_mask": near_depth_exempt_mask.astype(bool, copy=False),
        "reliable_track_mask": reliable_track_mask.astype(bool, copy=False),
        "ever_out_of_view_mask": ever_out_of_view_mask.astype(bool, copy=False),
        "first_invalid_step": first_invalid_step.astype(np.int32, copy=False),
        "summary": summary,
    }


def summarize_query_visibility_gate(
    *,
    gate_result: dict[str, Any],
    traj_valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    reliable_track_mask = np.asarray(gate_result["reliable_track_mask"], dtype=bool)
    query_world_valid_mask = np.asarray(gate_result["query_world_valid_mask"], dtype=bool)
    future_visible_ratio = np.asarray(gate_result["future_visible_ratio"], dtype=np.float32)
    query_border_dist_px = np.asarray(gate_result["query_border_dist_px"], dtype=np.float32)
    first_invalid_step = np.asarray(gate_result["first_invalid_step"], dtype=np.int32)

    if traj_valid_mask is None:
        tracked_valid_mask = np.ones(reliable_track_mask.shape[0], dtype=bool)
    else:
        tracked_valid_mask = np.asarray(traj_valid_mask, dtype=bool).reshape(-1)
        if tracked_valid_mask.shape != reliable_track_mask.shape:
            raise ValueError(
                f"Expected traj_valid_mask shape {reliable_track_mask.shape}, got {tracked_valid_mask.shape}"
            )

    removed_tracked_mask = tracked_valid_mask & (~reliable_track_mask)
    removed_tracked_first_invalid = first_invalid_step[removed_tracked_mask]
    if removed_tracked_first_invalid.size > 0:
        unique_steps, counts = np.unique(removed_tracked_first_invalid, return_counts=True)
        first_invalid_histogram = {
            str(int(step)): int(count) for step, count in zip(unique_steps.tolist(), counts.tolist(), strict=False)
        }
    else:
        first_invalid_histogram = {}

    return {
        "track_count": int(reliable_track_mask.shape[0]),
        "tracked_valid_count": int(np.count_nonzero(tracked_valid_mask)),
        "query_world_valid_count": int(np.count_nonzero(query_world_valid_mask)),
        "reliable_track_count": int(np.count_nonzero(reliable_track_mask)),
        "removed_track_count": int(np.count_nonzero(~reliable_track_mask)),
        "removed_tracked_count": int(np.count_nonzero(removed_tracked_mask)),
        "reliable_ratio": float(np.mean(reliable_track_mask.astype(np.float32))),
        "future_visible_ratio_summary": _finite_summary(future_visible_ratio),
        "removed_track_query_border_dist_px_summary": _finite_summary(query_border_dist_px[~reliable_track_mask]),
        "kept_track_query_border_dist_px_summary": _finite_summary(query_border_dist_px[reliable_track_mask]),
        "removed_tracked_first_invalid_histogram": first_invalid_histogram,
    }
