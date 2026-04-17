from __future__ import annotations

from typing import Any

import numpy as np

from utils.external_wobble_diagnostics import (
    build_query_anchor_bundle_from_keypoints,
    project_world_points,
    sample_image_at_keypoints,
    unproject_keypoints_to_world,
)


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


def compute_static_geometry_track_drift(
    depth_frames: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    keypoints: np.ndarray,
    query_frame: int = 0,
    min_query_depth_m: float = 0.01,
    min_border_dist_px: float = 0.0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
) -> dict[str, np.ndarray]:
    depth_frames = np.asarray(depth_frames, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    bundle = build_query_anchor_bundle_from_keypoints(
        depth_frames,
        intrinsics,
        extrinsics_w2c,
        keypoints=keypoints,
        query_frame=query_frame,
        min_query_depth_m=min_query_depth_m,
        min_border_dist_px=min_border_dist_px,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    anchor_mask = np.asarray(bundle["anchor_mask"], dtype=bool)
    world_points = np.asarray(bundle["world_points"], dtype=np.float32)
    frame_count = int(depth_frames.shape[0])
    track_count = int(keypoints.shape[0])
    per_track_drift_px = np.full((track_count, frame_count), np.nan, dtype=np.float32)
    per_track_valid = np.zeros((track_count, frame_count), dtype=bool)

    for frame_idx in range(frame_count):
        projected_uvz, projected_valid = project_world_points(
            world_points,
            intrinsics=intrinsics[frame_idx],
            w2c=extrinsics_w2c[frame_idx],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        observed_depth, observed_valid = sample_image_at_keypoints(depth_frames[frame_idx], projected_uvz[:, :2])
        observed_world, observed_world_valid = unproject_keypoints_to_world(
            projected_uvz[:, :2],
            observed_depth,
            intrinsics=intrinsics[frame_idx],
            w2c=extrinsics_w2c[frame_idx],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        query_reproj_uvz, query_reproj_valid = project_world_points(
            observed_world,
            intrinsics=intrinsics[query_frame],
            w2c=extrinsics_w2c[query_frame],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        valid = anchor_mask & projected_valid & observed_valid & observed_world_valid & query_reproj_valid
        if np.any(valid):
            drift = np.linalg.norm(query_reproj_uvz[valid, :2] - keypoints[valid], axis=1).astype(np.float32)
            per_track_drift_px[valid, frame_idx] = drift
            per_track_valid[valid, frame_idx] = True

    return {
        "anchor_mask": anchor_mask.astype(bool, copy=False),
        "query_depth_values": np.asarray(bundle["query_depth_values"], dtype=np.float32),
        "border_dist_px": np.asarray(bundle["border_dist_px"], dtype=np.float32),
        "per_track_query_reproj_drift_px": per_track_drift_px,
        "per_track_query_reproj_valid": per_track_valid.astype(bool, copy=False),
        "final_query_reproj_drift_px": per_track_drift_px[:, -1].astype(np.float32, copy=False),
        "final_query_reproj_valid": per_track_valid[:, -1].astype(bool, copy=False),
    }


def summarize_tracker_geometry_interaction(
    *,
    traj_uvz: np.ndarray,
    keypoints: np.ndarray,
    static_geometry_drift_px: np.ndarray,
    static_geometry_valid: np.ndarray,
    traj_valid_mask: np.ndarray | None = None,
    valid_steps: np.ndarray | None = None,
    geom_stable_threshold_px: float = 1.0,
    tracker_unstable_threshold_px: float = 3.0,
    excess_threshold_px: float = 2.0,
) -> dict[str, Any]:
    traj_uvz = np.asarray(traj_uvz, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    static_geometry_drift_px = np.asarray(static_geometry_drift_px, dtype=np.float32)
    static_geometry_valid = np.asarray(static_geometry_valid, dtype=bool)
    if traj_uvz.ndim != 3 or traj_uvz.shape[-1] < 2:
        raise ValueError(f"Expected traj_uvz shape (N,T,>=2), got {traj_uvz.shape}")
    if keypoints.shape != (traj_uvz.shape[0], 2):
        raise ValueError(f"Expected keypoints shape {(traj_uvz.shape[0], 2)}, got {keypoints.shape}")
    if static_geometry_drift_px.shape != (traj_uvz.shape[0],):
        raise ValueError(
            f"Expected static_geometry_drift_px shape {(traj_uvz.shape[0],)}, got {static_geometry_drift_px.shape}"
        )
    if static_geometry_valid.shape != (traj_uvz.shape[0],):
        raise ValueError(
            f"Expected static_geometry_valid shape {(traj_uvz.shape[0],)}, got {static_geometry_valid.shape}"
        )
    num_steps = int(traj_uvz.shape[1])
    if valid_steps is None:
        final_step_index = int(num_steps - 1)
    else:
        valid_steps = np.asarray(valid_steps, dtype=bool).reshape(-1)
        if valid_steps.shape != (num_steps,):
            raise ValueError(f"Expected valid_steps shape {(num_steps,)}, got {valid_steps.shape}")
        valid_step_indices = np.flatnonzero(valid_steps)
        final_step_index = int(valid_step_indices[-1]) if valid_step_indices.size > 0 else int(num_steps - 1)

    tracker_final_uv = np.asarray(traj_uvz[:, final_step_index, :2], dtype=np.float32)
    tracker_final_drift_px = np.linalg.norm(tracker_final_uv - keypoints, axis=1).astype(np.float32)
    tracker_final_valid = np.isfinite(tracker_final_uv).all(axis=1) & np.isfinite(tracker_final_drift_px)
    if traj_valid_mask is not None:
        tracker_final_valid &= np.asarray(traj_valid_mask, dtype=bool).reshape(-1)

    compare_mask = tracker_final_valid & static_geometry_valid
    excess_final_drift_px = np.full(tracker_final_drift_px.shape, np.nan, dtype=np.float32)
    excess_final_drift_px[compare_mask] = (
        tracker_final_drift_px[compare_mask] - static_geometry_drift_px[compare_mask]
    ).astype(np.float32)

    tracker_local_interaction_mask = (
        compare_mask
        & (static_geometry_drift_px <= float(geom_stable_threshold_px))
        & (tracker_final_drift_px >= float(tracker_unstable_threshold_px))
        & (excess_final_drift_px >= float(excess_threshold_px))
    )
    geometry_limited_mask = (
        compare_mask
        & (static_geometry_drift_px > float(geom_stable_threshold_px))
        & (tracker_final_drift_px >= float(tracker_unstable_threshold_px))
    )

    return {
        "track_count": int(traj_uvz.shape[0]),
        "final_step_index": final_step_index,
        "compare_count": int(np.count_nonzero(compare_mask)),
        "tracker_final_drift_px": tracker_final_drift_px.astype(np.float32, copy=False),
        "tracker_final_valid": tracker_final_valid.astype(bool, copy=False),
        "static_geometry_final_drift_px": static_geometry_drift_px.astype(np.float32, copy=False),
        "static_geometry_final_valid": static_geometry_valid.astype(bool, copy=False),
        "excess_final_drift_px": excess_final_drift_px.astype(np.float32, copy=False),
        "tracker_local_interaction_mask": tracker_local_interaction_mask.astype(bool, copy=False),
        "geometry_limited_mask": geometry_limited_mask.astype(bool, copy=False),
        "tracker_final_drift_summary": _finite_summary(tracker_final_drift_px[tracker_final_valid]),
        "static_geometry_final_drift_summary": _finite_summary(static_geometry_drift_px[static_geometry_valid]),
        "excess_final_drift_summary": _finite_summary(excess_final_drift_px[compare_mask]),
    }
