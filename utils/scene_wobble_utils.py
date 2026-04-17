from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_WOBBLE_MIN_QUERY_DEPTH_M = 0.2
DEFAULT_WOBBLE_MIN_BORDER_DIST_PX = 60.0
DEFAULT_WOBBLE_MIN_ANCHOR_COUNT = 32
DEFAULT_WOBBLE_GLOBAL_DISP_THRESHOLD_PX = 3.0


def _compute_query_border_distances_px(
    keypoints: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")
    x = keypoints[:, 0].astype(np.float32, copy=False)
    y = keypoints[:, 1].astype(np.float32, copy=False)
    border_dist = np.minimum.reduce(
        [
            np.maximum(x, 0.0),
            np.maximum(y, 0.0),
            np.maximum(float(width - 1) - x, 0.0),
            np.maximum(float(height - 1) - y, 0.0),
        ]
    )
    return border_dist.astype(np.float32, copy=False)


def _masked_percentile(values: np.ndarray, q: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float("nan")
    return float(np.percentile(arr[finite], q))


def compute_scene_wobble_summary(
    traj_uvz: np.ndarray,
    *,
    traj_valid_mask: np.ndarray | None = None,
    keypoints: np.ndarray | None = None,
    query_border_dist_px: np.ndarray | None = None,
    valid_steps: np.ndarray | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
    min_query_depth_m: float = DEFAULT_WOBBLE_MIN_QUERY_DEPTH_M,
    min_border_dist_px: float = DEFAULT_WOBBLE_MIN_BORDER_DIST_PX,
    min_anchor_count: int = DEFAULT_WOBBLE_MIN_ANCHOR_COUNT,
    global_disp_threshold_px: float = DEFAULT_WOBBLE_GLOBAL_DISP_THRESHOLD_PX,
) -> dict[str, Any]:
    """Measure fixed-view common drift over deep and central anchors.

    The intent is to separate:
    - bad seeds / local slip: large per-track motion but weak common drift
    - scene-level wobble: many deep-central tracks move together in fixed view
    """
    traj_uvz = np.asarray(traj_uvz, dtype=np.float32)
    if traj_uvz.ndim != 3 or traj_uvz.shape[-1] != 3:
        raise ValueError(f"Expected traj_uvz shape (N,T,3), got {traj_uvz.shape}")
    num_tracks, num_frames, _ = traj_uvz.shape

    if traj_valid_mask is None:
        traj_valid_mask = np.ones(num_tracks, dtype=bool)
    else:
        traj_valid_mask = np.asarray(traj_valid_mask, dtype=bool).reshape(-1)
        if traj_valid_mask.shape != (num_tracks,):
            raise ValueError(f"Expected traj_valid_mask shape {(num_tracks,)}, got {traj_valid_mask.shape}")

    if query_border_dist_px is None:
        if keypoints is None:
            raise ValueError("keypoints are required when query_border_dist_px is not provided")
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if keypoints.shape != (num_tracks, 2):
            raise ValueError(f"Expected keypoints shape {(num_tracks, 2)}, got {keypoints.shape}")
        if image_height is None or image_width is None:
            raise ValueError(
                "image_height and image_width are required when computing query_border_dist_px from keypoints"
            )
        query_border_dist_px = _compute_query_border_distances_px(
            keypoints,
            height=max(int(image_height), 1),
            width=max(int(image_width), 1),
        )
    else:
        query_border_dist_px = np.asarray(query_border_dist_px, dtype=np.float32).reshape(-1)
        if query_border_dist_px.shape != (num_tracks,):
            raise ValueError(
                f"Expected query_border_dist_px shape {(num_tracks,)}, got {query_border_dist_px.shape}"
            )

    if valid_steps is None:
        valid_steps = np.ones(num_frames, dtype=bool)
    else:
        valid_steps = np.asarray(valid_steps, dtype=bool).reshape(-1)
        if valid_steps.shape != (num_frames,):
            raise ValueError(f"Expected valid_steps shape {(num_frames,)}, got {valid_steps.shape}")

    finite_step_mask = np.isfinite(traj_uvz).all(axis=-1)
    query_depth = traj_uvz[:, 0, 2].astype(np.float32, copy=False)
    query_finite_mask = finite_step_mask[:, 0]
    anchor_mask = (
        traj_valid_mask
        & query_finite_mask
        & np.isfinite(query_border_dist_px)
        & (query_depth > float(min_query_depth_m))
        & (query_border_dist_px >= float(min_border_dist_px))
    )

    displacement_uv = traj_uvz[..., :2] - traj_uvz[:, :1, :2]
    per_step_anchor_count = np.zeros(num_frames, dtype=np.int32)
    per_step_global_dx_px = np.full(num_frames, np.nan, dtype=np.float32)
    per_step_global_dy_px = np.full(num_frames, np.nan, dtype=np.float32)
    per_step_global_disp_px = np.full(num_frames, np.nan, dtype=np.float32)
    per_step_track_disp_median_px = np.full(num_frames, np.nan, dtype=np.float32)
    per_step_track_disp_p95_px = np.full(num_frames, np.nan, dtype=np.float32)
    per_step_residual_median_px = np.full(num_frames, np.nan, dtype=np.float32)
    per_step_residual_p95_px = np.full(num_frames, np.nan, dtype=np.float32)

    for step_idx in range(num_frames):
        if not bool(valid_steps[step_idx]):
            continue
        step_mask = anchor_mask & finite_step_mask[:, step_idx]
        step_count = int(np.count_nonzero(step_mask))
        per_step_anchor_count[step_idx] = step_count
        if step_count == 0:
            continue

        step_disp = np.asarray(displacement_uv[step_mask, step_idx], dtype=np.float32)
        global_dx = float(np.median(step_disp[:, 0]))
        global_dy = float(np.median(step_disp[:, 1]))
        per_step_global_dx_px[step_idx] = global_dx
        per_step_global_dy_px[step_idx] = global_dy
        per_step_global_disp_px[step_idx] = float(np.hypot(global_dx, global_dy))

        step_norm = np.linalg.norm(step_disp, axis=1)
        per_step_track_disp_median_px[step_idx] = float(np.median(step_norm))
        per_step_track_disp_p95_px[step_idx] = _masked_percentile(step_norm, 95.0)

        residual = step_disp - np.array([global_dx, global_dy], dtype=np.float32)
        residual_norm = np.linalg.norm(residual, axis=1)
        per_step_residual_median_px[step_idx] = float(np.median(residual_norm))
        per_step_residual_p95_px[step_idx] = _masked_percentile(residual_norm, 95.0)

    valid_step_indices = np.flatnonzero(valid_steps)
    final_step_index = int(valid_step_indices[-1]) if valid_step_indices.size > 0 else int(num_frames - 1)
    final_anchor_count = int(per_step_anchor_count[final_step_index]) if num_frames > 0 else 0
    global_final_disp_px = (
        float(per_step_global_disp_px[final_step_index]) if num_frames > 0 else float("nan")
    )
    residual_final_p95_px = (
        float(per_step_residual_p95_px[final_step_index]) if num_frames > 0 else float("nan")
    )
    track_final_p95_px = (
        float(per_step_track_disp_p95_px[final_step_index]) if num_frames > 0 else float("nan")
    )

    has_sufficient_anchors = int(np.count_nonzero(anchor_mask)) >= int(min_anchor_count)
    geometry_unstable = bool(
        has_sufficient_anchors
        and np.isfinite(global_final_disp_px)
        and global_final_disp_px >= float(global_disp_threshold_px)
    )

    return {
        "anchor_mask": anchor_mask.astype(bool, copy=False),
        "anchor_count": int(np.count_nonzero(anchor_mask)),
        "has_sufficient_anchors": bool(has_sufficient_anchors),
        "geometry_unstable": geometry_unstable,
        "final_step_index": final_step_index,
        "final_anchor_count": final_anchor_count,
        "global_final_disp_px": global_final_disp_px,
        "residual_final_p95_px": residual_final_p95_px,
        "track_final_p95_px": track_final_p95_px,
        "global_disp_p95_px": _masked_percentile(per_step_global_disp_px, 95.0),
        "residual_disp_p95_px": _masked_percentile(per_step_residual_p95_px, 95.0),
        "track_disp_p95_px": _masked_percentile(per_step_track_disp_p95_px, 95.0),
        "per_step_anchor_count": per_step_anchor_count,
        "per_step_global_dx_px": per_step_global_dx_px,
        "per_step_global_dy_px": per_step_global_dy_px,
        "per_step_global_disp_px": per_step_global_disp_px,
        "per_step_track_disp_median_px": per_step_track_disp_median_px,
        "per_step_track_disp_p95_px": per_step_track_disp_p95_px,
        "per_step_residual_median_px": per_step_residual_median_px,
        "per_step_residual_p95_px": per_step_residual_p95_px,
    }
