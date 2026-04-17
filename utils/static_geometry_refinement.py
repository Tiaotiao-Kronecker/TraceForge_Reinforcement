from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from utils.external_wobble_diagnostics import (
    build_query_anchor_bundle,
    compute_border_distance_px,
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


def _maybe_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _normalize_query_frames(query_frames: Sequence[int] | np.ndarray) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for item in query_frames:
        value = int(item)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("Expected at least one query frame.")
    return values


def _camera_points_from_uv_depth(
    uv: np.ndarray,
    depth_values: np.ndarray,
    *,
    intrinsics: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    uv = np.asarray(uv, dtype=np.float32)
    depth_values = np.asarray(depth_values, dtype=np.float32).reshape(-1)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"Expected uv shape (N,2), got {uv.shape}")
    if depth_values.shape != (uv.shape[0],):
        raise ValueError(f"Expected depth_values shape {(uv.shape[0],)}, got {depth_values.shape}")
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    z = depth_values.astype(np.float32, copy=False)
    valid = np.isfinite(z) & (z > float(min_depth)) & (z < float(max_depth))
    x = np.where(valid, (uv[:, 0] - cx) * z / (fx + 1e-8), np.nan)
    y = np.where(valid, (uv[:, 1] - cy) * z / (fy + 1e-8), np.nan)
    camera_points = np.stack([x, y, np.where(valid, z, np.nan)], axis=1).astype(np.float32)
    return camera_points, valid.astype(bool)


def _estimate_rigid_transform_world_to_camera(
    world_points: np.ndarray,
    camera_points: np.ndarray,
) -> np.ndarray:
    world_points = np.asarray(world_points, dtype=np.float32)
    camera_points = np.asarray(camera_points, dtype=np.float32)
    if world_points.shape != camera_points.shape or world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError(
            f"Expected world_points/camera_points shape (N,3), got {world_points.shape} vs {camera_points.shape}"
        )
    if world_points.shape[0] < 3:
        raise ValueError(f"Expected at least 3 correspondences, got {world_points.shape[0]}")
    src_center = np.mean(world_points, axis=0)
    dst_center = np.mean(camera_points, axis=0)
    src_centered = world_points - src_center[None, :]
    dst_centered = camera_points - dst_center[None, :]
    covariance = src_centered.T @ dst_centered
    u, _, vh = np.linalg.svd(covariance.astype(np.float64), full_matrices=False)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1, :] *= -1.0
        rotation = vh.T @ u.T
    translation = dst_center.astype(np.float64) - rotation @ src_center.astype(np.float64)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = rotation.astype(np.float32)
    w2c[:3, 3] = translation.astype(np.float32)
    return w2c


def _smooth_valid_vectors(
    vectors: np.ndarray,
    valid_mask: np.ndarray,
    weights: np.ndarray,
    *,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if vectors.ndim != 2:
        raise ValueError(f"Expected vectors shape (T,D), got {vectors.shape}")
    if valid_mask.shape != (vectors.shape[0],):
        raise ValueError(f"Expected valid_mask shape {(vectors.shape[0],)}, got {valid_mask.shape}")
    if weights.shape != (vectors.shape[0],):
        raise ValueError(f"Expected weights shape {(vectors.shape[0],)}, got {weights.shape}")
    radius = int(radius)
    smoothed = np.zeros_like(vectors, dtype=np.float32)
    smooth_valid = np.zeros(vectors.shape[0], dtype=bool)
    if radius <= 0:
        smoothed[valid_mask] = vectors[valid_mask]
        smooth_valid[valid_mask] = True
        return smoothed, smooth_valid

    for frame_idx in range(vectors.shape[0]):
        start = max(0, frame_idx - radius)
        stop = min(vectors.shape[0], frame_idx + radius + 1)
        local_valid = valid_mask[start:stop]
        if not np.any(local_valid):
            continue
        local_weights = np.maximum(weights[start:stop], 1e-6).astype(np.float32)
        local_weights = np.where(local_valid, local_weights, 0.0)
        denom = float(np.sum(local_weights))
        if denom <= 0.0:
            continue
        smoothed[frame_idx] = (
            np.sum(vectors[start:stop] * local_weights[:, None], axis=0) / denom
        ).astype(np.float32)
        smooth_valid[frame_idx] = True
    return smoothed.astype(np.float32), smooth_valid.astype(bool)


def compute_static_geometry_track_consistency(
    depth_frames: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    query_anchor_bundle: dict[str, np.ndarray | int] | None = None,
    query_frame: int,
    grid_size: int = 80,
    min_query_depth_m: float = 0.2,
    min_border_dist_px: float = 60.0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
) -> dict[str, np.ndarray | int]:
    depth_frames = np.asarray(depth_frames, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
    if depth_frames.ndim != 3:
        raise ValueError(f"Expected depth_frames shape (T,H,W), got {depth_frames.shape}")
    if intrinsics.shape != (depth_frames.shape[0], 3, 3):
        raise ValueError(f"Expected intrinsics shape {(depth_frames.shape[0], 3, 3)}, got {intrinsics.shape}")
    if extrinsics_w2c.shape != (depth_frames.shape[0], 4, 4):
        raise ValueError(
            f"Expected extrinsics_w2c shape {(depth_frames.shape[0], 4, 4)}, got {extrinsics_w2c.shape}"
        )
    frame_count = int(depth_frames.shape[0])
    query_frame = int(query_frame)
    if query_anchor_bundle is None:
        query_anchor_bundle = build_query_anchor_bundle(
            depth_frames,
            intrinsics,
            extrinsics_w2c,
            query_frame=query_frame,
            grid_size=int(grid_size),
            min_query_depth_m=float(min_query_depth_m),
            min_border_dist_px=float(min_border_dist_px),
            min_depth=float(min_depth),
            max_depth=float(max_depth),
        )

    keypoints = np.asarray(query_anchor_bundle["keypoints"], dtype=np.float32)
    anchor_mask = np.asarray(query_anchor_bundle["anchor_mask"], dtype=bool)
    world_points = np.asarray(query_anchor_bundle["world_points"], dtype=np.float32)
    track_count = int(keypoints.shape[0])

    per_track_drift_px = np.full((track_count, frame_count), np.nan, dtype=np.float32)
    per_track_valid = np.zeros((track_count, frame_count), dtype=bool)
    per_frame_global_dx_px = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_global_dy_px = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_global_disp_px = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_drift_median_px = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_drift_p95_px = np.full(frame_count, np.nan, dtype=np.float32)

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
        if not np.any(valid):
            continue
        reproj_delta = query_reproj_uvz[valid, :2] - keypoints[valid]
        reproj_drift = np.linalg.norm(reproj_delta, axis=1).astype(np.float32)
        per_track_drift_px[valid, frame_idx] = reproj_drift
        per_track_valid[valid, frame_idx] = True
        global_dx = float(np.median(reproj_delta[:, 0]))
        global_dy = float(np.median(reproj_delta[:, 1]))
        per_frame_global_dx_px[frame_idx] = global_dx
        per_frame_global_dy_px[frame_idx] = global_dy
        per_frame_global_disp_px[frame_idx] = float(np.hypot(global_dx, global_dy))
        per_frame_drift_median_px[frame_idx] = float(np.median(reproj_drift))
        per_frame_drift_p95_px[frame_idx] = float(np.percentile(reproj_drift, 95))

    return {
        "query_frame": int(query_frame),
        "grid_size": int(query_anchor_bundle.get("grid_size", grid_size)),
        "anchor_count": int(np.count_nonzero(anchor_mask)),
        "keypoints": keypoints.astype(np.float32, copy=False),
        "anchor_mask": anchor_mask.astype(bool, copy=False),
        "per_track_query_reproj_drift_px": per_track_drift_px.astype(np.float32, copy=False),
        "per_track_query_reproj_valid": per_track_valid.astype(bool, copy=False),
        "per_frame_query_reproj_global_dx_px": per_frame_global_dx_px.astype(np.float32, copy=False),
        "per_frame_query_reproj_global_dy_px": per_frame_global_dy_px.astype(np.float32, copy=False),
        "per_frame_query_reproj_global_disp_px": per_frame_global_disp_px.astype(np.float32, copy=False),
        "per_frame_query_reproj_drift_median_px": per_frame_drift_median_px.astype(np.float32, copy=False),
        "per_frame_query_reproj_drift_p95_px": per_frame_drift_p95_px.astype(np.float32, copy=False),
    }


def summarize_spatial_tail_clusters(
    keypoints: np.ndarray,
    drift_px: np.ndarray,
    valid_mask: np.ndarray,
    *,
    image_height: int,
    image_width: int,
    cell_size_px: int = 64,
    tail_threshold_px: float = 20.0,
    top_k: int = 8,
) -> list[dict[str, object]]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    drift_px = np.asarray(drift_px, dtype=np.float32).reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")
    if drift_px.shape != (keypoints.shape[0],):
        raise ValueError(f"Expected drift_px shape {(keypoints.shape[0],)}, got {drift_px.shape}")
    if valid_mask.shape != (keypoints.shape[0],):
        raise ValueError(f"Expected valid_mask shape {(keypoints.shape[0],)}, got {valid_mask.shape}")
    cell_size_px = max(int(cell_size_px), 1)
    if int(image_height) <= 0 or int(image_width) <= 0:
        raise ValueError(f"Expected positive image size, got {image_width}x{image_height}")
    finite_mask = valid_mask & np.isfinite(drift_px)
    if not np.any(finite_mask):
        return []
    cell_cols = np.clip((keypoints[:, 0] / float(cell_size_px)).astype(np.int32), 0, (image_width - 1) // cell_size_px)
    cell_rows = np.clip((keypoints[:, 1] / float(cell_size_px)).astype(np.int32), 0, (image_height - 1) // cell_size_px)
    cell_payloads: list[dict[str, object]] = []
    for row in np.unique(cell_rows[finite_mask]):
        row_mask = finite_mask & (cell_rows == int(row))
        for col in np.unique(cell_cols[row_mask]):
            mask = row_mask & (cell_cols == int(col))
            cell_drift = drift_px[mask]
            tail_mask = cell_drift >= float(tail_threshold_px)
            cell_payloads.append(
                {
                    "cell_row": int(row),
                    "cell_col": int(col),
                    "x_min_px": int(col * cell_size_px),
                    "x_max_px": int(min(image_width - 1, (col + 1) * cell_size_px - 1)),
                    "y_min_px": int(row * cell_size_px),
                    "y_max_px": int(min(image_height - 1, (row + 1) * cell_size_px - 1)),
                    "track_count": int(cell_drift.size),
                    "tail_track_count": int(np.count_nonzero(tail_mask)),
                    "tail_track_ratio": float(np.count_nonzero(tail_mask) / max(int(cell_drift.size), 1)),
                    "drift_median_px": float(np.median(cell_drift)),
                    "drift_p95_px": float(np.percentile(cell_drift, 95)),
                    "drift_max_px": float(np.max(cell_drift)),
                }
            )
    cell_payloads.sort(
        key=lambda item: (
            int(item["tail_track_count"]),
            float(item["drift_p95_px"]),
            float(item["drift_max_px"]),
        ),
        reverse=True,
    )
    return cell_payloads[: max(int(top_k), 0)]


def audit_static_geometry_heavy_tail(
    depth_frames: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    query_frame: int,
    grid_size: int = 80,
    min_query_depth_m: float = 0.2,
    min_border_dist_px: float = 60.0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    cell_size_px: int = 64,
    tail_threshold_px: float = 20.0,
    top_k_frames: int = 5,
    top_k_cells: int = 8,
) -> dict[str, object]:
    depth_frames = np.asarray(depth_frames, dtype=np.float32)
    if depth_frames.ndim != 3:
        raise ValueError(f"Expected depth_frames shape (T,H,W), got {depth_frames.shape}")
    _, image_height, image_width = depth_frames.shape
    track_metrics = compute_static_geometry_track_consistency(
        depth_frames,
        intrinsics,
        extrinsics_w2c,
        query_frame=int(query_frame),
        grid_size=int(grid_size),
        min_query_depth_m=float(min_query_depth_m),
        min_border_dist_px=float(min_border_dist_px),
        min_depth=float(min_depth),
        max_depth=float(max_depth),
    )
    drift = np.asarray(track_metrics["per_track_query_reproj_drift_px"], dtype=np.float32)
    valid = np.asarray(track_metrics["per_track_query_reproj_valid"], dtype=bool)
    per_frame_p95 = np.asarray(track_metrics["per_frame_query_reproj_drift_p95_px"], dtype=np.float32)
    per_frame_median = np.asarray(track_metrics["per_frame_query_reproj_drift_median_px"], dtype=np.float32)
    per_frame_global_disp = np.asarray(track_metrics["per_frame_query_reproj_global_disp_px"], dtype=np.float32)
    keypoints = np.asarray(track_metrics["keypoints"], dtype=np.float32)
    final_frame_index = int(drift.shape[1] - 1)

    valid_frame_indices = np.flatnonzero(np.isfinite(per_frame_p95))
    order = valid_frame_indices[np.argsort(-per_frame_p95[valid_frame_indices], kind="stable")]
    selected_frame_indices: list[int] = []
    if final_frame_index not in selected_frame_indices:
        selected_frame_indices.append(final_frame_index)
    for frame_idx in order[: max(int(top_k_frames), 0)]:
        if int(frame_idx) not in selected_frame_indices:
            selected_frame_indices.append(int(frame_idx))

    frame_reports: list[dict[str, object]] = []
    for frame_idx in selected_frame_indices:
        frame_valid = valid[:, frame_idx]
        frame_drift = drift[:, frame_idx]
        tail_count = int(np.count_nonzero(frame_valid & (frame_drift >= float(tail_threshold_px))))
        frame_reports.append(
            {
                "frame_index": int(frame_idx),
                "anchor_valid_count": int(np.count_nonzero(frame_valid)),
                "global_disp_px": _maybe_float(per_frame_global_disp[frame_idx]),
                "drift_median_px": _maybe_float(per_frame_median[frame_idx]),
                "drift_p95_px": _maybe_float(per_frame_p95[frame_idx]),
                "tail_track_count": tail_count,
                "top_tail_cells": summarize_spatial_tail_clusters(
                    keypoints,
                    frame_drift,
                    frame_valid,
                    image_height=int(image_height),
                    image_width=int(image_width),
                    cell_size_px=int(cell_size_px),
                    tail_threshold_px=float(tail_threshold_px),
                    top_k=int(top_k_cells),
                ),
            }
        )

    final_report = next(item for item in frame_reports if int(item["frame_index"]) == final_frame_index)
    worst_report = max(
        frame_reports,
        key=lambda item: (-1.0 if item["drift_p95_px"] is None else float(item["drift_p95_px"])),
    )
    return {
        "query_frame": int(query_frame),
        "grid_size": int(grid_size),
        "anchor_count": int(track_metrics["anchor_count"]),
        "tail_threshold_px": float(tail_threshold_px),
        "cell_size_px": int(cell_size_px),
        "frame_count": int(drift.shape[1]),
        "drift_median_summary": _finite_summary(per_frame_median),
        "drift_p95_summary": _finite_summary(per_frame_p95),
        "global_disp_summary": _finite_summary(per_frame_global_disp),
        "final_frame": final_report,
        "worst_frame": worst_report,
        "selected_frames": frame_reports,
    }


def _collect_static_background_correspondences(
    depth_frames: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    query_anchor_bundle: dict[str, np.ndarray | int],
    target_frame: int,
    min_depth: float,
    max_depth: float,
    min_target_border_dist_px: float,
    max_depth_error_m: float,
    max_world_error_m: float,
    max_query_reproj_error_px: float,
) -> dict[str, np.ndarray | int | float]:
    depth_frames = np.asarray(depth_frames, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
    keypoints = np.asarray(query_anchor_bundle["keypoints"], dtype=np.float32)
    anchor_mask = np.asarray(query_anchor_bundle["anchor_mask"], dtype=bool)
    world_points = np.asarray(query_anchor_bundle["world_points"], dtype=np.float32)
    query_frame = int(query_anchor_bundle["query_frame"])
    height = int(depth_frames.shape[1])
    width = int(depth_frames.shape[2])

    projected_uvz, projected_valid = project_world_points(
        world_points,
        intrinsics=intrinsics[target_frame],
        w2c=extrinsics_w2c[target_frame],
        min_depth=min_depth,
        max_depth=max_depth,
    )
    projected_uv = projected_uvz[:, :2]
    observed_depth, observed_valid = sample_image_at_keypoints(depth_frames[target_frame], projected_uv)
    camera_points, camera_valid = _camera_points_from_uv_depth(
        projected_uv,
        observed_depth,
        intrinsics=intrinsics[target_frame],
        min_depth=min_depth,
        max_depth=max_depth,
    )
    observed_world, observed_world_valid = unproject_keypoints_to_world(
        projected_uv,
        observed_depth,
        intrinsics=intrinsics[target_frame],
        w2c=extrinsics_w2c[target_frame],
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

    projected_border_dist_px = compute_border_distance_px(projected_uv, height=height, width=width)
    depth_error_m = np.abs(observed_depth - projected_uvz[:, 2]).astype(np.float32)
    world_error_m = np.linalg.norm(observed_world - world_points, axis=1).astype(np.float32)
    query_reproj_error_px = np.linalg.norm(query_reproj_uvz[:, :2] - keypoints, axis=1).astype(np.float32)
    valid = (
        anchor_mask
        & projected_valid
        & observed_valid
        & camera_valid
        & observed_world_valid
        & query_reproj_valid
        & (projected_border_dist_px >= float(min_target_border_dist_px))
        & np.isfinite(depth_error_m)
        & np.isfinite(world_error_m)
        & np.isfinite(query_reproj_error_px)
        & (depth_error_m <= float(max_depth_error_m))
        & (world_error_m <= float(max_world_error_m))
        & (query_reproj_error_px <= float(max_query_reproj_error_px))
    )
    return {
        "world_points": world_points[valid].astype(np.float32, copy=False),
        "camera_points": camera_points[valid].astype(np.float32, copy=False),
        "support_count": int(np.count_nonzero(valid)),
        "depth_error_m": depth_error_m[valid].astype(np.float32, copy=False),
        "world_error_m": world_error_m[valid].astype(np.float32, copy=False),
        "query_reproj_error_px": query_reproj_error_px[valid].astype(np.float32, copy=False),
    }


def refine_extrinsics_w2c_static_background(
    depth_frames: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    query_frames: Sequence[int] | np.ndarray,
    grid_size: int = 80,
    min_query_depth_m: float = 0.2,
    min_border_dist_px: float = 60.0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    min_target_border_dist_px: float = 12.0,
    max_depth_error_m: float = 0.20,
    max_world_error_m: float = 0.20,
    max_query_reproj_error_px: float = 6.0,
    min_correspondences: int = 256,
    temporal_smooth_radius: int = 1,
    temporal_regularization_weight: float = 0.25,
    max_translation_delta_m: float = 0.05,
    max_rotation_delta_deg: float = 2.0,
) -> dict[str, Any]:
    depth_frames = np.asarray(depth_frames, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
    if depth_frames.ndim != 3:
        raise ValueError(f"Expected depth_frames shape (T,H,W), got {depth_frames.shape}")
    if intrinsics.shape != (depth_frames.shape[0], 3, 3):
        raise ValueError(f"Expected intrinsics shape {(depth_frames.shape[0], 3, 3)}, got {intrinsics.shape}")
    if extrinsics_w2c.shape != (depth_frames.shape[0], 4, 4):
        raise ValueError(
            f"Expected extrinsics_w2c shape {(depth_frames.shape[0], 4, 4)}, got {extrinsics_w2c.shape}"
        )
    query_frame_list = _normalize_query_frames(query_frames)
    frame_count = int(depth_frames.shape[0])
    for query_frame in query_frame_list:
        if query_frame < 0 or query_frame >= frame_count:
            raise IndexError(f"query_frame {query_frame} out of range for frame_count={frame_count}")

    query_bundles = [
        build_query_anchor_bundle(
            depth_frames,
            intrinsics,
            extrinsics_w2c,
            query_frame=query_frame,
            grid_size=int(grid_size),
            min_query_depth_m=float(min_query_depth_m),
            min_border_dist_px=float(min_border_dist_px),
            min_depth=float(min_depth),
            max_depth=float(max_depth),
        )
        for query_frame in query_frame_list
    ]

    orig_c2w = np.linalg.inv(extrinsics_w2c).astype(np.float32)
    orig_centers = orig_c2w[:, :3, 3].astype(np.float32)
    orig_rotations = orig_c2w[:, :3, :3].astype(np.float32)
    raw_centers = orig_centers.copy()
    raw_rotations = orig_rotations.copy()
    raw_valid = np.zeros(frame_count, dtype=bool)
    raw_support_count = np.zeros(frame_count, dtype=np.int32)
    raw_fit_median_m = np.full(frame_count, np.nan, dtype=np.float32)
    raw_fit_p95_m = np.full(frame_count, np.nan, dtype=np.float32)
    raw_query_support = np.zeros((frame_count, len(query_bundles)), dtype=np.int32)
    raw_query_reproj_error_median_px = np.full((frame_count, len(query_bundles)), np.nan, dtype=np.float32)

    for frame_idx in range(frame_count):
        world_chunks: list[np.ndarray] = []
        camera_chunks: list[np.ndarray] = []
        per_query_support: list[int] = []
        per_query_reproj_median: list[float] = []
        for bundle_idx, query_bundle in enumerate(query_bundles):
            corr = _collect_static_background_correspondences(
                depth_frames,
                intrinsics,
                extrinsics_w2c,
                query_anchor_bundle=query_bundle,
                target_frame=frame_idx,
                min_depth=float(min_depth),
                max_depth=float(max_depth),
                min_target_border_dist_px=float(min_target_border_dist_px),
                max_depth_error_m=float(max_depth_error_m),
                max_world_error_m=float(max_world_error_m),
                max_query_reproj_error_px=float(max_query_reproj_error_px),
            )
            support_count = int(corr["support_count"])
            raw_query_support[frame_idx, bundle_idx] = support_count
            per_query_support.append(support_count)
            if support_count == 0:
                per_query_reproj_median.append(float("nan"))
                continue
            query_world = np.asarray(corr["world_points"], dtype=np.float32)
            query_camera = np.asarray(corr["camera_points"], dtype=np.float32)
            query_reproj_error = np.asarray(corr["query_reproj_error_px"], dtype=np.float32)
            if query_reproj_error.size > 0:
                per_query_reproj_median.append(float(np.median(query_reproj_error)))
            else:
                per_query_reproj_median.append(float("nan"))
            world_chunks.append(query_world)
            camera_chunks.append(query_camera)
        if per_query_reproj_median:
            raw_query_reproj_error_median_px[frame_idx, : len(per_query_reproj_median)] = np.asarray(
                per_query_reproj_median, dtype=np.float32
            )

        total_support = int(sum(per_query_support))
        raw_support_count[frame_idx] = total_support
        if total_support < int(min_correspondences):
            continue

        world_points = np.concatenate(world_chunks, axis=0).astype(np.float32, copy=False)
        camera_points = np.concatenate(camera_chunks, axis=0).astype(np.float32, copy=False)
        candidate_w2c = _estimate_rigid_transform_world_to_camera(world_points, camera_points)
        predicted_camera = (
            candidate_w2c[:3, :3] @ world_points.T + candidate_w2c[:3, 3:4]
        ).T.astype(np.float32)
        fit_residual = np.linalg.norm(predicted_camera - camera_points, axis=1).astype(np.float32)
        raw_fit_median_m[frame_idx] = float(np.median(fit_residual))
        raw_fit_p95_m[frame_idx] = float(np.percentile(fit_residual, 95))

        candidate_c2w = np.linalg.inv(candidate_w2c).astype(np.float32)
        raw_centers[frame_idx] = candidate_c2w[:3, 3]
        raw_rotations[frame_idx] = candidate_c2w[:3, :3]
        raw_valid[frame_idx] = True

    raw_center_delta = (raw_centers - orig_centers).astype(np.float32)
    raw_rot_delta = np.zeros((frame_count, 3), dtype=np.float32)
    for frame_idx in range(frame_count):
        if not raw_valid[frame_idx]:
            continue
        relative = raw_rotations[frame_idx] @ orig_rotations[frame_idx].T
        raw_rot_delta[frame_idx] = Rotation.from_matrix(relative.astype(np.float64)).as_rotvec().astype(np.float32)

    support_weight = np.where(
        raw_valid,
        raw_support_count.astype(np.float32) / np.maximum(raw_fit_p95_m, 1e-3),
        0.0,
    ).astype(np.float32)
    smoothed_center_delta, smooth_center_valid = _smooth_valid_vectors(
        raw_center_delta,
        raw_valid,
        support_weight,
        radius=int(temporal_smooth_radius),
    )
    smoothed_rot_delta, smooth_rot_valid = _smooth_valid_vectors(
        raw_rot_delta,
        raw_valid,
        support_weight,
        radius=int(temporal_smooth_radius),
    )

    temporal_weight = float(np.clip(temporal_regularization_weight, 0.0, 1.0))
    final_centers = orig_centers.copy()
    final_rotations = orig_rotations.copy()
    final_valid = np.zeros(frame_count, dtype=bool)
    final_center_delta = np.zeros_like(raw_center_delta, dtype=np.float32)
    final_rot_delta = np.zeros_like(raw_rot_delta, dtype=np.float32)
    final_translation_delta_m = np.zeros(frame_count, dtype=np.float32)
    final_rotation_delta_deg = np.zeros(frame_count, dtype=np.float32)

    for frame_idx in range(frame_count):
        if not raw_valid[frame_idx]:
            continue
        combined_center_delta = raw_center_delta[frame_idx].copy()
        combined_rot_delta = raw_rot_delta[frame_idx].copy()
        if temporal_weight > 0.0 and smooth_center_valid[frame_idx]:
            combined_center_delta = (
                (1.0 - temporal_weight) * combined_center_delta
                + temporal_weight * smoothed_center_delta[frame_idx]
            ).astype(np.float32)
        if temporal_weight > 0.0 and smooth_rot_valid[frame_idx]:
            combined_rot_delta = (
                (1.0 - temporal_weight) * combined_rot_delta
                + temporal_weight * smoothed_rot_delta[frame_idx]
            ).astype(np.float32)

        translation_norm = float(np.linalg.norm(combined_center_delta))
        if translation_norm > float(max_translation_delta_m) > 0.0:
            combined_center_delta *= float(max_translation_delta_m) / max(translation_norm, 1e-8)

        rotation_deg = float(np.linalg.norm(combined_rot_delta) * (180.0 / np.pi))
        if rotation_deg > float(max_rotation_delta_deg) > 0.0:
            combined_rot_delta *= float(max_rotation_delta_deg) / max(rotation_deg, 1e-8)

        delta_rotation = Rotation.from_rotvec(combined_rot_delta.astype(np.float64)).as_matrix().astype(np.float32)
        final_centers[frame_idx] = (orig_centers[frame_idx] + combined_center_delta).astype(np.float32)
        final_rotations[frame_idx] = (delta_rotation @ orig_rotations[frame_idx]).astype(np.float32)
        final_valid[frame_idx] = True
        final_center_delta[frame_idx] = combined_center_delta
        final_rot_delta[frame_idx] = combined_rot_delta
        final_translation_delta_m[frame_idx] = float(np.linalg.norm(combined_center_delta))
        final_rotation_delta_deg[frame_idx] = float(np.linalg.norm(combined_rot_delta) * (180.0 / np.pi))

    refined_c2w = orig_c2w.copy()
    refined_c2w[:, :3, 3] = final_centers
    refined_c2w[:, :3, :3] = final_rotations
    refined_extrinsics = np.linalg.inv(refined_c2w).astype(np.float32)

    frame_reports = []
    for frame_idx in range(frame_count):
        frame_reports.append(
            {
                "frame_index": int(frame_idx),
                "raw_valid": bool(raw_valid[frame_idx]),
                "raw_support_count": int(raw_support_count[frame_idx]),
                "raw_fit_median_m": _maybe_float(raw_fit_median_m[frame_idx]),
                "raw_fit_p95_m": _maybe_float(raw_fit_p95_m[frame_idx]),
                "query_support_count": {
                    str(query_frame_list[bundle_idx]): int(raw_query_support[frame_idx, bundle_idx])
                    for bundle_idx in range(len(query_bundles))
                },
                "query_reproj_error_median_px": {
                    str(query_frame_list[bundle_idx]): _maybe_float(raw_query_reproj_error_median_px[frame_idx, bundle_idx])
                    for bundle_idx in range(len(query_bundles))
                },
                "translation_delta_m": float(final_translation_delta_m[frame_idx]),
                "rotation_delta_deg": float(final_rotation_delta_deg[frame_idx]),
            }
        )

    return {
        "query_frames": list(query_frame_list),
        "extrinsics_w2c": refined_extrinsics.astype(np.float32, copy=False),
        "raw_valid_mask": raw_valid.astype(bool, copy=False),
        "raw_support_count": raw_support_count.astype(np.int32, copy=False),
        "raw_fit_median_m": raw_fit_median_m.astype(np.float32, copy=False),
        "raw_fit_p95_m": raw_fit_p95_m.astype(np.float32, copy=False),
        "translation_delta_m": final_translation_delta_m.astype(np.float32, copy=False),
        "rotation_delta_deg": final_rotation_delta_deg.astype(np.float32, copy=False),
        "translation_delta_summary": _finite_summary(final_translation_delta_m[final_valid]),
        "rotation_delta_summary": _finite_summary(final_rotation_delta_deg[final_valid]),
        "support_count_summary": _finite_summary(raw_support_count[raw_valid]),
        "fit_median_summary": _finite_summary(raw_fit_median_m[raw_valid]),
        "fit_p95_summary": _finite_summary(raw_fit_p95_m[raw_valid]),
        "frame_reports": frame_reports,
    }
