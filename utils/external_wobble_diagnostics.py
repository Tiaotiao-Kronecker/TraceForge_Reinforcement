from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_EXTRINSICS_SMOOTH_RADIUS = 1
DEFAULT_DEPTH_MEDIAN_REPROJ_TOL_PX = 3.0
DEFAULT_DEPTH_MEDIAN_MIN_SUPPORT = 3
DEFAULT_DENSE_DEPTH_STABILIZATION_RADIUS = 2
DEFAULT_DENSE_DEPTH_STABILIZATION_MIN_SUPPORT = 3


def build_uniform_grid_keypoints(height: int, width: int, grid_size: int) -> np.ndarray:
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError(f"Expected positive height/width, got {height}x{width}")
    if int(grid_size) <= 0:
        raise ValueError(f"Expected positive grid_size, got {grid_size}")
    xs = np.linspace(0.0, float(width - 1), int(grid_size), dtype=np.float32)
    ys = np.linspace(0.0, float(height - 1), int(grid_size), dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float32)


def compute_border_distance_px(
    keypoints: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    border_dist = np.minimum.reduce(
        [
            np.maximum(x, 0.0),
            np.maximum(y, 0.0),
            np.maximum(float(width - 1) - x, 0.0),
            np.maximum(float(height - 1) - y, 0.0),
        ]
    )
    return border_dist.astype(np.float32, copy=False)


def sample_image_at_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected image shape (H,W), got {image.shape}")
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")
    height, width = image.shape
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= float(max(width - 1, 0)))
        & (y >= 0.0)
        & (y <= float(max(height - 1, 0)))
    )
    xs = np.clip(np.round(np.nan_to_num(x, nan=-1.0)).astype(np.int32), 0, max(width - 1, 0))
    ys = np.clip(np.round(np.nan_to_num(y, nan=-1.0)).astype(np.int32), 0, max(height - 1, 0))
    sampled = np.full(keypoints.shape[0], np.nan, dtype=np.float32)
    if np.any(valid):
        sampled[valid] = image[ys[valid], xs[valid]].astype(np.float32, copy=False)
    valid &= np.isfinite(sampled)
    return sampled.astype(np.float32), valid.astype(bool)


def unproject_keypoints_to_world(
    keypoints: np.ndarray,
    depth_values: np.ndarray,
    *,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    depth_values = np.asarray(depth_values, dtype=np.float32).reshape(-1)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")
    if depth_values.shape != (keypoints.shape[0],):
        raise ValueError(f"Expected depth_values shape {(keypoints.shape[0],)}, got {depth_values.shape}")

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    z = depth_values.astype(np.float32, copy=False)
    valid = np.isfinite(z) & (z > float(min_depth)) & (z < float(max_depth))
    x_cam = np.where(valid, (keypoints[:, 0] - cx) * z / (fx + 1e-8), np.nan)
    y_cam = np.where(valid, (keypoints[:, 1] - cy) * z / (fy + 1e-8), np.nan)
    pts_cam = np.stack([x_cam, y_cam, np.where(valid, z, np.nan)], axis=1)
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)], axis=1)
    c2w = np.linalg.inv(w2c).astype(np.float32)
    pts_world = (c2w @ pts_cam_h.T).T[:, :3].astype(np.float32)
    pts_world[~valid] = np.nan
    return pts_world, valid.astype(bool)


def project_world_points(
    world_points: np.ndarray,
    *,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    world_points = np.asarray(world_points, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError(f"Expected world_points shape (N,3), got {world_points.shape}")

    world_points_h = np.concatenate(
        [world_points, np.ones((world_points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    pts_cam_h = (w2c @ world_points_h.T).T
    pts_cam = pts_cam_h[:, :3]
    pts_img = (intrinsics @ pts_cam.T).T
    z = pts_cam[:, 2]
    u = pts_img[:, 0] / (z + 1e-8)
    v = pts_img[:, 1] / (z + 1e-8)
    uvz = np.stack([u, v, z], axis=1).astype(np.float32)
    valid = (
        np.isfinite(world_points).all(axis=1)
        & np.isfinite(uvz).all(axis=1)
        & (z > float(min_depth))
        & (z < float(max_depth))
    )
    uvz[~valid] = np.nan
    return uvz, valid.astype(bool)


def _unproject_depth_frame_to_world_points(
    depth_frame: np.ndarray,
    *,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
) -> np.ndarray:
    depth_frame = np.asarray(depth_frame, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)
    if depth_frame.ndim != 2:
        raise ValueError(f"Expected depth_frame shape (H,W), got {depth_frame.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3,3), got {intrinsics.shape}")
    if w2c.shape != (4, 4):
        raise ValueError(f"Expected w2c shape (4,4), got {w2c.shape}")

    valid = (
        np.isfinite(depth_frame)
        & (depth_frame > float(min_depth))
        & (depth_frame < float(max_depth))
    )
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    ys, xs = np.nonzero(valid)
    z = depth_frame[ys, xs].astype(np.float32, copy=False)
    xy_h = np.stack(
        [
            xs.astype(np.float32, copy=False),
            ys.astype(np.float32, copy=False),
            np.ones(xs.shape[0], dtype=np.float32),
        ],
        axis=1,
    )
    cam_dirs = (np.linalg.inv(intrinsics) @ xy_h.T).T.astype(np.float32)
    pts_cam = cam_dirs * z[:, None]
    pts_cam_h = np.concatenate(
        [pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    c2w = np.linalg.inv(w2c).astype(np.float32)
    pts_world = (c2w @ pts_cam_h.T).T[:, :3]
    return pts_world.astype(np.float32, copy=False)


def _rasterize_world_points_to_depth_image(
    world_points: np.ndarray,
    *,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    height: int,
    width: int,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
) -> np.ndarray:
    world_points = np.asarray(world_points, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError(f"Expected world_points shape (N,3), got {world_points.shape}")
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3,3), got {intrinsics.shape}")
    if w2c.shape != (4, 4):
        raise ValueError(f"Expected w2c shape (4,4), got {w2c.shape}")
    if int(height) <= 0 or int(width) <= 0:
        raise ValueError(f"Expected positive height/width, got {height}x{width}")

    warped_flat = np.full(int(height) * int(width), np.inf, dtype=np.float32)
    if world_points.shape[0] == 0:
        return warped_flat.reshape(int(height), int(width)).astype(np.float32)

    world_points_h = np.concatenate(
        [world_points, np.ones((world_points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    pts_cam_h = (w2c @ world_points_h.T).T
    pts_cam = pts_cam_h[:, :3]
    pts_img = (intrinsics @ pts_cam.T).T
    z = pts_cam[:, 2]
    u = pts_img[:, 0] / (z + 1e-8)
    v = pts_img[:, 1] / (z + 1e-8)
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    valid = (
        np.isfinite(world_points).all(axis=1)
        & np.isfinite(u)
        & np.isfinite(v)
        & np.isfinite(z)
        & (z > float(min_depth))
        & (z < float(max_depth))
        & (ui >= 0)
        & (ui < int(width))
        & (vi >= 0)
        & (vi < int(height))
    )
    if np.any(valid):
        linear_idx = vi[valid] * int(width) + ui[valid]
        np.minimum.at(warped_flat, linear_idx, z[valid].astype(np.float32, copy=False))
    warped = warped_flat.reshape(int(height), int(width)).astype(np.float32, copy=False)
    warped[~np.isfinite(warped)] = np.nan
    return warped


def stabilize_depth_frames_temporal_median_reproject(
    depth_frames: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    radius: int = DEFAULT_DENSE_DEPTH_STABILIZATION_RADIUS,
    min_support: int = DEFAULT_DENSE_DEPTH_STABILIZATION_MIN_SUPPORT,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
) -> dict[str, np.ndarray]:
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

    radius = int(radius)
    min_support = int(min_support)
    if radius < 0:
        raise ValueError(f"Expected nonnegative radius, got {radius}")
    if min_support < 1:
        raise ValueError(f"Expected min_support >= 1, got {min_support}")

    frame_count, height, width = depth_frames.shape
    stabilized_depths = depth_frames.astype(np.float32, copy=True)
    replace_ratio = np.zeros(frame_count, dtype=np.float32)
    replace_count = np.zeros(frame_count, dtype=np.int32)
    support_count_median = np.full(frame_count, np.nan, dtype=np.float32)
    support_count_p95 = np.full(frame_count, np.nan, dtype=np.float32)
    depth_delta_median_m = np.full(frame_count, np.nan, dtype=np.float32)
    depth_delta_p95_m = np.full(frame_count, np.nan, dtype=np.float32)

    if radius == 0 or frame_count == 0:
        return {
            "depth_frames": stabilized_depths,
            "replace_ratio": replace_ratio,
            "replace_count": replace_count,
            "support_count_median": support_count_median,
            "support_count_p95": support_count_p95,
            "depth_delta_median_m": depth_delta_median_m,
            "depth_delta_p95_m": depth_delta_p95_m,
        }

    cached_world_points = [
        _unproject_depth_frame_to_world_points(
            depth_frames[frame_idx],
            intrinsics=intrinsics[frame_idx],
            w2c=extrinsics_w2c[frame_idx],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        for frame_idx in range(frame_count)
    ]

    for target_idx in range(frame_count):
        support_layers = [depth_frames[target_idx].astype(np.float32, copy=False)]
        start = max(0, target_idx - radius)
        stop = min(frame_count, target_idx + radius + 1)
        for source_idx in range(start, stop):
            if source_idx == target_idx:
                continue
            support_layers.append(
                _rasterize_world_points_to_depth_image(
                    cached_world_points[source_idx],
                    intrinsics=intrinsics[target_idx],
                    w2c=extrinsics_w2c[target_idx],
                    height=height,
                    width=width,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )
            )

        support_stack = np.stack(support_layers, axis=0).astype(np.float32, copy=False)
        support_valid = (
            np.isfinite(support_stack)
            & (support_stack > float(min_depth))
            & (support_stack < float(max_depth))
        )
        support_counts = support_valid.sum(axis=0).astype(np.int16)
        masked_support = np.ma.masked_where(~support_valid, support_stack)
        median_depth = np.ma.median(masked_support, axis=0).filled(np.nan).astype(np.float32)

        original_depth = depth_frames[target_idx].astype(np.float32, copy=False)
        valid_original = (
            np.isfinite(original_depth)
            & (original_depth > float(min_depth))
            & (original_depth < float(max_depth))
        )
        replace_mask = (
            valid_original
            & (support_counts >= min_support)
            & np.isfinite(median_depth)
            & (median_depth > float(min_depth))
            & (median_depth < float(max_depth))
        )
        if np.any(replace_mask):
            stabilized_depths[target_idx, replace_mask] = median_depth[replace_mask]
            replaced_support = support_counts[replace_mask].astype(np.float32, copy=False)
            replaced_delta = np.abs(median_depth[replace_mask] - original_depth[replace_mask]).astype(np.float32)
            replace_count[target_idx] = int(np.count_nonzero(replace_mask))
            replace_ratio[target_idx] = float(
                replace_count[target_idx] / max(int(np.count_nonzero(valid_original)), 1)
            )
            support_count_median[target_idx] = float(np.median(replaced_support))
            support_count_p95[target_idx] = float(np.percentile(replaced_support, 95))
            depth_delta_median_m[target_idx] = float(np.median(replaced_delta))
            depth_delta_p95_m[target_idx] = float(np.percentile(replaced_delta, 95))

    return {
        "depth_frames": stabilized_depths.astype(np.float32, copy=False),
        "replace_ratio": replace_ratio,
        "replace_count": replace_count,
        "support_count_median": support_count_median,
        "support_count_p95": support_count_p95,
        "depth_delta_median_m": depth_delta_median_m,
        "depth_delta_p95_m": depth_delta_p95_m,
    }


def rotation_angle_deg(relative_rotation: np.ndarray) -> np.ndarray:
    relative_rotation = np.asarray(relative_rotation, dtype=np.float32)
    if relative_rotation.ndim < 2 or relative_rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices with trailing shape (3,3), got {relative_rotation.shape}")
    trace = np.trace(relative_rotation, axis1=-2, axis2=-1).astype(np.float32)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta)).astype(np.float32)


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


def _project_rotation_to_so3(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected rotation shape (3,3), got {rotation.shape}")
    u, _, vh = np.linalg.svd(rotation.astype(np.float64), full_matrices=False)
    projected = u @ vh
    if np.linalg.det(projected) < 0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected.astype(np.float32)


def smooth_extrinsics_w2c_moving_average(
    extrinsics_w2c: np.ndarray,
    *,
    radius: int = DEFAULT_EXTRINSICS_SMOOTH_RADIUS,
) -> np.ndarray:
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
    if extrinsics_w2c.ndim != 3 or extrinsics_w2c.shape[1:] != (4, 4):
        raise ValueError(f"Expected extrinsics_w2c shape (T,4,4), got {extrinsics_w2c.shape}")
    radius = int(radius)
    if radius <= 0:
        return extrinsics_w2c.astype(np.float32, copy=True)

    c2w = np.linalg.inv(extrinsics_w2c).astype(np.float32)
    centers = c2w[:, :3, 3]
    rotations = c2w[:, :3, :3]
    smoothed_c2w = c2w.copy()
    frame_count = int(extrinsics_w2c.shape[0])
    for frame_idx in range(frame_count):
        start = max(0, frame_idx - radius)
        stop = min(frame_count, frame_idx + radius + 1)
        smoothed_c2w[frame_idx, :3, 3] = np.mean(centers[start:stop], axis=0).astype(np.float32)
        mean_rot = np.mean(rotations[start:stop], axis=0).astype(np.float32)
        smoothed_c2w[frame_idx, :3, :3] = _project_rotation_to_so3(mean_rot)
        smoothed_c2w[frame_idx, 3] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return np.linalg.inv(smoothed_c2w).astype(np.float32)


def freeze_extrinsics_w2c_to_query_frame(
    extrinsics_w2c: np.ndarray,
    *,
    query_frame: int,
) -> np.ndarray:
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
    if extrinsics_w2c.ndim != 3 or extrinsics_w2c.shape[1:] != (4, 4):
        raise ValueError(f"Expected extrinsics_w2c shape (T,4,4), got {extrinsics_w2c.shape}")
    query_frame = int(query_frame)
    if query_frame < 0 or query_frame >= extrinsics_w2c.shape[0]:
        raise IndexError(f"query_frame {query_frame} out of range for frame_count={extrinsics_w2c.shape[0]}")
    return np.repeat(extrinsics_w2c[query_frame : query_frame + 1], extrinsics_w2c.shape[0], axis=0).astype(
        np.float32
    )


def build_query_anchor_bundle(
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
) -> dict[str, np.ndarray | int]:
    depth_frames = np.asarray(depth_frames, dtype=np.float32)
    if depth_frames.ndim != 3:
        raise ValueError(f"Expected depth_frames shape (T,H,W), got {depth_frames.shape}")
    _, height, width = depth_frames.shape
    keypoints = build_uniform_grid_keypoints(height, width, grid_size)
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
    bundle["grid_size"] = int(grid_size)
    return bundle


def build_query_anchor_bundle_from_keypoints(
    depth_frames: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    keypoints: np.ndarray,
    query_frame: int,
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
    query_frame = int(query_frame)
    frame_count, height, width = depth_frames.shape
    if query_frame < 0 or query_frame >= frame_count:
        raise IndexError(f"query_frame {query_frame} out of range for frame_count={frame_count}")

    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")
    border_dist = compute_border_distance_px(keypoints, height=height, width=width)
    query_depth_values, query_depth_valid = sample_image_at_keypoints(depth_frames[query_frame], keypoints)
    anchor_mask = (
        query_depth_valid
        & np.isfinite(query_depth_values)
        & (query_depth_values > float(min_query_depth_m))
        & (border_dist >= float(min_border_dist_px))
    )
    world_points, world_valid = unproject_keypoints_to_world(
        keypoints,
        query_depth_values,
        intrinsics=intrinsics[query_frame],
        w2c=extrinsics_w2c[query_frame],
        min_depth=min_depth,
        max_depth=max_depth,
    )
    anchor_mask &= world_valid
    return {
        "query_frame": query_frame,
        "grid_size": -1,
        "keypoints": keypoints.astype(np.float32, copy=False),
        "border_dist_px": border_dist.astype(np.float32, copy=False),
        "query_depth_values": query_depth_values.astype(np.float32, copy=False),
        "anchor_mask": anchor_mask.astype(bool, copy=False),
        "world_points": world_points.astype(np.float32, copy=False),
    }


def estimate_temporal_median_world_points(
    depth_frames: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    query_anchor_bundle: dict[str, np.ndarray | int],
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    reproj_tol_px: float = DEFAULT_DEPTH_MEDIAN_REPROJ_TOL_PX,
    min_support: int = DEFAULT_DEPTH_MEDIAN_MIN_SUPPORT,
) -> dict[str, np.ndarray]:
    depth_frames = np.asarray(depth_frames, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
    keypoints = np.asarray(query_anchor_bundle["keypoints"], dtype=np.float32)
    anchor_mask = np.asarray(query_anchor_bundle["anchor_mask"], dtype=bool)
    base_world_points = np.asarray(query_anchor_bundle["world_points"], dtype=np.float32)
    query_frame = int(query_anchor_bundle["query_frame"])
    frame_count = int(depth_frames.shape[0])
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")
    if anchor_mask.shape != (keypoints.shape[0],):
        raise ValueError(f"Expected anchor_mask shape {(keypoints.shape[0],)}, got {anchor_mask.shape}")
    if base_world_points.shape != (keypoints.shape[0], 3):
        raise ValueError(f"Expected world_points shape {(keypoints.shape[0], 3)}, got {base_world_points.shape}")

    support_world = np.full((frame_count, keypoints.shape[0], 3), np.nan, dtype=np.float32)
    support_mask = np.zeros((frame_count, keypoints.shape[0]), dtype=bool)

    for frame_idx in range(frame_count):
        projected_uvz, projected_valid = project_world_points(
            base_world_points,
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
        reproj_error = np.linalg.norm(query_reproj_uvz[:, :2] - keypoints, axis=1)
        valid = (
            anchor_mask
            & projected_valid
            & observed_valid
            & observed_world_valid
            & query_reproj_valid
            & np.isfinite(reproj_error)
            & (reproj_error <= float(reproj_tol_px))
        )
        if np.any(valid):
            support_world[frame_idx, valid] = observed_world[valid]
            support_mask[frame_idx, valid] = True

    support_counts = support_mask.sum(axis=0).astype(np.int32)
    stabilized_world_points = np.asarray(base_world_points, dtype=np.float32).copy()
    eligible = anchor_mask & (support_counts >= int(min_support))
    if np.any(eligible):
        with np.errstate(invalid="ignore"):
            stabilized_world_points[eligible] = np.nanmedian(support_world[:, eligible], axis=0).astype(np.float32)
    replace_mask = eligible & np.isfinite(stabilized_world_points).all(axis=1)
    return {
        "world_points": stabilized_world_points.astype(np.float32, copy=False),
        "support_counts": support_counts,
        "replace_mask": replace_mask.astype(bool, copy=False),
        "anchor_mask": anchor_mask.astype(bool, copy=False),
    }


def compute_extrinsics_temporal_metrics(extrinsics_w2c: np.ndarray) -> dict[str, Any]:
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
    if extrinsics_w2c.ndim != 3 or extrinsics_w2c.shape[1:] != (4, 4):
        raise ValueError(f"Expected extrinsics_w2c shape (T,4,4), got {extrinsics_w2c.shape}")
    frame_count = int(extrinsics_w2c.shape[0])
    c2w = np.linalg.inv(extrinsics_w2c).astype(np.float32)
    camera_centers = c2w[:, :3, 3]
    camera_rotations = c2w[:, :3, :3]

    if frame_count < 2:
        empty = np.zeros(0, dtype=np.float32)
        return {
            "frame_count": frame_count,
            "camera_center_path_length_m": 0.0,
            "step_translation_m": empty,
            "step_rotation_deg": empty,
            "jerk_translation_m": empty,
            "jerk_rotation_deg": empty,
            "step_translation_summary": _finite_summary(empty),
            "step_rotation_summary": _finite_summary(empty),
            "jerk_translation_summary": _finite_summary(empty),
            "jerk_rotation_summary": _finite_summary(empty),
        }

    step_translation_m = np.linalg.norm(np.diff(camera_centers, axis=0), axis=1).astype(np.float32)
    rel_rot = np.einsum("tij,tjk->tik", camera_rotations[1:], np.transpose(camera_rotations[:-1], (0, 2, 1)))
    step_rotation_deg = rotation_angle_deg(rel_rot)

    if frame_count < 3:
        jerk_translation_m = np.zeros(0, dtype=np.float32)
        jerk_rotation_deg = np.zeros(0, dtype=np.float32)
    else:
        jerk_translation_m = np.linalg.norm(
            camera_centers[2:] - 2.0 * camera_centers[1:-1] + camera_centers[:-2],
            axis=1,
        ).astype(np.float32)
        step_rel_rot = rel_rot
        jerk_rot = np.einsum(
            "tij,tjk->tik",
            step_rel_rot[1:],
            np.transpose(step_rel_rot[:-1], (0, 2, 1)),
        )
        jerk_rotation_deg = rotation_angle_deg(jerk_rot)

    return {
        "frame_count": frame_count,
        "camera_center_path_length_m": float(np.sum(step_translation_m)),
        "step_translation_m": step_translation_m,
        "step_rotation_deg": step_rotation_deg,
        "jerk_translation_m": jerk_translation_m,
        "jerk_rotation_deg": jerk_rotation_deg,
        "step_translation_summary": _finite_summary(step_translation_m),
        "step_rotation_summary": _finite_summary(step_rotation_deg),
        "jerk_translation_summary": _finite_summary(jerk_translation_m),
        "jerk_rotation_summary": _finite_summary(jerk_rotation_deg),
    }


def compute_static_geometry_consistency(
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
    query_frame = int(query_frame)
    if query_anchor_bundle is None:
        query_anchor_bundle = build_query_anchor_bundle(
            depth_frames,
            intrinsics,
            extrinsics_w2c,
            query_frame=query_frame,
            grid_size=grid_size,
            min_query_depth_m=min_query_depth_m,
            min_border_dist_px=min_border_dist_px,
            min_depth=min_depth,
            max_depth=max_depth,
        )
    frame_count = int(depth_frames.shape[0])
    keypoints = np.asarray(query_anchor_bundle["keypoints"], dtype=np.float32)
    anchor_mask = np.asarray(query_anchor_bundle["anchor_mask"], dtype=bool)
    world_points = np.asarray(query_anchor_bundle["world_points"], dtype=np.float32)
    if keypoints.shape != (world_points.shape[0], 2):
        raise ValueError(f"Expected keypoints/world_points to share track count, got {keypoints.shape} vs {world_points.shape}")
    if anchor_mask.shape != (keypoints.shape[0],):
        raise ValueError(f"Expected anchor_mask shape {(keypoints.shape[0],)}, got {anchor_mask.shape}")

    per_frame_anchor_count = np.zeros(frame_count, dtype=np.int32)
    per_frame_depth_error_median_m = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_depth_error_p95_m = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_world_error_median_m = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_world_error_p95_m = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_in_bounds_ratio = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_query_reproj_global_dx_px = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_query_reproj_global_dy_px = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_query_reproj_global_disp_px = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_query_reproj_drift_median_px = np.full(frame_count, np.nan, dtype=np.float32)
    per_frame_query_reproj_drift_p95_px = np.full(frame_count, np.nan, dtype=np.float32)

    for frame_idx in range(frame_count):
        projected_uvz, projected_valid = project_world_points(
            world_points,
            intrinsics=intrinsics[frame_idx],
            w2c=extrinsics_w2c[frame_idx],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        observed_depth, observed_valid = sample_image_at_keypoints(depth_frames[frame_idx], projected_uvz[:, :2])
        compare_mask = anchor_mask & projected_valid & observed_valid
        anchor_total = int(np.count_nonzero(anchor_mask))
        per_frame_anchor_count[frame_idx] = int(np.count_nonzero(compare_mask))
        per_frame_in_bounds_ratio[frame_idx] = (
            float(per_frame_anchor_count[frame_idx] / anchor_total) if anchor_total > 0 else float("nan")
        )
        if not np.any(compare_mask):
            continue

        expected_depth = projected_uvz[:, 2]
        depth_error = np.abs(observed_depth[compare_mask] - expected_depth[compare_mask]).astype(np.float32)
        per_frame_depth_error_median_m[frame_idx] = float(np.median(depth_error))
        per_frame_depth_error_p95_m[frame_idx] = float(np.percentile(depth_error, 95))

        observed_world, observed_world_valid = unproject_keypoints_to_world(
            projected_uvz[:, :2],
            observed_depth,
            intrinsics=intrinsics[frame_idx],
            w2c=extrinsics_w2c[frame_idx],
            min_depth=min_depth,
            max_depth=max_depth,
        )
        world_compare_mask = compare_mask & observed_world_valid
        if np.any(world_compare_mask):
            world_error = np.linalg.norm(
                observed_world[world_compare_mask] - world_points[world_compare_mask],
                axis=1,
            ).astype(np.float32)
            per_frame_world_error_median_m[frame_idx] = float(np.median(world_error))
            per_frame_world_error_p95_m[frame_idx] = float(np.percentile(world_error, 95))

            query_reproj_uvz, query_reproj_valid = project_world_points(
                observed_world,
                intrinsics=intrinsics[query_frame],
                w2c=extrinsics_w2c[query_frame],
                min_depth=min_depth,
                max_depth=max_depth,
            )
            reproj_mask = world_compare_mask & query_reproj_valid
            if np.any(reproj_mask):
                reproj_delta = query_reproj_uvz[reproj_mask, :2] - keypoints[reproj_mask]
                reproj_drift = np.linalg.norm(reproj_delta, axis=1).astype(np.float32)
                global_dx = float(np.median(reproj_delta[:, 0]))
                global_dy = float(np.median(reproj_delta[:, 1]))
                per_frame_query_reproj_global_dx_px[frame_idx] = global_dx
                per_frame_query_reproj_global_dy_px[frame_idx] = global_dy
                per_frame_query_reproj_global_disp_px[frame_idx] = float(np.hypot(global_dx, global_dy))
                per_frame_query_reproj_drift_median_px[frame_idx] = float(np.median(reproj_drift))
                per_frame_query_reproj_drift_p95_px[frame_idx] = float(np.percentile(reproj_drift, 95))

    return {
        "query_frame": query_frame,
        "grid_size": int(grid_size),
        "anchor_count": int(np.count_nonzero(anchor_mask)),
        "per_frame_anchor_count": per_frame_anchor_count,
        "per_frame_in_bounds_ratio": per_frame_in_bounds_ratio,
        "per_frame_depth_error_median_m": per_frame_depth_error_median_m,
        "per_frame_depth_error_p95_m": per_frame_depth_error_p95_m,
        "per_frame_world_error_median_m": per_frame_world_error_median_m,
        "per_frame_world_error_p95_m": per_frame_world_error_p95_m,
        "per_frame_query_reproj_global_dx_px": per_frame_query_reproj_global_dx_px,
        "per_frame_query_reproj_global_dy_px": per_frame_query_reproj_global_dy_px,
        "per_frame_query_reproj_global_disp_px": per_frame_query_reproj_global_disp_px,
        "per_frame_query_reproj_drift_median_px": per_frame_query_reproj_drift_median_px,
        "per_frame_query_reproj_drift_p95_px": per_frame_query_reproj_drift_p95_px,
        "final_depth_error_median_m": float(per_frame_depth_error_median_m[-1]),
        "final_depth_error_p95_m": float(per_frame_depth_error_p95_m[-1]),
        "final_world_error_median_m": float(per_frame_world_error_median_m[-1]),
        "final_world_error_p95_m": float(per_frame_world_error_p95_m[-1]),
        "final_query_reproj_global_disp_px": float(per_frame_query_reproj_global_disp_px[-1]),
        "final_query_reproj_drift_median_px": float(per_frame_query_reproj_drift_median_px[-1]),
        "final_query_reproj_drift_p95_px": float(per_frame_query_reproj_drift_p95_px[-1]),
        "depth_error_median_summary": _finite_summary(per_frame_depth_error_median_m),
        "depth_error_p95_summary": _finite_summary(per_frame_depth_error_p95_m),
        "world_error_median_summary": _finite_summary(per_frame_world_error_median_m),
        "world_error_p95_summary": _finite_summary(per_frame_world_error_p95_m),
        "query_reproj_global_disp_summary": _finite_summary(per_frame_query_reproj_global_disp_px),
        "query_reproj_drift_median_summary": _finite_summary(per_frame_query_reproj_drift_median_px),
        "query_reproj_drift_p95_summary": _finite_summary(per_frame_query_reproj_drift_p95_px),
    }
