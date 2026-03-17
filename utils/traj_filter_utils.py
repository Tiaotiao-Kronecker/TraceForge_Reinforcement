from __future__ import annotations

import warnings

import numpy as np


QUERY_DEPTH_PATCH_RADIUS = 2
QUERY_DEPTH_MIN_VALID_RATIO = 0.4
QUERY_DEPTH_ABS_TOL = 0.05
QUERY_DEPTH_REL_TOL = 0.10

TEMPORAL_DEPTH_ABS_TOL = 0.05
TEMPORAL_DEPTH_REL_TOL = 0.10
TEMPORAL_MIN_CONSISTENCY_RATIO = 0.95

TRAJ_FILTER_PROFILE_EXTERNAL = "external"
TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR = "external_manipulator"
TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR_V2 = "external_manipulator_v2"
TRAJ_FILTER_PROFILE_WRIST = "wrist"
TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR = "wrist_manipulator"

WRIST_MIN_PREFIX_FRAMES = 3
WRIST_MIN_SUPPORT_FRAMES = 3
WRIST_PREFIX_RATIO = 0.15
WRIST_SUPPORT_RATIO = 0.20

WRIST_MANIPULATOR_MAX_DEPTH_RANK = 0.50
WRIST_MANIPULATOR_MIN_MOTION_EXTENT = 0.03
WRIST_MANIPULATOR_CLUSTER_RADIUS_RATIO = 0.06
WRIST_MANIPULATOR_CLUSTER_RADIUS_MIN_PX = 24
WRIST_MANIPULATOR_MIN_COMPONENT_RATIO = 0.005
WRIST_MANIPULATOR_MIN_COMPONENT_SIZE = 2

EXTERNAL_MANIPULATOR_V2_MAX_DEPTH_RANK = 0.70
EXTERNAL_MANIPULATOR_V2_MIN_MOTION_EXTENT = 0.01
EXTERNAL_MANIPULATOR_V2_CLUSTER_RADIUS_RATIO = 0.06
EXTERNAL_MANIPULATOR_V2_CLUSTER_RADIUS_MIN_PX = 24
EXTERNAL_MANIPULATOR_V2_MIN_COMPONENT_RATIO = 0.002
EXTERNAL_MANIPULATOR_V2_MIN_COMPONENT_SIZE = 2
EXTERNAL_MANIPULATOR_V2_MAJOR_COMPONENT_RATIO = 0.15

VOLATILITY_LOW_PERCENTILE = 5.0
VOLATILITY_HIGH_PERCENTILE = 95.0
VOLATILITY_MASK_PERCENTILE = 99.0

MASK_REASON_BASE_GEOMETRY_FAIL = np.uint8(1 << 0)
MASK_REASON_QUERY_DEPTH_FAIL = np.uint8(1 << 1)
MASK_REASON_TEMPORAL_CONSISTENCY_FAIL = np.uint8(1 << 2)
MASK_REASON_STABLE_TEMPORAL_FAIL = np.uint8(1 << 3)
MASK_REASON_MANIPULATOR_DEPTH_FAIL = np.uint8(1 << 4)
MASK_REASON_MANIPULATOR_MOTION_FAIL = np.uint8(1 << 5)
MASK_REASON_MANIPULATOR_CLUSTER_FAIL = np.uint8(1 << 6)


def _normalize_visibility(
    visibs: np.ndarray | None,
    *,
    num_tracks: int,
    num_frames: int,
) -> np.ndarray | None:
    if visibs is None:
        return None

    visibility = np.asarray(visibs)
    if visibility.ndim == 3 and visibility.shape[-1] == 1:
        visibility = visibility.squeeze(-1)
    if visibility.shape == (num_frames, num_tracks):
        visibility = visibility.T
    if visibility.shape != (num_tracks, num_frames):
        raise ValueError(
            f"Expected visibility shape {(num_tracks, num_frames)} or {(num_frames, num_tracks)}, "
            f"got {visibility.shape}"
        )
    return visibility.astype(bool, copy=False)


def _require_segment_geometry(
    *,
    raw_depths_segment: np.ndarray | None,
    intrinsics_segment: np.ndarray | None,
    extrinsics_segment: np.ndarray | None,
    expected_num_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if raw_depths_segment is None or intrinsics_segment is None or extrinsics_segment is None:
        raise ValueError(
            "raw_depths_segment, intrinsics_segment, and extrinsics_segment are required "
            "when temporal consistency filtering is enabled"
        )

    raw_depths_segment = np.asarray(raw_depths_segment, dtype=np.float32)
    intrinsics_segment = np.asarray(intrinsics_segment, dtype=np.float32)
    extrinsics_segment = np.asarray(extrinsics_segment, dtype=np.float32)

    if raw_depths_segment.ndim != 3:
        raise ValueError(f"Expected raw_depths_segment shape (T,H,W), got {raw_depths_segment.shape}")
    if intrinsics_segment.shape != (expected_num_frames, 3, 3):
        raise ValueError(
            f"Expected intrinsics_segment shape {(expected_num_frames, 3, 3)}, got {intrinsics_segment.shape}"
        )
    if extrinsics_segment.shape != (expected_num_frames, 4, 4):
        raise ValueError(
            f"Expected extrinsics_segment shape {(expected_num_frames, 4, 4)}, got {extrinsics_segment.shape}"
        )
    if raw_depths_segment.shape[0] != expected_num_frames:
        raise ValueError(
            f"Expected raw_depths_segment first dimension {expected_num_frames}, got {raw_depths_segment.shape[0]}"
        )

    return raw_depths_segment, intrinsics_segment, extrinsics_segment


def _round_projected_coords(reprojected_uvz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round projected image coordinates without emitting NaN cast warnings."""
    rounded_u = np.rint(
        np.nan_to_num(reprojected_uvz[..., 0], nan=-1.0, posinf=-1.0, neginf=-1.0)
    ).astype(np.int32)
    rounded_v = np.rint(
        np.nan_to_num(reprojected_uvz[..., 1], nan=-1.0, posinf=-1.0, neginf=-1.0)
    ).astype(np.int32)
    return rounded_u, rounded_v


def _counts_to_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    ratio = np.full(denominator.shape, np.nan, dtype=np.float32)
    valid = denominator > 0
    ratio[valid] = numerator[valid] / denominator[valid]
    return ratio


def _compute_true_prefix_lengths(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape (N,T), got {mask.shape}")
    if mask.shape[1] == 0:
        return np.zeros(mask.shape[0], dtype=np.int32)
    false_mask = ~mask
    return np.where(false_mask.any(axis=1), false_mask.argmax(axis=1), mask.shape[1]).astype(np.int32)


def _resolve_support_frame_requirement(
    *,
    num_frames: int,
    min_frames: int,
    ratio: float,
) -> int:
    if num_frames <= 0:
        return 0
    return min(num_frames, max(int(min_frames), int(np.ceil(float(ratio) * num_frames))))


def _compute_query_depth_ranks(query_depth_values: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    query_depth_values = np.asarray(query_depth_values, dtype=np.float32).reshape(-1)
    seed_mask = np.asarray(seed_mask, dtype=bool).reshape(-1)
    if query_depth_values.shape != seed_mask.shape:
        raise ValueError(
            f"Expected query_depth_values and seed_mask to share shape, got "
            f"{query_depth_values.shape} and {seed_mask.shape}"
        )

    ranks = np.full(query_depth_values.shape, np.nan, dtype=np.float32)
    valid_seed = seed_mask & np.isfinite(query_depth_values)
    seed_indices = np.flatnonzero(valid_seed)
    if seed_indices.size == 0:
        return ranks

    order = seed_indices[np.argsort(query_depth_values[seed_indices], kind="stable")]
    if order.size == 1:
        ranks[order[0]] = 0.0
        return ranks

    denom = float(order.size - 1)
    ranks[order] = np.arange(order.size, dtype=np.float32) / denom
    return ranks


def _compute_supervised_motion_metrics(
    world_tracks: np.ndarray,
    supervision_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_tracks = np.asarray(world_tracks, dtype=np.float32)
    supervision_mask = np.asarray(supervision_mask, dtype=bool)
    if world_tracks.ndim != 3 or world_tracks.shape[-1] != 3:
        raise ValueError(f"Expected world_tracks shape (N,T,3), got {world_tracks.shape}")
    if supervision_mask.shape != world_tracks.shape[:2]:
        raise ValueError(
            f"Expected supervision_mask shape {world_tracks.shape[:2]}, got {supervision_mask.shape}"
        )

    num_tracks, num_frames, _ = world_tracks.shape
    valid = np.isfinite(world_tracks).all(axis=-1) & supervision_mask
    motion_extent = np.full(num_tracks, np.nan, dtype=np.float32)
    motion_step_median = np.full(num_tracks, np.nan, dtype=np.float32)

    step_norm = None
    pair_valid = None
    if num_frames > 1:
        step_norm = np.linalg.norm(np.diff(world_tracks, axis=1), axis=-1)
        pair_valid = valid[:, :-1] & valid[:, 1:]

    for track_idx in range(num_tracks):
        valid_indices = np.flatnonzero(valid[track_idx])
        if valid_indices.size >= 2:
            pts = world_tracks[track_idx, valid_indices]
            motion_extent[track_idx] = float(np.max(np.linalg.norm(pts - pts[0], axis=1)))

        if step_norm is None or pair_valid is None:
            continue

        valid_steps = step_norm[track_idx, pair_valid[track_idx]]
        if valid_steps.size > 0:
            motion_step_median[track_idx] = float(np.median(valid_steps))

    return motion_extent, motion_step_median


def _select_largest_spatial_component(
    keypoints: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    image_height: int,
    image_width: int,
    radius_ratio: float,
    radius_min_px: int,
    min_component_ratio: float,
    min_component_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")
    if candidate_mask.shape != (keypoints.shape[0],):
        raise ValueError(f"Expected candidate_mask shape {(keypoints.shape[0],)}, got {candidate_mask.shape}")

    num_tracks = int(keypoints.shape[0])
    final_mask = np.zeros(num_tracks, dtype=bool)
    component_ids = np.full(num_tracks, -1, dtype=np.int16)
    component_sizes = np.zeros(num_tracks, dtype=np.uint16)
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return final_mask, component_ids, component_sizes, False

    radius_px = float(max(int(radius_min_px), int(round(float(radius_ratio) * min(image_height, image_width)))))
    radius_sq = radius_px * radius_px
    coords = keypoints[candidate_indices]
    unvisited = np.ones(candidate_indices.size, dtype=bool)
    local_component_ids = np.full(candidate_indices.size, -1, dtype=np.int32)
    component_members: list[np.ndarray] = []

    while unvisited.any():
        start = int(np.flatnonzero(unvisited)[0])
        unvisited[start] = False
        stack = [start]
        members = [start]

        while stack:
            current = stack.pop()
            remaining = np.flatnonzero(unvisited)
            if remaining.size == 0:
                continue
            diff = coords[remaining] - coords[current]
            neighbors = remaining[np.sum(diff * diff, axis=1) <= radius_sq]
            if neighbors.size == 0:
                continue
            unvisited[neighbors] = False
            stack.extend(neighbors.tolist())
            members.extend(neighbors.tolist())

        component_id = len(component_members)
        member_array = np.asarray(members, dtype=np.int32)
        local_component_ids[member_array] = component_id
        component_members.append(member_array)

    component_sizes_local = np.asarray([len(members) for members in component_members], dtype=np.int32)
    component_ids[candidate_indices] = local_component_ids.astype(np.int16)
    for component_id, members in enumerate(component_members):
        component_sizes[candidate_indices[members]] = np.uint16(component_sizes_local[component_id])

    required_component_size = max(
        int(min_component_size),
        int(np.ceil(float(min_component_ratio) * float(num_tracks))),
    )
    largest_component_id = int(np.argmax(component_sizes_local))
    largest_component_size = int(component_sizes_local[largest_component_id])
    if largest_component_size >= required_component_size:
        final_mask[candidate_indices[component_members[largest_component_id]]] = True
        return final_mask, component_ids, component_sizes, False

    final_mask[candidate_indices] = True
    return final_mask, component_ids, component_sizes, True


def _select_major_spatial_components(
    keypoints: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    image_height: int,
    image_width: int,
    radius_ratio: float,
    radius_min_px: int,
    min_component_ratio: float,
    min_component_size: int,
    major_component_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")
    if candidate_mask.shape != (keypoints.shape[0],):
        raise ValueError(f"Expected candidate_mask shape {(keypoints.shape[0],)}, got {candidate_mask.shape}")

    num_tracks = int(keypoints.shape[0])
    final_mask = np.zeros(num_tracks, dtype=bool)
    component_ids = np.full(num_tracks, -1, dtype=np.int16)
    component_sizes = np.zeros(num_tracks, dtype=np.uint16)
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return final_mask, component_ids, component_sizes, False

    radius_px = float(max(int(radius_min_px), int(round(float(radius_ratio) * min(image_height, image_width)))))
    radius_sq = radius_px * radius_px
    coords = keypoints[candidate_indices]
    unvisited = np.ones(candidate_indices.size, dtype=bool)
    local_component_ids = np.full(candidate_indices.size, -1, dtype=np.int32)
    component_members: list[np.ndarray] = []

    while unvisited.any():
        start = int(np.flatnonzero(unvisited)[0])
        unvisited[start] = False
        stack = [start]
        members = [start]

        while stack:
            current = stack.pop()
            remaining = np.flatnonzero(unvisited)
            if remaining.size == 0:
                continue
            diff = coords[remaining] - coords[current]
            neighbors = remaining[np.sum(diff * diff, axis=1) <= radius_sq]
            if neighbors.size == 0:
                continue
            unvisited[neighbors] = False
            stack.extend(neighbors.tolist())
            members.extend(neighbors.tolist())

        component_id = len(component_members)
        member_array = np.asarray(members, dtype=np.int32)
        local_component_ids[member_array] = component_id
        component_members.append(member_array)

    component_sizes_local = np.asarray([len(members) for members in component_members], dtype=np.int32)
    component_ids[candidate_indices] = local_component_ids.astype(np.int16)
    for component_id, members in enumerate(component_members):
        component_sizes[candidate_indices[members]] = np.uint16(component_sizes_local[component_id])

    required_component_size = max(
        int(min_component_size),
        int(np.ceil(float(min_component_ratio) * float(num_tracks))),
    )
    largest_component_size = int(component_sizes_local.max())
    if largest_component_size < required_component_size:
        final_mask[candidate_indices] = True
        return final_mask, component_ids, component_sizes, True

    major_component_size = max(
        required_component_size,
        int(np.ceil(float(major_component_ratio) * float(largest_component_size))),
    )
    keep_component_ids = np.flatnonzero(component_sizes_local >= major_component_size)
    for component_id in keep_component_ids.tolist():
        final_mask[candidate_indices[component_members[component_id]]] = True
    return final_mask, component_ids, component_sizes, False


def _apply_manipulator_aware_filter(
    *,
    traj: np.ndarray,
    keypoints: np.ndarray,
    seed_mask: np.ndarray,
    supervision_mask: np.ndarray,
    intrinsics_segment: np.ndarray,
    extrinsics_segment: np.ndarray,
    image_height: int,
    image_width: int,
    min_depth: float,
    max_depth: float,
    max_depth_rank: float,
    min_motion_extent: float,
    cluster_radius_ratio: float,
    cluster_radius_min_px: int,
    min_component_ratio: float,
    min_component_size: int,
    component_keep_mode: str = "largest",
    major_component_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    traj = np.asarray(traj, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    seed_mask = np.asarray(seed_mask, dtype=bool)
    supervision_mask = np.asarray(supervision_mask, dtype=bool)
    intrinsics_segment = np.asarray(intrinsics_segment, dtype=np.float32)
    extrinsics_segment = np.asarray(extrinsics_segment, dtype=np.float32)

    query_depth_values = traj[:, 0, 2].astype(np.float32, copy=False)
    traj_query_depth_rank = _compute_query_depth_ranks(query_depth_values, seed_mask)
    near_depth_mask = (
        seed_mask
        & np.isfinite(traj_query_depth_rank)
        & (traj_query_depth_rank <= float(max_depth_rank))
    )

    world_tracks = traj_uvz_to_world_coordinates(
        traj,
        query_intrinsics=intrinsics_segment[0],
        query_w2c=extrinsics_segment[0],
        min_depth=min_depth,
        max_depth=max_depth,
    )
    traj_motion_extent, traj_motion_step_median = _compute_supervised_motion_metrics(
        world_tracks,
        supervision_mask,
    )
    motion_mask = (
        seed_mask
        & np.isfinite(traj_motion_extent)
        & (traj_motion_extent >= float(min_motion_extent))
    )
    traj_manipulator_candidate_mask = seed_mask & near_depth_mask & motion_mask
    if component_keep_mode == "largest":
        (
            final_mask,
            traj_manipulator_cluster_id,
            traj_manipulator_component_size,
            fallback_used,
        ) = _select_largest_spatial_component(
            keypoints,
            traj_manipulator_candidate_mask,
            image_height=image_height,
            image_width=image_width,
            radius_ratio=cluster_radius_ratio,
            radius_min_px=cluster_radius_min_px,
            min_component_ratio=min_component_ratio,
            min_component_size=min_component_size,
        )
    elif component_keep_mode == "major":
        if major_component_ratio is None:
            raise ValueError("major_component_ratio is required when component_keep_mode='major'")
        (
            final_mask,
            traj_manipulator_cluster_id,
            traj_manipulator_component_size,
            fallback_used,
        ) = _select_major_spatial_components(
            keypoints,
            traj_manipulator_candidate_mask,
            image_height=image_height,
            image_width=image_width,
            radius_ratio=cluster_radius_ratio,
            radius_min_px=cluster_radius_min_px,
            min_component_ratio=min_component_ratio,
            min_component_size=min_component_size,
            major_component_ratio=major_component_ratio,
        )
    else:
        raise ValueError(f"Unsupported component_keep_mode: {component_keep_mode}")
    return (
        final_mask,
        traj_query_depth_rank,
        traj_motion_extent,
        traj_motion_step_median,
        traj_manipulator_candidate_mask,
        traj_manipulator_cluster_id,
        traj_manipulator_component_size,
        near_depth_mask,
        motion_mask,
        bool(fallback_used),
    )


def compute_traj_base_geometry(
    traj: np.ndarray,
    *,
    visibs: np.ndarray | None = None,
    image_width: int = 1280,
    image_height: int = 720,
    min_valid_frames: int = 3,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    boundary_margin: int = 50,
    visibility_threshold: float = 0.5,
    check_depth_smoothness: bool = True,
    depth_change_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Compute base trajectory geometry checks and keep their individual masks."""
    traj = np.asarray(traj, dtype=np.float32)
    num_tracks, num_frames, _ = traj.shape
    visibility = _normalize_visibility(visibs, num_tracks=num_tracks, num_frames=num_frames)

    valid_frames = np.isfinite(traj).all(axis=-1)
    valid_counts = valid_frames.sum(axis=1).astype(np.int32)
    valid_count_mask = valid_counts >= int(min_valid_frames)

    depth_range_mask = np.ones(num_tracks, dtype=bool)
    boundary_mask = np.ones(num_tracks, dtype=bool)
    visibility_mask = np.ones(num_tracks, dtype=bool)
    depth_smooth_mask = np.ones(num_tracks, dtype=bool)

    depth_values = traj[:, :, 2]
    u_values = traj[:, :, 0]
    v_values = traj[:, :, 1]

    for track_idx in range(num_tracks):
        if not valid_count_mask[track_idx]:
            continue

        valid_depths = depth_values[track_idx, valid_frames[track_idx]]
        if len(valid_depths) > 0:
            if (valid_depths < min_depth).any() or (valid_depths > max_depth).any():
                depth_range_mask[track_idx] = False

        valid_u = u_values[track_idx, valid_frames[track_idx]]
        valid_v = v_values[track_idx, valid_frames[track_idx]]
        if len(valid_u) > 0:
            u_in_bounds = (valid_u >= -boundary_margin) & (valid_u <= image_width + boundary_margin)
            v_in_bounds = (valid_v >= -boundary_margin) & (valid_v <= image_height + boundary_margin)
            if not (u_in_bounds.all() and v_in_bounds.all()):
                boundary_mask[track_idx] = False

        if visibility is not None and valid_counts[track_idx] > 0:
            vis_count = int((visibility[track_idx] & valid_frames[track_idx]).sum())
            vis_ratio = vis_count / valid_counts[track_idx]
            if vis_ratio < visibility_threshold:
                visibility_mask[track_idx] = False

        if check_depth_smoothness and len(valid_depths) > 1:
            depth_diff = np.diff(valid_depths)
            if np.std(depth_diff) > depth_change_threshold:
                depth_smooth_mask[track_idx] = False

    traj_valid_mask = (
        valid_count_mask
        & depth_range_mask
        & boundary_mask
        & visibility_mask
        & depth_smooth_mask
    )
    return {
        "traj_valid_mask": traj_valid_mask.astype(bool),
        "valid_frames": valid_frames.astype(bool),
        "valid_counts": valid_counts.astype(np.int32),
        "valid_count_mask": valid_count_mask.astype(bool),
        "depth_range_mask": depth_range_mask.astype(bool),
        "boundary_mask": boundary_mask.astype(bool),
        "visibility_mask": visibility_mask.astype(bool),
        "depth_smooth_mask": depth_smooth_mask.astype(bool),
    }


def compute_traj_valid_mask(
    traj: np.ndarray,
    visibs: np.ndarray = None,
    image_width: int = 1280,
    image_height: int = 720,
    min_valid_frames: int = 3,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
    boundary_margin: int = 50,
    visibility_threshold: float = 0.5,
    check_depth_smoothness: bool = True,
    depth_change_threshold: float = 0.5,
) -> np.ndarray:
    """Compute trajectory validity mask (spatial dimension)."""
    return compute_traj_base_geometry(
        traj,
        visibs=visibs,
        image_width=image_width,
        image_height=image_height,
        min_valid_frames=min_valid_frames,
        min_depth=min_depth,
        max_depth=max_depth,
        boundary_margin=boundary_margin,
        visibility_threshold=visibility_threshold,
        check_depth_smoothness=check_depth_smoothness,
        depth_change_threshold=depth_change_threshold,
    )["traj_valid_mask"]


def resolve_traj_filter_config(filter_args) -> dict:
    """Resolve effective trajectory filtering settings from CLI args."""
    level = getattr(filter_args, "filter_level", "none") if filter_args is not None else "none"
    profile = (
        getattr(filter_args, "traj_filter_profile", TRAJ_FILTER_PROFILE_EXTERNAL)
        if filter_args is not None
        else TRAJ_FILTER_PROFILE_EXTERNAL
    )
    defaults = {
        "basic": {
            "enabled": True,
            "profile": profile,
            "min_valid_frames": 3,
            "min_depth": 0.01,
            "max_depth": 10.0,
            "boundary_margin": 50,
            "visibility_threshold": 0.0,
            "check_depth_smoothness": False,
            "depth_change_threshold": 0.5,
            "use_visibility": False,
            "use_query_depth_quality": True,
            "use_temporal_depth_consistency": True,
            "use_depth_volatility_guidance": True,
            "temporal_depth_abs_tol": TEMPORAL_DEPTH_ABS_TOL,
            "temporal_depth_rel_tol": TEMPORAL_DEPTH_REL_TOL,
            "temporal_min_consistency_ratio": TEMPORAL_MIN_CONSISTENCY_RATIO,
            "volatility_low_percentile": VOLATILITY_LOW_PERCENTILE,
            "volatility_high_percentile": VOLATILITY_HIGH_PERCENTILE,
            "volatility_mask_percentile": VOLATILITY_MASK_PERCENTILE,
            "wrist_min_prefix_frames": WRIST_MIN_PREFIX_FRAMES,
            "wrist_min_support_frames": WRIST_MIN_SUPPORT_FRAMES,
            "wrist_prefix_ratio": WRIST_PREFIX_RATIO,
            "wrist_support_ratio": WRIST_SUPPORT_RATIO,
        },
        "standard": {
            "enabled": True,
            "profile": profile,
            "min_valid_frames": 3,
            "min_depth": 0.01,
            "max_depth": 10.0,
            "boundary_margin": 50,
            "visibility_threshold": 0.5,
            "check_depth_smoothness": True,
            "depth_change_threshold": 0.5,
            "use_visibility": True,
            "use_query_depth_quality": True,
            "use_temporal_depth_consistency": True,
            "use_depth_volatility_guidance": True,
            "temporal_depth_abs_tol": TEMPORAL_DEPTH_ABS_TOL,
            "temporal_depth_rel_tol": TEMPORAL_DEPTH_REL_TOL,
            "temporal_min_consistency_ratio": TEMPORAL_MIN_CONSISTENCY_RATIO,
            "volatility_low_percentile": VOLATILITY_LOW_PERCENTILE,
            "volatility_high_percentile": VOLATILITY_HIGH_PERCENTILE,
            "volatility_mask_percentile": VOLATILITY_MASK_PERCENTILE,
            "wrist_min_prefix_frames": WRIST_MIN_PREFIX_FRAMES,
            "wrist_min_support_frames": WRIST_MIN_SUPPORT_FRAMES,
            "wrist_prefix_ratio": WRIST_PREFIX_RATIO,
            "wrist_support_ratio": WRIST_SUPPORT_RATIO,
        },
        "strict": {
            "enabled": True,
            "profile": profile,
            "min_valid_frames": 5,
            "min_depth": 0.01,
            "max_depth": 10.0,
            "boundary_margin": 20,
            "visibility_threshold": 0.6,
            "check_depth_smoothness": True,
            "depth_change_threshold": 0.3,
            "use_visibility": True,
            "use_query_depth_quality": True,
            "use_temporal_depth_consistency": True,
            "use_depth_volatility_guidance": True,
            "temporal_depth_abs_tol": TEMPORAL_DEPTH_ABS_TOL,
            "temporal_depth_rel_tol": TEMPORAL_DEPTH_REL_TOL,
            "temporal_min_consistency_ratio": TEMPORAL_MIN_CONSISTENCY_RATIO,
            "volatility_low_percentile": VOLATILITY_LOW_PERCENTILE,
            "volatility_high_percentile": VOLATILITY_HIGH_PERCENTILE,
            "volatility_mask_percentile": VOLATILITY_MASK_PERCENTILE,
            "wrist_min_prefix_frames": WRIST_MIN_PREFIX_FRAMES,
            "wrist_min_support_frames": WRIST_MIN_SUPPORT_FRAMES,
            "wrist_prefix_ratio": WRIST_PREFIX_RATIO,
            "wrist_support_ratio": WRIST_SUPPORT_RATIO,
        },
        "none": {
            "enabled": False,
            "profile": profile,
            "min_valid_frames": 0,
            "min_depth": 0.01,
            "max_depth": 10.0,
            "boundary_margin": 50,
            "visibility_threshold": 0.0,
            "check_depth_smoothness": False,
            "depth_change_threshold": 0.5,
            "use_visibility": False,
            "use_query_depth_quality": False,
            "use_temporal_depth_consistency": False,
            "use_depth_volatility_guidance": False,
            "temporal_depth_abs_tol": TEMPORAL_DEPTH_ABS_TOL,
            "temporal_depth_rel_tol": TEMPORAL_DEPTH_REL_TOL,
            "temporal_min_consistency_ratio": TEMPORAL_MIN_CONSISTENCY_RATIO,
            "volatility_low_percentile": VOLATILITY_LOW_PERCENTILE,
            "volatility_high_percentile": VOLATILITY_HIGH_PERCENTILE,
            "volatility_mask_percentile": VOLATILITY_MASK_PERCENTILE,
            "wrist_min_prefix_frames": WRIST_MIN_PREFIX_FRAMES,
            "wrist_min_support_frames": WRIST_MIN_SUPPORT_FRAMES,
            "wrist_prefix_ratio": WRIST_PREFIX_RATIO,
            "wrist_support_ratio": WRIST_SUPPORT_RATIO,
        },
    }
    config = defaults[level].copy()
    config.update(
        {
            "wrist_manipulator_max_depth_rank": WRIST_MANIPULATOR_MAX_DEPTH_RANK,
            "wrist_manipulator_min_motion_extent": WRIST_MANIPULATOR_MIN_MOTION_EXTENT,
            "wrist_manipulator_cluster_radius_ratio": WRIST_MANIPULATOR_CLUSTER_RADIUS_RATIO,
            "wrist_manipulator_cluster_radius_min_px": WRIST_MANIPULATOR_CLUSTER_RADIUS_MIN_PX,
            "wrist_manipulator_min_component_ratio": WRIST_MANIPULATOR_MIN_COMPONENT_RATIO,
            "wrist_manipulator_min_component_size": WRIST_MANIPULATOR_MIN_COMPONENT_SIZE,
            "external_manipulator_v2_max_depth_rank": EXTERNAL_MANIPULATOR_V2_MAX_DEPTH_RANK,
            "external_manipulator_v2_min_motion_extent": EXTERNAL_MANIPULATOR_V2_MIN_MOTION_EXTENT,
            "external_manipulator_v2_cluster_radius_ratio": EXTERNAL_MANIPULATOR_V2_CLUSTER_RADIUS_RATIO,
            "external_manipulator_v2_cluster_radius_min_px": EXTERNAL_MANIPULATOR_V2_CLUSTER_RADIUS_MIN_PX,
            "external_manipulator_v2_min_component_ratio": EXTERNAL_MANIPULATOR_V2_MIN_COMPONENT_RATIO,
            "external_manipulator_v2_min_component_size": EXTERNAL_MANIPULATOR_V2_MIN_COMPONENT_SIZE,
            "external_manipulator_v2_major_component_ratio": EXTERNAL_MANIPULATOR_V2_MAJOR_COMPONENT_RATIO,
        }
    )
    if filter_args is None:
        return config

    overrides = {
        "min_valid_frames": getattr(filter_args, "min_valid_frames", None),
        "visibility_threshold": getattr(filter_args, "visibility_threshold", None),
        "min_depth": getattr(filter_args, "min_depth", None),
        "max_depth": getattr(filter_args, "max_depth", None),
        "boundary_margin": getattr(filter_args, "boundary_margin", None),
        "depth_change_threshold": getattr(filter_args, "depth_change_threshold", None),
        "temporal_min_consistency_ratio": getattr(filter_args, "temporal_min_consistency_ratio", None),
        "volatility_mask_percentile": getattr(filter_args, "volatility_mask_percentile", None),
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def compute_query_depth_quality_mask(
    keypoints: np.ndarray,
    query_depth: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    patch_radius: int = QUERY_DEPTH_PATCH_RADIUS,
    min_patch_valid_ratio: float = QUERY_DEPTH_MIN_VALID_RATIO,
    median_abs_threshold: float = QUERY_DEPTH_ABS_TOL,
    median_rel_threshold: float = QUERY_DEPTH_REL_TOL,
) -> np.ndarray:
    """Reject query points whose raw query-frame depth is invalid or locally isolated."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    query_depth = np.asarray(query_depth, dtype=np.float32)
    if query_depth.ndim != 2:
        raise ValueError(f"Expected query_depth shape (H, W), got {query_depth.shape}")
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N, 2), got {keypoints.shape}")

    height, width = query_depth.shape
    if width == 0 or height == 0:
        return np.zeros(keypoints.shape[0], dtype=bool)

    xs = np.clip(np.round(keypoints[:, 0]).astype(np.int32), 0, width - 1)
    ys = np.clip(np.round(keypoints[:, 1]).astype(np.int32), 0, height - 1)
    valid_mask = np.zeros(keypoints.shape[0], dtype=bool)

    for track_idx, (x_coord, y_coord) in enumerate(zip(xs, ys)):
        query_value = float(query_depth[y_coord, x_coord])
        if not np.isfinite(query_value) or query_value <= min_depth or query_value >= max_depth:
            continue

        y0 = max(0, y_coord - patch_radius)
        y1 = min(height, y_coord + patch_radius + 1)
        x0 = max(0, x_coord - patch_radius)
        x1 = min(width, x_coord + patch_radius + 1)
        patch = query_depth[y0:y1, x0:x1]
        patch_valid = np.isfinite(patch) & (patch > min_depth) & (patch < max_depth)

        if patch_valid.size == 0 or float(patch_valid.mean()) < min_patch_valid_ratio:
            continue

        patch_values = patch[patch_valid]
        if patch_values.size == 0:
            continue

        patch_median = float(np.median(patch_values))
        deviation_limit = max(median_abs_threshold, median_rel_threshold * patch_median)
        if abs(query_value - patch_median) > deviation_limit:
            continue

        valid_mask[track_idx] = True

    return valid_mask


def compute_depth_volatility_map(
    full_depths: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    low_percentile: float = VOLATILITY_LOW_PERCENTILE,
    high_percentile: float = VOLATILITY_HIGH_PERCENTILE,
) -> np.ndarray:
    """Compute per-pixel temporal depth volatility from raw depth video."""
    full_depths = np.asarray(full_depths, dtype=np.float32)
    if full_depths.ndim != 3:
        raise ValueError(f"Expected full_depths shape (T,H,W), got {full_depths.shape}")

    valid = np.isfinite(full_depths) & (full_depths > min_depth) & (full_depths < max_depth)
    if not np.any(valid):
        return np.zeros(full_depths.shape[1:], dtype=np.float32)

    depths_nan = np.where(valid, full_depths, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        depth_lo = np.nanpercentile(depths_nan, low_percentile, axis=0)
        depth_hi = np.nanpercentile(depths_nan, high_percentile, axis=0)

    volatility = np.nan_to_num(depth_hi - depth_lo, nan=0.0, posinf=0.0, neginf=0.0)
    valid_counts = valid.sum(axis=0)
    volatility[valid_counts < 2] = 0.0
    return volatility.astype(np.float32)


def compute_high_volatility_mask(
    volatility_map: np.ndarray,
    *,
    percentile: float = VOLATILITY_MASK_PERCENTILE,
) -> tuple[np.ndarray, float]:
    """Threshold a volatility map by global percentile."""
    volatility_map = np.asarray(volatility_map, dtype=np.float32)
    if volatility_map.ndim != 2:
        raise ValueError(f"Expected volatility_map shape (H,W), got {volatility_map.shape}")

    finite = np.isfinite(volatility_map)
    values = volatility_map[finite]
    if values.size == 0:
        return np.zeros_like(volatility_map, dtype=bool), float("nan")

    threshold = float(np.percentile(values, percentile))
    if not np.isfinite(threshold) or threshold <= 0.0:
        return finite & (volatility_map > 0.0), threshold
    return finite & (volatility_map >= threshold), threshold


def traj_uvz_to_world_coordinates(
    traj_uvz: np.ndarray,
    *,
    query_intrinsics: np.ndarray,
    query_w2c: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Lift query-camera uvz trajectories back into world coordinates."""
    traj_uvz = np.asarray(traj_uvz, dtype=np.float32)
    query_intrinsics = np.asarray(query_intrinsics, dtype=np.float32)
    query_w2c = np.asarray(query_w2c, dtype=np.float32)
    if traj_uvz.ndim != 3 or traj_uvz.shape[-1] != 3:
        raise ValueError(f"Expected traj_uvz shape (N,T,3), got {traj_uvz.shape}")

    fx = float(query_intrinsics[0, 0])
    fy = float(query_intrinsics[1, 1])
    cx = float(query_intrinsics[0, 2])
    cy = float(query_intrinsics[1, 2])

    u = traj_uvz[..., 0]
    v = traj_uvz[..., 1]
    z = traj_uvz[..., 2]
    valid = np.isfinite(traj_uvz).all(axis=-1) & (z > min_depth) & (z < max_depth)

    x_cam = np.where(valid, (u - cx) * z / (fx + 1e-8), np.nan)
    y_cam = np.where(valid, (v - cy) * z / (fy + 1e-8), np.nan)
    pts_cam = np.stack([x_cam, y_cam, np.where(valid, z, np.nan)], axis=-1)
    pts_cam_h = np.concatenate(
        [pts_cam, np.ones((*pts_cam.shape[:2], 1), dtype=np.float32)],
        axis=-1,
    )
    c2w = np.linalg.inv(query_w2c).astype(np.float32)
    pts_world = (c2w @ pts_cam_h.reshape(-1, 4).T).T.reshape(*pts_cam.shape[:2], 4)[..., :3]
    pts_world = pts_world.astype(np.float32)
    pts_world[~valid] = np.nan
    return pts_world


def project_world_tracks_to_camera_uvz(
    world_tracks: np.ndarray,
    *,
    intrinsics_segment: np.ndarray,
    extrinsics_segment: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Project world-space trajectories into each segment frame camera."""
    world_tracks = np.asarray(world_tracks, dtype=np.float32)
    intrinsics_segment = np.asarray(intrinsics_segment, dtype=np.float32)
    extrinsics_segment = np.asarray(extrinsics_segment, dtype=np.float32)

    if world_tracks.ndim != 3 or world_tracks.shape[-1] != 3:
        raise ValueError(f"Expected world_tracks shape (N,T,3), got {world_tracks.shape}")
    num_tracks, num_frames, _ = world_tracks.shape
    if intrinsics_segment.shape != (num_frames, 3, 3):
        raise ValueError(
            f"Expected intrinsics_segment shape {(num_frames, 3, 3)}, got {intrinsics_segment.shape}"
        )
    if extrinsics_segment.shape != (num_frames, 4, 4):
        raise ValueError(
            f"Expected extrinsics_segment shape {(num_frames, 4, 4)}, got {extrinsics_segment.shape}"
        )

    world_tracks_h = np.concatenate(
        [world_tracks, np.ones((num_tracks, num_frames, 1), dtype=np.float32)],
        axis=-1,
    )
    tracks_cam_h = np.einsum("tij,ntj->nti", extrinsics_segment, world_tracks_h)
    tracks_cam = tracks_cam_h[..., :3]
    tracks_img = np.einsum("tij,ntj->nti", intrinsics_segment, tracks_cam)

    z = tracks_cam[..., 2]
    u = tracks_img[..., 0] / (z + 1e-8)
    v = tracks_img[..., 1] / (z + 1e-8)
    projected = np.stack([u, v, z], axis=-1).astype(np.float32)

    valid = (
        np.isfinite(world_tracks).all(axis=-1)
        & np.isfinite(projected).all(axis=-1)
        & (z > min_depth)
        & (z < max_depth)
    )
    projected[~valid] = np.nan
    return projected


def evaluate_temporal_depth_consistency(
    traj_uvz: np.ndarray,
    *,
    visibs: np.ndarray | None,
    raw_depths_segment: np.ndarray,
    intrinsics_segment: np.ndarray,
    extrinsics_segment: np.ndarray,
    min_depth: float,
    max_depth: float,
    min_valid_frames: int,
    min_consistency_ratio: float = TEMPORAL_MIN_CONSISTENCY_RATIO,
    depth_abs_tol: float = TEMPORAL_DEPTH_ABS_TOL,
    depth_rel_tol: float = TEMPORAL_DEPTH_REL_TOL,
    high_volatility_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | int]:
    """Check whether trajectories remain depth-consistent after per-frame reprojection."""
    traj_uvz = np.asarray(traj_uvz, dtype=np.float32)
    num_tracks, num_frames, _ = traj_uvz.shape
    visibility = _normalize_visibility(visibs, num_tracks=num_tracks, num_frames=num_frames)

    raw_depths_segment, intrinsics_segment, extrinsics_segment = _require_segment_geometry(
        raw_depths_segment=raw_depths_segment,
        intrinsics_segment=intrinsics_segment,
        extrinsics_segment=extrinsics_segment,
        expected_num_frames=num_frames,
    )

    world_tracks = traj_uvz_to_world_coordinates(
        traj_uvz,
        query_intrinsics=intrinsics_segment[0],
        query_w2c=extrinsics_segment[0],
        min_depth=min_depth,
        max_depth=max_depth,
    )
    reprojected_uvz = project_world_tracks_to_camera_uvz(
        world_tracks,
        intrinsics_segment=intrinsics_segment,
        extrinsics_segment=extrinsics_segment,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    proj_valid = np.isfinite(reprojected_uvz).all(axis=-1)
    xs, ys = _round_projected_coords(reprojected_uvz)
    height, width = raw_depths_segment.shape[1:]
    in_bounds = proj_valid & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

    xs_clip = np.clip(xs, 0, width - 1)
    ys_clip = np.clip(ys, 0, height - 1)
    observed_depth = raw_depths_segment[np.arange(num_frames)[None, :], ys_clip, xs_clip]
    observed_valid = np.isfinite(observed_depth) & (observed_depth > min_depth) & (observed_depth < max_depth)

    compare_mask = in_bounds & observed_valid
    if visibility is not None:
        compare_mask &= visibility

    depth_error = np.abs(reprojected_uvz[..., 2] - observed_depth)
    depth_limit = np.maximum(depth_abs_tol, depth_rel_tol * observed_depth)
    consistent_frame_mask = compare_mask & (depth_error <= depth_limit)

    if high_volatility_mask is None:
        volatility_frame_mask = np.zeros((num_tracks, num_frames), dtype=bool)
    else:
        high_volatility_mask = np.asarray(high_volatility_mask, dtype=bool)
        if high_volatility_mask.shape != (height, width):
            raise ValueError(
                f"Expected high_volatility_mask shape {(height, width)}, got {high_volatility_mask.shape}"
            )
        volatility_frame_mask = compare_mask & high_volatility_mask[ys_clip, xs_clip]

    stable_compare_mask = compare_mask & (~volatility_frame_mask)
    stable_consistent_frame_mask = consistent_frame_mask & (~volatility_frame_mask)

    compare_counts = compare_mask.sum(axis=1).astype(np.int32)
    consistent_counts = consistent_frame_mask.sum(axis=1).astype(np.int32)
    stable_compare_counts = stable_compare_mask.sum(axis=1).astype(np.int32)
    stable_consistent_counts = stable_consistent_frame_mask.sum(axis=1).astype(np.int32)
    volatility_counts = volatility_frame_mask.sum(axis=1).astype(np.int32)

    consistency_ratio = _counts_to_ratio(consistent_counts, compare_counts)
    stable_consistency_ratio = _counts_to_ratio(stable_consistent_counts, stable_compare_counts)
    volatility_exposure_ratio = _counts_to_ratio(volatility_counts, compare_counts)

    required_compare_frames = min(num_frames, max(3, int(min_valid_frames)))
    all_pass = (compare_counts >= required_compare_frames) & (consistency_ratio >= min_consistency_ratio)
    stable_frames_sufficient = stable_compare_counts >= required_compare_frames
    stable_pass = stable_frames_sufficient & (stable_consistency_ratio >= min_consistency_ratio)
    mask = np.where(stable_frames_sufficient, stable_pass, all_pass)

    return {
        "mask": mask.astype(bool),
        "consistency_ratio": consistency_ratio.astype(np.float32),
        "stable_consistency_ratio": stable_consistency_ratio.astype(np.float32),
        "compare_counts": compare_counts,
        "stable_compare_counts": stable_compare_counts,
        "required_compare_frames": int(required_compare_frames),
        "reprojected_uvz": reprojected_uvz.astype(np.float32),
        "compare_mask": compare_mask.astype(bool),
        "consistent_frame_mask": consistent_frame_mask.astype(bool),
        "high_volatility_hit": volatility_frame_mask.any(axis=1),
        "volatility_exposure_ratio": volatility_exposure_ratio.astype(np.float32),
        "stable_frames_sufficient": stable_frames_sufficient.astype(bool),
        "all_pass": all_pass.astype(bool),
        "stable_pass": stable_pass.astype(bool),
    }


def build_traj_filter_result(
    traj: np.ndarray,
    visibs: np.ndarray | None,
    image_width: int,
    image_height: int,
    filter_args,
    *,
    keypoints: np.ndarray | None = None,
    query_depth: np.ndarray | None = None,
    raw_depths_segment: np.ndarray | None = None,
    intrinsics_segment: np.ndarray | None = None,
    extrinsics_segment: np.ndarray | None = None,
    depth_volatility_map: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build per-trajectory mask plus debug metadata."""
    traj = np.asarray(traj, dtype=np.float32)
    num_tracks, num_frames, _ = traj.shape
    config = resolve_traj_filter_config(filter_args)

    default_ratio = np.full(num_tracks, np.nan, dtype=np.float32)
    default_counts = np.zeros(num_tracks, dtype=np.uint16)
    default_hits = np.zeros(num_tracks, dtype=bool)
    default_bits = np.zeros(num_tracks, dtype=np.uint8)
    default_supervision_mask = np.isfinite(traj).all(axis=-1).astype(bool)
    default_supervision_prefix_len = _compute_true_prefix_lengths(default_supervision_mask).astype(np.uint16)
    default_supervision_count = default_supervision_mask.sum(axis=1).astype(np.uint16)
    default_manipulator_mask = np.zeros(num_tracks, dtype=bool)
    default_manipulator_rank = np.full(num_tracks, np.nan, dtype=np.float32)
    default_cluster_id = np.full(num_tracks, -1, dtype=np.int16)
    default_component_size = np.zeros(num_tracks, dtype=np.uint16)
    default_fallback_used = np.asarray(False, dtype=bool)

    if not config["enabled"]:
        return {
            "traj_valid_mask": np.ones(num_tracks, dtype=bool),
            "traj_depth_consistency_ratio": default_ratio,
            "traj_stable_depth_consistency_ratio": default_ratio.copy(),
            "traj_high_volatility_hit": default_hits,
            "traj_volatility_exposure_ratio": default_ratio.copy(),
            "traj_compare_frame_count": default_counts.copy(),
            "traj_stable_compare_frame_count": default_counts.copy(),
            "traj_mask_reason_bits": default_bits,
            "traj_supervision_mask": default_supervision_mask,
            "traj_supervision_prefix_len": default_supervision_prefix_len,
            "traj_supervision_count": default_supervision_count,
            "traj_wrist_seed_mask": default_manipulator_mask.copy(),
            "traj_query_depth_rank": default_manipulator_rank.copy(),
            "traj_motion_extent": default_manipulator_rank.copy(),
            "traj_motion_step_median": default_manipulator_rank.copy(),
            "traj_manipulator_candidate_mask": default_manipulator_mask.copy(),
            "traj_manipulator_cluster_id": default_cluster_id.copy(),
            "traj_manipulator_component_size": default_component_size.copy(),
            "traj_manipulator_cluster_fallback_used": default_fallback_used.copy(),
        }

    visibility = _normalize_visibility(visibs, num_tracks=num_tracks, num_frames=num_frames)
    visibs_for_filter = visibility if config["use_visibility"] else None

    base_geometry = compute_traj_base_geometry(
        traj,
        visibs=visibs_for_filter,
        image_width=image_width,
        image_height=image_height,
        min_valid_frames=config["min_valid_frames"],
        min_depth=config["min_depth"],
        max_depth=config["max_depth"],
        boundary_margin=config["boundary_margin"],
        visibility_threshold=config["visibility_threshold"],
        check_depth_smoothness=config["check_depth_smoothness"],
        depth_change_threshold=config["depth_change_threshold"],
    )
    base_mask = np.asarray(base_geometry["traj_valid_mask"]).astype(bool, copy=False)
    wrist_base_mask = (
        np.asarray(base_geometry["valid_count_mask"]).astype(bool, copy=False)
        & np.asarray(base_geometry["depth_range_mask"]).astype(bool, copy=False)
        & np.asarray(base_geometry["depth_smooth_mask"]).astype(bool, copy=False)
    )

    query_depth_mask = np.ones(num_tracks, dtype=bool)
    if config["use_query_depth_quality"]:
        if keypoints is None or query_depth is None:
            raise ValueError("keypoints and query_depth are required when query-depth quality filtering is enabled")
        if keypoints.shape[0] != traj.shape[0]:
            raise ValueError(
                f"Expected keypoints and trajectories to share track count, got {keypoints.shape[0]} and {traj.shape[0]}"
            )
        query_depth_mask = compute_query_depth_quality_mask(
            keypoints,
            query_depth,
            min_depth=config["min_depth"],
            max_depth=config["max_depth"],
        )

    temporal_mask = np.ones(num_tracks, dtype=bool)
    depth_consistency_ratio = default_ratio.copy()
    stable_depth_consistency_ratio = default_ratio.copy()
    high_volatility_hit = default_hits.copy()
    volatility_exposure_ratio = default_ratio.copy()
    compare_frame_count = default_counts.copy()
    stable_compare_frame_count = default_counts.copy()
    stable_temporal_fail = np.zeros(num_tracks, dtype=bool)
    supervision_mask = default_supervision_mask.copy()

    if config["use_temporal_depth_consistency"]:
        high_volatility_mask = None
        if config["use_depth_volatility_guidance"]:
            if depth_volatility_map is None:
                raise ValueError("depth_volatility_map is required when volatility guidance is enabled")
            high_volatility_mask, _ = compute_high_volatility_mask(
                depth_volatility_map,
                percentile=config["volatility_mask_percentile"],
            )

        temporal_result = evaluate_temporal_depth_consistency(
            traj,
            visibs=visibility,
            raw_depths_segment=raw_depths_segment,
            intrinsics_segment=intrinsics_segment,
            extrinsics_segment=extrinsics_segment,
            min_depth=config["min_depth"],
            max_depth=config["max_depth"],
            min_valid_frames=config["min_valid_frames"],
            min_consistency_ratio=config["temporal_min_consistency_ratio"],
            depth_abs_tol=config["temporal_depth_abs_tol"],
            depth_rel_tol=config["temporal_depth_rel_tol"],
            high_volatility_mask=high_volatility_mask,
        )
        temporal_mask = np.asarray(temporal_result["mask"]).astype(bool, copy=False)
        depth_consistency_ratio = np.asarray(temporal_result["consistency_ratio"]).astype(np.float32, copy=False)
        stable_depth_consistency_ratio = (
            np.asarray(temporal_result["stable_consistency_ratio"]).astype(np.float32, copy=False)
        )
        high_volatility_hit = np.asarray(temporal_result["high_volatility_hit"]).astype(bool, copy=False)
        volatility_exposure_ratio = (
            np.asarray(temporal_result["volatility_exposure_ratio"]).astype(np.float32, copy=False)
        )
        compare_frame_count = np.asarray(temporal_result["compare_counts"]).astype(np.uint16, copy=False)
        stable_compare_frame_count = (
            np.asarray(temporal_result["stable_compare_counts"]).astype(np.uint16, copy=False)
        )
        supervision_mask = np.asarray(temporal_result["consistent_frame_mask"]).astype(bool, copy=False)
        stable_temporal_fail = (
            np.asarray(temporal_result["stable_frames_sufficient"]).astype(bool, copy=False)
            & (~np.asarray(temporal_result["stable_pass"]).astype(bool, copy=False))
        )
    elif visibility is not None:
        supervision_mask &= visibility

    supervision_prefix_len = _compute_true_prefix_lengths(supervision_mask).astype(np.uint16)
    supervision_count = supervision_mask.sum(axis=1).astype(np.uint16)

    reason_bits = np.zeros(num_tracks, dtype=np.uint8)
    profile = config["profile"]
    if profile not in {
        TRAJ_FILTER_PROFILE_EXTERNAL,
        TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR,
        TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR_V2,
        TRAJ_FILTER_PROFILE_WRIST,
        TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR,
    }:
        raise ValueError(f"Unsupported traj_filter_profile: {profile}")

    wrist_seed_mask = default_manipulator_mask.copy()
    traj_query_depth_rank = default_manipulator_rank.copy()
    traj_motion_extent = default_manipulator_rank.copy()
    traj_motion_step_median = default_manipulator_rank.copy()
    traj_manipulator_candidate_mask = default_manipulator_mask.copy()
    traj_manipulator_cluster_id = default_cluster_id.copy()
    traj_manipulator_component_size = default_component_size.copy()
    traj_manipulator_cluster_fallback_used = default_fallback_used.copy()
    external_seed_mask = base_mask & query_depth_mask & temporal_mask

    if profile in {
        TRAJ_FILTER_PROFILE_EXTERNAL,
        TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR,
        TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR_V2,
    }:
        reason_bits[~base_mask] |= MASK_REASON_BASE_GEOMETRY_FAIL
        reason_bits[~query_depth_mask] |= MASK_REASON_QUERY_DEPTH_FAIL
        reason_bits[~temporal_mask] |= MASK_REASON_TEMPORAL_CONSISTENCY_FAIL
        reason_bits[stable_temporal_fail & (~temporal_mask)] |= MASK_REASON_STABLE_TEMPORAL_FAIL
        if profile == TRAJ_FILTER_PROFILE_EXTERNAL:
            final_mask = external_seed_mask
        else:
            wrist_seed_mask = external_seed_mask.copy()
            raw_depths_segment, intrinsics_segment, extrinsics_segment = _require_segment_geometry(
                raw_depths_segment=raw_depths_segment,
                intrinsics_segment=intrinsics_segment,
                extrinsics_segment=extrinsics_segment,
                expected_num_frames=num_frames,
            )
            manipulator_filter_kwargs = {
                "traj": traj,
                "keypoints": keypoints,
                "seed_mask": wrist_seed_mask,
                "supervision_mask": supervision_mask,
                "intrinsics_segment": intrinsics_segment,
                "extrinsics_segment": extrinsics_segment,
                "image_height": image_height,
                "image_width": image_width,
                "min_depth": config["min_depth"],
                "max_depth": config["max_depth"],
            }
            if profile == TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR:
                manipulator_filter_kwargs.update(
                    {
                        "max_depth_rank": config["wrist_manipulator_max_depth_rank"],
                        "min_motion_extent": config["wrist_manipulator_min_motion_extent"],
                        "cluster_radius_ratio": config["wrist_manipulator_cluster_radius_ratio"],
                        "cluster_radius_min_px": config["wrist_manipulator_cluster_radius_min_px"],
                        "min_component_ratio": config["wrist_manipulator_min_component_ratio"],
                        "min_component_size": config["wrist_manipulator_min_component_size"],
                        "component_keep_mode": "largest",
                    }
                )
            else:
                manipulator_filter_kwargs.update(
                    {
                        "max_depth_rank": config["external_manipulator_v2_max_depth_rank"],
                        "min_motion_extent": config["external_manipulator_v2_min_motion_extent"],
                        "cluster_radius_ratio": config["external_manipulator_v2_cluster_radius_ratio"],
                        "cluster_radius_min_px": config["external_manipulator_v2_cluster_radius_min_px"],
                        "min_component_ratio": config["external_manipulator_v2_min_component_ratio"],
                        "min_component_size": config["external_manipulator_v2_min_component_size"],
                        "component_keep_mode": "major",
                        "major_component_ratio": config["external_manipulator_v2_major_component_ratio"],
                    }
                )
            (
                final_mask,
                traj_query_depth_rank,
                traj_motion_extent,
                traj_motion_step_median,
                traj_manipulator_candidate_mask,
                traj_manipulator_cluster_id,
                traj_manipulator_component_size,
                near_depth_mask,
                motion_mask,
                fallback_used,
            ) = _apply_manipulator_aware_filter(**manipulator_filter_kwargs)
            traj_manipulator_cluster_fallback_used = np.asarray(fallback_used, dtype=bool)

            reason_bits[wrist_seed_mask & (~near_depth_mask)] |= MASK_REASON_MANIPULATOR_DEPTH_FAIL
            reason_bits[wrist_seed_mask & (~motion_mask)] |= MASK_REASON_MANIPULATOR_MOTION_FAIL
            reason_bits[traj_manipulator_candidate_mask & (~final_mask)] |= MASK_REASON_MANIPULATOR_CLUSTER_FAIL
    else:
        required_prefix_frames = _resolve_support_frame_requirement(
            num_frames=num_frames,
            min_frames=config["wrist_min_prefix_frames"],
            ratio=config["wrist_prefix_ratio"],
        )
        required_support_frames = _resolve_support_frame_requirement(
            num_frames=num_frames,
            min_frames=config["wrist_min_support_frames"],
            ratio=config["wrist_support_ratio"],
        )
        supervision_support_mask = (
            supervision_prefix_len >= required_prefix_frames
        ) & (
            supervision_count >= required_support_frames
        )
        reason_bits[~wrist_base_mask] |= MASK_REASON_BASE_GEOMETRY_FAIL
        reason_bits[~query_depth_mask] |= MASK_REASON_QUERY_DEPTH_FAIL
        reason_bits[~supervision_support_mask] |= MASK_REASON_TEMPORAL_CONSISTENCY_FAIL
        wrist_seed_mask = wrist_base_mask & query_depth_mask & supervision_support_mask

        if profile == TRAJ_FILTER_PROFILE_WRIST:
            final_mask = wrist_seed_mask
        else:
            raw_depths_segment, intrinsics_segment, extrinsics_segment = _require_segment_geometry(
                raw_depths_segment=raw_depths_segment,
                intrinsics_segment=intrinsics_segment,
                extrinsics_segment=extrinsics_segment,
                expected_num_frames=num_frames,
            )
            (
                final_mask,
                traj_query_depth_rank,
                traj_motion_extent,
                traj_motion_step_median,
                traj_manipulator_candidate_mask,
                traj_manipulator_cluster_id,
                traj_manipulator_component_size,
                near_depth_mask,
                motion_mask,
                fallback_used,
            ) = _apply_manipulator_aware_filter(
                traj=traj,
                keypoints=keypoints,
                seed_mask=wrist_seed_mask,
                supervision_mask=supervision_mask,
                intrinsics_segment=intrinsics_segment,
                extrinsics_segment=extrinsics_segment,
                image_height=image_height,
                image_width=image_width,
                min_depth=config["min_depth"],
                max_depth=config["max_depth"],
                max_depth_rank=config["wrist_manipulator_max_depth_rank"],
                min_motion_extent=config["wrist_manipulator_min_motion_extent"],
                cluster_radius_ratio=config["wrist_manipulator_cluster_radius_ratio"],
                cluster_radius_min_px=config["wrist_manipulator_cluster_radius_min_px"],
                min_component_ratio=config["wrist_manipulator_min_component_ratio"],
                min_component_size=config["wrist_manipulator_min_component_size"],
            )
            traj_manipulator_cluster_fallback_used = np.asarray(fallback_used, dtype=bool)

            reason_bits[wrist_seed_mask & (~near_depth_mask)] |= MASK_REASON_MANIPULATOR_DEPTH_FAIL
            reason_bits[wrist_seed_mask & (~motion_mask)] |= MASK_REASON_MANIPULATOR_MOTION_FAIL
            reason_bits[traj_manipulator_candidate_mask & (~final_mask)] |= MASK_REASON_MANIPULATOR_CLUSTER_FAIL

    return {
        "traj_valid_mask": final_mask.astype(bool),
        "traj_depth_consistency_ratio": depth_consistency_ratio.astype(np.float32),
        "traj_stable_depth_consistency_ratio": stable_depth_consistency_ratio.astype(np.float32),
        "traj_high_volatility_hit": high_volatility_hit.astype(bool),
        "traj_volatility_exposure_ratio": volatility_exposure_ratio.astype(np.float32),
        "traj_compare_frame_count": compare_frame_count.astype(np.uint16),
        "traj_stable_compare_frame_count": stable_compare_frame_count.astype(np.uint16),
        "traj_mask_reason_bits": reason_bits.astype(np.uint8),
        "traj_supervision_mask": supervision_mask.astype(bool),
        "traj_supervision_prefix_len": supervision_prefix_len.astype(np.uint16),
        "traj_supervision_count": supervision_count.astype(np.uint16),
        "traj_wrist_seed_mask": wrist_seed_mask.astype(bool),
        "traj_query_depth_rank": traj_query_depth_rank.astype(np.float32),
        "traj_motion_extent": traj_motion_extent.astype(np.float32),
        "traj_motion_step_median": traj_motion_step_median.astype(np.float32),
        "traj_manipulator_candidate_mask": traj_manipulator_candidate_mask.astype(bool),
        "traj_manipulator_cluster_id": traj_manipulator_cluster_id.astype(np.int16),
        "traj_manipulator_component_size": traj_manipulator_component_size.astype(np.uint16),
        "traj_manipulator_cluster_fallback_used": np.asarray(
            traj_manipulator_cluster_fallback_used, dtype=bool
        ),
    }


def build_traj_valid_mask(
    traj: np.ndarray,
    visibs: np.ndarray | None,
    image_width: int,
    image_height: int,
    filter_args,
    *,
    keypoints: np.ndarray | None = None,
    query_depth: np.ndarray | None = None,
    raw_depths_segment: np.ndarray | None = None,
    intrinsics_segment: np.ndarray | None = None,
    extrinsics_segment: np.ndarray | None = None,
    depth_volatility_map: np.ndarray | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper returning only the final validity mask."""
    return build_traj_filter_result(
        traj=traj,
        visibs=visibs,
        image_width=image_width,
        image_height=image_height,
        filter_args=filter_args,
        keypoints=keypoints,
        query_depth=query_depth,
        raw_depths_segment=raw_depths_segment,
        intrinsics_segment=intrinsics_segment,
        extrinsics_segment=extrinsics_segment,
        depth_volatility_map=depth_volatility_map,
    )["traj_valid_mask"]
