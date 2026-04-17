from __future__ import annotations

import time
import warnings

import numpy as np

from utils.moge_utils3d import depth_edge


QUERY_DEPTH_PATCH_RADIUS = 2
QUERY_DEPTH_MIN_VALID_RATIO = 0.4
QUERY_DEPTH_ABS_TOL = 0.05
QUERY_DEPTH_REL_TOL = 0.10
QUERY_DEPTH_EDGE_RTOL = 0.03
QUERY_DEPTH_EDGE_PATCH_STD_THRESHOLD = 0.003

TEMPORAL_DEPTH_ABS_TOL = 0.05
TEMPORAL_DEPTH_REL_TOL = 0.10
TEMPORAL_MIN_CONSISTENCY_RATIO = 0.95

TRAJ_FILTER_PROFILE_EXTERNAL = "external"
TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR = "external_manipulator"
TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR_V2 = "external_manipulator_v2"
TRAJ_FILTER_PROFILE_EGOCENTRIC_OBJECT_INTERACTION_V1 = "egocentric_object_interaction_v1"
TRAJ_FILTER_PROFILE_WRIST = "wrist"
TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE = "wrist_pick_place"
TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE_NO_HEATMAP = "wrist_pick_place_no_heatmap"
TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR_TOP95 = "wrist_manipulator_top95"
TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR = "wrist_manipulator"

WRIST_MIN_PREFIX_FRAMES = 3
WRIST_MIN_SUPPORT_FRAMES = 3
WRIST_PREFIX_RATIO = 0.15
WRIST_SUPPORT_RATIO = 0.20
WRIST_MANIPULATOR_TOP95_KEEP_RATIO = 0.95

EGOCENTRIC_MIN_PREFIX_FRAMES = 1
EGOCENTRIC_MIN_SUPPORT_FRAMES = 1
EGOCENTRIC_PREFIX_RATIO = 0.05
EGOCENTRIC_SUPPORT_RATIO = 0.05

WRIST_MANIPULATOR_MAX_DEPTH_RANK = 0.50
WRIST_MANIPULATOR_MIN_MOTION_EXTENT = 0.03
WRIST_MANIPULATOR_CLUSTER_RADIUS_RATIO = 0.06
WRIST_MANIPULATOR_CLUSTER_RADIUS_MIN_PX = 24
WRIST_MANIPULATOR_MIN_COMPONENT_RATIO = 0.005
WRIST_MANIPULATOR_MIN_COMPONENT_SIZE = 2
WRIST_PICK_PLACE_MIN_HEATMAP_HITS = 2
WRIST_PICK_PLACE_MAX_MANIPULATOR_DISTANCE_M = 0.20
WRIST_PICK_PLACE_QUERY_DEPTH_MARGIN_M = 0.25
WRIST_PICK_PLACE_MAJOR_COMPONENT_RATIO = 0.15
WRIST_PICK_PLACE_MAJOR_COMPONENT_MIN_MOTION_RATIO = 0.75
WRIST_PICK_PLACE_MAJOR_COMPONENT_DEPTH_MARGIN_M = 0.08
WRIST_PICK_PLACE_NO_HEATMAP_MAX_DEPTH_RANK = 0.50
WRIST_PICK_PLACE_NO_HEATMAP_ANCHOR_MIN_MOTION_EXTENT = 0.03
WRIST_PICK_PLACE_NO_HEATMAP_BBOX_X_PAD_PX = 80
WRIST_PICK_PLACE_NO_HEATMAP_BBOX_Y_PAD_UP_PX = 40
WRIST_PICK_PLACE_NO_HEATMAP_BBOX_Y_PAD_DOWN_PX = 220
WRIST_PICK_PLACE_NO_HEATMAP_MIN_ANCHOR_COUNT = 8

EXTERNAL_MANIPULATOR_V2_MAX_DEPTH_RANK = 0.70
EXTERNAL_MANIPULATOR_V2_MIN_MOTION_EXTENT = 0.01
EXTERNAL_MANIPULATOR_V2_CLUSTER_RADIUS_RATIO = 0.06
EXTERNAL_MANIPULATOR_V2_CLUSTER_RADIUS_MIN_PX = 24
EXTERNAL_MANIPULATOR_V2_MIN_COMPONENT_RATIO = 0.002
EXTERNAL_MANIPULATOR_V2_MIN_COMPONENT_SIZE = 2
EXTERNAL_MANIPULATOR_V2_MAJOR_COMPONENT_RATIO = 0.15

EGOCENTRIC_MANIPULATOR_MAX_DEPTH_RANK = 0.82
EGOCENTRIC_MANIPULATOR_MIN_MOTION_EXTENT = 0.01
EGOCENTRIC_MANIPULATOR_CLUSTER_RADIUS_RATIO = 0.06
EGOCENTRIC_MANIPULATOR_CLUSTER_RADIUS_MIN_PX = 24
EGOCENTRIC_MANIPULATOR_MIN_COMPONENT_RATIO = 0.002
EGOCENTRIC_MANIPULATOR_MIN_COMPONENT_SIZE = 2
EGOCENTRIC_MANIPULATOR_MAJOR_COMPONENT_RATIO = 0.12
EGOCENTRIC_OBJECT_MAX_MANIPULATOR_DISTANCE_M = 0.20
EGOCENTRIC_OBJECT_QUERY_DEPTH_MARGIN_M = 0.25

STEREO_DEPTH_ABS_TOL = 0.05
STEREO_DEPTH_REL_TOL = 0.10
STEREO_MIN_CONSISTENCY_RATIO = 0.60
STEREO_MAX_PATCH_ERROR = 0.20

VOLATILITY_LOW_PERCENTILE = 5.0
VOLATILITY_HIGH_PERCENTILE = 95.0
VOLATILITY_MASK_PERCENTILE = 99.0

QUERY_PREFILTER_MODE_OFF = "off"
QUERY_PREFILTER_MODE_PROFILE_AWARE_STATIC_V1 = "profile_aware_static_v1"
QUERY_PREFILTER_MODE_EXTERNAL_DEPTH_STATIC_V1 = "external_depth_static_v1"
DEFAULT_QUERY_PREFILTER_MODE = QUERY_PREFILTER_MODE_OFF
DEFAULT_QUERY_PREFILTER_WRIST_RANK_KEEP_RATIO = 0.30
EXTERNAL_QUERY_PREFILTER_MIN_QUERY_DEPTH_M = 0.05
EXTERNAL_QUERY_PREFILTER_EDGE_BORDER_PX = 40

TRAJ_FILTER_ABLATION_MODE_NONE = "none"
TRAJ_FILTER_ABLATION_MODE_WRIST_SEED_TOP95 = "wrist_seed_top95"
TRAJ_FILTER_ABLATION_MODE_WRIST_NO_QUERY_EDGE = "wrist_no_query_edge"
TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_DEPTH = "wrist_no_manipulator_depth"
TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_MOTION = "wrist_no_manipulator_motion"
TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_CLUSTER = "wrist_no_manipulator_cluster"

TRAJ_FILTER_ABLATION_MODES = (
    TRAJ_FILTER_ABLATION_MODE_NONE,
    TRAJ_FILTER_ABLATION_MODE_WRIST_SEED_TOP95,
    TRAJ_FILTER_ABLATION_MODE_WRIST_NO_QUERY_EDGE,
    TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_DEPTH,
    TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_MOTION,
    TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_CLUSTER,
)

MASK_REASON_BASE_GEOMETRY_FAIL = np.uint8(1 << 0)
MASK_REASON_QUERY_DEPTH_FAIL = np.uint8(1 << 1)
MASK_REASON_TEMPORAL_CONSISTENCY_FAIL = np.uint8(1 << 2)
MASK_REASON_STABLE_TEMPORAL_FAIL = np.uint8(1 << 3)
MASK_REASON_MANIPULATOR_DEPTH_FAIL = np.uint8(1 << 4)
MASK_REASON_MANIPULATOR_MOTION_FAIL = np.uint8(1 << 5)
MASK_REASON_MANIPULATOR_CLUSTER_FAIL = np.uint8(1 << 6)
MASK_REASON_QUERY_DEPTH_EDGE_FAIL = np.uint8(1 << 7)


def _accumulate_profile_stat(
    profile_stats: dict[str, float] | None,
    key: str,
    seconds: float,
) -> None:
    if profile_stats is None:
        return
    profile_stats[key] = float(profile_stats.get(key, 0.0) + float(seconds))


def _get_profile_stat(
    profile_stats: dict[str, float] | None,
    key: str,
) -> float:
    if profile_stats is None:
        return 0.0
    return float(profile_stats.get(key, 0.0))


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


def is_tail_truncated_sample(
    *,
    num_frames: int,
    future_len: int | None = None,
    filter_args=None,
) -> bool:
    """Return whether a sample ends before the configured future window."""
    if future_len is None and filter_args is not None:
        future_len = getattr(filter_args, "future_len", None)
    if future_len is None:
        return False

    try:
        future_len = int(future_len)
    except (TypeError, ValueError):
        return False

    if future_len <= 0:
        return False
    return int(num_frames) < future_len


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


def _compute_motion_metrics_for_valid_masks(
    world_tracks: np.ndarray,
    valid_masks: tuple[np.ndarray, ...],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    world_tracks = np.asarray(world_tracks, dtype=np.float32)
    if world_tracks.ndim != 3 or world_tracks.shape[-1] != 3:
        raise ValueError(f"Expected world_tracks shape (N,T,3), got {world_tracks.shape}")

    num_tracks, num_frames, _ = world_tracks.shape
    finite_mask = np.isfinite(world_tracks).all(axis=-1)
    normalized_masks: list[np.ndarray] = []
    for valid_mask in valid_masks:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != world_tracks.shape[:2]:
            raise ValueError(
                f"Expected valid_mask shape {world_tracks.shape[:2]}, got {valid_mask.shape}"
            )
        normalized_masks.append(finite_mask & valid_mask)

    motion_extent_list: list[np.ndarray] = []
    motion_step_median_list: list[np.ndarray] = []

    step_norm = None
    if num_frames > 1:
        step_norm = np.linalg.norm(np.diff(world_tracks, axis=1), axis=-1)

    for valid_mask in normalized_masks:
        motion_extent = np.full(num_tracks, np.nan, dtype=np.float32)
        motion_step_median = np.full(num_tracks, np.nan, dtype=np.float32)

        valid_counts = np.count_nonzero(valid_mask, axis=1)
        extent_track_indices = np.flatnonzero(valid_counts >= 2)
        if extent_track_indices.size > 0:
            valid_subset = valid_mask[extent_track_indices]
            first_valid_idx = np.argmax(valid_subset, axis=1)
            start_points = world_tracks[extent_track_indices, first_valid_idx]
            distances = np.linalg.norm(
                world_tracks[extent_track_indices] - start_points[:, None, :],
                axis=-1,
            )
            masked_distances = np.where(valid_subset, distances, np.nan)
            motion_extent[extent_track_indices] = np.nanmax(masked_distances, axis=1).astype(np.float32)

        if step_norm is not None:
            pair_valid = valid_mask[:, :-1] & valid_mask[:, 1:]
            step_track_indices = np.flatnonzero(np.any(pair_valid, axis=1))
            if step_track_indices.size > 0:
                masked_steps = np.where(
                    pair_valid[step_track_indices],
                    step_norm[step_track_indices],
                    np.nan,
                )
                motion_step_median[step_track_indices] = np.nanmedian(masked_steps, axis=1).astype(
                    np.float32
                )

        motion_extent_list.append(motion_extent)
        motion_step_median_list.append(motion_step_median)

    return tuple(zip(motion_extent_list, motion_step_median_list))


def _apply_top_motion_extent_filter(
    *,
    seed_mask: np.ndarray,
    motion_extent: np.ndarray,
    keep_ratio: float,
) -> np.ndarray:
    seed_mask = np.asarray(seed_mask, dtype=bool)
    motion_extent = np.asarray(motion_extent, dtype=np.float32)

    final_mask = np.zeros(seed_mask.shape, dtype=bool)
    candidate_indices = np.flatnonzero(seed_mask & np.isfinite(motion_extent))
    if candidate_indices.size == 0:
        return final_mask

    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    order = candidate_indices[np.argsort(-motion_extent[candidate_indices], kind="stable")]
    keep_count = min(
        order.size,
        max(1, int(np.floor(keep_ratio * float(order.size)))),
    )
    final_mask[order[:keep_count]] = True
    return final_mask


def _build_anchor_query_region_mask(
    *,
    keypoints: np.ndarray,
    anchor_mask: np.ndarray,
    min_anchor_count: int,
    bbox_x_pad_px: int,
    bbox_y_pad_up_px: int,
    bbox_y_pad_down_px: int,
) -> tuple[np.ndarray, bool]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")
    if anchor_mask.shape != (keypoints.shape[0],):
        raise ValueError(f"Expected anchor_mask shape {(keypoints.shape[0],)}, got {anchor_mask.shape}")

    region_mask = np.ones(anchor_mask.shape, dtype=bool)
    anchor_points = keypoints[anchor_mask]
    finite_anchor_points = anchor_points[np.isfinite(anchor_points).all(axis=1)]
    if finite_anchor_points.shape[0] < int(min_anchor_count):
        return region_mask, True

    x_min = float(np.min(finite_anchor_points[:, 0]) - float(bbox_x_pad_px))
    x_max = float(np.max(finite_anchor_points[:, 0]) + float(bbox_x_pad_px))
    y_min = float(np.min(finite_anchor_points[:, 1]) - float(bbox_y_pad_up_px))
    y_max = float(np.max(finite_anchor_points[:, 1]) + float(bbox_y_pad_down_px))
    region_mask = (
        np.isfinite(keypoints).all(axis=1)
        & (keypoints[:, 0] >= x_min)
        & (keypoints[:, 0] <= x_max)
        & (keypoints[:, 1] >= y_min)
        & (keypoints[:, 1] <= y_max)
    )
    return region_mask.astype(bool, copy=False), False


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
    major_component_min_motion_ratio: float | None = None,
    major_component_depth_margin_m: float | None = None,
    motion_metric_mode: str = "supervised",
    apply_near_depth_filter: bool = True,
    apply_motion_filter: bool = True,
    apply_cluster_filter: bool = True,
    reusable_geometry: dict[str, np.ndarray] | None = None,
    profile_stats: dict[str, float] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
]:
    traj = np.asarray(traj, dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    seed_mask = np.asarray(seed_mask, dtype=bool)
    supervision_mask = np.asarray(supervision_mask, dtype=bool)
    intrinsics_segment = np.asarray(intrinsics_segment, dtype=np.float32)
    extrinsics_segment = np.asarray(extrinsics_segment, dtype=np.float32)

    near_depth_start = time.perf_counter()
    query_depth_values = traj[:, 0, 2].astype(np.float32, copy=False)
    traj_query_depth_rank = _compute_query_depth_ranks(query_depth_values, seed_mask)
    near_depth_mask_raw = (
        seed_mask
        & np.isfinite(traj_query_depth_rank)
        & (traj_query_depth_rank <= float(max_depth_rank))
    )
    near_depth_mask = near_depth_mask_raw if apply_near_depth_filter else seed_mask.copy()
    _accumulate_profile_stat(
        profile_stats,
        "filter_result_manipulator_near_depth_seconds",
        time.perf_counter() - near_depth_start,
    )

    world_lift_start = time.perf_counter()
    world_tracks = traj_uvz_to_world_coordinates(
        traj,
        query_intrinsics=intrinsics_segment[0],
        query_w2c=extrinsics_segment[0],
        min_depth=min_depth,
        max_depth=max_depth,
    )
    _accumulate_profile_stat(
        profile_stats,
        "filter_result_manipulator_world_lift_seconds",
        time.perf_counter() - world_lift_start,
    )
    if reusable_geometry is not None:
        reusable_geometry["world_tracks"] = world_tracks
    motion_start = time.perf_counter()
    (
        (traj_motion_extent, traj_motion_step_median),
        (traj_motion_extent_all_valid, traj_motion_step_median_all_valid),
    ) = _compute_motion_metrics_for_valid_masks(
        world_tracks,
        (
            supervision_mask,
            np.ones_like(supervision_mask, dtype=bool),
        ),
    )
    if motion_metric_mode == "supervised":
        motion_extent_for_gate = traj_motion_extent
    elif motion_metric_mode == "all_valid":
        motion_extent_for_gate = traj_motion_extent_all_valid
    else:
        raise ValueError(f"Unsupported motion_metric_mode: {motion_metric_mode}")
    motion_mask_raw = (
        seed_mask
        & np.isfinite(motion_extent_for_gate)
        & (motion_extent_for_gate >= float(min_motion_extent))
    )
    motion_mask = motion_mask_raw if apply_motion_filter else seed_mask.copy()
    _accumulate_profile_stat(
        profile_stats,
        "filter_result_manipulator_motion_seconds",
        time.perf_counter() - motion_start,
    )
    traj_manipulator_candidate_mask = seed_mask & near_depth_mask & motion_mask
    cluster_start = time.perf_counter()
    if not apply_cluster_filter:
        final_mask = traj_manipulator_candidate_mask.copy()
        traj_manipulator_cluster_id = np.full(seed_mask.shape, -1, dtype=np.int16)
        traj_manipulator_component_size = np.zeros(seed_mask.shape, dtype=np.uint16)
        if np.any(traj_manipulator_candidate_mask):
            candidate_count = int(np.count_nonzero(traj_manipulator_candidate_mask))
            traj_manipulator_cluster_id[traj_manipulator_candidate_mask] = np.int16(0)
            traj_manipulator_component_size[traj_manipulator_candidate_mask] = np.uint16(candidate_count)
        cluster_mask = final_mask.copy()
        fallback_used = False
    elif component_keep_mode == "largest":
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
        if (
            not fallback_used
            and np.any(final_mask)
            and major_component_min_motion_ratio is not None
            and major_component_depth_margin_m is not None
        ):
            refined_mask = np.zeros_like(final_mask)
            kept_component_ids = np.unique(traj_manipulator_cluster_id[final_mask])
            kept_component_ids = kept_component_ids[kept_component_ids >= 0]
            if kept_component_ids.size <= 1:
                refined_mask = final_mask.copy()
            else:
                component_motion = np.full(kept_component_ids.shape, np.nan, dtype=np.float32)
                component_depth = np.full(kept_component_ids.shape, np.nan, dtype=np.float32)
                for component_offset, component_id in enumerate(kept_component_ids.tolist()):
                    component_mask = final_mask & (traj_manipulator_cluster_id == component_id)
                    component_motion_values = motion_extent_for_gate[component_mask]
                    component_motion_values = component_motion_values[np.isfinite(component_motion_values)]
                    if component_motion_values.size > 0:
                        component_motion[component_offset] = float(np.median(component_motion_values))

                    component_depth_values = query_depth_values[component_mask]
                    component_depth_values = component_depth_values[np.isfinite(component_depth_values)]
                    if component_depth_values.size > 0:
                        component_depth[component_offset] = float(np.median(component_depth_values))

                keep_component = np.zeros(kept_component_ids.shape, dtype=bool)
                finite_motion = np.isfinite(component_motion)
                if np.any(finite_motion):
                    best_component_motion = float(np.max(component_motion[finite_motion]))
                    keep_component |= component_motion >= (
                        best_component_motion * float(major_component_min_motion_ratio)
                    )
                finite_depth = np.isfinite(component_depth)
                if np.any(finite_depth):
                    best_component_depth = float(np.min(component_depth[finite_depth]))
                    keep_component |= component_depth <= (
                        best_component_depth + float(major_component_depth_margin_m)
                    )
                if not np.any(keep_component):
                    refined_mask = final_mask.copy()
                else:
                    for component_id in kept_component_ids[keep_component].tolist():
                        refined_mask |= final_mask & (traj_manipulator_cluster_id == component_id)
            final_mask = refined_mask
    else:
        raise ValueError(f"Unsupported component_keep_mode: {component_keep_mode}")
    if apply_cluster_filter:
        cluster_mask = final_mask.copy()
    _accumulate_profile_stat(
        profile_stats,
        "filter_result_manipulator_cluster_seconds",
        time.perf_counter() - cluster_start,
    )
    return (
        final_mask,
        traj_query_depth_rank,
        traj_motion_extent,
        traj_motion_step_median,
        traj_motion_extent_all_valid,
        traj_motion_step_median_all_valid,
        traj_manipulator_candidate_mask,
        traj_manipulator_cluster_id,
        traj_manipulator_component_size,
        near_depth_mask,
        motion_mask,
        cluster_mask,
        bool(fallback_used),
    )


def _compute_binary_mask_track_hits(
    traj: np.ndarray,
    binary_mask_segment: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 3 or traj.shape[-1] != 3:
        raise ValueError(f"Expected traj shape (N,T,3), got {traj.shape}")

    num_tracks, num_frames, _ = traj.shape
    hit_mask = np.zeros((num_tracks, num_frames), dtype=bool)
    hit_count = np.zeros(num_tracks, dtype=np.uint16)
    if binary_mask_segment is None:
        return hit_mask, hit_count

    binary_mask_segment = np.asarray(binary_mask_segment, dtype=bool)
    if binary_mask_segment.shape[0] != num_frames or binary_mask_segment.ndim != 3:
        raise ValueError(
            f"Expected binary_mask_segment shape {(num_frames, 'H', 'W')}, got {binary_mask_segment.shape}"
        )

    height, width = binary_mask_segment.shape[1:]
    xs = np.rint(np.nan_to_num(traj[..., 0], nan=-1.0, posinf=-1.0, neginf=-1.0)).astype(np.int32)
    ys = np.rint(np.nan_to_num(traj[..., 1], nan=-1.0, posinf=-1.0, neginf=-1.0)).astype(np.int32)
    in_bounds = (
        np.isfinite(traj).all(axis=-1)
        & (xs >= 0)
        & (xs < width)
        & (ys >= 0)
        & (ys < height)
    )
    for frame_idx in range(num_frames):
        frame_mask = in_bounds[:, frame_idx]
        if not np.any(frame_mask):
            continue
        hit_mask[frame_mask, frame_idx] = binary_mask_segment[
            frame_idx,
            ys[frame_mask, frame_idx],
            xs[frame_mask, frame_idx],
        ]
    hit_count = hit_mask.sum(axis=1).astype(np.uint16)
    return hit_mask, hit_count


def _compute_pick_place_reference_geometry(
    *,
    traj: np.ndarray,
    manipulator_reference_mask: np.ndarray,
    manipulator_reference_component_ids: np.ndarray,
    min_depth: float,
    max_depth: float,
    query_depth_margin_m: float,
    world_tracks: np.ndarray | None = None,
    intrinsics_segment: np.ndarray | None = None,
    extrinsics_segment: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    traj = np.asarray(traj, dtype=np.float32)
    manipulator_reference_mask = np.asarray(manipulator_reference_mask, dtype=bool)
    manipulator_reference_component_ids = np.asarray(manipulator_reference_component_ids, dtype=np.int32)

    num_tracks, num_frames, _ = traj.shape
    if manipulator_reference_mask.shape != (num_tracks,):
        raise ValueError(
            f"Expected manipulator_reference_mask shape {(num_tracks,)}, got {manipulator_reference_mask.shape}"
        )
    if manipulator_reference_component_ids.shape != (num_tracks,):
        raise ValueError(
            "Expected manipulator_reference_component_ids to match manipulator_reference_mask shape, got "
            f"{manipulator_reference_component_ids.shape} and {manipulator_reference_mask.shape}"
        )

    if world_tracks is None:
        if intrinsics_segment is None or extrinsics_segment is None:
            raise ValueError(
                "intrinsics_segment and extrinsics_segment are required when world_tracks is not provided"
            )
        intrinsics_segment = np.asarray(intrinsics_segment, dtype=np.float32)
        extrinsics_segment = np.asarray(extrinsics_segment, dtype=np.float32)
        world_tracks = traj_uvz_to_world_coordinates(
            traj,
            query_intrinsics=intrinsics_segment[0],
            query_w2c=extrinsics_segment[0],
            min_depth=min_depth,
            max_depth=max_depth,
        )
    else:
        world_tracks = np.asarray(world_tracks, dtype=np.float32)
        if world_tracks.shape != (num_tracks, num_frames, 3):
            raise ValueError(
                f"Expected world_tracks shape {(num_tracks, num_frames, 3)}, got {world_tracks.shape}"
            )

    if not np.any(manipulator_reference_mask):
        return (
            world_tracks,
            np.full((0, num_frames, 3), np.nan, dtype=np.float32),
            np.full(0, np.nan, dtype=np.float32),
        )

    reference_component_ids = np.unique(manipulator_reference_component_ids[manipulator_reference_mask])
    reference_component_ids = reference_component_ids[reference_component_ids >= 0]
    if reference_component_ids.size == 0:
        component_masks = [manipulator_reference_mask]
    else:
        component_masks = [
            manipulator_reference_mask & (manipulator_reference_component_ids == component_id)
            for component_id in reference_component_ids.tolist()
        ]

    num_reference_components = len(component_masks)
    manipulator_component_centroids = np.full((num_reference_components, num_frames, 3), np.nan, dtype=np.float32)
    component_depth_upper = np.full(num_reference_components, np.nan, dtype=np.float32)
    query_depth_values = traj[:, 0, 2].astype(np.float32, copy=False)
    for component_offset, component_mask in enumerate(component_masks):
        component_query_depth = query_depth_values[component_mask]
        component_query_depth = component_query_depth[np.isfinite(component_query_depth)]
        if component_query_depth.size > 0:
            component_depth_upper[component_offset] = float(
                np.percentile(component_query_depth, 90)
            ) + float(query_depth_margin_m)
        for frame_idx in range(num_frames):
            frame_points = world_tracks[component_mask, frame_idx]
            frame_valid = np.isfinite(frame_points).all(axis=1)
            if np.any(frame_valid):
                manipulator_component_centroids[component_offset, frame_idx] = np.median(
                    frame_points[frame_valid], axis=0
                )

    return world_tracks, manipulator_component_centroids, component_depth_upper


def _compute_pick_place_contact_masks(
    *,
    traj: np.ndarray,
    candidate_mask: np.ndarray,
    manipulator_reference_mask: np.ndarray,
    manipulator_reference_component_ids: np.ndarray,
    intrinsics_segment: np.ndarray,
    extrinsics_segment: np.ndarray,
    min_depth: float,
    max_depth: float,
    max_manipulator_distance_m: float,
    query_depth_margin_m: float,
    world_tracks: np.ndarray | None = None,
    manipulator_component_centroids: np.ndarray | None = None,
    component_depth_upper: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traj = np.asarray(traj, dtype=np.float32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    manipulator_reference_mask = np.asarray(manipulator_reference_mask, dtype=bool)
    manipulator_reference_component_ids = np.asarray(manipulator_reference_component_ids, dtype=np.int32)

    num_tracks, num_frames, _ = traj.shape
    if candidate_mask.shape != (num_tracks,):
        raise ValueError(f"Expected candidate_mask shape {(num_tracks,)}, got {candidate_mask.shape}")
    if manipulator_reference_mask.shape != (num_tracks,):
        raise ValueError(
            f"Expected manipulator_reference_mask shape {(num_tracks,)}, got {manipulator_reference_mask.shape}"
        )
    if manipulator_reference_component_ids.shape != (num_tracks,):
        raise ValueError(
            "Expected manipulator_reference_component_ids to match candidate_mask shape, got "
            f"{manipulator_reference_component_ids.shape} and {candidate_mask.shape}"
        )

    min_manipulator_distance = np.full(num_tracks, np.nan, dtype=np.float32)
    contact_mask = np.zeros(num_tracks, dtype=bool)
    query_contact_mask = np.zeros(num_tracks, dtype=bool)
    delayed_contact_mask = np.zeros(num_tracks, dtype=bool)
    depth_guard_mask = np.zeros(num_tracks, dtype=bool)
    if not np.any(candidate_mask) or not np.any(manipulator_reference_mask):
        return (
            min_manipulator_distance,
            contact_mask,
            query_contact_mask,
            delayed_contact_mask,
            depth_guard_mask,
        )

    if manipulator_component_centroids is None or component_depth_upper is None:
        (
            world_tracks,
            manipulator_component_centroids,
            component_depth_upper,
        ) = _compute_pick_place_reference_geometry(
            traj=traj,
            manipulator_reference_mask=manipulator_reference_mask,
            manipulator_reference_component_ids=manipulator_reference_component_ids,
            min_depth=min_depth,
            max_depth=max_depth,
            query_depth_margin_m=query_depth_margin_m,
            world_tracks=world_tracks,
            intrinsics_segment=intrinsics_segment,
            extrinsics_segment=extrinsics_segment,
        )
    else:
        world_tracks = np.asarray(world_tracks, dtype=np.float32)
        manipulator_component_centroids = np.asarray(manipulator_component_centroids, dtype=np.float32)
        component_depth_upper = np.asarray(component_depth_upper, dtype=np.float32)

    candidate_indices = np.flatnonzero(candidate_mask)
    num_reference_components = int(manipulator_component_centroids.shape[0])
    component_min_distance_candidate = np.full(
        (candidate_indices.size, num_reference_components), np.nan, dtype=np.float32
    )
    per_frame_min_distance_candidate = np.full((candidate_indices.size, num_frames), np.nan, dtype=np.float32)
    if candidate_indices.size > 0 and num_reference_components > 0:
        candidate_world_tracks = world_tracks[candidate_indices]
        if num_reference_components == 1:
            distance_matrix = np.linalg.norm(
                candidate_world_tracks - manipulator_component_centroids[0][None, :, :],
                axis=-1,
            )
            finite_distance = np.isfinite(distance_matrix)
            if np.any(finite_distance):
                safe_distance = np.where(finite_distance, distance_matrix, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    per_frame_min_distance_candidate = safe_distance.astype(np.float32, copy=False)
                    component_min_distance_candidate[:, 0] = np.nanmin(
                        safe_distance,
                        axis=1,
                    ).astype(np.float32, copy=False)
        else:
            distance_matrix = np.linalg.norm(
                candidate_world_tracks[:, None, :, :] - manipulator_component_centroids[None, :, :, :],
                axis=-1,
            )
            finite_distance = np.isfinite(distance_matrix)
            if np.any(finite_distance):
                safe_distance = np.where(finite_distance, distance_matrix, np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    component_min_distance_candidate = np.nanmin(
                        safe_distance,
                        axis=2,
                    ).astype(np.float32, copy=False)
                    per_frame_min_distance_candidate = np.nanmin(
                        safe_distance,
                        axis=1,
                    ).astype(np.float32, copy=False)

    if candidate_indices.size > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            min_manipulator_distance[candidate_indices] = np.nanmin(
                component_min_distance_candidate,
                axis=1,
            ).astype(np.float32, copy=False)

    contact_mask = (
        candidate_mask
        & np.isfinite(min_manipulator_distance)
        & (min_manipulator_distance <= float(max_manipulator_distance_m))
    )
    if candidate_indices.size > 0:
        candidate_query_frame_distance = per_frame_min_distance_candidate[:, 0]
        query_contact_mask[candidate_indices] = (
            np.isfinite(candidate_query_frame_distance)
            & (candidate_query_frame_distance <= float(max_manipulator_distance_m))
        )
        if num_frames > 1:
            delayed_contact_mask[candidate_indices] = np.any(
                np.isfinite(per_frame_min_distance_candidate[:, 1:])
                & (per_frame_min_distance_candidate[:, 1:] <= float(max_manipulator_distance_m)),
                axis=1,
            )
            delayed_contact_mask[candidate_indices] &= ~query_contact_mask[candidate_indices]

    query_depth_values = traj[:, 0, 2].astype(np.float32, copy=False)
    if candidate_indices.size > 0 and num_reference_components > 0:
        nearest_component_offset = np.full(candidate_indices.size, -1, dtype=np.int32)
        valid_component_distance = np.isfinite(component_min_distance_candidate)
        if np.any(valid_component_distance):
            safe_component_distance = np.where(valid_component_distance, component_min_distance_candidate, np.inf)
            track_has_component = np.any(valid_component_distance, axis=1)
            nearest_component_offset[track_has_component] = np.argmin(
                safe_component_distance[track_has_component],
                axis=1,
            ).astype(np.int32, copy=False)

        per_track_depth_upper = np.full(candidate_indices.size, np.nan, dtype=np.float32)
        valid_nearest_component = nearest_component_offset >= 0
        if np.any(valid_nearest_component):
            per_track_depth_upper[valid_nearest_component] = component_depth_upper[
                nearest_component_offset[valid_nearest_component]
            ]
        depth_guard_mask[candidate_indices] = (
            np.isfinite(query_depth_values[candidate_indices])
            & np.isfinite(per_track_depth_upper)
            & (query_depth_values[candidate_indices] <= per_track_depth_upper)
        )

    return (
        min_manipulator_distance,
        contact_mask,
        query_contact_mask,
        delayed_contact_mask,
        depth_guard_mask,
    )


def _apply_pick_place_object_filter(
    *,
    traj: np.ndarray,
    seed_mask: np.ndarray,
    manipulator_reference_mask: np.ndarray,
    manipulator_reference_component_ids: np.ndarray,
    intrinsics_segment: np.ndarray,
    extrinsics_segment: np.ndarray,
    min_depth: float,
    max_depth: float,
    pick_place_heatmap_segment: np.ndarray | None,
    min_heatmap_hits: int,
    max_manipulator_distance_m: float,
    query_depth_margin_m: float,
    world_tracks: np.ndarray | None = None,
    manipulator_component_centroids: np.ndarray | None = None,
    component_depth_upper: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traj = np.asarray(traj, dtype=np.float32)
    seed_mask = np.asarray(seed_mask, dtype=bool)
    manipulator_reference_mask = np.asarray(manipulator_reference_mask, dtype=bool)
    manipulator_reference_component_ids = np.asarray(manipulator_reference_component_ids, dtype=np.int32)
    if manipulator_reference_component_ids.shape != seed_mask.shape:
        raise ValueError(
            "Expected manipulator_reference_component_ids to match seed_mask shape, got "
            f"{manipulator_reference_component_ids.shape} and {seed_mask.shape}"
        )

    num_tracks, num_frames, _ = traj.shape
    heatmap_support_mask = np.zeros(num_tracks, dtype=bool)
    heatmap_hit_count = np.zeros(num_tracks, dtype=np.uint16)
    min_manipulator_distance = np.full(num_tracks, np.nan, dtype=np.float32)
    contact_mask = np.zeros(num_tracks, dtype=bool)
    depth_guard_mask = np.zeros(num_tracks, dtype=bool)
    object_mask = np.zeros(num_tracks, dtype=bool)

    _, heatmap_hit_count = _compute_binary_mask_track_hits(traj, pick_place_heatmap_segment)
    if pick_place_heatmap_segment is not None:
        heatmap_support_mask = seed_mask & (heatmap_hit_count >= int(min_heatmap_hits))
    if not np.any(heatmap_support_mask):
        return (
            object_mask,
            heatmap_hit_count,
            heatmap_support_mask,
            min_manipulator_distance,
            contact_mask,
            depth_guard_mask,
        )

    (
        min_manipulator_distance,
        contact_mask,
        _query_contact_mask,
        _delayed_contact_mask,
        depth_guard_mask,
    ) = _compute_pick_place_contact_masks(
        traj=traj,
        candidate_mask=heatmap_support_mask,
        manipulator_reference_mask=manipulator_reference_mask,
        manipulator_reference_component_ids=manipulator_reference_component_ids,
        intrinsics_segment=intrinsics_segment,
        extrinsics_segment=extrinsics_segment,
        min_depth=min_depth,
        max_depth=max_depth,
        max_manipulator_distance_m=max_manipulator_distance_m,
        query_depth_margin_m=query_depth_margin_m,
        world_tracks=world_tracks,
        manipulator_component_centroids=manipulator_component_centroids,
        component_depth_upper=component_depth_upper,
    )
    object_mask = heatmap_support_mask & contact_mask & depth_guard_mask
    return (
        object_mask,
        heatmap_hit_count,
        heatmap_support_mask,
        min_manipulator_distance,
        contact_mask,
        depth_guard_mask,
    )


def _apply_delayed_contact_object_rescue_filter(
    *,
    traj: np.ndarray,
    visibs: np.ndarray | None,
    seed_mask: np.ndarray,
    local_keep_mask: np.ndarray,
    manipulator_reference_mask: np.ndarray,
    manipulator_reference_component_ids: np.ndarray,
    intrinsics_segment: np.ndarray,
    extrinsics_segment: np.ndarray,
    min_depth: float,
    max_depth: float,
    max_manipulator_distance_m: float,
    query_depth_margin_m: float,
    world_tracks: np.ndarray | None = None,
    manipulator_component_centroids: np.ndarray | None = None,
    component_depth_upper: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traj = np.asarray(traj, dtype=np.float32)
    seed_mask = np.asarray(seed_mask, dtype=bool)
    local_keep_mask = np.asarray(local_keep_mask, dtype=bool)

    num_tracks, num_frames, _ = traj.shape
    if seed_mask.shape != (num_tracks,):
        raise ValueError(f"Expected seed_mask shape {(num_tracks,)}, got {seed_mask.shape}")
    if local_keep_mask.shape != (num_tracks,):
        raise ValueError(f"Expected local_keep_mask shape {(num_tracks,)}, got {local_keep_mask.shape}")

    visibility = _normalize_visibility(visibs, num_tracks=num_tracks, num_frames=num_frames)
    query_visible_mask = np.isfinite(traj[:, 0]).all(axis=1)
    if visibility is not None:
        query_visible_mask &= visibility[:, 0]
    rescue_candidate_mask = seed_mask & query_visible_mask & (~local_keep_mask)

    rescue_mask = np.zeros(num_tracks, dtype=bool)
    min_manipulator_distance = np.full(num_tracks, np.nan, dtype=np.float32)
    contact_mask = np.zeros(num_tracks, dtype=bool)
    depth_guard_mask = np.zeros(num_tracks, dtype=bool)
    delayed_contact_mask = np.zeros(num_tracks, dtype=bool)
    if not np.any(rescue_candidate_mask):
        return (
            rescue_mask,
            min_manipulator_distance,
            contact_mask,
            depth_guard_mask,
            delayed_contact_mask,
        )

    (
        min_manipulator_distance,
        contact_mask,
        _query_contact_mask,
        delayed_contact_mask,
        depth_guard_mask,
    ) = _compute_pick_place_contact_masks(
        traj=traj,
        candidate_mask=rescue_candidate_mask,
        manipulator_reference_mask=manipulator_reference_mask,
        manipulator_reference_component_ids=manipulator_reference_component_ids,
        intrinsics_segment=intrinsics_segment,
        extrinsics_segment=extrinsics_segment,
        min_depth=min_depth,
        max_depth=max_depth,
        max_manipulator_distance_m=max_manipulator_distance_m,
        query_depth_margin_m=query_depth_margin_m,
        world_tracks=world_tracks,
        manipulator_component_centroids=manipulator_component_centroids,
        component_depth_upper=component_depth_upper,
    )
    rescue_mask = rescue_candidate_mask & delayed_contact_mask & depth_guard_mask
    return (
        rescue_mask,
        min_manipulator_distance,
        contact_mask,
        depth_guard_mask,
        delayed_contact_mask,
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
    eligible_tracks = valid_count_mask
    if np.any(eligible_tracks):
        valid_depth_range = (depth_values >= min_depth) & (depth_values <= max_depth)
        depth_range_mask[eligible_tracks] = np.all(
            (~valid_frames[eligible_tracks]) | valid_depth_range[eligible_tracks],
            axis=1,
        )

        in_bounds = (
            (u_values >= -boundary_margin)
            & (u_values <= image_width + boundary_margin)
            & (v_values >= -boundary_margin)
            & (v_values <= image_height + boundary_margin)
        )
        boundary_mask[eligible_tracks] = np.all(
            (~valid_frames[eligible_tracks]) | in_bounds[eligible_tracks],
            axis=1,
        )

        if visibility is not None:
            vis_count = (visibility & valid_frames).sum(axis=1).astype(np.float32, copy=False)
            vis_ratio = np.zeros(num_tracks, dtype=np.float32)
            positive_counts = valid_counts > 0
            vis_ratio[positive_counts] = vis_count[positive_counts] / valid_counts[positive_counts].astype(
                np.float32,
                copy=False,
            )
            visibility_mask[eligible_tracks] = vis_ratio[eligible_tracks] >= float(visibility_threshold)

        smooth_tracks = eligible_tracks & (valid_counts > 1)
        if check_depth_smoothness and np.any(smooth_tracks):
            compact_col_idx = np.cumsum(valid_frames, axis=1, dtype=np.int32) - 1
            compact_depths = np.full((num_tracks, num_frames), np.nan, dtype=np.float32)
            valid_track_idx, valid_frame_idx = np.nonzero(valid_frames)
            compact_depths[valid_track_idx, compact_col_idx[valid_track_idx, valid_frame_idx]] = depth_values[
                valid_track_idx,
                valid_frame_idx,
            ]
            depth_diffs = np.diff(compact_depths, axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                depth_diff_std = np.nanstd(depth_diffs, axis=1)
            depth_smooth_mask[smooth_tracks] = depth_diff_std[smooth_tracks] <= float(depth_change_threshold)

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
            "egocentric_min_prefix_frames": EGOCENTRIC_MIN_PREFIX_FRAMES,
            "egocentric_min_support_frames": EGOCENTRIC_MIN_SUPPORT_FRAMES,
            "egocentric_prefix_ratio": EGOCENTRIC_PREFIX_RATIO,
            "egocentric_support_ratio": EGOCENTRIC_SUPPORT_RATIO,
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
            "egocentric_min_prefix_frames": EGOCENTRIC_MIN_PREFIX_FRAMES,
            "egocentric_min_support_frames": EGOCENTRIC_MIN_SUPPORT_FRAMES,
            "egocentric_prefix_ratio": EGOCENTRIC_PREFIX_RATIO,
            "egocentric_support_ratio": EGOCENTRIC_SUPPORT_RATIO,
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
            "egocentric_min_prefix_frames": EGOCENTRIC_MIN_PREFIX_FRAMES,
            "egocentric_min_support_frames": EGOCENTRIC_MIN_SUPPORT_FRAMES,
            "egocentric_prefix_ratio": EGOCENTRIC_PREFIX_RATIO,
            "egocentric_support_ratio": EGOCENTRIC_SUPPORT_RATIO,
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
            "egocentric_min_prefix_frames": EGOCENTRIC_MIN_PREFIX_FRAMES,
            "egocentric_min_support_frames": EGOCENTRIC_MIN_SUPPORT_FRAMES,
            "egocentric_prefix_ratio": EGOCENTRIC_PREFIX_RATIO,
            "egocentric_support_ratio": EGOCENTRIC_SUPPORT_RATIO,
        },
    }
    config = defaults[level].copy()
    config.update(
        {
            "wrist_manipulator_top95_keep_ratio": WRIST_MANIPULATOR_TOP95_KEEP_RATIO,
            "wrist_manipulator_max_depth_rank": WRIST_MANIPULATOR_MAX_DEPTH_RANK,
            "wrist_manipulator_min_motion_extent": WRIST_MANIPULATOR_MIN_MOTION_EXTENT,
            "wrist_manipulator_cluster_radius_ratio": WRIST_MANIPULATOR_CLUSTER_RADIUS_RATIO,
            "wrist_manipulator_cluster_radius_min_px": WRIST_MANIPULATOR_CLUSTER_RADIUS_MIN_PX,
            "wrist_manipulator_min_component_ratio": WRIST_MANIPULATOR_MIN_COMPONENT_RATIO,
            "wrist_manipulator_min_component_size": WRIST_MANIPULATOR_MIN_COMPONENT_SIZE,
            "wrist_pick_place_min_heatmap_hits": WRIST_PICK_PLACE_MIN_HEATMAP_HITS,
            "wrist_pick_place_max_manipulator_distance_m": WRIST_PICK_PLACE_MAX_MANIPULATOR_DISTANCE_M,
            "wrist_pick_place_query_depth_margin_m": WRIST_PICK_PLACE_QUERY_DEPTH_MARGIN_M,
            "wrist_pick_place_major_component_ratio": WRIST_PICK_PLACE_MAJOR_COMPONENT_RATIO,
            "wrist_pick_place_major_component_min_motion_ratio": (
                WRIST_PICK_PLACE_MAJOR_COMPONENT_MIN_MOTION_RATIO
            ),
            "wrist_pick_place_major_component_depth_margin_m": WRIST_PICK_PLACE_MAJOR_COMPONENT_DEPTH_MARGIN_M,
            "wrist_pick_place_no_heatmap_max_depth_rank": WRIST_PICK_PLACE_NO_HEATMAP_MAX_DEPTH_RANK,
            "wrist_pick_place_no_heatmap_anchor_min_motion_extent": (
                WRIST_PICK_PLACE_NO_HEATMAP_ANCHOR_MIN_MOTION_EXTENT
            ),
            "wrist_pick_place_no_heatmap_bbox_x_pad_px": WRIST_PICK_PLACE_NO_HEATMAP_BBOX_X_PAD_PX,
            "wrist_pick_place_no_heatmap_bbox_y_pad_up_px": WRIST_PICK_PLACE_NO_HEATMAP_BBOX_Y_PAD_UP_PX,
            "wrist_pick_place_no_heatmap_bbox_y_pad_down_px": WRIST_PICK_PLACE_NO_HEATMAP_BBOX_Y_PAD_DOWN_PX,
            "wrist_pick_place_no_heatmap_min_anchor_count": WRIST_PICK_PLACE_NO_HEATMAP_MIN_ANCHOR_COUNT,
            "external_manipulator_v2_max_depth_rank": EXTERNAL_MANIPULATOR_V2_MAX_DEPTH_RANK,
            "external_manipulator_v2_min_motion_extent": EXTERNAL_MANIPULATOR_V2_MIN_MOTION_EXTENT,
            "external_manipulator_v2_cluster_radius_ratio": EXTERNAL_MANIPULATOR_V2_CLUSTER_RADIUS_RATIO,
            "external_manipulator_v2_cluster_radius_min_px": EXTERNAL_MANIPULATOR_V2_CLUSTER_RADIUS_MIN_PX,
            "external_manipulator_v2_min_component_ratio": EXTERNAL_MANIPULATOR_V2_MIN_COMPONENT_RATIO,
            "external_manipulator_v2_min_component_size": EXTERNAL_MANIPULATOR_V2_MIN_COMPONENT_SIZE,
            "external_manipulator_v2_major_component_ratio": EXTERNAL_MANIPULATOR_V2_MAJOR_COMPONENT_RATIO,
            "egocentric_manipulator_max_depth_rank": EGOCENTRIC_MANIPULATOR_MAX_DEPTH_RANK,
            "egocentric_manipulator_min_motion_extent": EGOCENTRIC_MANIPULATOR_MIN_MOTION_EXTENT,
            "egocentric_manipulator_cluster_radius_ratio": EGOCENTRIC_MANIPULATOR_CLUSTER_RADIUS_RATIO,
            "egocentric_manipulator_cluster_radius_min_px": EGOCENTRIC_MANIPULATOR_CLUSTER_RADIUS_MIN_PX,
            "egocentric_manipulator_min_component_ratio": EGOCENTRIC_MANIPULATOR_MIN_COMPONENT_RATIO,
            "egocentric_manipulator_min_component_size": EGOCENTRIC_MANIPULATOR_MIN_COMPONENT_SIZE,
            "egocentric_manipulator_major_component_ratio": EGOCENTRIC_MANIPULATOR_MAJOR_COMPONENT_RATIO,
            "egocentric_object_max_manipulator_distance_m": EGOCENTRIC_OBJECT_MAX_MANIPULATOR_DISTANCE_M,
            "egocentric_object_query_depth_margin_m": EGOCENTRIC_OBJECT_QUERY_DEPTH_MARGIN_M,
            "stereo_depth_abs_tol": STEREO_DEPTH_ABS_TOL,
            "stereo_depth_rel_tol": STEREO_DEPTH_REL_TOL,
            "stereo_min_consistency_ratio": STEREO_MIN_CONSISTENCY_RATIO,
            "stereo_max_patch_error": STEREO_MAX_PATCH_ERROR,
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


def resolve_traj_filter_ablation_mode(filter_args) -> str:
    """Resolve the optional save-time trajectory-filter ablation mode."""
    mode = (
        getattr(filter_args, "traj_filter_ablation_mode", TRAJ_FILTER_ABLATION_MODE_NONE)
        if filter_args is not None
        else TRAJ_FILTER_ABLATION_MODE_NONE
    )
    mode = str(mode)
    if mode not in TRAJ_FILTER_ABLATION_MODES:
        raise ValueError(
            f"Unsupported traj_filter_ablation_mode: {mode}. "
            f"Expected one of {list(TRAJ_FILTER_ABLATION_MODES)}"
        )
    return mode


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
    patch_stats: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Reject query points whose raw query-frame depth is invalid or locally isolated."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    query_depth = np.asarray(query_depth, dtype=np.float32)
    if query_depth.ndim != 2:
        raise ValueError(f"Expected query_depth shape (H, W), got {query_depth.shape}")
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N, 2), got {keypoints.shape}")

    if patch_stats is None:
        patch_stats = _compute_query_depth_patch_stats(
            keypoints,
            query_depth,
            min_depth=min_depth,
            max_depth=max_depth,
            patch_radius=patch_radius,
        )

    query_valid = np.asarray(patch_stats["query_valid"]).astype(bool, copy=False)
    patch_valid_ratio = np.asarray(patch_stats["patch_valid_ratio"]).astype(np.float32, copy=False)
    patch_median = np.asarray(patch_stats["patch_median"]).astype(np.float32, copy=False)
    query_values = np.asarray(patch_stats["query_values"]).astype(np.float32, copy=False)

    valid_mask = query_valid & (patch_valid_ratio >= float(min_patch_valid_ratio)) & np.isfinite(patch_median)
    deviation_limit = np.maximum(float(median_abs_threshold), float(median_rel_threshold) * patch_median)
    valid_mask &= np.isfinite(deviation_limit)
    valid_mask &= np.abs(query_values - patch_median) <= deviation_limit
    return valid_mask.astype(bool)


def _compute_query_depth_patch_stats(
    keypoints: np.ndarray,
    query_depth: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    patch_radius: int = QUERY_DEPTH_PATCH_RADIUS,
) -> dict[str, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    query_depth = np.asarray(query_depth, dtype=np.float32)
    if query_depth.ndim != 2:
        raise ValueError(f"Expected query_depth shape (H, W), got {query_depth.shape}")
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N, 2), got {keypoints.shape}")

    num_tracks = int(keypoints.shape[0])
    height, width = query_depth.shape
    if width == 0 or height == 0:
        empty_int = np.zeros(num_tracks, dtype=np.int32)
        empty_float = np.full(num_tracks, np.nan, dtype=np.float32)
        empty_bool = np.zeros(num_tracks, dtype=bool)
        return {
            "xs": empty_int,
            "ys": empty_int.copy(),
            "query_values": empty_float.copy(),
            "query_valid": empty_bool,
            "patch_valid_ratio": np.zeros(num_tracks, dtype=np.float32),
            "patch_median": empty_float.copy(),
            "patch_std": empty_float.copy(),
        }

    xs = np.clip(np.round(keypoints[:, 0]).astype(np.int32), 0, width - 1)
    ys = np.clip(np.round(keypoints[:, 1]).astype(np.int32), 0, height - 1)
    query_values = query_depth[ys, xs].astype(np.float32, copy=False)
    query_valid = np.isfinite(query_values) & (query_values > min_depth) & (query_values < max_depth)
    offsets = np.arange(-patch_radius, patch_radius + 1, dtype=np.int32)
    offset_y, offset_x = np.meshgrid(offsets, offsets, indexing="ij")
    patch_y = ys[:, None] + offset_y.reshape(1, -1)
    patch_x = xs[:, None] + offset_x.reshape(1, -1)
    in_bounds = (
        (patch_y >= 0)
        & (patch_y < height)
        & (patch_x >= 0)
        & (patch_x < width)
    )
    patch_y_clip = np.clip(patch_y, 0, height - 1)
    patch_x_clip = np.clip(patch_x, 0, width - 1)
    patch_values = query_depth[patch_y_clip, patch_x_clip].astype(np.float32, copy=False)
    patch_valid = (
        in_bounds
        & np.isfinite(patch_values)
        & (patch_values > min_depth)
        & (patch_values < max_depth)
    )
    patch_area = in_bounds.sum(axis=1).astype(np.float32, copy=False)
    patch_valid_ratio = np.zeros(num_tracks, dtype=np.float32)
    valid_patch_area = patch_area > 0
    patch_valid_ratio[valid_patch_area] = (
        patch_valid.sum(axis=1)[valid_patch_area].astype(np.float32, copy=False)
        / patch_area[valid_patch_area]
    )
    patch_values_masked = np.where(patch_valid, patch_values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        patch_median = np.nanmedian(patch_values_masked, axis=1).astype(np.float32, copy=False)
        patch_std = np.nanstd(patch_values_masked, axis=1).astype(np.float32, copy=False)

    return {
        "xs": xs,
        "ys": ys,
        "query_values": query_values,
        "query_valid": query_valid.astype(bool, copy=False),
        "patch_valid_ratio": patch_valid_ratio.astype(np.float32),
        "patch_median": patch_median.astype(np.float32),
        "patch_std": patch_std.astype(np.float32),
    }


def compute_query_depth_edge_risk_mask(
    keypoints: np.ndarray,
    query_depth: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    edge_rtol: float = QUERY_DEPTH_EDGE_RTOL,
    patch_radius: int = QUERY_DEPTH_PATCH_RADIUS,
    min_patch_valid_ratio: float = QUERY_DEPTH_MIN_VALID_RATIO,
    patch_std_threshold: float = QUERY_DEPTH_EDGE_PATCH_STD_THRESHOLD,
    patch_stats: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    query_depth = np.asarray(query_depth, dtype=np.float32)
    if query_depth.ndim != 2:
        raise ValueError(f"Expected query_depth shape (H, W), got {query_depth.shape}")
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N, 2), got {keypoints.shape}")

    if patch_stats is None:
        patch_stats = _compute_query_depth_patch_stats(
            keypoints,
            query_depth,
            min_depth=min_depth,
            max_depth=max_depth,
            patch_radius=patch_radius,
        )

    xs = np.asarray(patch_stats["xs"]).astype(np.int32, copy=False)
    ys = np.asarray(patch_stats["ys"]).astype(np.int32, copy=False)
    query_valid = np.asarray(patch_stats["query_valid"]).astype(bool, copy=False)
    patch_valid_ratio = np.asarray(patch_stats["patch_valid_ratio"]).astype(np.float32, copy=False)
    patch_std = np.asarray(patch_stats["patch_std"]).astype(np.float32, copy=False)

    valid_depth = np.isfinite(query_depth) & (query_depth > min_depth) & (query_depth < max_depth)
    if query_depth.shape[0] == 0 or query_depth.shape[1] == 0:
        query_edge_mask = np.zeros(keypoints.shape[0], dtype=bool)
    else:
        depth_in = query_depth.copy()
        depth_in[~valid_depth] = 1e9
        edge_mask = depth_edge(depth_in, rtol=edge_rtol, mask=valid_depth)
        query_edge_mask = edge_mask[ys, xs]

    risk_mask = (
        query_valid
        & query_edge_mask
        & (patch_valid_ratio >= float(min_patch_valid_ratio))
        & np.isfinite(patch_std)
        & (patch_std >= float(patch_std_threshold))
    )
    return {
        "mask": risk_mask.astype(bool),
        "query_edge_mask": query_edge_mask.astype(bool),
        "patch_valid_ratio": patch_valid_ratio.astype(np.float32),
        "patch_std": patch_std.astype(np.float32),
    }


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
            x,
            y,
            np.maximum(float(width - 1) - x, 0.0),
            np.maximum(float(height - 1) - y, 0.0),
        ]
    )
    return np.clip(np.round(border_dist), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def build_query_prefilter_result(
    keypoints: np.ndarray,
    query_depth: np.ndarray,
    *,
    filter_args,
    filter_config: dict | None = None,
    query_prefilter_mode: str | None = None,
    wrist_rank_keep_ratio: float | None = None,
) -> dict[str, np.ndarray]:
    """Build a light-weight query prefilter using only query-frame static signals."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    query_depth = np.asarray(query_depth, dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"Expected keypoints shape (N,2), got {keypoints.shape}")
    if query_depth.ndim != 2:
        raise ValueError(f"Expected query_depth shape (H,W), got {query_depth.shape}")

    num_tracks = int(keypoints.shape[0])
    default_prefilter_mask = np.ones(num_tracks, dtype=bool)
    default_reason_bits = np.zeros(num_tracks, dtype=np.uint8)
    default_rank = np.full(num_tracks, np.nan, dtype=np.float32)
    default_ratio = np.full(num_tracks, np.nan, dtype=np.float32)
    default_std = np.full(num_tracks, np.nan, dtype=np.float32)
    default_bool = np.zeros(num_tracks, dtype=bool)
    default_border_dist = _compute_query_border_distances_px(
        keypoints,
        height=int(query_depth.shape[0]),
        width=int(query_depth.shape[1]),
    )

    def _default_result() -> dict[str, np.ndarray]:
        return {
            "prefilter_mask": default_prefilter_mask,
            "reason_bits": default_reason_bits,
            "query_depth_quality_mask": default_prefilter_mask.copy(),
            "query_depth_edge_mask": default_bool.copy(),
            "query_depth_edge_risk_mask": default_bool.copy(),
            "query_depth_patch_valid_ratio": default_ratio.copy(),
            "query_depth_patch_std": default_std.copy(),
            "query_depth_rank": default_rank.copy(),
            "query_border_dist_px": default_border_dist.copy(),
        }

    if filter_config is None:
        filter_config = resolve_traj_filter_config(filter_args)
    profile = str(filter_config["profile"])
    mode = (
        str(query_prefilter_mode)
        if query_prefilter_mode is not None
        else str(getattr(filter_args, "query_prefilter_mode", DEFAULT_QUERY_PREFILTER_MODE))
    )

    if mode == QUERY_PREFILTER_MODE_OFF:
        return _default_result()
    if mode not in {
        QUERY_PREFILTER_MODE_PROFILE_AWARE_STATIC_V1,
        QUERY_PREFILTER_MODE_EXTERNAL_DEPTH_STATIC_V1,
    }:
        raise ValueError(f"Unsupported query_prefilter_mode: {mode}")
    if (
        not bool(filter_config["enabled"])
        and mode != QUERY_PREFILTER_MODE_EXTERNAL_DEPTH_STATIC_V1
    ):
        return _default_result()
    if (
        mode == QUERY_PREFILTER_MODE_PROFILE_AWARE_STATIC_V1
        and profile in {
        TRAJ_FILTER_PROFILE_EXTERNAL,
        TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR,
        TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR_V2,
        }
    ):
        return _default_result()

    patch_stats = _compute_query_depth_patch_stats(
        keypoints,
        query_depth,
        min_depth=float(filter_config["min_depth"]),
        max_depth=float(filter_config["max_depth"]),
    )
    query_depth_quality_mask = compute_query_depth_quality_mask(
        keypoints,
        query_depth,
        min_depth=float(filter_config["min_depth"]),
        max_depth=float(filter_config["max_depth"]),
        patch_stats=patch_stats,
    )
    reason_bits = np.zeros(num_tracks, dtype=np.uint8)
    reason_bits[~query_depth_quality_mask] |= MASK_REASON_QUERY_DEPTH_FAIL

    query_depth_edge_mask = np.zeros(num_tracks, dtype=bool)
    query_depth_edge_risk_mask = np.zeros(num_tracks, dtype=bool)
    query_border_dist_px = default_border_dist.copy()
    prefilter_mask = query_depth_quality_mask.copy()
    if mode == QUERY_PREFILTER_MODE_EXTERNAL_DEPTH_STATIC_V1:
        query_depth_values = np.asarray(patch_stats["query_values"]).astype(np.float32, copy=False)
        min_query_depth_mask = np.isfinite(query_depth_values) & (
            query_depth_values > float(EXTERNAL_QUERY_PREFILTER_MIN_QUERY_DEPTH_M)
        )
        prefilter_mask &= min_query_depth_mask
        reason_bits[~min_query_depth_mask] |= MASK_REASON_QUERY_DEPTH_FAIL
        edge_result = compute_query_depth_edge_risk_mask(
            keypoints,
            query_depth,
            min_depth=float(filter_config["min_depth"]),
            max_depth=float(filter_config["max_depth"]),
            patch_stats=patch_stats,
        )
        query_depth_edge_mask = np.asarray(edge_result["query_edge_mask"]).astype(bool, copy=False)
        query_depth_edge_risk_mask = np.asarray(edge_result["mask"]).astype(bool, copy=False)
        external_edge_fail_mask = query_depth_edge_risk_mask & (
            query_border_dist_px <= np.uint16(EXTERNAL_QUERY_PREFILTER_EDGE_BORDER_PX)
        )
        prefilter_mask &= ~external_edge_fail_mask
        reason_bits[external_edge_fail_mask] |= MASK_REASON_QUERY_DEPTH_EDGE_FAIL
    elif profile in {
        TRAJ_FILTER_PROFILE_WRIST,
        TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE,
        TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE_NO_HEATMAP,
        TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR_TOP95,
        TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR,
    }:
        edge_result = compute_query_depth_edge_risk_mask(
            keypoints,
            query_depth,
            min_depth=float(filter_config["min_depth"]),
            max_depth=float(filter_config["max_depth"]),
            patch_stats=patch_stats,
        )
        query_depth_edge_mask = np.asarray(edge_result["query_edge_mask"]).astype(bool, copy=False)
        query_depth_edge_risk_mask = np.asarray(edge_result["mask"]).astype(bool, copy=False)
        prefilter_mask &= ~query_depth_edge_risk_mask
        reason_bits[query_depth_edge_risk_mask] |= MASK_REASON_QUERY_DEPTH_EDGE_FAIL

    query_depth_rank = np.full(num_tracks, np.nan, dtype=np.float32)
    if profile in {
        TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR_TOP95,
        TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR,
    }:
        rank_keep_ratio = (
            DEFAULT_QUERY_PREFILTER_WRIST_RANK_KEEP_RATIO
            if wrist_rank_keep_ratio is None
            else float(wrist_rank_keep_ratio)
        )
        rank_keep_ratio = float(np.clip(rank_keep_ratio, 0.0, 1.0))
        rank_input_mask = prefilter_mask.copy()
        query_depth_values = np.asarray(patch_stats["query_values"]).astype(np.float32, copy=False)
        query_depth_rank = _compute_query_depth_ranks(query_depth_values, rank_input_mask)
        rank_keep_mask = rank_input_mask & np.isfinite(query_depth_rank) & (query_depth_rank <= rank_keep_ratio)
        reason_bits[rank_input_mask & (~rank_keep_mask)] |= MASK_REASON_MANIPULATOR_DEPTH_FAIL
        prefilter_mask = rank_keep_mask

    return {
        "prefilter_mask": prefilter_mask.astype(bool, copy=False),
        "reason_bits": reason_bits.astype(np.uint8, copy=False),
        "query_depth_quality_mask": query_depth_quality_mask.astype(bool, copy=False),
        "query_depth_edge_mask": query_depth_edge_mask.astype(bool, copy=False),
        "query_depth_edge_risk_mask": query_depth_edge_risk_mask.astype(bool, copy=False),
        "query_depth_patch_valid_ratio": np.asarray(patch_stats["patch_valid_ratio"]).astype(np.float32, copy=False),
        "query_depth_patch_std": np.asarray(patch_stats["patch_std"]).astype(np.float32, copy=False),
        "query_depth_rank": query_depth_rank.astype(np.float32, copy=False),
        "query_border_dist_px": query_border_dist_px.astype(np.uint16, copy=False),
    }


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
        depth_lo, depth_hi = np.nanpercentile(
            depths_nan,
            [low_percentile, high_percentile],
            axis=0,
        )

    volatility = np.nan_to_num(depth_hi - depth_lo, nan=0.0, posinf=0.0, neginf=0.0)
    valid_counts = valid.sum(axis=0)
    volatility[valid_counts < 2] = 0.0
    return volatility.astype(np.float32)


def _threshold_high_volatility_values(
    values: np.ndarray,
    *,
    percentile: float,
) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        return np.zeros(values.shape, dtype=bool), float("nan")

    threshold = float(np.percentile(finite_values, percentile))
    if not np.isfinite(threshold) or threshold <= 0.0:
        return finite & (values > 0.0), threshold
    return finite & (values >= threshold), threshold


def compute_high_volatility_mask(
    volatility_map: np.ndarray,
    *,
    percentile: float = VOLATILITY_MASK_PERCENTILE,
) -> tuple[np.ndarray, float]:
    """Threshold a volatility map by global percentile."""
    volatility_map = np.asarray(volatility_map, dtype=np.float32)
    if volatility_map.ndim != 2:
        raise ValueError(f"Expected volatility_map shape (H,W), got {volatility_map.shape}")

    hit_mask, threshold = _threshold_high_volatility_values(
        volatility_map.reshape(-1),
        percentile=percentile,
    )
    return hit_mask.reshape(volatility_map.shape), threshold


def _compute_linear_percentiles_for_masked_columns(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    percentiles: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Match NumPy's default linear percentile interpolation for a masked (T, M) matrix."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"Expected values shape (T,M), got {values.shape}")
    if valid.shape != values.shape:
        raise ValueError(f"Expected valid shape {values.shape}, got {valid.shape}")

    percentiles_np = np.asarray(percentiles, dtype=np.float64).reshape(-1)
    if percentiles_np.size == 0:
        raise ValueError("percentiles must contain at least one value")

    valid_counts = valid.sum(axis=0).astype(np.int32, copy=False)
    result = np.full((percentiles_np.size, values.shape[1]), np.nan, dtype=np.float64)
    valid_columns = valid_counts > 0
    if not np.any(valid_columns):
        return result, valid_counts

    sorted_values = np.sort(np.where(valid, values, np.inf), axis=0)
    cols = np.flatnonzero(valid_columns).astype(np.int32, copy=False)
    counts = valid_counts[cols].astype(np.float64, copy=False)
    max_indices = counts - 1.0
    for percentile_idx, percentile in enumerate(percentiles_np.tolist()):
        rank = np.clip(float(percentile), 0.0, 100.0) / 100.0 * max_indices
        low_idx = np.floor(rank).astype(np.int32, copy=False)
        high_idx = np.ceil(rank).astype(np.int32, copy=False)
        interp = rank - low_idx.astype(np.float64, copy=False)
        low_values = sorted_values[low_idx, cols].astype(np.float64, copy=False)
        high_values = sorted_values[high_idx, cols].astype(np.float64, copy=False)
        result[percentile_idx, cols] = low_values + interp * (high_values - low_values)

    return result, valid_counts


def compute_accessed_high_volatility_mask(
    full_depths: np.ndarray,
    *,
    accessed_pixel_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
    low_percentile: float = VOLATILITY_LOW_PERCENTILE,
    high_percentile: float = VOLATILITY_HIGH_PERCENTILE,
    mask_percentile: float = VOLATILITY_MASK_PERCENTILE,
    return_stats: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, dict[str, float]]:
    """Compute a dense high-volatility mask using only accessed pixel locations."""
    full_depths = np.asarray(full_depths, dtype=np.float32)
    accessed_pixel_mask = np.asarray(accessed_pixel_mask, dtype=bool)
    if full_depths.ndim != 3:
        raise ValueError(f"Expected full_depths shape (T,H,W), got {full_depths.shape}")
    if accessed_pixel_mask.shape != full_depths.shape[1:]:
        raise ValueError(
            f"Expected accessed_pixel_mask shape {full_depths.shape[1:]}, got {accessed_pixel_mask.shape}"
        )

    high_volatility_mask = np.zeros(full_depths.shape[1:], dtype=bool)
    accessed_pixel_count = int(np.count_nonzero(accessed_pixel_mask))
    ys, xs = np.nonzero(accessed_pixel_mask)
    if ys.size == 0:
        stats = {
            "accessed_pixel_count": float(accessed_pixel_count),
            "valid_pixel_count": 0.0,
            "threshold": float("nan"),
        }
        if return_stats:
            return high_volatility_mask, float("nan"), stats
        return high_volatility_mask, float("nan")

    accessed_depths = full_depths[:, ys, xs]
    valid = np.isfinite(accessed_depths) & (accessed_depths > min_depth) & (accessed_depths < max_depth)
    valid_pixel_count = int(np.count_nonzero(valid.any(axis=0)))
    if not np.any(valid):
        stats = {
            "accessed_pixel_count": float(accessed_pixel_count),
            "valid_pixel_count": float(valid_pixel_count),
            "threshold": float("nan"),
        }
        if return_stats:
            return high_volatility_mask, float("nan"), stats
        return high_volatility_mask, float("nan")

    percentile_values, valid_counts = _compute_linear_percentiles_for_masked_columns(
        accessed_depths,
        valid,
        percentiles=(float(low_percentile), float(high_percentile)),
    )
    depth_lo, depth_hi = percentile_values

    volatility_values = np.nan_to_num(depth_hi - depth_lo, nan=0.0, posinf=0.0, neginf=0.0)
    volatility_values[valid_counts < 2] = 0.0
    hit_mask, threshold = _threshold_high_volatility_values(
        volatility_values,
        percentile=mask_percentile,
    )
    high_volatility_mask[ys[hit_mask], xs[hit_mask]] = True
    stats = {
        "accessed_pixel_count": float(accessed_pixel_count),
        "valid_pixel_count": float(valid_pixel_count),
        "threshold": float(threshold),
    }
    if return_stats:
        return high_volatility_mask, threshold, stats
    return high_volatility_mask, threshold


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


def _rgb_segment_to_gray(rgb_segment: np.ndarray) -> np.ndarray:
    rgb_segment = np.asarray(rgb_segment, dtype=np.float32)
    if rgb_segment.ndim != 4:
        raise ValueError(f"Expected rgb_segment shape (T,C,H,W) or (T,H,W,C), got {rgb_segment.shape}")
    if rgb_segment.shape[1] == 3:
        red, green, blue = rgb_segment[:, 0], rgb_segment[:, 1], rgb_segment[:, 2]
    elif rgb_segment.shape[-1] == 3:
        red, green, blue = rgb_segment[..., 0], rgb_segment[..., 1], rgb_segment[..., 2]
    else:
        raise ValueError(f"Expected RGB channel dimension of size 3, got {rgb_segment.shape}")
    if np.nanmax(rgb_segment) > 1.5:
        red = red / 255.0
        green = green / 255.0
        blue = blue / 255.0
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def _sample_segment_values_at_uv(
    values_segment: np.ndarray,
    uvz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values_segment = np.asarray(values_segment, dtype=np.float32)
    uvz = np.asarray(uvz, dtype=np.float32)
    if values_segment.ndim != 3:
        raise ValueError(f"Expected values_segment shape (T,H,W), got {values_segment.shape}")
    if uvz.ndim != 3 or uvz.shape[-1] < 2:
        raise ValueError(f"Expected uvz shape (N,T,>=2), got {uvz.shape}")
    num_tracks, num_frames = uvz.shape[:2]
    if values_segment.shape[0] != num_frames:
        raise ValueError(
            f"Expected values_segment first dim {num_frames}, got {values_segment.shape[0]}"
        )

    h, w = values_segment.shape[1:]
    u = uvz[..., 0]
    v = uvz[..., 1]
    in_bounds = (
        np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u <= float(max(w - 1, 0)))
        & (v >= 0.0)
        & (v <= float(max(h - 1, 0)))
    )
    xs = np.clip(np.round(np.nan_to_num(u, nan=-1.0)).astype(np.int32), 0, max(w - 1, 0))
    ys = np.clip(np.round(np.nan_to_num(v, nan=-1.0)).astype(np.int32), 0, max(h - 1, 0))
    sampled = np.full((num_tracks, num_frames), np.nan, dtype=np.float32)
    for frame_idx in range(num_frames):
        valid_mask = in_bounds[:, frame_idx]
        if not np.any(valid_mask):
            continue
        sampled[valid_mask, frame_idx] = values_segment[frame_idx, ys[valid_mask, frame_idx], xs[valid_mask, frame_idx]]
    sampled_valid = in_bounds & np.isfinite(sampled)
    return sampled.astype(np.float32), sampled_valid.astype(bool)


def evaluate_stereo_consistency(
    traj_uvz: np.ndarray,
    *,
    visibs: np.ndarray | None,
    intrinsics_segment: np.ndarray,
    extrinsics_segment: np.ndarray,
    stereo_context: dict[str, np.ndarray | str] | None,
    min_depth: float,
    max_depth: float,
    min_valid_frames: int,
    depth_abs_tol: float,
    depth_rel_tol: float,
    min_consistency_ratio: float,
    max_patch_error: float,
) -> dict[str, np.ndarray]:
    traj_uvz = np.asarray(traj_uvz, dtype=np.float32)
    num_tracks, num_frames, _ = traj_uvz.shape
    default_ratio = np.full(num_tracks, np.nan, dtype=np.float32)
    default_counts = np.zeros(num_tracks, dtype=np.uint16)
    if stereo_context is None:
        return {
            "compare_counts": default_counts,
            "depth_consistency_ratio": default_ratio,
            "patch_error": default_ratio.copy(),
            "mask": np.ones(num_tracks, dtype=bool),
        }

    stereo_depths = np.asarray(stereo_context["depth_segment"], dtype=np.float32)
    stereo_intrinsics = np.asarray(stereo_context["intrinsics_segment"], dtype=np.float32)
    stereo_extrinsics = np.asarray(stereo_context["extrinsics_segment"], dtype=np.float32)
    if stereo_depths.shape[0] != num_frames:
        raise ValueError(
            f"Stereo context frame count mismatch: traj has {num_frames}, stereo depth has {stereo_depths.shape[0]}"
        )

    visibility = _normalize_visibility(visibs, num_tracks=num_tracks, num_frames=num_frames)
    world_tracks = traj_uvz_to_world_coordinates(
        traj_uvz,
        query_intrinsics=intrinsics_segment[0],
        query_w2c=extrinsics_segment[0],
        min_depth=min_depth,
        max_depth=max_depth,
    )
    stereo_uvz = project_world_tracks_to_camera_uvz(
        world_tracks,
        intrinsics_segment=stereo_intrinsics,
        extrinsics_segment=stereo_extrinsics,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    compare_mask = np.isfinite(traj_uvz).all(axis=-1) & np.isfinite(stereo_uvz).all(axis=-1)
    if visibility is not None:
        compare_mask &= visibility

    sampled_depths, sampled_depth_valid = _sample_segment_values_at_uv(stereo_depths, stereo_uvz)
    compare_mask &= sampled_depth_valid
    expected_depth = stereo_uvz[..., 2]
    depth_tol = np.maximum(float(depth_abs_tol), float(depth_rel_tol) * np.abs(expected_depth))
    depth_agree = compare_mask & np.isfinite(expected_depth) & np.isfinite(sampled_depths)
    depth_agree &= np.abs(sampled_depths - expected_depth) <= depth_tol
    compare_counts = compare_mask.sum(axis=1).astype(np.uint16)
    depth_consistency_ratio = _counts_to_ratio(depth_agree.sum(axis=1), compare_counts)

    patch_error = np.full(num_tracks, np.nan, dtype=np.float32)
    current_rgb_segment = stereo_context.get("current_rgb_segment")
    stereo_rgb_segment = stereo_context.get("rgb_segment")
    if current_rgb_segment is not None and stereo_rgb_segment is not None:
        current_gray = _rgb_segment_to_gray(np.asarray(current_rgb_segment, dtype=np.float32))
        stereo_gray = _rgb_segment_to_gray(np.asarray(stereo_rgb_segment, dtype=np.float32))
        current_gray_samples, current_gray_valid = _sample_segment_values_at_uv(current_gray, traj_uvz)
        stereo_gray_samples, stereo_gray_valid = _sample_segment_values_at_uv(stereo_gray, stereo_uvz)
        patch_mask = compare_mask & current_gray_valid & stereo_gray_valid
        if np.any(patch_mask):
            gray_diff = np.abs(current_gray_samples - stereo_gray_samples).astype(np.float32)
            gray_diff[~patch_mask] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                patch_error = np.nanmedian(gray_diff, axis=1).astype(np.float32)

    enough_support_mask = compare_counts >= np.uint16(max(1, int(min_valid_frames)))
    patch_pass_mask = (~np.isfinite(patch_error)) | (patch_error <= float(max_patch_error))
    stereo_mask = (~enough_support_mask) | (
        np.isfinite(depth_consistency_ratio)
        & (depth_consistency_ratio >= float(min_consistency_ratio))
        & patch_pass_mask
    )
    return {
        "compare_counts": compare_counts.astype(np.uint16),
        "depth_consistency_ratio": depth_consistency_ratio.astype(np.float32),
        "patch_error": patch_error.astype(np.float32),
        "mask": stereo_mask.astype(bool),
    }


def prepare_temporal_depth_consistency_context(
    traj_uvz: np.ndarray,
    *,
    visibs: np.ndarray | None,
    raw_depths_segment: np.ndarray,
    intrinsics_segment: np.ndarray,
    extrinsics_segment: np.ndarray,
    min_depth: float,
    max_depth: float,
    depth_abs_tol: float = TEMPORAL_DEPTH_ABS_TOL,
    depth_rel_tol: float = TEMPORAL_DEPTH_REL_TOL,
    include_reprojected_uvz: bool = False,
) -> dict[str, np.ndarray | int]:
    """Prepare reusable per-frame temporal depth consistency comparisons."""
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

    context: dict[str, np.ndarray | int] = {
        "compare_mask": compare_mask.astype(bool),
        "consistent_frame_mask": consistent_frame_mask.astype(bool),
        "xs_clip": xs_clip.astype(np.int32),
        "ys_clip": ys_clip.astype(np.int32),
        "height": int(height),
        "width": int(width),
    }
    if include_reprojected_uvz:
        context["reprojected_uvz"] = reprojected_uvz.astype(np.float32)
    return context


def _evaluate_temporal_depth_consistency_from_context(
    temporal_compare_context: dict[str, np.ndarray | int],
    *,
    high_volatility_mask: np.ndarray | None,
    min_valid_frames: int,
    min_consistency_ratio: float = TEMPORAL_MIN_CONSISTENCY_RATIO,
) -> dict[str, np.ndarray | int]:
    compare_mask = np.asarray(temporal_compare_context["compare_mask"]).astype(bool, copy=False)
    consistent_frame_mask = np.asarray(temporal_compare_context["consistent_frame_mask"]).astype(bool, copy=False)
    xs_clip = np.asarray(temporal_compare_context["xs_clip"]).astype(np.int32, copy=False)
    ys_clip = np.asarray(temporal_compare_context["ys_clip"]).astype(np.int32, copy=False)
    if compare_mask.shape != consistent_frame_mask.shape or compare_mask.shape != xs_clip.shape or compare_mask.shape != ys_clip.shape:
        raise ValueError("Temporal compare context arrays must share shape (N,T)")

    num_tracks, num_frames = compare_mask.shape
    height = int(temporal_compare_context["height"])
    width = int(temporal_compare_context["width"])

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

    result: dict[str, np.ndarray | int] = {
        "mask": mask.astype(bool),
        "consistency_ratio": consistency_ratio.astype(np.float32),
        "stable_consistency_ratio": stable_consistency_ratio.astype(np.float32),
        "compare_counts": compare_counts,
        "stable_compare_counts": stable_compare_counts,
        "required_compare_frames": int(required_compare_frames),
        "compare_mask": compare_mask.astype(bool),
        "consistent_frame_mask": consistent_frame_mask.astype(bool),
        "high_volatility_hit": volatility_frame_mask.any(axis=1),
        "volatility_exposure_ratio": volatility_exposure_ratio.astype(np.float32),
        "stable_frames_sufficient": stable_frames_sufficient.astype(bool),
        "all_pass": all_pass.astype(bool),
        "stable_pass": stable_pass.astype(bool),
    }
    if "reprojected_uvz" in temporal_compare_context:
        result["reprojected_uvz"] = np.asarray(temporal_compare_context["reprojected_uvz"]).astype(
            np.float32,
            copy=False,
        )
    return result


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
    temporal_compare_context: dict[str, np.ndarray | int] | None = None,
) -> dict[str, np.ndarray | int]:
    """Check whether trajectories remain depth-consistent after per-frame reprojection."""
    if temporal_compare_context is None:
        temporal_compare_context = prepare_temporal_depth_consistency_context(
            traj_uvz,
            visibs=visibs,
            raw_depths_segment=raw_depths_segment,
            intrinsics_segment=intrinsics_segment,
            extrinsics_segment=extrinsics_segment,
            min_depth=min_depth,
            max_depth=max_depth,
            depth_abs_tol=depth_abs_tol,
            depth_rel_tol=depth_rel_tol,
            include_reprojected_uvz=True,
        )
    return _evaluate_temporal_depth_consistency_from_context(
        temporal_compare_context,
        high_volatility_mask=high_volatility_mask,
        min_valid_frames=min_valid_frames,
        min_consistency_ratio=min_consistency_ratio,
    )


def build_traj_filter_result(
    traj: np.ndarray,
    visibs: np.ndarray | None,
    image_width: int,
    image_height: int,
    filter_args,
    *,
    keypoints: np.ndarray | None = None,
    query_depth: np.ndarray | None = None,
    pick_place_heatmap_segment: np.ndarray | None = None,
    raw_depths_segment: np.ndarray | None = None,
    intrinsics_segment: np.ndarray | None = None,
    extrinsics_segment: np.ndarray | None = None,
    stereo_context: dict[str, np.ndarray | str] | None = None,
    high_volatility_mask: np.ndarray | None = None,
    depth_volatility_map: np.ndarray | None = None,
    temporal_compare_context: dict[str, np.ndarray | int] | None = None,
    profile_stats: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Build per-trajectory mask plus debug metadata."""
    filter_result_start = time.perf_counter()
    traj = np.asarray(traj, dtype=np.float32)
    num_tracks, num_frames, _ = traj.shape
    config = resolve_traj_filter_config(filter_args)
    profile = config["profile"]
    if profile not in {
        TRAJ_FILTER_PROFILE_EXTERNAL,
        TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR,
        TRAJ_FILTER_PROFILE_EXTERNAL_MANIPULATOR_V2,
        TRAJ_FILTER_PROFILE_EGOCENTRIC_OBJECT_INTERACTION_V1,
        TRAJ_FILTER_PROFILE_WRIST,
        TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE,
        TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE_NO_HEATMAP,
        TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR_TOP95,
        TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR,
    }:
        raise ValueError(f"Unsupported traj_filter_profile: {profile}")
    ablation_mode = resolve_traj_filter_ablation_mode(filter_args)
    wrist_like_profiles = {
        TRAJ_FILTER_PROFILE_WRIST,
        TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE,
        TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE_NO_HEATMAP,
        TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR_TOP95,
        TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR,
    }
    wrist_manipulator_profiles = {
        TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR_TOP95,
        TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR,
    }
    manipulator_stage_ablation_modes = {
        TRAJ_FILTER_ABLATION_MODE_WRIST_SEED_TOP95,
        TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_DEPTH,
        TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_MOTION,
        TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_CLUSTER,
    }
    if ablation_mode != TRAJ_FILTER_ABLATION_MODE_NONE and profile not in wrist_like_profiles:
        raise ValueError(
            f"traj_filter_ablation_mode='{ablation_mode}' requires a wrist-like traj_filter_profile, got '{profile}'"
        )
    if (
        ablation_mode in manipulator_stage_ablation_modes
        and profile not in wrist_manipulator_profiles
    ):
        raise ValueError(
            f"traj_filter_ablation_mode='{ablation_mode}' requires traj_filter_profile to be one of "
            f"{sorted(wrist_manipulator_profiles)}, got '{profile}'"
        )

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
    default_query_depth_edge_mask = np.zeros(num_tracks, dtype=bool)
    default_query_depth_patch_valid_ratio = np.full(num_tracks, np.nan, dtype=np.float32)
    default_query_depth_patch_std = np.full(num_tracks, np.nan, dtype=np.float32)
    default_all_true_mask = np.ones(num_tracks, dtype=bool)
    default_pick_place_count = np.zeros(num_tracks, dtype=np.uint16)
    default_pick_place_distance = np.full(num_tracks, np.nan, dtype=np.float32)
    default_stereo_compare_count = np.zeros(num_tracks, dtype=np.uint16)
    default_stereo_ratio = np.full(num_tracks, np.nan, dtype=np.float32)
    default_stereo_mask = np.ones(num_tracks, dtype=bool)
    filter_result_base_geometry_seconds = 0.0
    filter_result_query_depth_patch_stats_seconds = 0.0
    filter_result_query_depth_quality_seconds = 0.0
    filter_result_query_depth_edge_risk_seconds = 0.0
    filter_result_temporal_seconds = 0.0
    filter_result_manipulator_near_depth_seconds = 0.0
    filter_result_manipulator_world_lift_seconds = 0.0
    filter_result_manipulator_motion_seconds = 0.0
    filter_result_manipulator_cluster_seconds = 0.0
    filter_result_top95_seconds = 0.0

    if not config["enabled"]:
        filter_result_total_seconds = time.perf_counter() - filter_result_start
        result = {
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
            "traj_motion_extent_all_valid": default_manipulator_rank.copy(),
            "traj_motion_step_median_all_valid": default_manipulator_rank.copy(),
            "traj_manipulator_candidate_mask": default_manipulator_mask.copy(),
            "traj_manipulator_cluster_id": default_cluster_id.copy(),
            "traj_manipulator_component_size": default_component_size.copy(),
            "traj_manipulator_cluster_fallback_used": default_fallback_used.copy(),
            "traj_query_depth_edge_mask": default_query_depth_edge_mask.copy(),
            "traj_query_depth_patch_valid_ratio": default_query_depth_patch_valid_ratio.copy(),
            "traj_query_depth_patch_std": default_query_depth_patch_std.copy(),
            "traj_query_depth_edge_risk_mask": default_query_depth_edge_mask.copy(),
            "traj_base_mask": default_all_true_mask.copy(),
            "traj_query_depth_quality_mask": default_all_true_mask.copy(),
            "traj_query_depth_keep_mask": default_all_true_mask.copy(),
            "traj_supervision_support_mask": default_all_true_mask.copy(),
            "traj_near_depth_mask": default_manipulator_mask.copy(),
            "traj_motion_mask": default_manipulator_mask.copy(),
            "traj_cluster_mask": default_manipulator_mask.copy(),
            "traj_pre_top95_mask": default_all_true_mask.copy(),
            "traj_pick_place_heatmap_hit_count": default_pick_place_count.copy(),
            "traj_pick_place_heatmap_support_mask": default_manipulator_mask.copy(),
            "traj_pick_place_min_manipulator_distance": default_pick_place_distance.copy(),
            "traj_pick_place_contact_mask": default_manipulator_mask.copy(),
            "traj_pick_place_depth_guard_mask": default_manipulator_mask.copy(),
            "traj_pick_place_delayed_contact_rescue_mask": default_manipulator_mask.copy(),
            "traj_pick_place_object_mask": default_manipulator_mask.copy(),
            "traj_stereo_compare_frame_count": default_stereo_compare_count.copy(),
            "traj_stereo_depth_consistency_ratio": default_stereo_ratio.copy(),
            "traj_stereo_patch_error": default_stereo_ratio.copy(),
            "traj_stereo_consistency_mask": default_stereo_mask.copy(),
        }
        _accumulate_profile_stat(
            profile_stats,
            "filter_result_total_seconds",
            filter_result_total_seconds,
        )
        _accumulate_profile_stat(profile_stats, "filter_result_other_seconds", filter_result_total_seconds)
        return result

    base_geometry_start = time.perf_counter()
    visibility = _normalize_visibility(visibs, num_tracks=num_tracks, num_frames=num_frames)
    tail_truncated_sample = is_tail_truncated_sample(num_frames=num_frames, filter_args=filter_args)
    visibs_for_filter = visibility if (config["use_visibility"] and not tail_truncated_sample) else None
    visibs_for_temporal = None if tail_truncated_sample else visibility
    if profile == TRAJ_FILTER_PROFILE_EGOCENTRIC_OBJECT_INTERACTION_V1:
        # Egocentric tracks often get near-zero learned visibility; use geometric
        # temporal/stereo support here instead of letting visibility collapse all support.
        visibs_for_temporal = None

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
    filter_result_base_geometry_seconds = time.perf_counter() - base_geometry_start
    _accumulate_profile_stat(
        profile_stats,
        "filter_result_base_geometry_seconds",
        filter_result_base_geometry_seconds,
    )
    base_mask = np.asarray(base_geometry["traj_valid_mask"]).astype(bool, copy=False)
    wrist_base_mask = (
        np.asarray(base_geometry["valid_count_mask"]).astype(bool, copy=False)
        & np.asarray(base_geometry["depth_range_mask"]).astype(bool, copy=False)
        & np.asarray(base_geometry["depth_smooth_mask"]).astype(bool, copy=False)
    )
    wrist_pick_place_base_mask = (
        np.asarray(base_geometry["valid_count_mask"]).astype(bool, copy=False)
        & np.asarray(base_geometry["depth_smooth_mask"]).astype(bool, copy=False)
    )

    query_depth_quality_mask = np.ones(num_tracks, dtype=bool)
    query_depth_mask = np.ones(num_tracks, dtype=bool)
    query_depth_patch_stats: dict[str, np.ndarray] | None = None
    traj_query_depth_edge_mask = default_query_depth_edge_mask.copy()
    traj_query_depth_patch_valid_ratio = default_query_depth_patch_valid_ratio.copy()
    traj_query_depth_patch_std = default_query_depth_patch_std.copy()
    traj_query_depth_edge_risk_mask = default_query_depth_edge_mask.copy()
    if config["use_query_depth_quality"]:
        if keypoints is None or query_depth is None:
            raise ValueError("keypoints and query_depth are required when query-depth quality filtering is enabled")
        if keypoints.shape[0] != traj.shape[0]:
            raise ValueError(
                f"Expected keypoints and trajectories to share track count, got {keypoints.shape[0]} and {traj.shape[0]}"
            )
        query_depth_patch_start = time.perf_counter()
        query_depth_patch_stats = _compute_query_depth_patch_stats(
            keypoints,
            query_depth,
            min_depth=config["min_depth"],
            max_depth=config["max_depth"],
        )
        filter_result_query_depth_patch_stats_seconds = time.perf_counter() - query_depth_patch_start
        _accumulate_profile_stat(
            profile_stats,
            "filter_result_query_depth_patch_stats_seconds",
            filter_result_query_depth_patch_stats_seconds,
        )
        query_depth_quality_start = time.perf_counter()
        query_depth_quality_mask = compute_query_depth_quality_mask(
            keypoints,
            query_depth,
            min_depth=config["min_depth"],
            max_depth=config["max_depth"],
            patch_stats=query_depth_patch_stats,
        )
        filter_result_query_depth_quality_seconds = time.perf_counter() - query_depth_quality_start
        _accumulate_profile_stat(
            profile_stats,
            "filter_result_query_depth_quality_seconds",
            filter_result_query_depth_quality_seconds,
        )
        query_depth_mask = query_depth_quality_mask.copy()
        if profile in wrist_like_profiles:
            query_depth_edge_start = time.perf_counter()
            query_depth_edge_result = compute_query_depth_edge_risk_mask(
                keypoints,
                query_depth,
                min_depth=config["min_depth"],
                max_depth=config["max_depth"],
                patch_stats=query_depth_patch_stats,
            )
            traj_query_depth_edge_mask = np.asarray(query_depth_edge_result["query_edge_mask"]).astype(
                bool, copy=False
            )
            traj_query_depth_patch_valid_ratio = np.asarray(
                query_depth_edge_result["patch_valid_ratio"]
            ).astype(np.float32, copy=False)
            traj_query_depth_patch_std = np.asarray(query_depth_edge_result["patch_std"]).astype(
                np.float32, copy=False
            )
            traj_query_depth_edge_risk_mask = np.asarray(query_depth_edge_result["mask"]).astype(
                bool, copy=False
            )
            filter_result_query_depth_edge_risk_seconds = time.perf_counter() - query_depth_edge_start
            _accumulate_profile_stat(
                profile_stats,
                "filter_result_query_depth_edge_risk_seconds",
                filter_result_query_depth_edge_risk_seconds,
            )
            if ablation_mode != TRAJ_FILTER_ABLATION_MODE_WRIST_NO_QUERY_EDGE:
                query_depth_mask &= ~traj_query_depth_edge_risk_mask
    traj_query_depth_keep_mask = query_depth_mask.copy()

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
        temporal_high_volatility_mask = None
        if config["use_depth_volatility_guidance"]:
            if high_volatility_mask is None and depth_volatility_map is not None:
                high_volatility_mask, _ = compute_high_volatility_mask(
                    depth_volatility_map,
                    percentile=config["volatility_mask_percentile"],
                )
            if high_volatility_mask is None:
                raise ValueError("high_volatility_mask is required when volatility guidance is enabled")
            temporal_high_volatility_mask = high_volatility_mask

        temporal_start = time.perf_counter()
        temporal_result = evaluate_temporal_depth_consistency(
            traj,
            visibs=visibs_for_temporal,
            raw_depths_segment=raw_depths_segment,
            intrinsics_segment=intrinsics_segment,
            extrinsics_segment=extrinsics_segment,
            min_depth=config["min_depth"],
            max_depth=config["max_depth"],
            min_valid_frames=config["min_valid_frames"],
            min_consistency_ratio=config["temporal_min_consistency_ratio"],
            depth_abs_tol=config["temporal_depth_abs_tol"],
            depth_rel_tol=config["temporal_depth_rel_tol"],
            high_volatility_mask=temporal_high_volatility_mask,
            temporal_compare_context=temporal_compare_context,
        )
        filter_result_temporal_seconds = time.perf_counter() - temporal_start
        _accumulate_profile_stat(
            profile_stats,
            "filter_result_temporal_seconds",
            filter_result_temporal_seconds,
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
    elif visibs_for_temporal is not None:
        supervision_mask &= visibs_for_temporal

    supervision_prefix_len = _compute_true_prefix_lengths(supervision_mask).astype(np.uint16)
    supervision_count = supervision_mask.sum(axis=1).astype(np.uint16)

    stereo_compare_frame_count = default_stereo_compare_count.copy()
    stereo_depth_consistency_ratio = default_stereo_ratio.copy()
    stereo_patch_error = default_stereo_ratio.copy()
    stereo_consistency_mask = default_stereo_mask.copy()
    if stereo_context is not None:
        stereo_result = evaluate_stereo_consistency(
            traj,
            visibs=visibs_for_temporal,
            intrinsics_segment=intrinsics_segment,
            extrinsics_segment=extrinsics_segment,
            stereo_context=stereo_context,
            min_depth=config["min_depth"],
            max_depth=config["max_depth"],
            min_valid_frames=config["min_valid_frames"],
            depth_abs_tol=config["stereo_depth_abs_tol"],
            depth_rel_tol=config["stereo_depth_rel_tol"],
            min_consistency_ratio=config["stereo_min_consistency_ratio"],
            max_patch_error=config["stereo_max_patch_error"],
        )
        stereo_compare_frame_count = np.asarray(stereo_result["compare_counts"]).astype(np.uint16, copy=False)
        stereo_depth_consistency_ratio = np.asarray(stereo_result["depth_consistency_ratio"]).astype(
            np.float32,
            copy=False,
        )
        stereo_patch_error = np.asarray(stereo_result["patch_error"]).astype(np.float32, copy=False)
        stereo_consistency_mask = np.asarray(stereo_result["mask"]).astype(bool, copy=False)

    reason_bits = np.zeros(num_tracks, dtype=np.uint8)

    wrist_seed_mask = default_manipulator_mask.copy()
    traj_query_depth_rank = default_manipulator_rank.copy()
    traj_motion_extent = default_manipulator_rank.copy()
    traj_motion_step_median = default_manipulator_rank.copy()
    traj_motion_extent_all_valid = default_manipulator_rank.copy()
    traj_motion_step_median_all_valid = default_manipulator_rank.copy()
    traj_manipulator_candidate_mask = default_manipulator_mask.copy()
    traj_manipulator_cluster_id = default_cluster_id.copy()
    traj_manipulator_component_size = default_component_size.copy()
    traj_manipulator_cluster_fallback_used = default_fallback_used.copy()
    traj_base_mask = base_mask.copy()
    traj_supervision_support_mask = temporal_mask.copy()
    traj_near_depth_mask = default_manipulator_mask.copy()
    traj_motion_mask = default_manipulator_mask.copy()
    traj_cluster_mask = default_manipulator_mask.copy()
    traj_pre_top95_mask = default_manipulator_mask.copy()
    traj_pick_place_heatmap_hit_count = default_pick_place_count.copy()
    traj_pick_place_heatmap_support_mask = default_manipulator_mask.copy()
    traj_pick_place_min_manipulator_distance = default_pick_place_distance.copy()
    traj_pick_place_contact_mask = default_manipulator_mask.copy()
    traj_pick_place_depth_guard_mask = default_manipulator_mask.copy()
    traj_pick_place_delayed_contact_rescue_mask = default_manipulator_mask.copy()
    traj_pick_place_object_mask = default_manipulator_mask.copy()
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
            traj_pre_top95_mask = final_mask.copy()
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
                "motion_metric_mode": "supervised",
                "reusable_geometry": None,
                "profile_stats": profile_stats,
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
            manipulator_near_depth_before = _get_profile_stat(
                profile_stats, "filter_result_manipulator_near_depth_seconds"
            )
            manipulator_world_lift_before = _get_profile_stat(
                profile_stats, "filter_result_manipulator_world_lift_seconds"
            )
            manipulator_motion_before = _get_profile_stat(
                profile_stats, "filter_result_manipulator_motion_seconds"
            )
            manipulator_cluster_before = _get_profile_stat(
                profile_stats, "filter_result_manipulator_cluster_seconds"
            )
            (
                final_mask,
                traj_query_depth_rank,
                traj_motion_extent,
                traj_motion_step_median,
                traj_motion_extent_all_valid,
                traj_motion_step_median_all_valid,
                traj_manipulator_candidate_mask,
                traj_manipulator_cluster_id,
                traj_manipulator_component_size,
                traj_near_depth_mask,
                traj_motion_mask,
                traj_cluster_mask,
                fallback_used,
            ) = _apply_manipulator_aware_filter(**manipulator_filter_kwargs)
            filter_result_manipulator_near_depth_seconds += max(
                0.0,
                _get_profile_stat(profile_stats, "filter_result_manipulator_near_depth_seconds")
                - manipulator_near_depth_before,
            )
            filter_result_manipulator_world_lift_seconds += max(
                0.0,
                _get_profile_stat(profile_stats, "filter_result_manipulator_world_lift_seconds")
                - manipulator_world_lift_before,
            )
            filter_result_manipulator_motion_seconds += max(
                0.0,
                _get_profile_stat(profile_stats, "filter_result_manipulator_motion_seconds")
                - manipulator_motion_before,
            )
            filter_result_manipulator_cluster_seconds += max(
                0.0,
                _get_profile_stat(profile_stats, "filter_result_manipulator_cluster_seconds")
                - manipulator_cluster_before,
            )
            traj_manipulator_cluster_fallback_used = np.asarray(fallback_used, dtype=bool)
            traj_pre_top95_mask = final_mask.copy()

            reason_bits[wrist_seed_mask & (~traj_near_depth_mask)] |= MASK_REASON_MANIPULATOR_DEPTH_FAIL
            reason_bits[wrist_seed_mask & (~traj_motion_mask)] |= MASK_REASON_MANIPULATOR_MOTION_FAIL
            reason_bits[traj_manipulator_candidate_mask & (~traj_cluster_mask)] |= MASK_REASON_MANIPULATOR_CLUSTER_FAIL
    elif profile == TRAJ_FILTER_PROFILE_EGOCENTRIC_OBJECT_INTERACTION_V1:
        traj_base_mask = wrist_base_mask.copy()
        required_prefix_frames = _resolve_support_frame_requirement(
            num_frames=num_frames,
            min_frames=config["egocentric_min_prefix_frames"],
            ratio=config["egocentric_prefix_ratio"],
        )
        required_support_frames = _resolve_support_frame_requirement(
            num_frames=num_frames,
            min_frames=config["egocentric_min_support_frames"],
            ratio=config["egocentric_support_ratio"],
        )
        supervision_support_mask = (
            supervision_prefix_len >= required_prefix_frames
        ) & (
            supervision_count >= required_support_frames
        )
        traj_supervision_support_mask = supervision_support_mask.copy()
        reason_bits[~traj_base_mask] |= MASK_REASON_BASE_GEOMETRY_FAIL
        reason_bits[~query_depth_quality_mask] |= MASK_REASON_QUERY_DEPTH_FAIL
        reason_bits[~supervision_support_mask] |= MASK_REASON_TEMPORAL_CONSISTENCY_FAIL
        wrist_seed_mask = traj_base_mask & query_depth_mask & supervision_support_mask

        raw_depths_segment, intrinsics_segment, extrinsics_segment = _require_segment_geometry(
            raw_depths_segment=raw_depths_segment,
            intrinsics_segment=intrinsics_segment,
            extrinsics_segment=extrinsics_segment,
            expected_num_frames=num_frames,
        )
        reusable_geometry: dict[str, np.ndarray] | None = {}
        manipulator_near_depth_before = _get_profile_stat(
            profile_stats, "filter_result_manipulator_near_depth_seconds"
        )
        manipulator_world_lift_before = _get_profile_stat(
            profile_stats, "filter_result_manipulator_world_lift_seconds"
        )
        manipulator_motion_before = _get_profile_stat(
            profile_stats, "filter_result_manipulator_motion_seconds"
        )
        manipulator_cluster_before = _get_profile_stat(
            profile_stats, "filter_result_manipulator_cluster_seconds"
        )
        (
            manipulator_final_mask,
            traj_query_depth_rank,
            traj_motion_extent,
            traj_motion_step_median,
            traj_motion_extent_all_valid,
            traj_motion_step_median_all_valid,
            traj_manipulator_candidate_mask,
            traj_manipulator_cluster_id,
            traj_manipulator_component_size,
            traj_near_depth_mask,
            traj_motion_mask,
            traj_cluster_mask,
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
            max_depth_rank=config["egocentric_manipulator_max_depth_rank"],
            min_motion_extent=config["egocentric_manipulator_min_motion_extent"],
            cluster_radius_ratio=config["egocentric_manipulator_cluster_radius_ratio"],
            cluster_radius_min_px=config["egocentric_manipulator_cluster_radius_min_px"],
            min_component_ratio=config["egocentric_manipulator_min_component_ratio"],
            min_component_size=config["egocentric_manipulator_min_component_size"],
            component_keep_mode="major",
            major_component_ratio=config["egocentric_manipulator_major_component_ratio"],
            motion_metric_mode="all_valid",
            reusable_geometry=reusable_geometry,
            profile_stats=profile_stats,
        )
        filter_result_manipulator_near_depth_seconds += max(
            0.0,
            _get_profile_stat(profile_stats, "filter_result_manipulator_near_depth_seconds")
            - manipulator_near_depth_before,
        )
        filter_result_manipulator_world_lift_seconds += max(
            0.0,
            _get_profile_stat(profile_stats, "filter_result_manipulator_world_lift_seconds")
            - manipulator_world_lift_before,
        )
        filter_result_manipulator_motion_seconds += max(
            0.0,
            _get_profile_stat(profile_stats, "filter_result_manipulator_motion_seconds")
            - manipulator_motion_before,
        )
        filter_result_manipulator_cluster_seconds += max(
            0.0,
            _get_profile_stat(profile_stats, "filter_result_manipulator_cluster_seconds")
            - manipulator_cluster_before,
        )
        traj_manipulator_cluster_fallback_used = np.asarray(fallback_used, dtype=bool)

        reason_bits[wrist_seed_mask & (~traj_near_depth_mask)] |= MASK_REASON_MANIPULATOR_DEPTH_FAIL
        reason_bits[wrist_seed_mask & (~traj_motion_mask)] |= MASK_REASON_MANIPULATOR_MOTION_FAIL
        reason_bits[traj_manipulator_candidate_mask & (~traj_cluster_mask)] |= MASK_REASON_MANIPULATOR_CLUSTER_FAIL

        pick_place_world_tracks = None if reusable_geometry is None else reusable_geometry.get("world_tracks")
        (
            traj_pick_place_delayed_contact_rescue_mask,
            traj_pick_place_min_manipulator_distance,
            traj_pick_place_contact_mask,
            traj_pick_place_depth_guard_mask,
            _traj_pick_place_delayed_contact_mask,
        ) = _apply_delayed_contact_object_rescue_filter(
            traj=traj,
            visibs=visibility,
            seed_mask=wrist_seed_mask,
            local_keep_mask=manipulator_final_mask,
            manipulator_reference_mask=manipulator_final_mask,
            manipulator_reference_component_ids=traj_manipulator_cluster_id,
            intrinsics_segment=intrinsics_segment,
            extrinsics_segment=extrinsics_segment,
            min_depth=config["min_depth"],
            max_depth=config["max_depth"],
            max_manipulator_distance_m=config["egocentric_object_max_manipulator_distance_m"],
            query_depth_margin_m=config["egocentric_object_query_depth_margin_m"],
            world_tracks=pick_place_world_tracks,
        )
        traj_pick_place_object_mask = traj_pick_place_delayed_contact_rescue_mask.copy()
        final_mask = manipulator_final_mask | traj_pick_place_object_mask
        traj_pre_top95_mask = final_mask.copy()
        if stereo_context is not None:
            final_mask = final_mask & stereo_consistency_mask
    else:
        wrist_profile_base_mask = wrist_base_mask
        if profile in {
            TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE,
            TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE_NO_HEATMAP,
        }:
            wrist_profile_base_mask = wrist_pick_place_base_mask
        traj_base_mask = wrist_profile_base_mask.copy()
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
        traj_supervision_support_mask = supervision_support_mask.copy()
        reason_bits[~wrist_profile_base_mask] |= MASK_REASON_BASE_GEOMETRY_FAIL
        reason_bits[~query_depth_quality_mask] |= MASK_REASON_QUERY_DEPTH_FAIL
        if ablation_mode != TRAJ_FILTER_ABLATION_MODE_WRIST_NO_QUERY_EDGE:
            reason_bits[traj_query_depth_edge_risk_mask] |= MASK_REASON_QUERY_DEPTH_EDGE_FAIL
        reason_bits[~supervision_support_mask] |= MASK_REASON_TEMPORAL_CONSISTENCY_FAIL
        wrist_seed_mask = wrist_profile_base_mask & query_depth_mask & supervision_support_mask

        if profile == TRAJ_FILTER_PROFILE_WRIST:
            final_mask = wrist_seed_mask
            traj_pre_top95_mask = final_mask.copy()
        else:
            raw_depths_segment, intrinsics_segment, extrinsics_segment = _require_segment_geometry(
                raw_depths_segment=raw_depths_segment,
                intrinsics_segment=intrinsics_segment,
                extrinsics_segment=extrinsics_segment,
                expected_num_frames=num_frames,
            )
            reusable_geometry = (
                {}
                if profile in {
                    TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE,
                    TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE_NO_HEATMAP,
                }
                else None
            )
            apply_near_depth_filter = (
                ablation_mode != TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_DEPTH
                and ablation_mode != TRAJ_FILTER_ABLATION_MODE_WRIST_SEED_TOP95
            )
            apply_motion_filter = (
                ablation_mode != TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_MOTION
                and ablation_mode != TRAJ_FILTER_ABLATION_MODE_WRIST_SEED_TOP95
            )
            apply_cluster_filter = (
                ablation_mode != TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_CLUSTER
                and ablation_mode != TRAJ_FILTER_ABLATION_MODE_WRIST_SEED_TOP95
            )
            if profile == TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE_NO_HEATMAP:
                apply_cluster_filter = False
            manipulator_near_depth_before = _get_profile_stat(
                profile_stats, "filter_result_manipulator_near_depth_seconds"
            )
            manipulator_world_lift_before = _get_profile_stat(
                profile_stats, "filter_result_manipulator_world_lift_seconds"
            )
            manipulator_motion_before = _get_profile_stat(
                profile_stats, "filter_result_manipulator_motion_seconds"
            )
            manipulator_cluster_before = _get_profile_stat(
                profile_stats, "filter_result_manipulator_cluster_seconds"
            )
            manipulator_filter_kwargs = {}
            if profile == TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE:
                manipulator_filter_kwargs.update(
                    {
                        "component_keep_mode": "major",
                        "major_component_ratio": config["wrist_pick_place_major_component_ratio"],
                        "major_component_min_motion_ratio": (
                            config["wrist_pick_place_major_component_min_motion_ratio"]
                        ),
                        "major_component_depth_margin_m": (
                            config["wrist_pick_place_major_component_depth_margin_m"]
                        ),
                    }
                )
            max_depth_rank = config["wrist_manipulator_max_depth_rank"]
            min_motion_extent = config["wrist_manipulator_min_motion_extent"]
            if profile == TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE_NO_HEATMAP:
                max_depth_rank = config["wrist_pick_place_no_heatmap_max_depth_rank"]
                min_motion_extent = config["wrist_pick_place_no_heatmap_anchor_min_motion_extent"]
            (
                manipulator_final_mask,
                traj_query_depth_rank,
                traj_motion_extent,
                traj_motion_step_median,
                traj_motion_extent_all_valid,
                traj_motion_step_median_all_valid,
                traj_manipulator_candidate_mask,
                traj_manipulator_cluster_id,
                traj_manipulator_component_size,
                traj_near_depth_mask,
                traj_motion_mask,
                traj_cluster_mask,
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
                max_depth_rank=max_depth_rank,
                min_motion_extent=min_motion_extent,
                cluster_radius_ratio=config["wrist_manipulator_cluster_radius_ratio"],
                cluster_radius_min_px=config["wrist_manipulator_cluster_radius_min_px"],
                min_component_ratio=config["wrist_manipulator_min_component_ratio"],
                min_component_size=config["wrist_manipulator_min_component_size"],
                motion_metric_mode="all_valid",
                apply_near_depth_filter=apply_near_depth_filter,
                apply_motion_filter=apply_motion_filter,
                apply_cluster_filter=apply_cluster_filter,
                reusable_geometry=reusable_geometry,
                profile_stats=profile_stats,
                **manipulator_filter_kwargs,
            )
            pick_place_world_tracks = None if reusable_geometry is None else reusable_geometry.get("world_tracks")
            filter_result_manipulator_near_depth_seconds += max(
                0.0,
                _get_profile_stat(profile_stats, "filter_result_manipulator_near_depth_seconds")
                - manipulator_near_depth_before,
            )
            filter_result_manipulator_world_lift_seconds += max(
                0.0,
                _get_profile_stat(profile_stats, "filter_result_manipulator_world_lift_seconds")
                - manipulator_world_lift_before,
            )
            filter_result_manipulator_motion_seconds += max(
                0.0,
                _get_profile_stat(profile_stats, "filter_result_manipulator_motion_seconds")
                - manipulator_motion_before,
            )
            filter_result_manipulator_cluster_seconds += max(
                0.0,
                _get_profile_stat(profile_stats, "filter_result_manipulator_cluster_seconds")
                - manipulator_cluster_before,
            )
            traj_manipulator_cluster_fallback_used = np.asarray(fallback_used, dtype=bool)

            if profile == TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE_NO_HEATMAP:
                traj_cluster_mask, region_fallback_used = _build_anchor_query_region_mask(
                    keypoints=keypoints,
                    anchor_mask=traj_manipulator_candidate_mask,
                    min_anchor_count=config["wrist_pick_place_no_heatmap_min_anchor_count"],
                    bbox_x_pad_px=config["wrist_pick_place_no_heatmap_bbox_x_pad_px"],
                    bbox_y_pad_up_px=config["wrist_pick_place_no_heatmap_bbox_y_pad_up_px"],
                    bbox_y_pad_down_px=config["wrist_pick_place_no_heatmap_bbox_y_pad_down_px"],
                )
                local_keep_mask = traj_near_depth_mask & traj_cluster_mask
                final_mask = local_keep_mask.copy()
                traj_manipulator_cluster_fallback_used = np.asarray(region_fallback_used, dtype=bool)
                reason_bits[wrist_seed_mask & (~traj_near_depth_mask)] |= MASK_REASON_MANIPULATOR_DEPTH_FAIL
                reason_bits[traj_near_depth_mask & (~traj_cluster_mask)] |= MASK_REASON_MANIPULATOR_CLUSTER_FAIL
            else:
                if apply_near_depth_filter:
                    reason_bits[wrist_seed_mask & (~traj_near_depth_mask)] |= MASK_REASON_MANIPULATOR_DEPTH_FAIL
                if apply_motion_filter:
                    reason_bits[wrist_seed_mask & (~traj_motion_mask)] |= MASK_REASON_MANIPULATOR_MOTION_FAIL
                if apply_cluster_filter:
                    reason_bits[traj_manipulator_candidate_mask & (~traj_cluster_mask)] |= (
                        MASK_REASON_MANIPULATOR_CLUSTER_FAIL
                    )

            if profile == TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE:
                (
                    traj_pick_place_object_mask,
                    traj_pick_place_heatmap_hit_count,
                    traj_pick_place_heatmap_support_mask,
                    traj_pick_place_min_manipulator_distance,
                    traj_pick_place_contact_mask,
                    traj_pick_place_depth_guard_mask,
                ) = _apply_pick_place_object_filter(
                    traj=traj,
                    seed_mask=wrist_seed_mask,
                    manipulator_reference_mask=manipulator_final_mask,
                    manipulator_reference_component_ids=traj_manipulator_cluster_id,
                    intrinsics_segment=intrinsics_segment,
                    extrinsics_segment=extrinsics_segment,
                    min_depth=config["min_depth"],
                    max_depth=config["max_depth"],
                    pick_place_heatmap_segment=pick_place_heatmap_segment,
                    min_heatmap_hits=config["wrist_pick_place_min_heatmap_hits"],
                    max_manipulator_distance_m=config["wrist_pick_place_max_manipulator_distance_m"],
                    query_depth_margin_m=config["wrist_pick_place_query_depth_margin_m"],
                    world_tracks=pick_place_world_tracks,
                )
                final_mask = manipulator_final_mask | traj_pick_place_object_mask
                traj_pre_top95_mask = final_mask.copy()
            elif profile == TRAJ_FILTER_PROFILE_WRIST_PICK_PLACE_NO_HEATMAP:
                (
                    traj_pick_place_delayed_contact_rescue_mask,
                    traj_pick_place_min_manipulator_distance,
                    traj_pick_place_contact_mask,
                    traj_pick_place_depth_guard_mask,
                    _traj_pick_place_delayed_contact_mask,
                ) = _apply_delayed_contact_object_rescue_filter(
                    traj=traj,
                    visibs=visibility,
                    seed_mask=wrist_seed_mask,
                    local_keep_mask=final_mask,
                    manipulator_reference_mask=traj_manipulator_candidate_mask,
                    manipulator_reference_component_ids=traj_manipulator_cluster_id,
                    intrinsics_segment=intrinsics_segment,
                    extrinsics_segment=extrinsics_segment,
                    min_depth=config["min_depth"],
                    max_depth=config["max_depth"],
                    max_manipulator_distance_m=config["wrist_pick_place_max_manipulator_distance_m"],
                    query_depth_margin_m=config["wrist_pick_place_query_depth_margin_m"],
                    world_tracks=pick_place_world_tracks,
                )
                traj_pick_place_object_mask = traj_pick_place_delayed_contact_rescue_mask.copy()
                final_mask = final_mask | traj_pick_place_delayed_contact_rescue_mask
                traj_pre_top95_mask = final_mask.copy()
            elif ablation_mode == TRAJ_FILTER_ABLATION_MODE_WRIST_SEED_TOP95:
                traj_pre_top95_mask = wrist_seed_mask.copy()
                top95_start = time.perf_counter()
                final_mask = _apply_top_motion_extent_filter(
                    seed_mask=wrist_seed_mask,
                    motion_extent=traj_motion_extent_all_valid,
                    keep_ratio=config["wrist_manipulator_top95_keep_ratio"],
                )
                filter_result_top95_seconds = time.perf_counter() - top95_start
                _accumulate_profile_stat(
                    profile_stats,
                    "filter_result_top95_seconds",
                    filter_result_top95_seconds,
                )
            elif profile == TRAJ_FILTER_PROFILE_WRIST_MANIPULATOR_TOP95:
                traj_pre_top95_mask = manipulator_final_mask.copy()
                top95_start = time.perf_counter()
                final_mask = _apply_top_motion_extent_filter(
                    seed_mask=traj_pre_top95_mask,
                    motion_extent=traj_motion_extent_all_valid,
                    keep_ratio=config["wrist_manipulator_top95_keep_ratio"],
                )
                filter_result_top95_seconds = time.perf_counter() - top95_start
                _accumulate_profile_stat(
                    profile_stats,
                    "filter_result_top95_seconds",
                    filter_result_top95_seconds,
                )
            else:
                final_mask = manipulator_final_mask
                traj_pre_top95_mask = final_mask.copy()

    reason_bits[final_mask] = 0

    result = {
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
        "traj_motion_extent_all_valid": traj_motion_extent_all_valid.astype(np.float32),
        "traj_motion_step_median_all_valid": traj_motion_step_median_all_valid.astype(np.float32),
        "traj_manipulator_candidate_mask": traj_manipulator_candidate_mask.astype(bool),
        "traj_manipulator_cluster_id": traj_manipulator_cluster_id.astype(np.int16),
        "traj_manipulator_component_size": traj_manipulator_component_size.astype(np.uint16),
        "traj_manipulator_cluster_fallback_used": np.asarray(
            traj_manipulator_cluster_fallback_used, dtype=bool
        ),
        "traj_query_depth_edge_mask": traj_query_depth_edge_mask.astype(bool),
        "traj_query_depth_patch_valid_ratio": traj_query_depth_patch_valid_ratio.astype(np.float32),
        "traj_query_depth_patch_std": traj_query_depth_patch_std.astype(np.float32),
        "traj_query_depth_edge_risk_mask": traj_query_depth_edge_risk_mask.astype(bool),
        "traj_base_mask": traj_base_mask.astype(bool),
        "traj_query_depth_quality_mask": query_depth_quality_mask.astype(bool),
        "traj_query_depth_keep_mask": traj_query_depth_keep_mask.astype(bool),
        "traj_supervision_support_mask": traj_supervision_support_mask.astype(bool),
        "traj_near_depth_mask": traj_near_depth_mask.astype(bool),
        "traj_motion_mask": traj_motion_mask.astype(bool),
        "traj_cluster_mask": traj_cluster_mask.astype(bool),
        "traj_pre_top95_mask": traj_pre_top95_mask.astype(bool),
        "traj_pick_place_heatmap_hit_count": traj_pick_place_heatmap_hit_count.astype(np.uint16),
        "traj_pick_place_heatmap_support_mask": traj_pick_place_heatmap_support_mask.astype(bool),
        "traj_pick_place_min_manipulator_distance": traj_pick_place_min_manipulator_distance.astype(np.float32),
        "traj_pick_place_contact_mask": traj_pick_place_contact_mask.astype(bool),
        "traj_pick_place_depth_guard_mask": traj_pick_place_depth_guard_mask.astype(bool),
        "traj_pick_place_delayed_contact_rescue_mask": traj_pick_place_delayed_contact_rescue_mask.astype(bool),
        "traj_pick_place_object_mask": traj_pick_place_object_mask.astype(bool),
        "traj_stereo_compare_frame_count": stereo_compare_frame_count.astype(np.uint16),
        "traj_stereo_depth_consistency_ratio": stereo_depth_consistency_ratio.astype(np.float32),
        "traj_stereo_patch_error": stereo_patch_error.astype(np.float32),
        "traj_stereo_consistency_mask": stereo_consistency_mask.astype(bool),
    }
    filter_result_total_seconds = time.perf_counter() - filter_result_start
    _accumulate_profile_stat(profile_stats, "filter_result_total_seconds", filter_result_total_seconds)
    filter_result_explicit_seconds = (
        filter_result_base_geometry_seconds
        + filter_result_query_depth_patch_stats_seconds
        + filter_result_query_depth_quality_seconds
        + filter_result_query_depth_edge_risk_seconds
        + filter_result_temporal_seconds
        + filter_result_manipulator_near_depth_seconds
        + filter_result_manipulator_world_lift_seconds
        + filter_result_manipulator_motion_seconds
        + filter_result_manipulator_cluster_seconds
        + filter_result_top95_seconds
    )
    _accumulate_profile_stat(
        profile_stats,
        "filter_result_other_seconds",
        max(0.0, filter_result_total_seconds - filter_result_explicit_seconds),
    )
    return result


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
    high_volatility_mask: np.ndarray | None = None,
    depth_volatility_map: np.ndarray | None = None,
    temporal_compare_context: dict[str, np.ndarray | int] | None = None,
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
        high_volatility_mask=high_volatility_mask,
        depth_volatility_map=depth_volatility_map,
        temporal_compare_context=temporal_compare_context,
    )["traj_valid_mask"]
