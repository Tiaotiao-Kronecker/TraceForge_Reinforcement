from types import SimpleNamespace
import unittest
import warnings

import numpy as np

from utils.traj_filter_utils import (
    DEFAULT_QUERY_PREFILTER_MODE,
    QUERY_PREFILTER_MODE_PROFILE_AWARE_STATIC_V1,
    MASK_REASON_MANIPULATOR_CLUSTER_FAIL,
    MASK_REASON_MANIPULATOR_DEPTH_FAIL,
    MASK_REASON_MANIPULATOR_MOTION_FAIL,
    MASK_REASON_QUERY_DEPTH_EDGE_FAIL,
    MASK_REASON_QUERY_DEPTH_FAIL,
    MASK_REASON_STABLE_TEMPORAL_FAIL,
    MASK_REASON_TEMPORAL_CONSISTENCY_FAIL,
    TRAJ_FILTER_ABLATION_MODE_NONE,
    TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_CLUSTER,
    TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_DEPTH,
    TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_MOTION,
    TRAJ_FILTER_ABLATION_MODE_WRIST_NO_QUERY_EDGE,
    TRAJ_FILTER_ABLATION_MODE_WRIST_SEED_TOP95,
    _compute_motion_metrics_for_valid_masks,
    _apply_manipulator_aware_filter,
    _apply_pick_place_object_filter,
    _compute_linear_percentiles_for_masked_columns,
    _compute_query_depth_patch_stats,
    build_traj_filter_result,
    build_query_prefilter_result,
    build_traj_valid_mask,
    compute_traj_base_geometry,
    compute_accessed_high_volatility_mask,
    compute_depth_volatility_map,
    compute_query_depth_quality_mask,
    is_tail_truncated_sample,
    prepare_temporal_depth_consistency_context,
)


def _make_identity_intrinsics(num_frames: int) -> np.ndarray:
    return np.repeat(np.eye(3, dtype=np.float32)[None], num_frames, axis=0)


def _make_scaled_intrinsics(
    num_frames: int,
    *,
    fx: float = 100.0,
    fy: float = 100.0,
    cx: float = 0.0,
    cy: float = 0.0,
) -> np.ndarray:
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], num_frames, axis=0)
    intrinsics[:, 0, 0] = float(fx)
    intrinsics[:, 1, 1] = float(fy)
    intrinsics[:, 0, 2] = float(cx)
    intrinsics[:, 1, 2] = float(cy)
    return intrinsics


def _make_identity_extrinsics(num_frames: int) -> np.ndarray:
    return np.repeat(np.eye(4, dtype=np.float32)[None], num_frames, axis=0)


def _make_track(
    *,
    u_values: list[float] | np.ndarray,
    v: float = 2.0,
    depth: float = 1.0,
) -> np.ndarray:
    u_values = np.asarray(u_values, dtype=np.float32)
    num_frames = int(u_values.shape[0])
    traj = np.zeros((1, num_frames, 3), dtype=np.float32)
    traj[0, :, 0] = u_values
    traj[0, :, 1] = v
    traj[0, :, 2] = depth
    return traj


def _make_base_fixture(
    *,
    u_values: list[float] | np.ndarray | None = None,
    height: int = 8,
    width: int = 8,
) -> dict[str, np.ndarray]:
    if u_values is None:
        u_values = [2.0, 2.0, 2.0, 2.0]
    traj = _make_track(u_values=u_values)
    num_frames = int(traj.shape[1])
    keypoints = np.array([[float(traj[0, 0, 0]), float(traj[0, 0, 1])]], dtype=np.float32)
    query_depth = np.ones((height, width), dtype=np.float32)
    raw_depths_segment = np.ones((num_frames, height, width), dtype=np.float32)
    intrinsics_segment = _make_identity_intrinsics(num_frames)
    extrinsics_segment = _make_identity_extrinsics(num_frames)
    depth_volatility_map = np.zeros((height, width), dtype=np.float32)
    visibs = np.ones((1, num_frames), dtype=bool)
    return {
        "traj": traj,
        "keypoints": keypoints,
        "query_depth": query_depth,
        "raw_depths_segment": raw_depths_segment,
        "intrinsics_segment": intrinsics_segment,
        "extrinsics_segment": extrinsics_segment,
        "depth_volatility_map": depth_volatility_map,
        "visibs": visibs,
        "image_width": width,
        "image_height": height,
    }


def _paint_patch(image: np.ndarray, *, x: float, y: float, value: float, radius: int = 2) -> None:
    height, width = image.shape
    x_coord = int(np.clip(np.round(x), 0, width - 1))
    y_coord = int(np.clip(np.round(y), 0, height - 1))
    y0 = max(0, y_coord - radius)
    y1 = min(height, y_coord + radius + 1)
    x0 = max(0, x_coord - radius)
    x1 = min(width, x_coord + radius + 1)
    image[y0:y1, x0:x1] = float(value)


def _make_multi_track_fixture(
    *,
    traj: np.ndarray,
    keypoints: np.ndarray | None = None,
    height: int = 64,
    width: int = 64,
    visibs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 3 or traj.shape[-1] != 3:
        raise ValueError(f"Expected traj shape (N,T,3), got {traj.shape}")

    num_tracks, num_frames, _ = traj.shape
    if keypoints is None:
        keypoints = traj[:, 0, :2].astype(np.float32, copy=True)
    else:
        keypoints = np.asarray(keypoints, dtype=np.float32)

    query_depth = np.ones((height, width), dtype=np.float32)
    raw_depths_segment = np.ones((num_frames, height, width), dtype=np.float32)
    for track_idx in range(num_tracks):
        _paint_patch(
            query_depth,
            x=float(keypoints[track_idx, 0]),
            y=float(keypoints[track_idx, 1]),
            value=float(traj[track_idx, 0, 2]),
            radius=2,
        )
        for frame_idx in range(num_frames):
            if not np.isfinite(traj[track_idx, frame_idx]).all():
                continue
            _paint_patch(
                raw_depths_segment[frame_idx],
                x=float(traj[track_idx, frame_idx, 0]),
                y=float(traj[track_idx, frame_idx, 1]),
                value=float(traj[track_idx, frame_idx, 2]),
                radius=0,
            )

    if visibs is None:
        visibs = np.ones((num_tracks, num_frames), dtype=bool)
    else:
        visibs = np.asarray(visibs, dtype=bool)

    return {
        "traj": traj,
        "keypoints": keypoints,
        "query_depth": query_depth,
        "raw_depths_segment": raw_depths_segment,
        "intrinsics_segment": _make_identity_intrinsics(num_frames),
        "extrinsics_segment": _make_identity_extrinsics(num_frames),
        "depth_volatility_map": np.zeros((height, width), dtype=np.float32),
        "visibs": visibs,
        "image_width": width,
        "image_height": height,
    }


def _make_filter_args(**overrides) -> SimpleNamespace:
    values = {
        "filter_level": "basic",
        "traj_filter_profile": "external",
        "min_valid_frames": None,
        "visibility_threshold": None,
        "min_depth": 0.01,
        "max_depth": 10.0,
        "boundary_margin": None,
        "depth_change_threshold": None,
        "temporal_min_consistency_ratio": None,
        "volatility_mask_percentile": None,
        "traj_filter_ablation_mode": TRAJ_FILTER_ABLATION_MODE_NONE,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class QueryDepthQualityMaskTests(unittest.TestCase):
    def test_keeps_consistent_query_depth_at_image_boundary(self):
        keypoints = np.array([[0.0, 0.0]], dtype=np.float32)
        query_depth = np.ones((6, 6), dtype=np.float32)

        mask = compute_query_depth_quality_mask(
            keypoints,
            query_depth,
            min_depth=0.01,
            max_depth=10.0,
        )

        np.testing.assert_array_equal(mask, np.array([True]))

    def test_rejects_invalid_query_depth_value(self):
        keypoints = np.array([[2.0, 2.0]], dtype=np.float32)
        query_depth = np.ones((6, 6), dtype=np.float32)
        query_depth[2, 2] = 0.0

        mask = compute_query_depth_quality_mask(
            keypoints,
            query_depth,
            min_depth=0.01,
            max_depth=10.0,
        )

        np.testing.assert_array_equal(mask, np.array([False]))


class TailTruncationTests(unittest.TestCase):
    def test_identifies_tail_truncated_sample_from_future_len(self):
        self.assertTrue(is_tail_truncated_sample(num_frames=4, future_len=6))
        self.assertFalse(is_tail_truncated_sample(num_frames=4, future_len=4))
        self.assertFalse(is_tail_truncated_sample(num_frames=4, future_len=None))

    def test_tail_truncated_external_sample_ignores_visibility_hard_gates(self):
        fixture = _make_base_fixture(u_values=[2.0, 2.0, 2.0, 2.0])
        fixture["visibs"][0] = np.array([True, False, False, False], dtype=bool)

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                filter_level="standard",
                traj_filter_profile="external",
                future_len=6,
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_base_mask"], np.array([True]))
        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True]))
        np.testing.assert_array_equal(result["traj_compare_frame_count"], np.array([4], dtype=np.uint16))
        np.testing.assert_array_equal(result["traj_supervision_mask"], np.array([[True, True, True, True]]))

    def test_non_tail_external_sample_still_uses_visibility_hard_gates(self):
        fixture = _make_base_fixture(u_values=[2.0, 2.0, 2.0, 2.0])
        fixture["visibs"][0] = np.array([True, False, False, False], dtype=bool)

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                filter_level="standard",
                traj_filter_profile="external",
                future_len=4,
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_base_mask"], np.array([False]))
        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([False]))
        np.testing.assert_array_equal(result["traj_compare_frame_count"], np.array([1], dtype=np.uint16))

    def test_tail_truncated_precomputed_context_matches_direct_path_with_visibility_relaxation(self):
        fixture = _make_base_fixture(u_values=[2.0, 2.0, 2.0, 2.0])
        fixture["visibs"][0] = np.array([True, False, False, False], dtype=bool)

        precomputed_context = prepare_temporal_depth_consistency_context(
            fixture["traj"],
            visibs=None,
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            min_depth=0.01,
            max_depth=10.0,
        )
        precomputed_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                filter_level="standard",
                traj_filter_profile="external",
                future_len=6,
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
            temporal_compare_context=precomputed_context,
        )
        direct_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                filter_level="standard",
                traj_filter_profile="external",
                future_len=6,
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(precomputed_result["traj_valid_mask"], direct_result["traj_valid_mask"])
        np.testing.assert_array_equal(
            precomputed_result["traj_compare_frame_count"],
            direct_result["traj_compare_frame_count"],
        )
        np.testing.assert_array_equal(precomputed_result["traj_supervision_mask"], direct_result["traj_supervision_mask"])

    def test_rejects_local_depth_outlier(self):
        keypoints = np.array([[2.0, 2.0]], dtype=np.float32)
        query_depth = np.ones((6, 6), dtype=np.float32)
        query_depth[2, 2] = 2.0

        mask = compute_query_depth_quality_mask(
            keypoints,
            query_depth,
            min_depth=0.01,
            max_depth=10.0,
        )

        np.testing.assert_array_equal(mask, np.array([False]))

    def test_rejects_patch_with_too_few_valid_neighbors(self):
        keypoints = np.array([[2.0, 2.0]], dtype=np.float32)
        query_depth = np.zeros((6, 6), dtype=np.float32)
        query_depth[2, 2] = 1.0

        mask = compute_query_depth_quality_mask(
            keypoints,
            query_depth,
            min_depth=0.01,
            max_depth=10.0,
        )

        np.testing.assert_array_equal(mask, np.array([False]))


class PatchStatsAndBaseGeometryRegressionTests(unittest.TestCase):
    def test_query_depth_patch_stats_uses_clipped_patch_area_at_boundary(self):
        keypoints = np.array([[0.0, 0.0]], dtype=np.float32)
        query_depth = np.ones((6, 6), dtype=np.float32)

        stats = _compute_query_depth_patch_stats(
            keypoints,
            query_depth,
            min_depth=0.01,
            max_depth=10.0,
        )

        self.assertAlmostEqual(float(stats["patch_valid_ratio"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(stats["patch_median"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(stats["patch_std"][0]), 0.0, places=6)

    def test_compute_traj_base_geometry_compresses_valid_depth_sequence_for_smoothness(self):
        traj = np.array(
            [
                [
                    [2.0, 2.0, 1.0],
                    [np.nan, np.nan, np.nan],
                    [2.0, 2.0, 2.0],
                    [2.0, 2.0, 3.0],
                ]
            ],
            dtype=np.float32,
        )

        result = compute_traj_base_geometry(
            traj,
            image_width=8,
            image_height=8,
            min_valid_frames=3,
            min_depth=0.01,
            max_depth=10.0,
            boundary_margin=0,
            visibility_threshold=0.0,
            check_depth_smoothness=True,
            depth_change_threshold=0.1,
        )

        np.testing.assert_array_equal(result["valid_count_mask"], np.array([True]))
        np.testing.assert_array_equal(result["depth_smooth_mask"], np.array([True]))
        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True]))


class QueryPrefilterTests(unittest.TestCase):
    def test_off_mode_bypasses_prefilter(self):
        keypoints = np.array([[2.0, 2.0]], dtype=np.float32)
        query_depth = np.zeros((6, 6), dtype=np.float32)

        result = build_query_prefilter_result(
            keypoints,
            query_depth,
            filter_args=SimpleNamespace(
                filter_level="basic",
                traj_filter_profile="wrist_manipulator_top95",
                query_prefilter_mode=DEFAULT_QUERY_PREFILTER_MODE,
            ),
        )

        np.testing.assert_array_equal(result["prefilter_mask"], np.array([True]))
        np.testing.assert_array_equal(result["reason_bits"], np.array([0], dtype=np.uint8))
        self.assertTrue(np.isnan(result["query_depth_rank"]).all())

    def test_external_profile_keeps_dense_grid_under_profile_aware_mode(self):
        keypoints = np.array([[2.0, 2.0], [4.0, 4.0]], dtype=np.float32)
        query_depth = np.ones((8, 8), dtype=np.float32)
        query_depth[2, 2] = 0.0

        result = build_query_prefilter_result(
            keypoints,
            query_depth,
            filter_args=SimpleNamespace(
                filter_level="basic",
                traj_filter_profile="external",
                query_prefilter_mode=QUERY_PREFILTER_MODE_PROFILE_AWARE_STATIC_V1,
            ),
        )

        np.testing.assert_array_equal(result["prefilter_mask"], np.array([True, True]))
        np.testing.assert_array_equal(result["reason_bits"], np.array([0, 0], dtype=np.uint8))
        self.assertTrue(np.isnan(result["query_depth_patch_valid_ratio"]).all())

    def test_wrist_profile_prefilter_marks_query_depth_and_edge_failures(self):
        keypoints = np.array(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [5.0, 5.0],
            ],
            dtype=np.float32,
        )
        query_depth = np.ones((8, 8), dtype=np.float32)
        query_depth[:, :2] = 0.05
        query_depth[1, 1] = 0.0

        result = build_query_prefilter_result(
            keypoints,
            query_depth,
            filter_args=SimpleNamespace(
                filter_level="basic",
                traj_filter_profile="wrist",
                query_prefilter_mode=QUERY_PREFILTER_MODE_PROFILE_AWARE_STATIC_V1,
            ),
        )

        np.testing.assert_array_equal(result["prefilter_mask"], np.array([False, False, True]))
        self.assertTrue(bool(result["reason_bits"][0] & MASK_REASON_QUERY_DEPTH_FAIL))
        self.assertTrue(bool(result["reason_bits"][1] & MASK_REASON_QUERY_DEPTH_EDGE_FAIL))
        np.testing.assert_array_equal(result["query_depth_edge_risk_mask"], np.array([False, True, False]))

    def test_wrist_manipulator_prefilter_keeps_nearest_depth_rank_slice(self):
        keypoints = np.array(
            [
                [5.0, 5.0],
                [15.0, 5.0],
                [25.0, 5.0],
                [35.0, 5.0],
            ],
            dtype=np.float32,
        )
        query_depth = np.ones((48, 48), dtype=np.float32)
        for keypoint, value in zip(keypoints, [0.2, 0.4, 0.6, 0.8]):
            _paint_patch(query_depth, x=float(keypoint[0]), y=float(keypoint[1]), value=float(value), radius=2)

        result = build_query_prefilter_result(
            keypoints,
            query_depth,
            filter_args=SimpleNamespace(
                filter_level="basic",
                traj_filter_profile="wrist_manipulator_top95",
                query_prefilter_mode=QUERY_PREFILTER_MODE_PROFILE_AWARE_STATIC_V1,
                query_prefilter_wrist_rank_keep_ratio=0.40,
            ),
            wrist_rank_keep_ratio=0.40,
        )

        np.testing.assert_array_equal(result["prefilter_mask"], np.array([True, True, False, False]))
        self.assertAlmostEqual(float(result["query_depth_rank"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(result["query_depth_rank"][1]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(result["query_depth_rank"][2]), 2.0 / 3.0, places=6)
        self.assertTrue(bool(result["reason_bits"][2] & MASK_REASON_MANIPULATOR_DEPTH_FAIL))
        self.assertTrue(bool(result["reason_bits"][3] & MASK_REASON_MANIPULATOR_DEPTH_FAIL))


class BuildTrajValidMaskTests(unittest.TestCase):
    def test_none_level_bypasses_temporal_and_volatility_checks(self):
        traj = np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.1], [1.0, 1.0, 1.2]],
                [[2.0, 2.0, 1.0], [2.0, 2.0, 1.1], [2.0, 2.0, 1.2]],
            ],
            dtype=np.float32,
        )

        result = build_traj_filter_result(
            traj=traj,
            visibs=None,
            image_width=10,
            image_height=10,
            filter_args=SimpleNamespace(filter_level="none"),
        )

        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True, True]))
        self.assertTrue(np.isnan(result["traj_depth_consistency_ratio"]).all())
        self.assertTrue(np.isnan(result["traj_stable_depth_consistency_ratio"]).all())
        self.assertTrue(np.isnan(result["traj_volatility_exposure_ratio"]).all())
        np.testing.assert_array_equal(result["traj_high_volatility_hit"], np.array([False, False]))
        np.testing.assert_array_equal(result["traj_compare_frame_count"], np.array([0, 0], dtype=np.uint16))
        np.testing.assert_array_equal(result["traj_stable_compare_frame_count"], np.array([0, 0], dtype=np.uint16))
        np.testing.assert_array_equal(result["traj_mask_reason_bits"], np.array([0, 0], dtype=np.uint8))

    def test_basic_level_rejects_query_depth_outlier(self):
        fixture = _make_base_fixture()
        fixture["query_depth"][2, 2] = 2.0

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([False]))
        self.assertEqual(int(result["traj_mask_reason_bits"][0]), int(MASK_REASON_QUERY_DEPTH_FAIL))

    def test_stable_track_passes_full_filter(self):
        fixture = _make_base_fixture()

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=SimpleNamespace(filter_level="basic"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True]))
        self.assertAlmostEqual(float(result["traj_depth_consistency_ratio"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(result["traj_stable_depth_consistency_ratio"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(result["traj_volatility_exposure_ratio"][0]), 0.0, places=6)
        np.testing.assert_array_equal(result["traj_high_volatility_hit"], np.array([False]))
        np.testing.assert_array_equal(result["traj_compare_frame_count"], np.array([4], dtype=np.uint16))
        np.testing.assert_array_equal(result["traj_stable_compare_frame_count"], np.array([4], dtype=np.uint16))
        np.testing.assert_array_equal(result["traj_supervision_mask"], np.array([[True, True, True, True]]))
        np.testing.assert_array_equal(result["traj_supervision_prefix_len"], np.array([4], dtype=np.uint16))
        np.testing.assert_array_equal(result["traj_supervision_count"], np.array([4], dtype=np.uint16))
        self.assertEqual(int(result["traj_mask_reason_bits"][0]), 0)

    def test_external_profile_populates_timing_profile_stats(self):
        fixture = _make_base_fixture()
        profile_stats: dict[str, float] = {}

        build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="external"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
            profile_stats=profile_stats,
        )

        expected_keys = {
            "filter_result_total_seconds",
            "filter_result_other_seconds",
            "filter_result_base_geometry_seconds",
            "filter_result_query_depth_patch_stats_seconds",
            "filter_result_query_depth_quality_seconds",
            "filter_result_temporal_seconds",
        }
        self.assertTrue(expected_keys.issubset(profile_stats))
        for key in expected_keys:
            self.assertGreaterEqual(float(profile_stats[key]), 0.0)
        self.assertNotIn("filter_result_top95_seconds", profile_stats)

    def test_wrist_manipulator_top95_populates_timing_profile_stats(self):
        fixture = _make_base_fixture(u_values=[2.0, 2.5, 3.0, 3.5], width=12, height=12)
        profile_stats: dict[str, float] = {}

        build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                boundary_margin=0,
                traj_filter_profile="wrist_manipulator_top95",
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
            profile_stats=profile_stats,
        )

        expected_keys = {
            "filter_result_total_seconds",
            "filter_result_other_seconds",
            "filter_result_base_geometry_seconds",
            "filter_result_query_depth_patch_stats_seconds",
            "filter_result_query_depth_quality_seconds",
            "filter_result_query_depth_edge_risk_seconds",
            "filter_result_temporal_seconds",
            "filter_result_manipulator_near_depth_seconds",
            "filter_result_manipulator_world_lift_seconds",
            "filter_result_manipulator_motion_seconds",
            "filter_result_manipulator_cluster_seconds",
            "filter_result_top95_seconds",
        }
        self.assertTrue(expected_keys.issubset(profile_stats))
        for key in expected_keys:
            self.assertGreaterEqual(float(profile_stats[key]), 0.0)

    def test_high_volatility_only_does_not_veto_track(self):
        fixture = _make_base_fixture()
        fixture["depth_volatility_map"][2, 2] = 10.0

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=SimpleNamespace(filter_level="basic"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True]))
        np.testing.assert_array_equal(result["traj_high_volatility_hit"], np.array([True]))
        self.assertAlmostEqual(float(result["traj_depth_consistency_ratio"][0]), 1.0, places=6)
        self.assertTrue(np.isnan(result["traj_stable_depth_consistency_ratio"][0]))
        self.assertAlmostEqual(float(result["traj_volatility_exposure_ratio"][0]), 1.0, places=6)
        np.testing.assert_array_equal(result["traj_compare_frame_count"], np.array([4], dtype=np.uint16))
        np.testing.assert_array_equal(result["traj_stable_compare_frame_count"], np.array([0], dtype=np.uint16))
        self.assertEqual(int(result["traj_mask_reason_bits"][0]), 0)

    def test_stable_frame_consistency_can_override_bad_volatile_frames(self):
        fixture = _make_base_fixture(u_values=[1.0, 2.0, 3.0, 5.0])
        fixture["depth_volatility_map"][2, 5] = 10.0
        fixture["raw_depths_segment"][3, 2, 5] = 2.0

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=SimpleNamespace(filter_level="basic"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True]))
        self.assertAlmostEqual(float(result["traj_depth_consistency_ratio"][0]), 0.75, places=6)
        self.assertAlmostEqual(float(result["traj_stable_depth_consistency_ratio"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(result["traj_volatility_exposure_ratio"][0]), 0.25, places=6)
        np.testing.assert_array_equal(result["traj_stable_compare_frame_count"], np.array([3], dtype=np.uint16))
        self.assertEqual(int(result["traj_mask_reason_bits"][0]), 0)

    def test_stable_frame_failure_filters_track_when_stable_frames_are_sufficient(self):
        fixture = _make_base_fixture(u_values=[1.0, 2.0, 3.0, 5.0])
        fixture["depth_volatility_map"][2, 5] = 10.0
        fixture["raw_depths_segment"][1, 2, 2] = 2.0

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=SimpleNamespace(filter_level="basic"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([False]))
        self.assertAlmostEqual(float(result["traj_depth_consistency_ratio"][0]), 0.75, places=6)
        self.assertAlmostEqual(float(result["traj_stable_depth_consistency_ratio"][0]), 2.0 / 3.0, places=5)
        self.assertTrue(result["traj_mask_reason_bits"][0] & MASK_REASON_TEMPORAL_CONSISTENCY_FAIL)
        self.assertTrue(result["traj_mask_reason_bits"][0] & MASK_REASON_STABLE_TEMPORAL_FAIL)

    def test_falls_back_to_all_frame_consistency_when_stable_frames_are_insufficient(self):
        fixture = _make_base_fixture(u_values=[1.0, 2.0, 5.0, 6.0])
        fixture["depth_volatility_map"][2, 5] = 10.0
        fixture["depth_volatility_map"][2, 6] = 10.0
        fixture["raw_depths_segment"][3, 2, 6] = 2.0

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=SimpleNamespace(filter_level="basic"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([False]))
        self.assertAlmostEqual(float(result["traj_depth_consistency_ratio"][0]), 0.75, places=6)
        self.assertTrue(np.isnan(result["traj_stable_depth_consistency_ratio"][0]) or float(result["traj_stable_depth_consistency_ratio"][0]) == 1.0)
        np.testing.assert_array_equal(result["traj_stable_compare_frame_count"], np.array([2], dtype=np.uint16))
        self.assertTrue(result["traj_mask_reason_bits"][0] & MASK_REASON_TEMPORAL_CONSISTENCY_FAIL)
        self.assertFalse(bool(result["traj_mask_reason_bits"][0] & MASK_REASON_STABLE_TEMPORAL_FAIL))

    def test_too_few_comparable_frames_filters_track(self):
        fixture = _make_base_fixture()
        fixture["raw_depths_segment"][1:, 2, 2] = 0.0
        fixture["raw_depths_segment"][3, 2, 2] = 1.0

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=SimpleNamespace(filter_level="basic"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([False]))
        self.assertAlmostEqual(float(result["traj_depth_consistency_ratio"][0]), 1.0, places=6)
        self.assertTrue(result["traj_mask_reason_bits"][0] & MASK_REASON_TEMPORAL_CONSISTENCY_FAIL)

    def test_visibility_false_frames_are_excluded_from_temporal_check(self):
        fixture = _make_base_fixture()
        fixture["raw_depths_segment"][3, 2, 2] = 2.0
        fixture["visibs"][0, 3] = False

        mask = build_traj_valid_mask(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(mask, np.array([True]))

    def test_wrist_profile_keeps_track_with_supported_prefix_even_if_tail_leaves_query_view(self):
        fixture = _make_base_fixture(u_values=[2.0, 2.0, 2.0, 20.0], width=8)

        external_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="external"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )
        np.testing.assert_array_equal(external_result["traj_valid_mask"], np.array([False]))

        wrist_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(wrist_result["traj_valid_mask"], np.array([True]))
        np.testing.assert_array_equal(wrist_result["traj_supervision_mask"], np.array([[True, True, True, False]]))
        np.testing.assert_array_equal(wrist_result["traj_supervision_prefix_len"], np.array([3], dtype=np.uint16))
        np.testing.assert_array_equal(wrist_result["traj_supervision_count"], np.array([3], dtype=np.uint16))
        self.assertEqual(int(wrist_result["traj_mask_reason_bits"][0]), 0)

    def test_wrist_profile_rejects_track_when_supported_prefix_is_too_short(self):
        fixture = _make_base_fixture(u_values=[2.0, 2.0, 20.0, 20.0], width=8)

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([False]))
        np.testing.assert_array_equal(result["traj_supervision_mask"], np.array([[True, True, False, False]]))
        np.testing.assert_array_equal(result["traj_supervision_prefix_len"], np.array([2], dtype=np.uint16))
        np.testing.assert_array_equal(result["traj_supervision_count"], np.array([2], dtype=np.uint16))
        self.assertTrue(result["traj_mask_reason_bits"][0] & MASK_REASON_TEMPORAL_CONSISTENCY_FAIL)

    def test_wrist_profile_rejects_query_depth_edge_risk_seed(self):
        fixture = _make_base_fixture()
        fixture["query_depth"][:, :2] = 0.05

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([False]))
        np.testing.assert_array_equal(result["traj_query_depth_edge_mask"], np.array([True]))
        np.testing.assert_array_equal(result["traj_query_depth_edge_risk_mask"], np.array([True]))
        self.assertAlmostEqual(float(result["traj_query_depth_patch_valid_ratio"][0]), 1.0, places=6)
        self.assertGreater(float(result["traj_query_depth_patch_std"][0]), 0.003)
        self.assertTrue(result["traj_mask_reason_bits"][0] & MASK_REASON_QUERY_DEPTH_EDGE_FAIL)
        self.assertFalse(bool(result["traj_mask_reason_bits"][0] & MASK_REASON_QUERY_DEPTH_FAIL))

    def test_wrist_profile_keeps_low_variance_query_depth_edge_seed(self):
        fixture = _make_base_fixture()
        fixture["traj"][0, :, 2] = 0.05
        fixture["query_depth"][:, :] = 0.05
        fixture["raw_depths_segment"][:, :, :] = 0.05
        fixture["query_depth"][:, :2] = 0.052

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True]))
        np.testing.assert_array_equal(result["traj_query_depth_edge_mask"], np.array([True]))
        np.testing.assert_array_equal(result["traj_query_depth_edge_risk_mask"], np.array([False]))
        self.assertLess(float(result["traj_query_depth_patch_std"][0]), 0.003)

    def test_wrist_profile_keeps_default_motion_debug_fields(self):
        fixture = _make_base_fixture()

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        self.assertTrue(np.isnan(result["traj_motion_extent"]).all())
        self.assertTrue(np.isnan(result["traj_motion_step_median"]).all())
        self.assertTrue(np.isnan(result["traj_motion_extent_all_valid"]).all())
        self.assertTrue(np.isnan(result["traj_motion_step_median_all_valid"]).all())
        np.testing.assert_array_equal(result["traj_manipulator_candidate_mask"], np.array([False]))

    def test_wrist_manipulator_uses_all_valid_motion_for_late_motion(self):
        fixture = _make_multi_track_fixture(
            traj=_make_track(u_values=[5.0, 5.05, 5.10, 40.0], v=5.0, depth=0.20),
            height=64,
            width=64,
        )
        fixture["raw_depths_segment"][3] = 1.0

        external_manipulator_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="external_manipulator"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )
        wrist_manipulator_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist_manipulator"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(
            wrist_manipulator_result["traj_supervision_mask"],
            np.array([[True, True, True, False]]),
        )
        self.assertLess(float(wrist_manipulator_result["traj_motion_extent"][0]), 0.03)
        self.assertGreater(float(wrist_manipulator_result["traj_motion_extent_all_valid"][0]), 0.03)
        np.testing.assert_array_equal(external_manipulator_result["traj_motion_mask"], np.array([False]))
        np.testing.assert_array_equal(wrist_manipulator_result["traj_motion_mask"], np.array([True]))
        np.testing.assert_array_equal(external_manipulator_result["traj_valid_mask"], np.array([False]))
        np.testing.assert_array_equal(wrist_manipulator_result["traj_valid_mask"], np.array([True]))

    def test_wrist_manipulator_top95_uses_wrist_manipulator_as_baseline(self):
        near_tracks = []
        far_tracks = []
        for track_idx in range(20):
            motion = float(track_idx + 1)
            start_u = 10.0 + float(track_idx * 6)
            near_tracks.append(
                _make_track(
                    u_values=[start_u, start_u + motion, start_u + 2.0 * motion, start_u + 3.0 * motion],
                    v=24.0,
                    depth=0.20 + 0.01 * float(track_idx),
                )
            )
            far_start_u = 12.0 + float(track_idx * 6)
            far_tracks.append(
                _make_track(
                    u_values=[far_start_u, far_start_u + 5.0, far_start_u + 10.0, far_start_u + 15.0],
                    v=120.0,
                    depth=1.00 + 0.01 * float(track_idx),
                )
            )
        traj = np.concatenate(near_tracks + far_tracks, axis=0)
        fixture = _make_multi_track_fixture(traj=traj, height=192, width=192)

        manipulator_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist_manipulator"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        top95_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist_manipulator_top95"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(top95_result["traj_wrist_seed_mask"], np.ones(40, dtype=bool))
        np.testing.assert_array_equal(
            manipulator_result["traj_valid_mask"],
            np.array([True] * 20 + [False] * 20, dtype=bool),
        )
        np.testing.assert_array_equal(
            top95_result["traj_valid_mask"],
            np.array([False] + [True] * 19 + [False] * 20, dtype=bool),
        )
        self.assertEqual(int(np.count_nonzero(top95_result["traj_valid_mask"])), 19)
        self.assertTrue(np.isfinite(top95_result["traj_motion_extent"]).all())
        self.assertTrue(np.isfinite(top95_result["traj_motion_step_median"]).all())
        self.assertTrue(np.isfinite(top95_result["traj_motion_extent_all_valid"]).all())
        self.assertTrue(np.isfinite(top95_result["traj_motion_step_median_all_valid"]).all())
        np.testing.assert_array_equal(
            top95_result["traj_valid_mask"] & (~manipulator_result["traj_valid_mask"]),
            np.zeros(40, dtype=bool),
        )
        self.assertLess(
            float(top95_result["traj_motion_extent"][0]),
            float(np.min(top95_result["traj_motion_extent"][1:20])),
        )

    def test_wrist_manipulator_top95_ranks_by_all_valid_motion(self):
        near_tracks = [
            _make_track(
                u_values=[10.0, 10.05, 10.10, 40.0],
                v=24.0,
                depth=0.20,
            )
        ]
        for track_idx in range(1, 20):
            motion = float(track_idx + 1)
            start_u = 10.0 + float(track_idx * 6)
            near_tracks.append(
                _make_track(
                    u_values=[start_u, start_u + motion, start_u + 2.0 * motion, start_u + 3.0 * motion],
                    v=24.0,
                    depth=0.20 + 0.01 * float(track_idx),
                )
            )
        traj = np.concatenate(near_tracks, axis=0)
        fixture = _make_multi_track_fixture(traj=traj, height=192, width=192)
        fixture["raw_depths_segment"][3] = 1.0

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist_manipulator_top95"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(
            result["traj_pre_top95_mask"],
            np.array([True] * 10 + [False] * 10, dtype=bool),
        )
        self.assertEqual(int(np.count_nonzero(result["traj_valid_mask"])), 9)
        self.assertLess(float(result["traj_motion_extent"][0]), float(result["traj_motion_extent"][1]))
        self.assertGreater(
            float(result["traj_motion_extent_all_valid"][0]),
            float(result["traj_motion_extent_all_valid"][1]),
        )
        self.assertTrue(bool(result["traj_valid_mask"][0]))
        self.assertFalse(bool(result["traj_valid_mask"][1]))

    def test_wrist_seed_top95_ranks_directly_from_wrist_seed(self):
        near_tracks = []
        far_tracks = []
        for track_idx in range(20):
            motion = float(track_idx + 1)
            start_u = 10.0 + float(track_idx * 6)
            near_tracks.append(
                _make_track(
                    u_values=[start_u, start_u + motion, start_u + 2.0 * motion, start_u + 3.0 * motion],
                    v=24.0,
                    depth=0.20 + 0.01 * float(track_idx),
                )
            )
            far_start_u = 12.0 + float(track_idx * 6)
            far_tracks.append(
                _make_track(
                    u_values=[far_start_u, far_start_u + 5.0, far_start_u + 10.0, far_start_u + 15.0],
                    v=120.0,
                    depth=1.00 + 0.01 * float(track_idx),
                )
            )
        traj = np.concatenate(near_tracks + far_tracks, axis=0)
        fixture = _make_multi_track_fixture(traj=traj, height=192, width=192)

        manipulator_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist_manipulator"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )
        ablated_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                boundary_margin=0,
                traj_filter_profile="wrist_manipulator_top95",
                traj_filter_ablation_mode=TRAJ_FILTER_ABLATION_MODE_WRIST_SEED_TOP95,
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        self.assertEqual(int(np.count_nonzero(ablated_result["traj_wrist_seed_mask"])), 40)
        np.testing.assert_array_equal(
            ablated_result["traj_pre_top95_mask"],
            ablated_result["traj_wrist_seed_mask"],
        )
        self.assertTrue(
            np.any(
                ablated_result["traj_valid_mask"]
                & (~manipulator_result["traj_valid_mask"])
            )
        )

    def test_wrist_no_query_edge_only_disables_edge_rejection(self):
        fixture = _make_base_fixture()
        fixture["query_depth"][:, :2] = 0.05

        baseline = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )
        ablated = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                boundary_margin=0,
                traj_filter_profile="wrist",
                traj_filter_ablation_mode=TRAJ_FILTER_ABLATION_MODE_WRIST_NO_QUERY_EDGE,
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(baseline["traj_valid_mask"], np.array([False]))
        np.testing.assert_array_equal(ablated["traj_valid_mask"], np.array([True]))
        self.assertTrue(bool(baseline["traj_mask_reason_bits"][0] & MASK_REASON_QUERY_DEPTH_EDGE_FAIL))
        self.assertFalse(bool(ablated["traj_mask_reason_bits"][0] & MASK_REASON_QUERY_DEPTH_EDGE_FAIL))

    def test_wrist_no_manipulator_depth_only_bypasses_near_depth_stage(self):
        traj = np.concatenate(
            [
                _make_track(u_values=[8.0, 9.0, 10.0, 11.0], v=8.0, depth=0.20),
                _make_track(u_values=[24.0, 25.0, 26.0, 27.0], v=40.0, depth=0.22),
                _make_track(u_values=[64.0, 65.0, 66.0, 67.0], v=16.0, depth=0.24),
                _make_track(u_values=[48.0, 54.0, 60.0, 66.0], v=48.0, depth=1.00),
                _make_track(u_values=[52.0, 58.0, 64.0, 70.0], v=52.0, depth=1.02),
            ],
            axis=0,
        )
        fixture = _make_multi_track_fixture(traj=traj, height=96, width=96)

        baseline = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist_manipulator"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )
        ablated = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                boundary_margin=0,
                traj_filter_profile="wrist_manipulator",
                traj_filter_ablation_mode=TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_DEPTH,
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(
            baseline["traj_valid_mask"],
            np.array([True, True, True, False, False]),
        )
        np.testing.assert_array_equal(
            ablated["traj_valid_mask"],
            np.array([False, False, False, True, True]),
        )
        np.testing.assert_array_equal(ablated["traj_near_depth_mask"], np.ones(5, dtype=bool))
        self.assertFalse(bool(ablated["traj_mask_reason_bits"][3] & MASK_REASON_MANIPULATOR_DEPTH_FAIL))
        self.assertFalse(bool(ablated["traj_mask_reason_bits"][4] & MASK_REASON_MANIPULATOR_DEPTH_FAIL))

    def test_wrist_no_manipulator_motion_only_bypasses_motion_stage(self):
        traj = np.concatenate(
            [
                _make_track(u_values=[10.0, 14.0, 18.0, 22.0], v=10.0, depth=0.20),
                _make_track(u_values=[48.0, 48.0, 48.0, 48.0], v=48.0, depth=0.21),
                _make_track(u_values=[68.0, 68.0, 68.0, 68.0], v=48.0, depth=0.22),
                _make_track(u_values=[80.0, 81.0, 82.0, 83.0], v=20.0, depth=1.00),
                _make_track(u_values=[86.0, 87.0, 88.0, 89.0], v=70.0, depth=1.20),
            ],
            axis=0,
        )
        fixture = _make_multi_track_fixture(traj=traj, height=96, width=96)

        baseline = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist_manipulator"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )
        ablated = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                boundary_margin=0,
                traj_filter_profile="wrist_manipulator",
                traj_filter_ablation_mode=TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_MOTION,
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(
            baseline["traj_valid_mask"],
            np.array([True, False, False, False, False]),
        )
        np.testing.assert_array_equal(
            ablated["traj_valid_mask"],
            np.array([False, True, True, False, False]),
        )
        np.testing.assert_array_equal(ablated["traj_motion_mask"], np.ones(5, dtype=bool))
        self.assertFalse(bool(ablated["traj_mask_reason_bits"][1] & MASK_REASON_MANIPULATOR_MOTION_FAIL))
        self.assertFalse(bool(ablated["traj_mask_reason_bits"][2] & MASK_REASON_MANIPULATOR_MOTION_FAIL))

    def test_wrist_no_manipulator_cluster_only_bypasses_cluster_stage(self):
        traj = np.concatenate(
            [
                _make_track(u_values=[5.0, 6.0, 7.0, 8.0], v=5.0, depth=0.20),
                _make_track(u_values=[11.0, 12.0, 13.0, 14.0], v=6.0, depth=0.21),
                _make_track(u_values=[50.0, 51.0, 52.0, 53.0], v=50.0, depth=0.22),
                _make_track(u_values=[20.0, 20.0, 20.0, 20.0], v=45.0, depth=1.00),
                _make_track(u_values=[45.0, 45.0, 45.0, 45.0], v=20.0, depth=1.20),
            ],
            axis=0,
        )
        fixture = _make_multi_track_fixture(traj=traj, height=64, width=64)

        baseline = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="wrist_manipulator", boundary_margin=0),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )
        ablated = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                traj_filter_profile="wrist_manipulator",
                boundary_margin=0,
                traj_filter_ablation_mode=TRAJ_FILTER_ABLATION_MODE_WRIST_NO_MANIPULATOR_CLUSTER,
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(baseline["traj_valid_mask"], np.array([True, True, False, False, False]))
        np.testing.assert_array_equal(
            ablated["traj_valid_mask"],
            np.array([True, True, True, False, False]),
        )
        np.testing.assert_array_equal(
            ablated["traj_cluster_mask"],
            ablated["traj_manipulator_candidate_mask"],
        )
        self.assertFalse(bool(ablated["traj_mask_reason_bits"][2] & MASK_REASON_MANIPULATOR_CLUSTER_FAIL))

    def test_external_profile_ignores_query_depth_edge_risk_rule(self):
        fixture = _make_base_fixture()
        fixture["query_depth"][:, :2] = 0.05

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="external"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True]))
        np.testing.assert_array_equal(result["traj_query_depth_edge_mask"], np.array([False]))
        np.testing.assert_array_equal(result["traj_query_depth_edge_risk_mask"], np.array([False]))
        self.assertTrue(np.isnan(result["traj_query_depth_patch_valid_ratio"]).all())
        self.assertTrue(np.isnan(result["traj_query_depth_patch_std"]).all())

    def test_wrist_manipulator_keeps_near_moving_adjacent_cluster(self):
        traj = np.concatenate(
            [
                _make_track(u_values=[5.0, 6.0, 7.0, 8.0], v=5.0, depth=0.20),
                _make_track(u_values=[10.0, 11.0, 12.0, 13.0], v=6.0, depth=0.22),
                _make_track(u_values=[40.0, 40.0, 40.0, 40.0], v=40.0, depth=1.00),
            ],
            axis=0,
        )
        fixture = _make_multi_track_fixture(traj=traj, height=48, width=48)

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="wrist_manipulator", boundary_margin=0),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_wrist_seed_mask"], np.array([True, True, True]))
        np.testing.assert_array_equal(result["traj_manipulator_candidate_mask"], np.array([True, True, False]))
        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True, True, False]))
        self.assertFalse(bool(np.asarray(result["traj_manipulator_cluster_fallback_used"]).reshape(-1)[0]))
        self.assertTrue(result["traj_mask_reason_bits"][2] & MASK_REASON_MANIPULATOR_DEPTH_FAIL)
        self.assertTrue(result["traj_mask_reason_bits"][2] & MASK_REASON_MANIPULATOR_MOTION_FAIL)

    def test_wrist_manipulator_filters_out_isolated_candidate_when_main_cluster_exists(self):
        traj = np.concatenate(
            [
                _make_track(u_values=[5.0, 6.0, 7.0, 8.0], v=5.0, depth=0.20),
                _make_track(u_values=[11.0, 12.0, 13.0, 14.0], v=6.0, depth=0.21),
                _make_track(u_values=[50.0, 51.0, 52.0, 53.0], v=50.0, depth=0.22),
                _make_track(u_values=[20.0, 20.0, 20.0, 20.0], v=45.0, depth=1.00),
                _make_track(u_values=[45.0, 45.0, 45.0, 45.0], v=20.0, depth=1.20),
            ],
            axis=0,
        )
        fixture = _make_multi_track_fixture(traj=traj, height=64, width=64)

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="wrist_manipulator", boundary_margin=0),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(
            result["traj_manipulator_candidate_mask"],
            np.array([True, True, True, False, False]),
        )
        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True, True, False, False, False]))
        self.assertTrue(result["traj_mask_reason_bits"][2] & MASK_REASON_MANIPULATOR_CLUSTER_FAIL)
        self.assertEqual(int(result["traj_manipulator_component_size"][2]), 1)

    def test_wrist_manipulator_falls_back_to_candidate_mask_for_small_samples(self):
        traj = _make_track(u_values=[5.0, 6.0, 7.0, 8.0], v=5.0, depth=0.20)
        fixture = _make_multi_track_fixture(traj=traj, height=32, width=32)

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="wrist_manipulator", boundary_margin=0),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_manipulator_candidate_mask"], np.array([True]))
        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True]))
        self.assertTrue(bool(np.asarray(result["traj_manipulator_cluster_fallback_used"]).reshape(-1)[0]))
        self.assertEqual(int(result["traj_manipulator_cluster_id"][0]), 0)

    def test_wrist_manipulator_preserves_wrist_supported_prefix_behavior(self):
        fixture = _make_multi_track_fixture(
            traj=_make_track(u_values=[5.0, 6.0, 7.0, 40.0], v=5.0, depth=0.20),
            height=32,
            width=32,
        )

        external_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="external"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )
        np.testing.assert_array_equal(external_result["traj_valid_mask"], np.array([False]))

        manipulator_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="wrist_manipulator"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(manipulator_result["traj_valid_mask"], np.array([True]))
        np.testing.assert_array_equal(
            manipulator_result["traj_supervision_mask"],
            np.array([[True, True, True, False]]),
        )
        np.testing.assert_array_equal(manipulator_result["traj_manipulator_candidate_mask"], np.array([True]))

    def test_wrist_pick_place_keeps_heatmap_guided_object_and_filters_far_background(self):
        traj = np.concatenate(
            [
                _make_track(u_values=[8.0, 18.0, 28.0, 38.0], v=20.0, depth=0.20),
                _make_track(u_values=[12.0, 22.0, 32.0, 42.0], v=22.0, depth=0.21),
                _make_track(u_values=[24.0, 34.0, 44.0, 54.0], v=24.0, depth=0.30),
                _make_track(u_values=[24.0, 34.0, 44.0, 54.0], v=40.0, depth=0.70),
            ],
            axis=0,
        )
        fixture = _make_multi_track_fixture(traj=traj, height=64, width=64)
        fixture["intrinsics_segment"] = _make_scaled_intrinsics(traj.shape[1], fx=100.0, fy=100.0)
        pick_heatmap = np.zeros((traj.shape[1], 64, 64), dtype=bool)
        for frame_idx in range(traj.shape[1]):
            for x_coord, y_coord in ((24 + 10 * frame_idx, 24), (24 + 10 * frame_idx, 40)):
                x0 = max(0, int(x_coord) - 1)
                x1 = min(64, int(x_coord) + 2)
                y0 = max(0, int(y_coord) - 1)
                y1 = min(64, int(y_coord) + 2)
                pick_heatmap[frame_idx, y0:y1, x0:x1] = True

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="wrist_pick_place", boundary_margin=0),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            pick_place_heatmap_segment=pick_heatmap,
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_manipulator_candidate_mask"], np.array([True, True, False, False]))
        np.testing.assert_array_equal(result["traj_pick_place_heatmap_hit_count"], np.array([0, 0, 4, 4], dtype=np.uint16))
        np.testing.assert_array_equal(result["traj_pick_place_heatmap_support_mask"], np.array([False, False, True, True]))
        np.testing.assert_array_equal(result["traj_pick_place_object_mask"], np.array([False, False, True, False]))
        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True, True, True, False]))
        self.assertFalse(np.isnan(result["traj_pick_place_min_manipulator_distance"][2]))
        self.assertEqual(int(result["traj_mask_reason_bits"][2]), 0)

    def test_wrist_pick_place_without_heatmap_falls_back_to_manipulator_branch(self):
        traj = np.concatenate(
            [
                _make_track(u_values=[8.0, 18.0, 28.0, 38.0], v=20.0, depth=0.20),
                _make_track(u_values=[12.0, 22.0, 32.0, 42.0], v=22.0, depth=0.21),
                _make_track(u_values=[24.0, 34.0, 44.0, 54.0], v=24.0, depth=0.30),
            ],
            axis=0,
        )
        fixture = _make_multi_track_fixture(traj=traj, height=64, width=64)
        fixture["intrinsics_segment"] = _make_scaled_intrinsics(traj.shape[1], fx=100.0, fy=100.0)

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="wrist_pick_place", boundary_margin=0),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_pick_place_heatmap_hit_count"], np.zeros(3, dtype=np.uint16))
        np.testing.assert_array_equal(result["traj_pick_place_object_mask"], np.array([False, False, False]))
        np.testing.assert_array_equal(result["traj_valid_mask"], np.array([True, True, False]))

    def test_wrist_pick_place_keeps_symmetric_second_arm_object_with_major_component_reference(self):
        traj = np.concatenate(
            [
                _make_track(u_values=[10.0, 20.0, 30.0, 40.0], v=20.0, depth=0.20),
                _make_track(u_values=[14.0, 24.0, 34.0, 44.0], v=24.0, depth=0.21),
                _make_track(u_values=[90.0, 100.0, 110.0, 120.0], v=68.0, depth=0.22),
                _make_track(u_values=[94.0, 104.0, 114.0, 124.0], v=72.0, depth=0.23),
                _make_track(u_values=[98.0, 108.0, 118.0, 128.0], v=70.0, depth=0.24),
                _make_track(u_values=[50.0, 50.0, 50.0, 50.0], v=40.0, depth=1.00),
                _make_track(u_values=[130.0, 130.0, 130.0, 130.0], v=30.0, depth=1.20),
            ],
            axis=0,
        )
        fixture = _make_multi_track_fixture(traj=traj, height=96, width=160)
        fixture["intrinsics_segment"] = _make_scaled_intrinsics(traj.shape[1], fx=50.0, fy=50.0)
        pick_heatmap = np.zeros((traj.shape[1], 96, 160), dtype=bool)
        for frame_idx in range(traj.shape[1]):
            x_coord = 98 + 10 * frame_idx
            y_coord = 70
            x0 = max(0, int(x_coord) - 1)
            x1 = min(160, int(x_coord) + 2)
            y0 = max(0, int(y_coord) - 1)
            y1 = min(96, int(y_coord) + 2)
            pick_heatmap[frame_idx, y0:y1, x0:x1] = True

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="wrist_pick_place", boundary_margin=0),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            pick_place_heatmap_segment=pick_heatmap,
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(
            result["traj_manipulator_candidate_mask"],
            np.array([True, True, True, True, False, False, False]),
        )
        np.testing.assert_array_equal(
            result["traj_pick_place_object_mask"],
            np.array([False, False, False, False, True, False, False]),
        )
        np.testing.assert_array_equal(
            result["traj_valid_mask"],
            np.array([True, True, True, True, True, False, False]),
        )
        self.assertTrue(result["traj_pick_place_contact_mask"][4])
        self.assertLess(float(result["traj_pick_place_min_manipulator_distance"][4]), 0.20)
        self.assertEqual(int(result["traj_mask_reason_bits"][4]), 0)

    def test_wrist_pick_place_no_heatmap_keeps_low_motion_local_object_and_filters_far_noise(self):
        anchor_tracks = []
        for track_idx in range(10):
            start_u = 20.0 + 8.0 * float(track_idx)
            start_v = 20.0 + 4.0 * float(track_idx % 5)
            depth = 0.20 + 0.001 * float(track_idx)
            anchor_tracks.append(
                _make_track(
                    u_values=[start_u, start_u + 18.0, start_u + 36.0, start_u + 54.0],
                    v=start_v,
                    depth=depth,
                )
            )
        traj = np.concatenate(
            anchor_tracks
            + [
                _make_track(u_values=[130.0, 131.0, 131.0, 132.0], v=170.0, depth=0.2065),
                _make_track(u_values=[210.0, 211.0, 211.0, 212.0], v=50.0, depth=0.2075),
                _make_track(u_values=[40.0, 40.0, 40.0, 40.0], v=220.0, depth=0.80),
                _make_track(u_values=[80.0, 80.0, 80.0, 80.0], v=220.0, depth=0.90),
                _make_track(u_values=[120.0, 120.0, 120.0, 120.0], v=220.0, depth=1.00),
                _make_track(u_values=[160.0, 160.0, 160.0, 160.0], v=220.0, depth=1.10),
                _make_track(u_values=[200.0, 200.0, 200.0, 200.0], v=220.0, depth=1.20),
                _make_track(u_values=[220.0, 220.0, 220.0, 220.0], v=160.0, depth=1.30),
                _make_track(u_values=[236.0, 236.0, 236.0, 236.0], v=120.0, depth=1.40),
            ],
            axis=0,
        )
        fixture = _make_multi_track_fixture(traj=traj, height=256, width=256)
        fixture["intrinsics_segment"] = _make_scaled_intrinsics(traj.shape[1], fx=100.0, fy=100.0)

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                traj_filter_profile="wrist_pick_place_no_heatmap",
                boundary_margin=0,
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        self.assertEqual(int(result["traj_manipulator_candidate_mask"].sum()), 8)
        self.assertFalse(bool(result["traj_motion_mask"][10]))
        self.assertFalse(bool(result["traj_manipulator_candidate_mask"][10]))
        self.assertTrue(bool(result["traj_valid_mask"][10]))
        self.assertFalse(bool(result["traj_valid_mask"][11]))
        self.assertTrue(bool(result["traj_near_depth_mask"][11]))
        self.assertFalse(bool(result["traj_cluster_mask"][11]))
        self.assertEqual(
            int(result["traj_mask_reason_bits"][11]),
            int(MASK_REASON_MANIPULATOR_CLUSTER_FAIL),
        )
        np.testing.assert_array_equal(
            result["traj_pick_place_heatmap_hit_count"],
            np.zeros(traj.shape[0], dtype=np.uint16),
        )
        np.testing.assert_array_equal(
            result["traj_pick_place_object_mask"],
            np.zeros(traj.shape[0], dtype=bool),
        )
        np.testing.assert_array_equal(
            result["traj_pick_place_delayed_contact_rescue_mask"],
            np.zeros(traj.shape[0], dtype=bool),
        )

    def test_wrist_pick_place_no_heatmap_falls_back_to_rank_only_when_anchor_count_is_small(self):
        traj = np.concatenate(
            [
                _make_track(u_values=[20.0, 38.0, 56.0, 74.0], v=20.0, depth=0.20),
                _make_track(u_values=[28.0, 46.0, 64.0, 82.0], v=24.0, depth=0.21),
                _make_track(u_values=[36.0, 54.0, 72.0, 90.0], v=28.0, depth=0.22),
                _make_track(u_values=[44.0, 62.0, 80.0, 98.0], v=32.0, depth=0.23),
                _make_track(u_values=[220.0, 221.0, 221.0, 222.0], v=180.0, depth=0.24),
                _make_track(u_values=[20.0, 20.0, 20.0, 20.0], v=220.0, depth=0.90),
                _make_track(u_values=[60.0, 60.0, 60.0, 60.0], v=220.0, depth=1.00),
                _make_track(u_values=[100.0, 100.0, 100.0, 100.0], v=220.0, depth=1.10),
                _make_track(u_values=[140.0, 140.0, 140.0, 140.0], v=220.0, depth=1.20),
            ],
            axis=0,
        )
        fixture = _make_multi_track_fixture(traj=traj, height=256, width=256)
        fixture["intrinsics_segment"] = _make_scaled_intrinsics(traj.shape[1], fx=100.0, fy=100.0)

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                traj_filter_profile="wrist_pick_place_no_heatmap",
                boundary_margin=0,
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        self.assertEqual(int(result["traj_manipulator_candidate_mask"].sum()), 4)
        self.assertTrue(bool(np.asarray(result["traj_manipulator_cluster_fallback_used"]).reshape(-1)[0]))
        self.assertTrue(bool(result["traj_valid_mask"][4]))
        self.assertTrue(bool(result["traj_cluster_mask"][4]))
        self.assertFalse(bool(result["traj_motion_mask"][4]))
        np.testing.assert_array_equal(
            result["traj_pick_place_heatmap_support_mask"],
            np.zeros(traj.shape[0], dtype=bool),
        )
        np.testing.assert_array_equal(
            result["traj_pick_place_delayed_contact_rescue_mask"],
            np.zeros(traj.shape[0], dtype=bool),
        )

    def test_wrist_pick_place_no_heatmap_rescues_query_visible_pre_grasp_object_after_delayed_contact(self):
        anchor_tracks = []
        for track_idx in range(10):
            start_u = 20.0 + 8.0 * float(track_idx)
            start_v = 20.0 + 4.0 * float(track_idx % 5)
            depth = 0.20 + 0.001 * float(track_idx)
            anchor_tracks.append(
                _make_track(
                    u_values=[start_u, start_u + 18.0, start_u + 36.0, start_u + 54.0],
                    v=start_v,
                    depth=depth,
                )
            )

        rescued_track = np.array(
            [
                [210.0, 50.0, 0.44],
                [82.0, 32.0, 0.23],
                [100.0, 32.0, 0.23],
                [118.0, 32.0, 0.23],
            ],
            dtype=np.float32,
        )[None, ...]
        too_deep_track = np.array(
            [
                [214.0, 54.0, 0.60],
                [84.0, 34.0, 0.23],
                [102.0, 34.0, 0.23],
                [120.0, 34.0, 0.23],
            ],
            dtype=np.float32,
        )[None, ...]
        hidden_query_track = np.array(
            [
                [218.0, 58.0, 0.44],
                [86.0, 36.0, 0.23],
                [104.0, 36.0, 0.23],
                [122.0, 36.0, 0.23],
            ],
            dtype=np.float32,
        )[None, ...]
        traj = np.concatenate(
            anchor_tracks
            + [
                rescued_track,
                too_deep_track,
                _make_track(u_values=[230.0, 231.0, 231.0, 232.0], v=80.0, depth=0.2075),
                _make_track(u_values=[40.0, 40.0, 40.0, 40.0], v=220.0, depth=0.80),
                _make_track(u_values=[80.0, 80.0, 80.0, 80.0], v=220.0, depth=0.90),
                _make_track(u_values=[120.0, 120.0, 120.0, 120.0], v=220.0, depth=1.00),
                _make_track(u_values=[160.0, 160.0, 160.0, 160.0], v=220.0, depth=1.10),
                _make_track(u_values=[200.0, 200.0, 200.0, 200.0], v=220.0, depth=1.20),
                _make_track(u_values=[220.0, 220.0, 220.0, 220.0], v=160.0, depth=1.30),
                _make_track(u_values=[236.0, 236.0, 236.0, 236.0], v=120.0, depth=1.40),
                hidden_query_track,
            ],
            axis=0,
        )
        visibs = np.ones(traj.shape[:2], dtype=bool)
        visibs[20, 0] = False
        fixture = _make_multi_track_fixture(traj=traj, height=256, width=256, visibs=visibs)
        fixture["intrinsics_segment"] = _make_scaled_intrinsics(traj.shape[1], fx=100.0, fy=100.0)

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(
                traj_filter_profile="wrist_pick_place_no_heatmap",
                boundary_margin=0,
            ),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        rescued_idx = 10
        too_deep_idx = 11
        far_noise_idx = 12
        hidden_query_idx = 20

        self.assertEqual(int(result["traj_manipulator_candidate_mask"].sum()), 9)
        self.assertFalse(bool(result["traj_cluster_mask"][rescued_idx]))
        self.assertTrue(bool(result["traj_pick_place_contact_mask"][rescued_idx]))
        self.assertTrue(bool(result["traj_pick_place_depth_guard_mask"][rescued_idx]))
        self.assertTrue(bool(result["traj_pick_place_delayed_contact_rescue_mask"][rescued_idx]))
        self.assertTrue(bool(result["traj_pick_place_object_mask"][rescued_idx]))
        self.assertTrue(bool(result["traj_valid_mask"][rescued_idx]))
        self.assertEqual(int(result["traj_mask_reason_bits"][rescued_idx]), 0)

        self.assertTrue(bool(result["traj_pick_place_contact_mask"][too_deep_idx]))
        self.assertFalse(bool(result["traj_pick_place_depth_guard_mask"][too_deep_idx]))
        self.assertFalse(bool(result["traj_pick_place_delayed_contact_rescue_mask"][too_deep_idx]))
        self.assertFalse(bool(result["traj_valid_mask"][too_deep_idx]))

        self.assertFalse(bool(result["traj_pick_place_contact_mask"][far_noise_idx]))
        self.assertFalse(bool(result["traj_pick_place_delayed_contact_rescue_mask"][far_noise_idx]))
        self.assertFalse(bool(result["traj_valid_mask"][far_noise_idx]))

        self.assertFalse(bool(result["traj_pick_place_contact_mask"][hidden_query_idx]))
        self.assertFalse(bool(result["traj_pick_place_delayed_contact_rescue_mask"][hidden_query_idx]))
        self.assertFalse(bool(result["traj_valid_mask"][hidden_query_idx]))


class MotionMetricHelperTests(unittest.TestCase):
    def test_compute_motion_metrics_keeps_first_valid_anchor_and_nan_semantics(self):
        world_tracks = np.array(
            [
                [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [3.0, 0.0, 0.5], [6.0, 0.0, 0.5]],
                [[5.0, 1.0, 0.4], [np.nan, np.nan, np.nan], [8.0, 1.0, 0.4], [10.0, 1.0, 0.4]],
                [[2.0, 2.0, 0.3], [2.0, 2.0, 0.3], [2.0, 2.0, 0.3], [2.0, 2.0, 0.3]],
            ],
            dtype=np.float32,
        )
        supervision_mask = np.array(
            [
                [False, True, True, False],
                [True, True, True, True],
                [True, False, False, False],
            ],
            dtype=bool,
        )
        late_mask = np.array(
            [
                [False, False, True, True],
                [False, False, True, False],
                [True, True, True, True],
            ],
            dtype=bool,
        )

        (
            (motion_extent, motion_step_median),
            (motion_extent_late, motion_step_median_late),
        ) = _compute_motion_metrics_for_valid_masks(
            world_tracks,
            (supervision_mask, late_mask),
        )

        np.testing.assert_allclose(
            motion_extent,
            np.array([2.0, 5.0, np.nan], dtype=np.float32),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            motion_step_median,
            np.array([2.0, 2.0, np.nan], dtype=np.float32),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            motion_extent_late,
            np.array([3.0, np.nan, 0.0], dtype=np.float32),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            motion_step_median_late,
            np.array([3.0, np.nan, 0.0], dtype=np.float32),
            equal_nan=True,
        )


class PickPlaceFilterRegressionTests(unittest.TestCase):
    def test_major_component_refinement_drops_far_low_motion_component(self):
        traj = np.array(
            [
                [[10.0, 20.0, 0.20], [20.0, 20.0, 0.20], [30.0, 20.0, 0.20], [40.0, 20.0, 0.20]],
                [[14.0, 24.0, 0.21], [24.0, 24.0, 0.21], [34.0, 24.0, 0.21], [44.0, 24.0, 0.21]],
                [[120.0, 44.0, 0.40], [123.0, 44.0, 0.40], [125.0, 44.0, 0.40], [128.0, 44.0, 0.40]],
                [[124.0, 48.0, 0.42], [127.0, 48.0, 0.42], [129.0, 48.0, 0.42], [132.0, 48.0, 0.42]],
                [[70.0, 32.0, 1.20], [70.0, 32.0, 1.20], [70.0, 32.0, 1.20], [70.0, 32.0, 1.20]],
            ],
            dtype=np.float32,
        )
        keypoints = traj[:, 0, :2].astype(np.float32)
        supervision_mask = np.ones(traj.shape[:2], dtype=bool)
        seed_mask = np.array([True, True, True, True, False], dtype=bool)

        (
            final_mask,
            _traj_query_depth_rank,
            _traj_motion_extent,
            _traj_motion_step_median,
            _traj_motion_extent_all_valid,
            _traj_motion_step_median_all_valid,
            traj_manipulator_candidate_mask,
            traj_manipulator_cluster_id,
            _traj_manipulator_component_size,
            _traj_near_depth_mask,
            _traj_motion_mask,
            traj_cluster_mask,
            fallback_used,
        ) = _apply_manipulator_aware_filter(
            traj=traj,
            keypoints=keypoints,
            seed_mask=seed_mask,
            supervision_mask=supervision_mask,
            intrinsics_segment=_make_scaled_intrinsics(traj.shape[1], fx=100.0, fy=100.0),
            extrinsics_segment=_make_identity_extrinsics(traj.shape[1]),
            image_height=96,
            image_width=160,
            min_depth=0.01,
            max_depth=10.0,
            max_depth_rank=1.0,
            min_motion_extent=0.03,
            cluster_radius_ratio=0.06,
            cluster_radius_min_px=24,
            min_component_ratio=0.005,
            min_component_size=2,
            component_keep_mode="major",
            major_component_ratio=0.15,
            major_component_min_motion_ratio=0.75,
            major_component_depth_margin_m=0.08,
            motion_metric_mode="all_valid",
        )

        np.testing.assert_array_equal(
            traj_manipulator_candidate_mask,
            np.array([True, True, True, True, False]),
        )
        np.testing.assert_array_equal(
            traj_manipulator_cluster_id,
            np.array([0, 0, 1, 1, -1], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            traj_cluster_mask,
            np.array([True, True, False, False, False]),
        )
        np.testing.assert_array_equal(
            final_mask,
            np.array([True, True, False, False, False]),
        )
        self.assertFalse(bool(fallback_used))

    def test_object_depth_guard_uses_nearest_manipulator_component(self):
        traj = np.array(
            [
                [[10.0, 20.0, 0.20], [20.0, 20.0, 0.20], [30.0, 20.0, 0.20], [40.0, 20.0, 0.20]],
                [[14.0, 24.0, 0.21], [24.0, 24.0, 0.21], [34.0, 24.0, 0.21], [44.0, 24.0, 0.21]],
                [[18.0, 28.0, 0.22], [28.0, 28.0, 0.22], [38.0, 28.0, 0.22], [48.0, 28.0, 0.22]],
                [[120.0, 70.0, 0.55], [130.0, 70.0, 0.55], [140.0, 70.0, 0.55], [150.0, 70.0, 0.55]],
                [[124.0, 74.0, 0.56], [134.0, 74.0, 0.56], [144.0, 74.0, 0.56], [154.0, 74.0, 0.56]],
                [[18.0, 26.0, 0.48], [24.0, 26.0, 0.30], [34.0, 26.0, 0.30], [44.0, 26.0, 0.30]],
            ],
            dtype=np.float32,
        )
        pick_heatmap = np.zeros((traj.shape[1], 96, 192), dtype=bool)
        for frame_idx, (u_coord, v_coord) in enumerate(traj[5, :, :2].tolist()):
            x0 = max(0, int(round(u_coord)) - 1)
            x1 = min(192, int(round(u_coord)) + 2)
            y0 = max(0, int(round(v_coord)) - 1)
            y1 = min(96, int(round(v_coord)) + 2)
            pick_heatmap[frame_idx, y0:y1, x0:x1] = True

        (
            object_mask,
            heatmap_hit_count,
            heatmap_support_mask,
            min_manipulator_distance,
            contact_mask,
            depth_guard_mask,
        ) = _apply_pick_place_object_filter(
            traj=traj,
            seed_mask=np.ones(traj.shape[0], dtype=bool),
            manipulator_reference_mask=np.array([True, True, True, True, True, False]),
            manipulator_reference_component_ids=np.array([0, 0, 0, 1, 1, -1], dtype=np.int32),
            intrinsics_segment=_make_scaled_intrinsics(traj.shape[1], fx=100.0, fy=100.0),
            extrinsics_segment=_make_identity_extrinsics(traj.shape[1]),
            min_depth=0.01,
            max_depth=10.0,
            pick_place_heatmap_segment=pick_heatmap,
            min_heatmap_hits=2,
            max_manipulator_distance_m=0.20,
            query_depth_margin_m=0.25,
        )

        np.testing.assert_array_equal(heatmap_hit_count, np.array([0, 0, 0, 0, 0, 4], dtype=np.uint16))
        np.testing.assert_array_equal(heatmap_support_mask, np.array([False, False, False, False, False, True]))
        np.testing.assert_array_equal(contact_mask, np.array([False, False, False, False, False, True]))
        np.testing.assert_array_equal(depth_guard_mask, np.array([False, False, False, False, False, False]))
        np.testing.assert_array_equal(object_mask, np.array([False, False, False, False, False, False]))
        self.assertLess(float(min_manipulator_distance[5]), 0.20)

    def test_external_manipulator_is_subset_of_external(self):
        traj = np.concatenate(
            [
                _make_track(u_values=[5.0, 6.0, 7.0, 8.0], v=5.0, depth=0.20),
                _make_track(u_values=[10.0, 11.0, 12.0, 13.0], v=6.0, depth=0.22),
                _make_track(u_values=[25.0, 25.0, 25.0, 25.0], v=25.0, depth=1.00),
            ],
            axis=0,
        )
        fixture = _make_multi_track_fixture(traj=traj, height=48, width=48)

        external_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="external", boundary_margin=0),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )
        np.testing.assert_array_equal(external_result["traj_valid_mask"], np.array([True, True, True]))

        external_manipulator_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="external_manipulator", boundary_margin=0),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(
            external_manipulator_result["traj_wrist_seed_mask"],
            np.array([True, True, True]),
        )
        np.testing.assert_array_equal(
            external_manipulator_result["traj_manipulator_candidate_mask"],
            np.array([True, True, False]),
        )
        np.testing.assert_array_equal(
            external_manipulator_result["traj_valid_mask"],
            np.array([True, True, False]),
        )
        self.assertTrue(
            external_manipulator_result["traj_mask_reason_bits"][2] & MASK_REASON_MANIPULATOR_DEPTH_FAIL
        )
        self.assertTrue(
            external_manipulator_result["traj_mask_reason_bits"][2] & MASK_REASON_MANIPULATOR_MOTION_FAIL
        )

    def test_external_manipulator_does_not_use_wrist_supported_prefix_relaxation(self):
        fixture = _make_multi_track_fixture(
            traj=_make_track(u_values=[5.0, 6.0, 7.0, 40.0], v=5.0, depth=0.20),
            height=32,
            width=32,
        )

        external_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="external"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )
        np.testing.assert_array_equal(external_result["traj_valid_mask"], np.array([False]))

        external_manipulator_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(boundary_margin=0, traj_filter_profile="external_manipulator"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(external_manipulator_result["traj_wrist_seed_mask"], np.array([False]))
        np.testing.assert_array_equal(
            external_manipulator_result["traj_manipulator_candidate_mask"],
            np.array([False]),
        )
        np.testing.assert_array_equal(external_manipulator_result["traj_valid_mask"], np.array([False]))

    def test_external_manipulator_v2_keeps_multiple_major_components(self):
        traj = np.concatenate(
            [
                _make_track(u_values=[10.0, 11.0, 12.0, 13.0], v=10.0, depth=0.20),
                _make_track(u_values=[90.0, 91.0, 92.0, 93.0], v=90.0, depth=0.21),
                _make_track(u_values=[14.0, 15.0, 16.0, 17.0], v=14.0, depth=0.22),
                _make_track(u_values=[94.0, 95.0, 96.0, 97.0], v=94.0, depth=0.23),
                _make_track(u_values=[18.0, 19.0, 20.0, 21.0], v=18.0, depth=0.24),
                _make_track(u_values=[60.0, 60.0, 60.0, 60.0], v=60.0, depth=1.00),
            ],
            axis=0,
        )
        fixture = _make_multi_track_fixture(traj=traj, height=128, width=128)

        external_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="external", boundary_margin=0),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )
        np.testing.assert_array_equal(
            external_result["traj_valid_mask"],
            np.array([True, True, True, True, True, True]),
        )

        external_manipulator_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="external_manipulator", boundary_margin=0),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )
        np.testing.assert_array_equal(
            external_manipulator_result["traj_valid_mask"],
            np.array([True, False, True, False, False, False]),
        )

        external_manipulator_v2_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="external_manipulator_v2", boundary_margin=0),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(
            external_manipulator_v2_result["traj_wrist_seed_mask"],
            np.array([True, True, True, True, True, True]),
        )
        np.testing.assert_array_equal(
            external_manipulator_v2_result["traj_manipulator_candidate_mask"],
            np.array([True, True, True, True, False, False]),
        )
        np.testing.assert_array_equal(
            external_manipulator_v2_result["traj_valid_mask"],
            np.array([True, True, True, True, False, False]),
        )
        self.assertTrue(
            external_manipulator_v2_result["traj_mask_reason_bits"][4] & MASK_REASON_MANIPULATOR_DEPTH_FAIL
        )
        self.assertTrue(
            external_manipulator_v2_result["traj_mask_reason_bits"][5] & MASK_REASON_MANIPULATOR_DEPTH_FAIL
        )
        self.assertTrue(
            external_manipulator_v2_result["traj_mask_reason_bits"][5] & MASK_REASON_MANIPULATOR_MOTION_FAIL
        )

    def test_non_manipulator_profiles_return_default_manipulator_debug_fields(self):
        fixture = _make_base_fixture()

        result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(traj_filter_profile="external"),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            depth_volatility_map=fixture["depth_volatility_map"],
        )

        np.testing.assert_array_equal(result["traj_wrist_seed_mask"], np.array([False]))
        self.assertTrue(np.isnan(result["traj_query_depth_rank"]).all())
        np.testing.assert_array_equal(result["traj_query_depth_edge_mask"], np.array([False]))
        self.assertTrue(np.isnan(result["traj_query_depth_patch_valid_ratio"]).all())
        self.assertTrue(np.isnan(result["traj_query_depth_patch_std"]).all())
        np.testing.assert_array_equal(result["traj_query_depth_edge_risk_mask"], np.array([False]))
        self.assertTrue(np.isnan(result["traj_motion_extent"]).all())
        self.assertTrue(np.isnan(result["traj_motion_step_median"]).all())
        self.assertTrue(np.isnan(result["traj_motion_extent_all_valid"]).all())
        self.assertTrue(np.isnan(result["traj_motion_step_median_all_valid"]).all())
        np.testing.assert_array_equal(result["traj_manipulator_candidate_mask"], np.array([False]))
        np.testing.assert_array_equal(result["traj_manipulator_cluster_id"], np.array([-1], dtype=np.int16))
        np.testing.assert_array_equal(result["traj_manipulator_component_size"], np.array([0], dtype=np.uint16))
        self.assertFalse(bool(np.asarray(result["traj_manipulator_cluster_fallback_used"]).reshape(-1)[0]))


class DepthVolatilityHelperTests(unittest.TestCase):
    def test_masked_linear_percentiles_match_nanpercentile_baseline(self):
        values = np.array(
            [
                [1.0, 10.0, 4.0, -2.0],
                [2.0, 20.0, 5.0, -1.0],
                [3.0, 30.0, 6.0, 0.0],
                [4.0, 40.0, 7.0, 1.0],
            ],
            dtype=np.float32,
        )
        valid = np.array(
            [
                [True, True, False, False],
                [True, False, False, False],
                [True, True, True, False],
                [True, True, False, False],
            ],
            dtype=bool,
        )

        actual, actual_counts = _compute_linear_percentiles_for_masked_columns(
            values,
            valid,
            percentiles=(5.0, 95.0),
        )

        masked_values = np.where(valid, values, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            expected = np.nanpercentile(masked_values, [5.0, 95.0], axis=0)

        np.testing.assert_allclose(actual[:, :3], expected[:, :3], atol=1e-6, rtol=0.0)
        self.assertTrue(np.isnan(actual[:, 3]).all())
        np.testing.assert_array_equal(actual_counts, np.array([4, 3, 1, 0], dtype=np.int32))

    def test_joint_percentile_matches_two_pass_baseline(self):
        full_depths = np.array(
            [
                [[1.0, 2.0], [3.0, np.nan]],
                [[2.0, 3.0], [4.0, 0.0]],
                [[3.0, 4.0], [5.0, 12.0]],
                [[4.0, 5.0], [6.0, 1.0]],
            ],
            dtype=np.float32,
        )
        min_depth = 0.5
        max_depth = 10.0

        actual = compute_depth_volatility_map(
            full_depths,
            min_depth=min_depth,
            max_depth=max_depth,
            low_percentile=5.0,
            high_percentile=95.0,
        )

        valid = np.isfinite(full_depths) & (full_depths > min_depth) & (full_depths < max_depth)
        depths_nan = np.where(valid, full_depths, np.nan)
        with np.errstate(invalid="ignore"):
            expected_lo = np.nanpercentile(depths_nan, 5.0, axis=0)
            expected_hi = np.nanpercentile(depths_nan, 95.0, axis=0)
        expected = np.nan_to_num(expected_hi - expected_lo, nan=0.0, posinf=0.0, neginf=0.0)
        expected[valid.sum(axis=0) < 2] = 0.0

        np.testing.assert_allclose(actual, expected.astype(np.float32), atol=1e-6, rtol=0.0)

    def test_accessed_high_volatility_mask_only_uses_accessed_pixels(self):
        full_depths = np.array(
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, 9.0], [1.0, 2.0, 1.0]],
                [[1.0, 1.0, 1.0], [1.0, 3.0, 1.0]],
                [[1.0, 1.0, 9.0], [1.0, 4.0, 1.0]],
            ],
            dtype=np.float32,
        )
        accessed_pixel_mask = np.array(
            [
                [True, False, False],
                [False, True, False],
            ],
            dtype=bool,
        )

        high_volatility_mask, threshold = compute_accessed_high_volatility_mask(
            full_depths,
            accessed_pixel_mask=accessed_pixel_mask,
            min_depth=0.01,
            max_depth=10.0,
            low_percentile=5.0,
            high_percentile=95.0,
            mask_percentile=50.0,
        )

        expected_mask = np.array(
            [
                [False, False, False],
                [False, True, False],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(high_volatility_mask, expected_mask)
        self.assertAlmostEqual(float(threshold), 1.35, places=6)

    def test_accessed_high_volatility_mask_return_stats_reports_counts_and_threshold(self):
        full_depths = np.array(
            [
                [[1.0, 1.0, np.nan], [1.0, 1.0, 1.0]],
                [[1.0, 1.0, np.nan], [1.0, 2.0, 1.0]],
                [[1.0, 1.0, np.nan], [1.0, 3.0, 1.0]],
                [[1.0, 1.0, np.nan], [1.0, 4.0, 1.0]],
            ],
            dtype=np.float32,
        )
        accessed_pixel_mask = np.array(
            [
                [True, False, True],
                [False, True, False],
            ],
            dtype=bool,
        )

        high_volatility_mask, threshold, stats = compute_accessed_high_volatility_mask(
            full_depths,
            accessed_pixel_mask=accessed_pixel_mask,
            min_depth=0.01,
            max_depth=10.0,
            low_percentile=5.0,
            high_percentile=95.0,
            mask_percentile=50.0,
            return_stats=True,
        )

        expected_mask = np.array(
            [
                [False, False, False],
                [False, True, False],
            ],
            dtype=bool,
        )
        np.testing.assert_array_equal(high_volatility_mask, expected_mask)
        self.assertAlmostEqual(float(threshold), 0.0, places=6)
        self.assertEqual(stats["accessed_pixel_count"], 3.0)
        self.assertEqual(stats["valid_pixel_count"], 2.0)
        self.assertAlmostEqual(float(stats["threshold"]), 0.0, places=6)

    def test_precomputed_temporal_context_matches_direct_temporal_evaluation(self):
        fixture = _make_base_fixture(u_values=[1.0, 2.0, 3.0, 5.0])
        high_volatility_mask = np.zeros(
            (fixture["image_height"], fixture["image_width"]),
            dtype=bool,
        )
        high_volatility_mask[2, 5] = True
        fixture["raw_depths_segment"][3, 2, 5] = 2.0

        precomputed_context = prepare_temporal_depth_consistency_context(
            fixture["traj"],
            visibs=fixture["visibs"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            min_depth=0.01,
            max_depth=10.0,
        )
        precomputed_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            high_volatility_mask=high_volatility_mask,
            temporal_compare_context=precomputed_context,
        )
        direct_result = build_traj_filter_result(
            traj=fixture["traj"],
            visibs=fixture["visibs"],
            image_width=fixture["image_width"],
            image_height=fixture["image_height"],
            filter_args=_make_filter_args(),
            keypoints=fixture["keypoints"],
            query_depth=fixture["query_depth"],
            raw_depths_segment=fixture["raw_depths_segment"],
            intrinsics_segment=fixture["intrinsics_segment"],
            extrinsics_segment=fixture["extrinsics_segment"],
            high_volatility_mask=high_volatility_mask,
        )

        np.testing.assert_array_equal(
            precomputed_result["traj_valid_mask"],
            direct_result["traj_valid_mask"],
        )
        np.testing.assert_array_equal(
            precomputed_result["traj_high_volatility_hit"],
            direct_result["traj_high_volatility_hit"],
        )
        np.testing.assert_allclose(
            precomputed_result["traj_volatility_exposure_ratio"],
            direct_result["traj_volatility_exposure_ratio"],
            atol=1e-6,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            precomputed_result["traj_stable_depth_consistency_ratio"],
            direct_result["traj_stable_depth_consistency_ratio"],
            atol=1e-6,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
