from types import SimpleNamespace
import unittest

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
    build_traj_filter_result,
    build_query_prefilter_result,
    build_traj_valid_mask,
    compute_accessed_high_volatility_mask,
    compute_depth_volatility_map,
    compute_query_depth_quality_mask,
    prepare_temporal_depth_consistency_context,
)


def _make_identity_intrinsics(num_frames: int) -> np.ndarray:
    return np.repeat(np.eye(3, dtype=np.float32)[None], num_frames, axis=0)


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
        np.testing.assert_array_equal(result["traj_manipulator_candidate_mask"], np.array([False]))

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
        np.testing.assert_array_equal(
            top95_result["traj_valid_mask"] & (~manipulator_result["traj_valid_mask"]),
            np.zeros(40, dtype=bool),
        )
        self.assertLess(
            float(top95_result["traj_motion_extent"][0]),
            float(np.min(top95_result["traj_motion_extent"][1:20])),
        )

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
        np.testing.assert_array_equal(result["traj_manipulator_candidate_mask"], np.array([False]))
        np.testing.assert_array_equal(result["traj_manipulator_cluster_id"], np.array([-1], dtype=np.int16))
        np.testing.assert_array_equal(result["traj_manipulator_component_size"], np.array([0], dtype=np.uint16))
        self.assertFalse(bool(np.asarray(result["traj_manipulator_cluster_fallback_used"]).reshape(-1)[0]))


class DepthVolatilityHelperTests(unittest.TestCase):
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
