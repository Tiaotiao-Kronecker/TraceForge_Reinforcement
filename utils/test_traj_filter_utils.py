from types import SimpleNamespace
import unittest

import numpy as np

from utils.traj_filter_utils import (
    MASK_REASON_QUERY_DEPTH_FAIL,
    MASK_REASON_STABLE_TEMPORAL_FAIL,
    MASK_REASON_TEMPORAL_CONSISTENCY_FAIL,
    build_traj_filter_result,
    build_traj_valid_mask,
    compute_query_depth_quality_mask,
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


if __name__ == "__main__":
    unittest.main()
