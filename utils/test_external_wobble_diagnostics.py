import unittest

import numpy as np

from utils.external_wobble_diagnostics import (
    build_query_anchor_bundle,
    build_query_anchor_bundle_from_keypoints,
    compute_extrinsics_temporal_metrics,
    compute_static_geometry_consistency,
    estimate_temporal_median_world_points,
    freeze_extrinsics_w2c_to_query_frame,
    smooth_extrinsics_w2c_moving_average,
    stabilize_depth_frames_temporal_median_reproject,
)


def _translation_w2c(tx: float, ty: float, tz: float) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return np.linalg.inv(c2w).astype(np.float32)


class ComputeExtrinsicsTemporalMetricsTests(unittest.TestCase):
    def test_constant_translation_has_zero_jerk(self):
        extrinsics = np.stack(
            [
                _translation_w2c(0.0, 0.0, 0.0),
                _translation_w2c(1.0, 0.0, 0.0),
                _translation_w2c(2.0, 0.0, 0.0),
                _translation_w2c(3.0, 0.0, 0.0),
            ],
            axis=0,
        )

        metrics = compute_extrinsics_temporal_metrics(extrinsics)

        np.testing.assert_allclose(metrics["step_translation_m"], np.ones(3, dtype=np.float32))
        np.testing.assert_allclose(metrics["jerk_translation_m"], np.zeros(2, dtype=np.float32))
        np.testing.assert_allclose(metrics["step_rotation_deg"], np.zeros(3, dtype=np.float32), atol=1e-5)
        np.testing.assert_allclose(metrics["jerk_rotation_deg"], np.zeros(2, dtype=np.float32), atol=1e-5)

    def test_smoothing_preserves_constant_sequence(self):
        extrinsics = np.repeat(_translation_w2c(1.0, 2.0, 3.0)[None], 5, axis=0)

        smoothed = smooth_extrinsics_w2c_moving_average(extrinsics, radius=1)

        np.testing.assert_allclose(smoothed, extrinsics, atol=1e-6)

    def test_freeze_repeats_query_frame_pose(self):
        extrinsics = np.stack(
            [
                _translation_w2c(0.0, 0.0, 0.0),
                _translation_w2c(1.0, 0.0, 0.0),
                _translation_w2c(2.0, 0.0, 0.0),
            ],
            axis=0,
        )

        frozen = freeze_extrinsics_w2c_to_query_frame(extrinsics, query_frame=1)

        expected = np.repeat(extrinsics[1:2], 3, axis=0)
        np.testing.assert_allclose(frozen, expected, atol=1e-6)


class ComputeStaticGeometryConsistencyTests(unittest.TestCase):
    def test_keypoint_anchor_bundle_matches_uniform_grid_bundle(self):
        depth_frames = np.ones((2, 5, 5), dtype=np.float32)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)

        uniform_bundle = build_query_anchor_bundle(
            depth_frames,
            intrinsics,
            extrinsics,
            query_frame=0,
            grid_size=3,
            min_query_depth_m=0.2,
            min_border_dist_px=0.0,
        )
        keypoint_bundle = build_query_anchor_bundle_from_keypoints(
            depth_frames,
            intrinsics,
            extrinsics,
            keypoints=np.asarray(uniform_bundle["keypoints"], dtype=np.float32),
            query_frame=0,
            min_query_depth_m=0.2,
            min_border_dist_px=0.0,
        )

        np.testing.assert_allclose(keypoint_bundle["keypoints"], uniform_bundle["keypoints"], atol=1e-6)
        np.testing.assert_allclose(keypoint_bundle["world_points"], uniform_bundle["world_points"], atol=1e-6)
        np.testing.assert_array_equal(keypoint_bundle["anchor_mask"], uniform_bundle["anchor_mask"])

    def test_perfect_static_plane_has_zero_depth_and_world_error(self):
        depth_frames = np.ones((3, 5, 5), dtype=np.float32)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)

        result = compute_static_geometry_consistency(
            depth_frames,
            intrinsics,
            extrinsics,
            query_frame=0,
            grid_size=3,
            min_query_depth_m=0.2,
            min_border_dist_px=0.0,
        )

        self.assertEqual(result["anchor_count"], 9)
        np.testing.assert_allclose(result["per_frame_depth_error_median_m"], np.zeros(3, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(result["per_frame_world_error_median_m"], np.zeros(3, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(result["per_frame_in_bounds_ratio"], np.ones(3, dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(
            result["per_frame_query_reproj_drift_median_px"],
            np.zeros(3, dtype=np.float32),
            atol=1e-6,
        )

    def test_temporal_median_world_points_can_recover_from_noisy_query_depth(self):
        depth_frames = np.ones((3, 5, 5), dtype=np.float32)
        depth_frames[0, 2, 2] = 2.0
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)

        bundle = build_query_anchor_bundle(
            depth_frames,
            intrinsics,
            extrinsics,
            query_frame=0,
            grid_size=3,
            min_query_depth_m=0.2,
            min_border_dist_px=0.0,
        )
        keypoints = np.asarray(bundle["keypoints"], dtype=np.float32)
        center_idx = int(np.argmin(np.linalg.norm(keypoints - np.array([2.0, 2.0], dtype=np.float32), axis=1)))
        self.assertTrue(bool(np.asarray(bundle["anchor_mask"], dtype=bool)[center_idx]))
        self.assertAlmostEqual(float(np.asarray(bundle["world_points"], dtype=np.float32)[center_idx, 2]), 2.0, places=6)

        stabilized = estimate_temporal_median_world_points(
            depth_frames,
            intrinsics,
            extrinsics,
            query_anchor_bundle=bundle,
            reproj_tol_px=0.5,
            min_support=2,
        )

        self.assertTrue(bool(stabilized["replace_mask"][center_idx]))
        self.assertGreaterEqual(int(stabilized["support_counts"][center_idx]), 2)
        self.assertAlmostEqual(float(stabilized["world_points"][center_idx, 2]), 1.0, places=6)

        stabilized_bundle = dict(bundle)
        stabilized_bundle["world_points"] = np.asarray(stabilized["world_points"], dtype=np.float32)
        result = compute_static_geometry_consistency(
            depth_frames,
            intrinsics,
            extrinsics,
            query_frame=0,
            query_anchor_bundle=stabilized_bundle,
            grid_size=3,
            min_query_depth_m=0.2,
            min_border_dist_px=0.0,
        )
        self.assertLessEqual(float(result["final_query_reproj_drift_median_px"]), 1e-6)

    def test_dense_depth_temporal_median_reproject_recovers_noisy_center_depth(self):
        depth_frames = np.ones((3, 5, 5), dtype=np.float32)
        depth_frames[1, 2, 2] = 2.0
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)

        stabilized = stabilize_depth_frames_temporal_median_reproject(
            depth_frames,
            intrinsics,
            extrinsics,
            radius=1,
            min_support=3,
            min_depth=0.2,
            max_depth=10.0,
        )

        self.assertEqual(stabilized["depth_frames"].shape, depth_frames.shape)
        self.assertAlmostEqual(float(stabilized["depth_frames"][1, 2, 2]), 1.0, places=6)
        self.assertEqual(int(stabilized["replace_count"][1]), 25)
        self.assertAlmostEqual(float(stabilized["replace_ratio"][1]), 1.0, places=6)
        self.assertAlmostEqual(float(stabilized["support_count_median"][1]), 3.0, places=6)
        self.assertAlmostEqual(float(stabilized["depth_delta_p95_m"][1]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
