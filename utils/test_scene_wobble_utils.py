import unittest

import numpy as np

from utils.scene_wobble_utils import compute_scene_wobble_summary


class ComputeSceneWobbleSummaryTests(unittest.TestCase):
    def test_detects_common_fixed_view_drift_over_deep_central_anchors(self):
        keypoints = np.array(
            [
                [100.0, 100.0],
                [140.0, 100.0],
                [180.0, 100.0],
                [220.0, 100.0],
                [10.0, 10.0],
            ],
            dtype=np.float32,
        )
        traj_uvz = np.array(
            [
                [[100.0, 100.0, 1.0], [102.0, 100.0, 1.0], [104.0, 100.0, 1.0]],
                [[140.0, 100.0, 1.2], [142.0, 100.0, 1.2], [144.0, 100.0, 1.2]],
                [[180.0, 100.0, 1.4], [182.0, 100.0, 1.4], [184.0, 100.0, 1.4]],
                [[220.0, 100.0, 1.6], [222.0, 100.0, 1.6], [224.0, 100.0, 1.6]],
                [[10.0, 10.0, 0.01], [50.0, 10.0, 0.01], [90.0, 10.0, 0.01]],
            ],
            dtype=np.float32,
        )

        summary = compute_scene_wobble_summary(
            traj_uvz,
            traj_valid_mask=np.ones(5, dtype=bool),
            query_border_dist_px=np.array([100.0, 100.0, 100.0, 100.0, 10.0], dtype=np.float32),
            min_query_depth_m=0.2,
            min_border_dist_px=60.0,
            min_anchor_count=4,
            global_disp_threshold_px=3.0,
        )

        self.assertEqual(summary["anchor_count"], 4)
        self.assertTrue(summary["has_sufficient_anchors"])
        self.assertTrue(summary["geometry_unstable"])
        self.assertEqual(summary["final_step_index"], 2)
        self.assertEqual(summary["final_anchor_count"], 4)
        self.assertAlmostEqual(summary["global_final_disp_px"], 4.0, places=4)
        self.assertAlmostEqual(summary["residual_final_p95_px"], 0.0, places=4)
        np.testing.assert_array_equal(
            summary["anchor_mask"],
            np.array([True, True, True, True, False]),
        )

    def test_uses_last_true_valid_step_for_final_metrics(self):
        keypoints = np.array([[100.0, 100.0], [140.0, 100.0]], dtype=np.float32)
        traj_uvz = np.array(
            [
                [[100.0, 100.0, 1.0], [101.0, 100.0, 1.0], [999.0, 100.0, 1.0]],
                [[140.0, 100.0, 1.0], [141.0, 100.0, 1.0], [999.0, 100.0, 1.0]],
            ],
            dtype=np.float32,
        )

        summary = compute_scene_wobble_summary(
            traj_uvz,
            query_border_dist_px=np.array([100.0, 100.0], dtype=np.float32),
            valid_steps=np.array([True, True, False]),
            min_anchor_count=2,
            global_disp_threshold_px=0.5,
        )

        self.assertEqual(summary["final_step_index"], 1)
        self.assertAlmostEqual(summary["global_final_disp_px"], 1.0, places=4)

    def test_requires_keypoints_when_border_distance_is_missing(self):
        traj_uvz = np.ones((2, 3, 3), dtype=np.float32)

        with self.assertRaises(ValueError):
            compute_scene_wobble_summary(traj_uvz)


if __name__ == "__main__":
    unittest.main()
