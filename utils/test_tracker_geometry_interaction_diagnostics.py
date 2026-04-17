import unittest

import numpy as np

from utils.tracker_geometry_interaction_diagnostics import (
    compute_static_geometry_track_drift,
    summarize_tracker_geometry_interaction,
)


class TrackerGeometryInteractionDiagnosticsTests(unittest.TestCase):
    def test_static_geometry_track_drift_is_zero_for_perfect_plane(self):
        depth_frames = np.ones((3, 5, 5), dtype=np.float32)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
        keypoints = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)

        result = compute_static_geometry_track_drift(
            depth_frames,
            intrinsics,
            extrinsics,
            keypoints=keypoints,
            min_query_depth_m=0.2,
            min_border_dist_px=0.0,
        )

        np.testing.assert_allclose(result["final_query_reproj_drift_px"], np.zeros(3, dtype=np.float32), atol=1e-6)
        np.testing.assert_array_equal(result["final_query_reproj_valid"], np.ones(3, dtype=bool))

    def test_summary_flags_tracker_local_interaction_when_geometry_is_stable(self):
        keypoints = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        traj_uvz = np.array(
            [
                [[1.0, 1.0, 1.0], [1.1, 1.1, 1.0], [5.5, 5.5, 1.0]],
                [[2.0, 2.0, 1.0], [2.0, 2.0, 1.0], [2.1, 2.1, 1.0]],
            ],
            dtype=np.float32,
        )
        summary = summarize_tracker_geometry_interaction(
            traj_uvz=traj_uvz,
            keypoints=keypoints,
            static_geometry_drift_px=np.array([0.2, 0.1], dtype=np.float32),
            static_geometry_valid=np.array([True, True]),
            traj_valid_mask=np.array([True, True]),
            geom_stable_threshold_px=1.0,
            tracker_unstable_threshold_px=3.0,
            excess_threshold_px=2.0,
        )

        np.testing.assert_array_equal(summary["tracker_local_interaction_mask"], np.array([True, False]))
        np.testing.assert_array_equal(summary["geometry_limited_mask"], np.array([False, False]))

    def test_summary_uses_last_valid_step_when_valid_steps_is_present(self):
        keypoints = np.array([[1.0, 1.0]], dtype=np.float32)
        traj_uvz = np.array(
            [
                [
                    [1.0, 1.0, 1.0],
                    [5.5, 5.5, 1.0],
                    [np.inf, np.inf, np.inf],
                ]
            ],
            dtype=np.float32,
        )
        summary = summarize_tracker_geometry_interaction(
            traj_uvz=traj_uvz,
            keypoints=keypoints,
            static_geometry_drift_px=np.array([0.2], dtype=np.float32),
            static_geometry_valid=np.array([True], dtype=bool),
            traj_valid_mask=np.array([True], dtype=bool),
            valid_steps=np.array([True, True, False], dtype=bool),
            geom_stable_threshold_px=1.0,
            tracker_unstable_threshold_px=3.0,
            excess_threshold_px=2.0,
        )

        self.assertEqual(summary["final_step_index"], 1)
        self.assertEqual(summary["tracker_final_drift_summary"]["finite_count"], 1)
        np.testing.assert_array_equal(summary["tracker_local_interaction_mask"], np.array([True]))


if __name__ == "__main__":
    unittest.main()
