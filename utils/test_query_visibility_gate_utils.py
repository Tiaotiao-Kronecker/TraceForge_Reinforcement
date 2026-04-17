from __future__ import annotations

import unittest

import numpy as np

from utils.query_visibility_gate_utils import (
    compute_query_visibility_gate,
    summarize_query_visibility_gate,
)


class QueryVisibilityGateUtilsTest(unittest.TestCase):
    def test_static_camera_keeps_center_track(self) -> None:
        depth_frames = np.ones((3, 9, 9), dtype=np.float32)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 3, axis=0)
        intrinsics[:, 0, 0] = 4.0
        intrinsics[:, 1, 1] = 4.0
        intrinsics[:, 0, 2] = 4.0
        intrinsics[:, 1, 2] = 4.0
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
        keypoints = np.array([[4.0, 4.0]], dtype=np.float32)

        result = compute_query_visibility_gate(
            depth_frames,
            intrinsics,
            extrinsics,
            keypoints=keypoints,
            query_frame=0,
        )

        self.assertTrue(bool(result["query_world_valid_mask"][0]))
        self.assertTrue(bool(result["reliable_track_mask"][0]))
        np.testing.assert_array_equal(result["first_invalid_step"], np.array([-1], dtype=np.int32))

    def test_camera_shift_marks_track_unreliable(self) -> None:
        depth_frames = np.ones((2, 9, 9), dtype=np.float32)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0)
        intrinsics[:, 0, 0] = 4.0
        intrinsics[:, 1, 1] = 4.0
        intrinsics[:, 0, 2] = 4.0
        intrinsics[:, 1, 2] = 4.0
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
        extrinsics[1, 0, 3] = 2.0
        keypoints = np.array([[4.0, 4.0]], dtype=np.float32)

        result = compute_query_visibility_gate(
            depth_frames,
            intrinsics,
            extrinsics,
            keypoints=keypoints,
            query_frame=0,
        )

        self.assertTrue(bool(result["query_world_valid_mask"][0]))
        self.assertFalse(bool(result["reliable_track_mask"][0]))
        self.assertFalse(bool(result["projected_in_bounds_mask"][0, 1]))
        np.testing.assert_array_equal(result["first_invalid_step"], np.array([1], dtype=np.int32))

    def test_invalid_query_depth_is_removed(self) -> None:
        depth_frames = np.ones((2, 9, 9), dtype=np.float32)
        depth_frames[0, 4, 4] = 0.0
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0)
        intrinsics[:, 0, 0] = 4.0
        intrinsics[:, 1, 1] = 4.0
        intrinsics[:, 0, 2] = 4.0
        intrinsics[:, 1, 2] = 4.0
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
        keypoints = np.array([[4.0, 4.0]], dtype=np.float32)

        result = compute_query_visibility_gate(
            depth_frames,
            intrinsics,
            extrinsics,
            keypoints=keypoints,
            query_frame=0,
        )
        summary = summarize_query_visibility_gate(
            gate_result=result,
            traj_valid_mask=np.array([True], dtype=bool),
        )

        self.assertFalse(bool(result["query_world_valid_mask"][0]))
        self.assertFalse(bool(result["reliable_track_mask"][0]))
        np.testing.assert_array_equal(result["first_invalid_step"], np.array([0], dtype=np.int32))
        self.assertEqual(summary["removed_tracked_count"], 1)
        self.assertEqual(summary["removed_tracked_first_invalid_histogram"], {"0": 1})


if __name__ == "__main__":
    unittest.main()
