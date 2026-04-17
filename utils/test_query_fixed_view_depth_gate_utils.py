from __future__ import annotations

import unittest

import numpy as np

from utils.query_fixed_view_depth_gate_utils import (
    compute_query_fixed_view_depth_gate,
    summarize_query_fixed_view_depth_gate,
)


class QueryFixedViewDepthGateUtilsTest(unittest.TestCase):
    def _build_intrinsics(self, frame_count: int) -> np.ndarray:
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], frame_count, axis=0)
        intrinsics[:, 0, 0] = 4.0
        intrinsics[:, 1, 1] = 4.0
        intrinsics[:, 0, 2] = 4.0
        intrinsics[:, 1, 2] = 4.0
        return intrinsics

    def test_static_camera_without_depth_jump_keeps_track(self) -> None:
        depth_frames = np.ones((3, 9, 9), dtype=np.float32)
        intrinsics = self._build_intrinsics(3)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 3, axis=0)
        keypoints = np.array([[4.0, 4.0]], dtype=np.float32)

        result = compute_query_fixed_view_depth_gate(
            depth_frames,
            intrinsics,
            extrinsics,
            keypoints=keypoints,
            query_frame=0,
        )

        self.assertTrue(bool(result["query_world_valid_mask"][0]))
        self.assertTrue(bool(result["reliable_track_mask"][0]))
        self.assertEqual(int(result["depth_anomaly_hit_count"][0]), 0)
        np.testing.assert_array_equal(result["first_anomaly_step"], np.array([-1], dtype=np.int32))

    def test_static_camera_depth_jump_marks_track_unreliable(self) -> None:
        depth_frames = np.ones((2, 9, 9), dtype=np.float32)
        depth_frames[1, 4, 4] = 1.25
        intrinsics = self._build_intrinsics(2)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
        keypoints = np.array([[4.0, 4.0]], dtype=np.float32)

        result = compute_query_fixed_view_depth_gate(
            depth_frames,
            intrinsics,
            extrinsics,
            keypoints=keypoints,
            query_frame=0,
            uv_threshold_px=1.0,
            depth_threshold_m=0.10,
        )
        summary = summarize_query_fixed_view_depth_gate(
            gate_result=result,
            tracked_mask=np.array([True], dtype=bool),
        )

        self.assertTrue(bool(result["query_world_valid_mask"][0]))
        self.assertFalse(bool(result["reliable_track_mask"][0]))
        self.assertEqual(int(result["compare_frame_count"][0]), 2)
        self.assertEqual(int(result["uv_stable_hit_count"][0]), 2)
        self.assertEqual(int(result["depth_jump_hit_count"][0]), 1)
        self.assertEqual(int(result["depth_anomaly_hit_count"][0]), 1)
        self.assertAlmostEqual(float(result["max_depth_delta_m"][0]), 0.25, places=5)
        np.testing.assert_array_equal(result["first_anomaly_step"], np.array([1], dtype=np.int32))
        self.assertEqual(summary["removed_tracked_count"], 1)
        self.assertEqual(summary["removed_tracked_first_anomaly_histogram"], {"1": 1})

    def test_subthreshold_depth_delta_does_not_trigger_gate(self) -> None:
        depth_frames = np.ones((2, 9, 9), dtype=np.float32)
        depth_frames[1, 4, 4] = 1.05
        intrinsics = self._build_intrinsics(2)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
        keypoints = np.array([[4.0, 4.0]], dtype=np.float32)

        result = compute_query_fixed_view_depth_gate(
            depth_frames,
            intrinsics,
            extrinsics,
            keypoints=keypoints,
            query_frame=0,
            uv_threshold_px=1.0,
            depth_threshold_m=0.10,
        )

        self.assertTrue(bool(result["reliable_track_mask"][0]))
        self.assertEqual(int(result["depth_anomaly_hit_count"][0]), 0)
        np.testing.assert_array_equal(result["first_anomaly_step"], np.array([-1], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
