from __future__ import annotations

import unittest

import numpy as np

from utils.traj_uvd_gate_utils import (
    compute_traj_uvd_motion_gate,
    summarize_traj_uvd_motion_gate,
)


class TrajUvdGateUtilsTest(unittest.TestCase):
    def test_low_uv_high_depth_std_marks_track_unreliable(self) -> None:
        traj_uvz = np.array(
            [
                [
                    [10.0, 10.0, 1.00],
                    [10.5, 10.0, 1.02],
                    [11.0, 10.0, 1.20],
                    [11.5, 10.0, 1.21],
                ]
            ],
            dtype=np.float32,
        )

        result = compute_traj_uvd_motion_gate(
            traj_uvz,
            uv_mean_threshold_px=3.0,
            depth_std_threshold_m=0.01,
            max_depth_threshold_m=1.5,
        )
        summary = summarize_traj_uvd_motion_gate(
            gate_result=result,
            traj_valid_mask=np.array([True], dtype=bool),
        )

        self.assertFalse(bool(result["reliable_track_mask"][0]))
        self.assertTrue(bool(result["uv_depth_anomaly_mask"][0]))
        self.assertFalse(bool(result["far_depth_mask"][0]))
        self.assertAlmostEqual(float(result["uv_mean_delta_px"][0]), 0.5, places=5)
        self.assertGreater(float(result["depth_delta_std_m"][0]), 0.01)
        self.assertEqual(summary["removed_valid_count"], 1)

    def test_far_depth_marks_track_unreliable(self) -> None:
        traj_uvz = np.array(
            [
                [
                    [10.0, 10.0, 1.40],
                    [15.0, 10.0, 1.60],
                    [20.0, 10.0, 1.70],
                ]
            ],
            dtype=np.float32,
        )

        result = compute_traj_uvd_motion_gate(
            traj_uvz,
            uv_mean_threshold_px=3.0,
            depth_std_threshold_m=0.01,
            max_depth_threshold_m=1.5,
        )

        self.assertFalse(bool(result["reliable_track_mask"][0]))
        self.assertFalse(bool(result["uv_depth_anomaly_mask"][0]))
        self.assertTrue(bool(result["far_depth_mask"][0]))
        self.assertAlmostEqual(float(result["max_depth_m"][0]), 1.7, places=5)

    def test_large_uv_motion_does_not_trigger_uv_depth_anomaly(self) -> None:
        traj_uvz = np.array(
            [
                [
                    [10.0, 10.0, 1.00],
                    [20.0, 10.0, 1.10],
                    [30.0, 10.0, 1.25],
                    [40.0, 10.0, 1.30],
                ]
            ],
            dtype=np.float32,
        )

        result = compute_traj_uvd_motion_gate(
            traj_uvz,
            uv_mean_threshold_px=3.0,
            depth_std_threshold_m=0.01,
            max_depth_threshold_m=2.0,
        )

        self.assertTrue(bool(result["reliable_track_mask"][0]))
        self.assertFalse(bool(result["uv_depth_anomaly_mask"][0]))
        self.assertFalse(bool(result["far_depth_mask"][0]))
        self.assertGreater(float(result["uv_mean_delta_px"][0]), 3.0)

    def test_near_depth_relaxed_std_threshold_recovers_near_track(self) -> None:
        traj_uvz = np.array(
            [
                [
                    [10.0, 10.0, 0.55],
                    [10.5, 10.0, 0.57],
                    [11.0, 10.0, 0.59],
                    [11.5, 10.0, 0.61],
                    [12.0, 10.0, 0.653],
                ]
            ],
            dtype=np.float32,
        )

        strict = compute_traj_uvd_motion_gate(
            traj_uvz,
            uv_mean_threshold_px=3.0,
            depth_std_threshold_m=0.01,
            max_depth_threshold_m=1.5,
        )
        relaxed = compute_traj_uvd_motion_gate(
            traj_uvz,
            uv_mean_threshold_px=3.0,
            depth_std_threshold_m=0.01,
            max_depth_threshold_m=1.5,
            near_depth_threshold_m=0.8,
            near_depth_relaxed_std_threshold_m=0.015,
        )

        self.assertFalse(bool(strict["reliable_track_mask"][0]))
        self.assertTrue(bool(strict["uv_depth_anomaly_mask"][0]))
        self.assertTrue(bool(relaxed["reliable_track_mask"][0]))
        self.assertFalse(bool(relaxed["uv_depth_anomaly_mask"][0]))
        self.assertTrue(bool(relaxed["near_depth_relaxed_mask"][0]))
        self.assertAlmostEqual(float(relaxed["effective_depth_std_threshold_m"][0]), 0.015, places=6)

    def test_near_depth_exempt_threshold_bypasses_uv_depth_anomaly(self) -> None:
        traj_uvz = np.array(
            [
                [
                    [10.0, 10.0, 0.55],
                    [10.5, 10.0, 0.57],
                    [11.0, 10.0, 0.59],
                    [11.5, 10.0, 0.61],
                    [12.0, 10.0, 0.70],
                ]
            ],
            dtype=np.float32,
        )

        strict = compute_traj_uvd_motion_gate(
            traj_uvz,
            uv_mean_threshold_px=3.0,
            depth_std_threshold_m=0.01,
            max_depth_threshold_m=1.5,
        )
        exempt = compute_traj_uvd_motion_gate(
            traj_uvz,
            uv_mean_threshold_px=3.0,
            depth_std_threshold_m=0.01,
            max_depth_threshold_m=1.5,
            near_depth_exempt_threshold_m=0.8,
        )

        self.assertFalse(bool(strict["reliable_track_mask"][0]))
        self.assertTrue(bool(strict["uv_depth_anomaly_mask"][0]))
        self.assertTrue(bool(exempt["reliable_track_mask"][0]))
        self.assertFalse(bool(exempt["uv_depth_anomaly_mask"][0]))
        self.assertTrue(bool(exempt["near_depth_exempt_mask"][0]))


if __name__ == "__main__":
    unittest.main()
