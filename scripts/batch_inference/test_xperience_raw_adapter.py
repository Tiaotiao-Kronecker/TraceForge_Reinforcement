import sys
import unittest
from pathlib import Path

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.batch_inference.adapters.xperience_raw import _sample_window_query_source_indices
from scripts.batch_inference.adapters.xperience_raw import parse_args


class SampleWindowQuerySourceIndicesTests(unittest.TestCase):
    def test_samples_against_window_relative_time_and_keeps_window_start(self):
        source_frame_indices = np.arange(100, 200, dtype=np.int32)

        candidate_source_indices, short_tail_source_indices, sampled_source_indices = (
            _sample_window_query_source_indices(
                source_frame_indices=source_frame_indices,
                episode_fps=20.0,
                keyframes_per_sec_min=2,
                keyframes_per_sec_max=2,
                seed=7,
            )
        )

        np.testing.assert_array_equal(
            candidate_source_indices,
            np.arange(100, 192, dtype=np.int32),
        )
        np.testing.assert_array_equal(
            short_tail_source_indices,
            np.arange(192, 200, dtype=np.int32),
        )
        self.assertEqual(int(sampled_source_indices[0]), 100)
        relative_sampled = sampled_source_indices - int(source_frame_indices[0])
        per_second_counts = [
            int(np.sum((relative_sampled >= sec * 20) & (relative_sampled < (sec + 1) * 20)))
            for sec in range(5)
        ]
        self.assertIn(per_second_counts[0], [2, 3])
        self.assertEqual(per_second_counts[1:], [2, 2, 2, 2])

    def test_raises_when_tail_filter_removes_all_candidates(self):
        with self.assertRaisesRegex(ValueError, "No candidate query frames remain"):
            _sample_window_query_source_indices(
                source_frame_indices=np.arange(8, dtype=np.int32),
                episode_fps=20.0,
                keyframes_per_sec_min=2,
                keyframes_per_sec_max=3,
                seed=0,
            )


class XperienceRawParserTests(unittest.TestCase):
    def test_parse_args_accepts_multi_gpu_batch_flags(self):
        args = parse_args(
            [
                "--dataset_root",
                "/tmp/xperience_dataset",
                "--checkpoint",
                "/tmp/checkpoint.pth",
                "--out_dir",
                "/tmp/xperience_out",
                "--gpu_id",
                "0,2,7",
                "--workers_per_gpu",
                "2",
                "--min_free_gpu_mem_gb",
                "64",
                "--gpu_recovery_poll_sec",
                "15",
                "--telemetry_out_dir",
                "/tmp/xperience_telemetry",
                "--hardware_telemetry_interval_sec",
                "5",
                "--grid_width",
                "96",
                "--grid_height",
                "72",
            ]
        )

        self.assertEqual(args.gpu_id, "0,2,7")
        self.assertEqual(args.workers_per_gpu, 2)
        self.assertEqual(args.min_free_gpu_mem_gb, 64.0)
        self.assertEqual(args.gpu_recovery_poll_sec, 15.0)
        self.assertEqual(args.telemetry_out_dir, "/tmp/xperience_telemetry")
        self.assertEqual(args.hardware_telemetry_interval_sec, 5.0)
        self.assertEqual(args.future_len, 16)
        self.assertEqual(args.query_grid_hw, (72, 96))

    def test_parse_args_rejects_non_positive_workers_per_gpu(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--dataset_root",
                    "/tmp/xperience_dataset",
                    "--checkpoint",
                    "/tmp/checkpoint.pth",
                    "--out_dir",
                    "/tmp/xperience_out",
                    "--workers_per_gpu",
                    "0",
                ]
            )


if __name__ == "__main__":
    unittest.main()
