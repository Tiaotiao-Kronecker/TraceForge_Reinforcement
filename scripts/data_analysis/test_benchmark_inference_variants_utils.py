import unittest

import numpy as np

from scripts.data_analysis.benchmark_inference_variants import (
    build_variant_specs,
    compare_sample_pair,
    parse_support_grid_ratios,
    summarize_case_samples,
    summarize_sample,
)


def _make_camera_arrays(length: int = 8) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], length, axis=0)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], length, axis=0)
    return intrinsics, extrinsics


def _make_sample(
    *,
    dense_query_count: int = 4,
    tracked_query_count: int = 2,
    valid_mask: np.ndarray | None = None,
    reason_bits: np.ndarray | None = None,
    supervision_count: np.ndarray | None = None,
    supervision_mask: np.ndarray | None = None,
    traj_uvz: np.ndarray | None = None,
    query_frame_index: int = 1,
) -> dict:
    if traj_uvz is None:
        traj_uvz = np.array(
            [
                [[1.0, 1.0, 1.0], [2.0, 2.0, 1.0]],
                [[3.0, 3.0, 1.0], [4.0, 4.0, 1.0]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            ],
            dtype=np.float32,
        )
    if valid_mask is None:
        valid_mask = np.array([True, False, False, False], dtype=bool)
    if reason_bits is None:
        reason_bits = np.array([0, 2, 128, 130], dtype=np.uint8)
    if supervision_count is None:
        supervision_count = np.array([2, 1, 0, 0], dtype=np.uint16)
    if supervision_mask is None:
        supervision_mask = np.array(
            [
                [True, True],
                [True, False],
                [False, False],
                [False, False],
            ],
            dtype=bool,
        )
    return {
        "traj_uvz": np.asarray(traj_uvz, dtype=np.float32),
        "traj_valid_mask": np.asarray(valid_mask, dtype=bool),
        "traj_mask_reason_bits": np.asarray(reason_bits, dtype=np.uint8),
        "traj_supervision_count": np.asarray(supervision_count, dtype=np.uint16),
        "traj_supervision_mask": np.asarray(supervision_mask, dtype=bool),
        "query_frame_index": query_frame_index,
        "dense_query_count": int(dense_query_count),
        "tracked_query_count": int(tracked_query_count),
        "support_grid_size": 48,
    }


class VariantSpecTests(unittest.TestCase):
    def test_parse_support_grid_ratios_requires_baseline(self):
        with self.assertRaises(ValueError):
            parse_support_grid_ratios("0.6,0.4,0.0")

    def test_build_variant_specs_generates_support_sweep_names(self):
        specs = build_variant_specs(
            support_grid_ratios=parse_support_grid_ratios("0.8,0.6,0.05,0.0"),
        )

        self.assertEqual([spec["name"] for spec in specs], ["baseline", "support_r060", "support_r005", "support_r000"])
        self.assertEqual(specs[0]["support_grid_ratio"], 0.8)
        self.assertEqual(specs[1]["support_grid_ratio"], 0.6)
        self.assertEqual(specs[2]["support_grid_ratio"], 0.05)
        self.assertEqual(specs[3]["support_grid_ratio"], 0.0)
        self.assertTrue(all(spec["query_prefilter_mode"] == "off" for spec in specs))


class SampleSummaryTests(unittest.TestCase):
    def test_summarize_sample_reports_dense_and_reason_metrics(self):
        summary = summarize_sample(_make_sample())

        self.assertEqual(summary["dense_query_count"], 4)
        self.assertEqual(summary["tracked_query_count"], 2)
        self.assertEqual(summary["valid_track_count"], 1)
        self.assertAlmostEqual(summary["tracked_ratio"], 0.5)
        self.assertAlmostEqual(summary["valid_ratio_dense"], 0.25)
        self.assertAlmostEqual(summary["valid_ratio_tracked"], 0.5)
        self.assertEqual(summary["traj_mask_reason_bit_counts"]["bit_1"], 2)
        self.assertEqual(summary["traj_mask_reason_bit_counts"]["bit_7"], 2)

    def test_summarize_case_samples_aggregates_support_sizes(self):
        overview = summarize_case_samples(
            {
                "7": summarize_sample(_make_sample()),
                "15": summarize_sample(_make_sample(tracked_query_count=3, dense_query_count=4)),
            }
        )

        self.assertEqual(overview["sample_count"], 2)
        self.assertEqual(overview["support_grid_size_set"], [48])
        self.assertAlmostEqual(overview["tracked_query_count_mean"], 2.5)


class PairwiseComparisonTests(unittest.TestCase):
    def test_compare_sample_pair_reports_jaccard_and_world_metrics(self):
        baseline = _make_sample(
            valid_mask=np.array([True, True, False, False], dtype=bool),
            tracked_query_count=2,
        )
        variant_traj = np.array(
            [
                [[1.2, 1.0, 1.0], [2.2, 2.0, 1.0]],
                [[3.0, 3.0, 1.0], [4.0, 4.0, 1.0]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            ],
            dtype=np.float32,
        )
        variant = _make_sample(
            valid_mask=np.array([True, False, False, False], dtype=bool),
            tracked_query_count=2,
            traj_uvz=variant_traj,
        )

        comparison = compare_sample_pair(
            baseline,
            variant,
            baseline_camera_arrays=_make_camera_arrays(),
            variant_camera_arrays=_make_camera_arrays(),
        )

        self.assertEqual(comparison["tracked_query_count_delta"], 0)
        self.assertEqual(comparison["valid_track_count_delta"], -1)
        self.assertAlmostEqual(comparison["traj_valid_mask_jaccard"], 0.5)
        self.assertAlmostEqual(comparison["traj_uvz_mae"], 0.06666667, places=6)
        self.assertAlmostEqual(comparison["traj_2d_l2_mean"], 0.2, places=6)
        self.assertAlmostEqual(comparison["traj_depth_abs_mean"], 0.0, places=6)
        self.assertAlmostEqual(comparison["traj_world_l2_mean"], 0.2, places=6)
        self.assertAlmostEqual(comparison["traj_world_l2_p95"], 0.2, places=6)
        self.assertAlmostEqual(comparison["traj_world_error_var_mean"], 0.0, places=6)
        self.assertAlmostEqual(comparison["traj_world_endpoint_l2_mean"], 0.2, places=6)
        self.assertAlmostEqual(comparison["traj_world_step_delta_l2_mean"], 0.0, places=6)

    def test_compare_sample_pair_identical_samples_have_zero_error(self):
        sample = _make_sample(
            valid_mask=np.array([True, True, False, False], dtype=bool),
            tracked_query_count=2,
        )
        comparison = compare_sample_pair(
            sample,
            sample,
            baseline_camera_arrays=_make_camera_arrays(),
            variant_camera_arrays=_make_camera_arrays(),
        )

        self.assertEqual(comparison["common_valid_track_count"], 2)
        self.assertEqual(comparison["common_valid_step_count"], 3)
        self.assertAlmostEqual(comparison["traj_valid_mask_jaccard"], 1.0)
        self.assertAlmostEqual(comparison["traj_uvz_mae"], 0.0)
        self.assertAlmostEqual(comparison["traj_world_l2_mean"], 0.0)
        self.assertAlmostEqual(comparison["traj_world_step_delta_l2_mean"], 0.0)

    def test_compare_sample_pair_requires_two_steps_for_error_variance(self):
        sample = _make_sample(
            valid_mask=np.array([True, False, False, False], dtype=bool),
            tracked_query_count=1,
            supervision_mask=np.array(
                [
                    [True, False],
                    [False, False],
                    [False, False],
                    [False, False],
                ],
                dtype=bool,
            ),
        )
        comparison = compare_sample_pair(
            sample,
            sample,
            baseline_camera_arrays=_make_camera_arrays(),
            variant_camera_arrays=_make_camera_arrays(),
        )

        self.assertIsNone(comparison["traj_world_error_var_mean"])


if __name__ == "__main__":
    unittest.main()
