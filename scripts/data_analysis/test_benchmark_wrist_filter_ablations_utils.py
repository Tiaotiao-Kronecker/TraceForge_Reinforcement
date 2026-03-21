import unittest
from unittest.mock import patch

from scripts.data_analysis.benchmark_wrist_filter_ablations import (
    DEFAULT_EXTERNAL_CAMERA_NAME,
    DEFAULT_WRIST_CAMERA_NAME,
    WRIST_MANIPULATOR_STAGE_ORDER,
    build_save_variant_specs,
    choose_recommended_variant,
    parse_args,
    summarize_stage_debug_records,
)


class VariantSpecTests(unittest.TestCase):
    def test_parse_args_defaults_support_grid_ratio_to_zero(self):
        with patch("sys.argv", ["benchmark_wrist_filter_ablations.py"]):
            args = parse_args()

        self.assertEqual(args.support_grid_ratio, 0.0)

    def test_build_save_variant_specs_includes_external_control_and_wrist_ablations(self):
        specs = build_save_variant_specs(
            external_camera_name=DEFAULT_EXTERNAL_CAMERA_NAME,
            wrist_camera_name=DEFAULT_WRIST_CAMERA_NAME,
        )

        self.assertEqual(specs[0]["name"], "external_control")
        self.assertEqual(specs[0]["camera_name"], DEFAULT_EXTERNAL_CAMERA_NAME)
        self.assertEqual(specs[0]["traj_filter_profile"], "external")

        by_name = {spec["name"]: spec for spec in specs}
        self.assertEqual(by_name["wrist_external"]["traj_filter_profile"], "external")
        self.assertEqual(by_name["wrist_seed_top95"]["traj_filter_ablation_mode"], "wrist_seed_top95")
        self.assertEqual(by_name["wrist_no_query_edge"]["traj_filter_ablation_mode"], "wrist_no_query_edge")
        self.assertEqual(
            tuple(by_name["wrist_manipulator_top95"]["stage_order"]),
            WRIST_MANIPULATOR_STAGE_ORDER,
        )


class StageSummaryTests(unittest.TestCase):
    def test_summarize_stage_debug_records_reports_stage_counts_and_drops(self):
        records = [
            {
                "query_frame_index": 7,
                "dense_query_count": 100,
                "tracked_query_count": 100,
                "valid_track_count": 18,
                "stage_counts": {
                    "base_mask": 80,
                    "query_depth_quality": 70,
                    "query_depth_keep": 60,
                    "supervision_support": 50,
                    "wrist_seed": 50,
                    "near_depth": 30,
                    "motion": 24,
                    "cluster": 20,
                    "pre_top95": 20,
                    "final": 18,
                },
            },
            {
                "query_frame_index": 15,
                "dense_query_count": 100,
                "tracked_query_count": 100,
                "valid_track_count": 20,
                "stage_counts": {
                    "base_mask": 82,
                    "query_depth_quality": 74,
                    "query_depth_keep": 64,
                    "supervision_support": 54,
                    "wrist_seed": 54,
                    "near_depth": 32,
                    "motion": 26,
                    "cluster": 22,
                    "pre_top95": 22,
                    "final": 20,
                },
            },
        ]

        summary = summarize_stage_debug_records(
            records,
            stage_order=WRIST_MANIPULATOR_STAGE_ORDER,
        )

        self.assertEqual(summary["sample_count"], 2)
        self.assertAlmostEqual(summary["dense_query_count_mean"], 100.0)
        self.assertAlmostEqual(summary["stage_count_means"]["base_mask"], 81.0)
        self.assertAlmostEqual(summary["stage_count_means"]["final"], 19.0)
        self.assertAlmostEqual(summary["stage_drop_means"]["base_mask"], 19.0)
        self.assertAlmostEqual(summary["stage_drop_means"]["query_depth_quality"], 9.0)
        self.assertAlmostEqual(summary["stage_drop_means"]["final"], 2.0)


class RecommendationTests(unittest.TestCase):
    def test_choose_recommended_variant_picks_simplest_candidate_that_meets_thresholds(self):
        pairwise_aggregates = [
            {
                "variant_name": "wrist_external",
                "save_seconds_mean": 6.0,
                "traj_valid_mask_jaccard_mean": 0.85,
                "valid_track_count_delta_mean": -600.0,
                "dense_query_count_mean": 6400.0,
            },
            {
                "variant_name": "wrist",
                "save_seconds_mean": 5.1,
                "traj_valid_mask_jaccard_mean": 0.93,
                "valid_track_count_delta_mean": -120.0,
                "dense_query_count_mean": 6400.0,
            },
            {
                "variant_name": "wrist_seed_top95",
                "save_seconds_mean": 5.4,
                "traj_valid_mask_jaccard_mean": 0.97,
                "valid_track_count_delta_mean": -20.0,
                "dense_query_count_mean": 6400.0,
            },
        ]
        variant_rows = [
            {"variant_name": "external_control", "save_seconds_mean": 4.8},
            {"variant_name": "wrist_external", "save_seconds_mean": 6.0},
            {"variant_name": "wrist", "save_seconds_mean": 5.1},
            {"variant_name": "wrist_seed_top95", "save_seconds_mean": 5.4},
        ]

        recommendation = choose_recommended_variant(
            pairwise_aggregates=pairwise_aggregates,
            variant_rows=variant_rows,
        )

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation["variant_name"], "wrist")


if __name__ == "__main__":
    unittest.main()
