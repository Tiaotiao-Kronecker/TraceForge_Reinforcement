import unittest

from scripts.data_analysis.compare_traceforge_output_roots import (
    aggregate_quality_rows,
    aggregate_timing_records,
    build_worst_quality_cases,
    validate_identical_manifest_hashes,
    validate_identical_query_frames,
)


class TimingAggregationTests(unittest.TestCase):
    def test_aggregate_timing_records_reports_mean_and_variance(self):
        records = [
            {
                "status": "success",
                "episode_name": "episode_00001_green",
                "camera_name": "varied_camera_1",
                "query_frame_count": 10,
                "process_seconds": 50.0,
                "save_seconds": 2.0,
                "total_seconds": 52.0,
                "process_seconds_per_query": 5.0,
                "save_seconds_per_query": 0.2,
                "total_seconds_per_query": 5.2,
            },
            {
                "status": "success",
                "episode_name": "episode_00002_blue",
                "camera_name": "varied_camera_1",
                "query_frame_count": 8,
                "process_seconds": 48.0,
                "save_seconds": 4.0,
                "total_seconds": 52.0,
                "process_seconds_per_query": 6.0,
                "save_seconds_per_query": 0.5,
                "total_seconds_per_query": 6.5,
            },
            {
                "status": "failed",
                "episode_name": "episode_00002_blue",
                "camera_name": "varied_camera_3",
                "query_frame_count": 8,
                "process_seconds": 10.0,
                "save_seconds": 1.0,
                "total_seconds": 11.0,
                "process_seconds_per_query": 1.25,
                "save_seconds_per_query": 0.125,
                "total_seconds_per_query": 1.375,
            },
        ]

        summary = aggregate_timing_records(
            records,
            episodes=["episode_00001_green", "episode_00002_blue"],
            camera_names=["varied_camera_1", "varied_camera_3"],
        )

        self.assertEqual(summary["task_count"], 2)
        camera_row = summary["by_camera"][0]
        self.assertEqual(camera_row["camera_name"], "varied_camera_1")
        self.assertAlmostEqual(camera_row["metric_summaries"]["query_frame_count"]["mean"], 9.0)
        self.assertAlmostEqual(camera_row["metric_summaries"]["total_seconds_per_query"]["mean"], 5.85)
        self.assertAlmostEqual(camera_row["metric_summaries"]["total_seconds_per_query"]["variance"], 0.845)
        self.assertEqual(summary["by_camera"][1]["task_count"], 0)


class QualityAggregationTests(unittest.TestCase):
    def test_aggregate_quality_rows_groups_by_camera(self):
        rows = [
            {
                "camera_name": "varied_camera_1",
                "valid_track_count_delta_mean": 0.5,
                "traj_valid_mask_jaccard_mean": 0.99,
                "traj_world_l2_mean": 1e-4,
                "traj_world_step_delta_l2_p95": 2e-4,
                "traj_world_l2_p95": 3e-4,
                "traj_world_error_var_mean": 4e-4,
                "traj_world_endpoint_l2_mean": 5e-4,
                "traj_2d_l2_mean": 0.2,
                "traj_depth_abs_mean": 0.1,
                "traj_uvz_mae_mean": 0.05,
                "common_valid_track_count_mean": 100.0,
                "common_valid_step_count_mean": 200.0,
            },
            {
                "camera_name": "varied_camera_1",
                "valid_track_count_delta_mean": 1.5,
                "traj_valid_mask_jaccard_mean": 0.97,
                "traj_world_l2_mean": 2e-4,
                "traj_world_step_delta_l2_p95": 4e-4,
                "traj_world_l2_p95": 6e-4,
                "traj_world_error_var_mean": 8e-4,
                "traj_world_endpoint_l2_mean": 1e-3,
                "traj_2d_l2_mean": 0.4,
                "traj_depth_abs_mean": 0.2,
                "traj_uvz_mae_mean": 0.1,
                "common_valid_track_count_mean": 110.0,
                "common_valid_step_count_mean": 210.0,
            },
        ]

        aggregate_rows = aggregate_quality_rows(rows, camera_names=["varied_camera_1"])
        metric_summaries = aggregate_rows[0]["metric_summaries"]

        self.assertAlmostEqual(metric_summaries["traj_valid_mask_jaccard_mean"]["mean"], 0.98)
        self.assertAlmostEqual(metric_summaries["valid_track_count_delta_mean"]["mean"], 1.0)
        self.assertAlmostEqual(metric_summaries["traj_world_l2_mean"]["mean"], 1.5e-4)

    def test_build_worst_quality_cases_selects_expected_extremes(self):
        rows = [
            {
                "episode_name": "episode_00001_green",
                "camera_name": "varied_camera_1",
                "query_frame": 3,
                "traj_valid_mask_jaccard": 0.95,
                "traj_world_l2_mean": 1e-4,
                "traj_world_step_delta_l2_p95": 2e-4,
            },
            {
                "episode_name": "episode_00002_blue",
                "camera_name": "varied_camera_3",
                "query_frame": 7,
                "traj_valid_mask_jaccard": 0.80,
                "traj_world_l2_mean": 4e-4,
                "traj_world_step_delta_l2_p95": 6e-4,
            },
        ]

        worst_cases = build_worst_quality_cases(rows)
        selectors = {row["selector"] for row in worst_cases}
        self.assertEqual(selectors, {"min_jaccard", "max_world_l2", "max_step_delta_p95"})
        self.assertTrue(all(row["episode_name"] == "episode_00002_blue" for row in worst_cases))

    def test_validate_identical_query_frames_raises_on_mismatch(self):
        with self.assertRaises(ValueError):
            validate_identical_query_frames(
                {
                    "baseline_query_frames": [1, 2, 3],
                    "variant_query_frames": [1, 3],
                },
                episode_name="episode_00001_green",
                camera_name="varied_camera_1",
            )

    def test_validate_identical_manifest_hashes_raises_on_mismatch(self):
        with self.assertRaises(ValueError):
            validate_identical_manifest_hashes(
                {"query_task_manifest_sha256": "abc"},
                {"query_task_manifest_sha256": "def"},
            )


if __name__ == "__main__":
    unittest.main()
