import unittest

from scripts.data_analysis.analyze_batch_run_telemetry import (
    summarize_profile_records,
    summarize_run_overview,
    summarize_task_groups,
)


class BatchTelemetryOverviewTests(unittest.TestCase):
    def test_summarize_run_overview_reports_three_core_throughput_views(self):
        summary = {
            "wall_clock_seconds": 400.0,
            "telemetry_gpu_ids": [0, 1],
            "worker_slot_count": 8,
        }
        task_records = [
            {
                "status": "success",
                "query_frame_count": 10,
                "total_seconds": 120.0,
                "process_seconds": 118.0,
                "save_seconds": 2.0,
            },
            {
                "status": "success",
                "query_frame_count": 30,
                "total_seconds": 360.0,
                "process_seconds": 350.0,
                "save_seconds": 10.0,
            },
        ]

        overview = summarize_run_overview(summary, task_records)

        self.assertEqual(overview["total_query_count"], 40)
        self.assertAlmostEqual(overview["cluster_seconds_per_query"], 10.0)
        self.assertAlmostEqual(overview["single_gpu_seconds_per_query"], 20.0)
        self.assertAlmostEqual(overview["slot_seconds_per_query"], 12.0)
        self.assertAlmostEqual(overview["process_slot_seconds_per_query"], 11.7)
        self.assertAlmostEqual(overview["save_slot_seconds_per_query"], 0.3)


class BatchTelemetryGroupingTests(unittest.TestCase):
    def test_summarize_task_groups_aggregates_weighted_per_query_metrics(self):
        task_records = [
            {
                "camera_name": "varied_camera_1",
                "traj_filter_profile": "external",
                "gpu_id": 0,
                "worker_label": "GPU 0 slot 1/2",
                "status": "success",
                "query_frame_count": 10,
                "total_seconds": 100.0,
                "process_seconds": 98.0,
                "save_seconds": 2.0,
                "total_seconds_per_query": 10.0,
            },
            {
                "camera_name": "varied_camera_1",
                "traj_filter_profile": "external",
                "gpu_id": 0,
                "worker_label": "GPU 0 slot 2/2",
                "status": "success",
                "query_frame_count": 20,
                "total_seconds": 260.0,
                "process_seconds": 254.0,
                "save_seconds": 6.0,
                "total_seconds_per_query": 13.0,
            },
            {
                "camera_name": "varied_camera_3",
                "traj_filter_profile": "wrist_pick_place_no_heatmap",
                "gpu_id": 1,
                "worker_label": "GPU 1 slot 1/2",
                "status": "failed",
                "query_frame_count": 0,
                "total_seconds": None,
                "process_seconds": None,
                "save_seconds": None,
                "total_seconds_per_query": None,
            },
        ]

        grouped = summarize_task_groups(task_records, group_fields=("camera_name",))

        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped[0]["group_label"], "varied_camera_1")
        self.assertEqual(grouped[0]["query_count"], 30)
        self.assertAlmostEqual(grouped[0]["slot_seconds_per_query"], 12.0)
        self.assertAlmostEqual(grouped[0]["process_slot_seconds_per_query"], 11.7333333333)
        self.assertAlmostEqual(grouped[0]["save_slot_seconds_per_query"], 0.2666666667)
        self.assertEqual(grouped[1]["group_label"], "varied_camera_3")
        self.assertEqual(grouped[1]["success_task_count"], 0)


class BatchTelemetryProfileTests(unittest.TestCase):
    def test_summarize_profile_records_computes_seconds_per_query(self):
        profile_records = [
            {
                "status": "success",
                "query_frame_count": 10,
                "profile_stats": {
                    "tracker_model_forward_seconds": 50.0,
                    "prepare_inputs_seconds": 20.0,
                },
            },
            {
                "status": "success",
                "query_frame_count": 30,
                "profile_stats": {
                    "tracker_model_forward_seconds": 120.0,
                    "prepare_inputs_seconds": 40.0,
                },
            },
        ]

        rows = summarize_profile_records(
            profile_records,
            profile_field="profile_stats",
            profile_keys=("tracker_model_forward_seconds", "prepare_inputs_seconds"),
        )

        self.assertEqual(rows[0]["profile_key"], "tracker_model_forward_seconds")
        self.assertAlmostEqual(rows[0]["total_seconds"], 170.0)
        self.assertAlmostEqual(rows[0]["seconds_per_query"], 4.25)
        self.assertEqual(rows[1]["profile_key"], "prepare_inputs_seconds")
        self.assertAlmostEqual(rows[1]["seconds_per_query"], 1.5)

    def test_summarize_profile_records_ignores_non_timing_scalars(self):
        profile_records = [
            {
                "status": "success",
                "query_frame_count": 5,
                "profile_stats": {
                    "tracker_model_forward_seconds": 25.0,
                    "prepare_depth_filter_cache_hit_frames": 100.0,
                    "high_volatility_threshold": 8.0,
                },
            }
        ]

        rows = summarize_profile_records(
            profile_records,
            profile_field="profile_stats",
            profile_keys=("tracker_model_forward_seconds",),
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["profile_key"], "tracker_model_forward_seconds")
        self.assertAlmostEqual(rows[0]["seconds_per_query"], 5.0)


if __name__ == "__main__":
    unittest.main()
