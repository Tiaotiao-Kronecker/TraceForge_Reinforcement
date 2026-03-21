import json
import tempfile
import unittest
from pathlib import Path

from scripts.data_analysis.benchmark_num_iters_manifest import (
    build_aggregate_case_results,
    build_volatility_summary,
    load_benchmark_manifest,
)


def _make_case(
    *,
    camera_name: str,
    variant_name: str,
    process_mean: float,
    save_mean: float,
    total_mean: float,
    depth_filter_mean: float,
    points_to_normals_mean: float,
    edge_mask_mean: float,
    measured_points_to_normals: list[float],
    measured_depth_filter: list[float],
    measured_edge_mask: list[float],
) -> dict:
    measured_runs = []
    for index, (ptn, depth_filter, edge_mask) in enumerate(
        zip(measured_points_to_normals, measured_depth_filter, measured_edge_mask),
        start=1,
    ):
        measured_runs.append(
            {
                "run_index": index,
                "warmup": False,
                "process_seconds": process_mean,
                "save_seconds": save_mean,
                "process_profile_stats": {
                    "process_total_seconds": process_mean,
                    "prepare_depth_filter_seconds": depth_filter,
                    "prepare_depth_filter_worker_total_seconds": depth_filter + 10.0,
                    "prepare_depth_filter_points_to_normals_seconds": ptn,
                    "prepare_depth_filter_edge_mask_seconds": edge_mask,
                },
            }
        )
    return {
        "camera_name": camera_name,
        "variant_name": variant_name,
        "variant_config": {"name": variant_name, "num_iters": 5},
        "traj_filter_profile": "external" if camera_name == "varied_camera_1" else "wrist_manipulator_top95",
        "aggregates": {
            "process_seconds_mean": process_mean,
            "save_seconds_mean": save_mean,
            "total_seconds_mean": total_mean,
        },
        "process_profile_aggregates": {
            "process_total_seconds": {"mean": process_mean, "stdev": 0.0},
            "prepare_depth_filter_seconds": {"mean": depth_filter_mean, "stdev": 0.0},
            "prepare_depth_filter_worker_total_seconds": {"mean": depth_filter_mean + 10.0, "stdev": 0.0},
            "prepare_depth_filter_points_to_normals_seconds": {"mean": points_to_normals_mean, "stdev": 0.0},
            "prepare_depth_filter_edge_mask_seconds": {"mean": edge_mask_mean, "stdev": 0.0},
        },
        "save_profile_aggregates": {},
        "measured_runs": measured_runs,
    }


class ManifestLoaderTests(unittest.TestCase):
    def test_load_benchmark_manifest_resolves_relative_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            episode_dir = dataset_root / "episode_00001_green"
            episode_dir.mkdir(parents=True)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_root": "dataset",
                        "episodes": ["episode_00001_green"],
                    }
                ),
                encoding="utf-8",
            )

            manifest = load_benchmark_manifest(manifest_path)

            self.assertEqual(manifest["dataset_root"], dataset_root.resolve())
            self.assertEqual(manifest["episodes"], ["episode_00001_green"])
            self.assertEqual(manifest["episode_dirs"], [episode_dir.resolve()])

    def test_load_benchmark_manifest_requires_dataset_root_and_episodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(json.dumps({"dataset_root": "/tmp"}), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_benchmark_manifest(manifest_path)


class AggregateCaseResultTests(unittest.TestCase):
    def test_build_aggregate_case_results_aggregates_across_episodes(self):
        episode_results = [
            {
                "episode_name": "episode_a",
                "summary": {
                    "case_results": [
                        _make_case(
                            camera_name="varied_camera_1",
                            variant_name="iters_5",
                            process_mean=10.0,
                            save_mean=1.0,
                            total_mean=11.0,
                            depth_filter_mean=2.0,
                            points_to_normals_mean=5.0,
                            edge_mask_mean=1.0,
                            measured_points_to_normals=[4.0, 6.0],
                            measured_depth_filter=[1.5, 2.5],
                            measured_edge_mask=[0.8, 1.2],
                        )
                    ]
                },
            },
            {
                "episode_name": "episode_b",
                "summary": {
                    "case_results": [
                        _make_case(
                            camera_name="varied_camera_1",
                            variant_name="iters_5",
                            process_mean=14.0,
                            save_mean=3.0,
                            total_mean=17.0,
                            depth_filter_mean=4.0,
                            points_to_normals_mean=9.0,
                            edge_mask_mean=2.0,
                            measured_points_to_normals=[8.0, 10.0],
                            measured_depth_filter=[3.5, 4.5],
                            measured_edge_mask=[1.8, 2.2],
                        )
                    ]
                },
            },
        ]

        rows = build_aggregate_case_results(episode_results)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["camera_name"], "varied_camera_1")
        self.assertEqual(row["variant_name"], "iters_5")
        self.assertEqual(row["episode_count"], 2)
        self.assertEqual(row["episodes"], ["episode_a", "episode_b"])
        self.assertAlmostEqual(row["aggregates"]["process_seconds_mean"]["mean"], 12.0)
        self.assertAlmostEqual(row["aggregates"]["save_seconds_mean"]["mean"], 2.0)
        self.assertAlmostEqual(
            row["process_profile_aggregates"]["prepare_depth_filter_points_to_normals_seconds"]["mean"],
            7.0,
        )


class VolatilitySummaryTests(unittest.TestCase):
    def test_build_volatility_summary_uses_measured_runs_only(self):
        case = _make_case(
            camera_name="varied_camera_3",
            variant_name="iters_5",
            process_mean=20.0,
            save_mean=2.0,
            total_mean=22.0,
            depth_filter_mean=6.0,
            points_to_normals_mean=12.0,
            edge_mask_mean=3.0,
            measured_points_to_normals=[10.0, 14.0],
            measured_depth_filter=[5.0, 7.0],
            measured_edge_mask=[2.5, 3.5],
        )
        case["raw_runs"] = [
            {
                "run_index": 0,
                "warmup": True,
                "process_profile_stats": {
                    "prepare_depth_filter_points_to_normals_seconds": 100.0,
                    "prepare_depth_filter_seconds": 100.0,
                    "prepare_depth_filter_edge_mask_seconds": 100.0,
                    "prepare_depth_filter_worker_total_seconds": 100.0,
                    "process_total_seconds": 100.0,
                },
            }
        ] + case["measured_runs"]
        episode_results = [
            {
                "episode_name": "episode_00001_green",
                "summary": {"case_results": [case]},
            }
        ]

        summary = build_volatility_summary(episode_results)

        by_episode_case = summary["by_episode_case"][0]
        metric = by_episode_case["metric_summaries"]["prepare_depth_filter_points_to_normals_seconds"]
        self.assertEqual(metric["sample_count"], 2)
        self.assertAlmostEqual(metric["mean"], 12.0)
        self.assertAlmostEqual(metric["min"], 10.0)
        self.assertAlmostEqual(metric["max"], 14.0)

        by_camera_variant = summary["by_camera_variant"][0]
        metric_cross = by_camera_variant["metric_summaries"]["prepare_depth_filter_points_to_normals_seconds"]
        self.assertEqual(metric_cross["sample_count"], 2)
        self.assertAlmostEqual(metric_cross["mean"], 12.0)


if __name__ == "__main__":
    unittest.main()
