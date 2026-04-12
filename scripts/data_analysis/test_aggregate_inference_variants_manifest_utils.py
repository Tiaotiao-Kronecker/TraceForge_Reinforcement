import unittest

from scripts.data_analysis.aggregate_inference_variants_manifest import (
    build_aggregate_pairwise_comparisons,
    build_aggregate_variant_rows,
    find_aggregate_pareto_candidates,
)


def _make_variant_row(
    *,
    variant_name: str,
    support_grid_ratio: float,
    process_seconds_mean: float,
    save_seconds_mean: float,
    total_seconds_mean: float,
    effective_support_query_count_mean: float,
    tracker_forward_seconds: float,
) -> dict:
    return {
        "camera_name": "varied_camera_1",
        "variant_name": variant_name,
        "traj_filter_profile": "external",
        "variant_config": {
            "name": variant_name,
            "support_grid_ratio": support_grid_ratio,
        },
        "aggregates": {
            "process_seconds_mean": process_seconds_mean,
            "save_seconds_mean": save_seconds_mean,
            "total_seconds_mean": total_seconds_mean,
            "effective_support_query_count_mean": effective_support_query_count_mean,
        },
        "profile_aggregates": {
            "prepare_inputs_seconds": {"mean": 1.0, "stdev": 0.0},
            "tracker_model_forward_seconds": {"mean": tracker_forward_seconds, "stdev": 0.0},
        },
    }


def _make_pairwise(
    *,
    variant_name: str,
    support_grid_ratio: float,
    total_speedup: float,
    forward_speedup: float,
    valid_delta: float,
    jaccard: float,
    world_l2: float,
    step_delta: float,
    endpoint_error: float,
    worst_queries: dict[str, float],
) -> dict:
    per_sample = {
        query_frame: {
            "traj_valid_mask_jaccard": jaccard_value,
            "valid_track_count_delta": valid_delta,
            "traj_world_l2_mean": world_l2,
        }
        for query_frame, jaccard_value in worst_queries.items()
    }
    return {
        "camera_name": "varied_camera_1",
        "variant_name": variant_name,
        "traj_filter_profile": "external",
        "variant_config": {
            "name": variant_name,
            "support_grid_ratio": support_grid_ratio,
        },
        "process_speedup_vs_baseline": total_speedup,
        "save_speedup_vs_baseline": 1.0,
        "total_speedup_vs_baseline": total_speedup,
        "prepare_inputs_speedup_vs_baseline": 1.0,
        "tracker_inference_speedup_vs_baseline": forward_speedup,
        "tracker_forward_speedup_vs_baseline": forward_speedup,
        "sample_diff": {
            "aggregates": {
                "valid_track_count_delta_mean": valid_delta,
                "traj_valid_mask_jaccard_mean": jaccard,
                "traj_world_l2_mean": world_l2,
                "traj_world_step_delta_l2_mean": step_delta,
                "traj_world_endpoint_l2_mean": endpoint_error,
            },
            "per_sample": per_sample,
        },
    }


def _make_episode_result(
    *,
    episode_name: str,
    baseline_total: float,
    support_r040_total: float,
    support_r000_total: float,
    support_r040_jaccard: float,
    support_r000_jaccard: float,
    support_r040_worst: dict[str, float],
    support_r000_worst: dict[str, float],
) -> dict:
    return {
        "episode_name": episode_name,
        "summary": {
            "variant_rows": [
                _make_variant_row(
                    variant_name="baseline",
                    support_grid_ratio=0.8,
                    process_seconds_mean=baseline_total - 1.0,
                    save_seconds_mean=1.0,
                    total_seconds_mean=baseline_total,
                    effective_support_query_count_mean=3200.0,
                    tracker_forward_seconds=9.0,
                ),
                _make_variant_row(
                    variant_name="support_r040",
                    support_grid_ratio=0.4,
                    process_seconds_mean=support_r040_total - 1.0,
                    save_seconds_mean=1.0,
                    total_seconds_mean=support_r040_total,
                    effective_support_query_count_mean=900.0,
                    tracker_forward_seconds=6.0,
                ),
                _make_variant_row(
                    variant_name="support_r000",
                    support_grid_ratio=0.0,
                    process_seconds_mean=support_r000_total - 1.0,
                    save_seconds_mean=1.0,
                    total_seconds_mean=support_r000_total,
                    effective_support_query_count_mean=0.0,
                    tracker_forward_seconds=6.5,
                ),
            ],
            "pairwise_comparisons": [
                _make_pairwise(
                    variant_name="support_r040",
                    support_grid_ratio=0.4,
                    total_speedup=baseline_total / support_r040_total,
                    forward_speedup=9.0 / 6.0,
                    valid_delta=-2.0,
                    jaccard=support_r040_jaccard,
                    world_l2=0.001,
                    step_delta=0.002,
                    endpoint_error=0.003,
                    worst_queries=support_r040_worst,
                ),
                _make_pairwise(
                    variant_name="support_r000",
                    support_grid_ratio=0.0,
                    total_speedup=baseline_total / support_r000_total,
                    forward_speedup=9.0 / 6.5,
                    valid_delta=-10.0,
                    jaccard=support_r000_jaccard,
                    world_l2=0.010,
                    step_delta=0.020,
                    endpoint_error=0.030,
                    worst_queries=support_r000_worst,
                ),
            ],
        },
    }


class AggregateInferenceVariantManifestTests(unittest.TestCase):
    def test_build_aggregate_variant_rows_averages_runtime_metrics(self):
        episode_results = [
            _make_episode_result(
                episode_name="00000",
                baseline_total=11.0,
                support_r040_total=8.0,
                support_r000_total=9.5,
                support_r040_jaccard=0.985,
                support_r000_jaccard=0.900,
                support_r040_worst={"12": 0.93},
                support_r000_worst={"12": 0.70},
            ),
            _make_episode_result(
                episode_name="00001",
                baseline_total=13.0,
                support_r040_total=9.0,
                support_r000_total=10.5,
                support_r040_jaccard=0.980,
                support_r000_jaccard=0.890,
                support_r040_worst={"18": 0.83},
                support_r000_worst={"18": 0.62},
            ),
        ]

        aggregate_rows = build_aggregate_variant_rows(episode_results)
        by_variant = {row["variant_name"]: row for row in aggregate_rows}

        self.assertEqual(by_variant["baseline"]["episode_count"], 2)
        self.assertAlmostEqual(
            by_variant["baseline"]["aggregates"]["total_seconds_mean"]["mean"],
            12.0,
        )
        self.assertAlmostEqual(
            by_variant["support_r040"]["aggregates"]["effective_support_query_count_mean"]["mean"],
            900.0,
        )

    def test_build_aggregate_pairwise_comparisons_keeps_global_worst_query(self):
        episode_results = [
            _make_episode_result(
                episode_name="00000",
                baseline_total=11.0,
                support_r040_total=8.0,
                support_r000_total=9.5,
                support_r040_jaccard=0.985,
                support_r000_jaccard=0.900,
                support_r040_worst={"12": 0.93, "16": 0.91},
                support_r000_worst={"12": 0.70},
            ),
            _make_episode_result(
                episode_name="00001",
                baseline_total=13.0,
                support_r040_total=9.0,
                support_r000_total=10.5,
                support_r040_jaccard=0.980,
                support_r000_jaccard=0.890,
                support_r040_worst={"18": 0.83, "20": 0.88},
                support_r000_worst={"18": 0.62},
            ),
        ]

        aggregate_rows = build_aggregate_pairwise_comparisons(episode_results)
        by_variant = {row["variant_name"]: row for row in aggregate_rows}
        worst_query = by_variant["support_r040"]["worst_query"]

        self.assertEqual(worst_query["episode_name"], "00001")
        self.assertEqual(worst_query["query_frame"], 18)
        self.assertAlmostEqual(worst_query["traj_valid_mask_jaccard"], 0.83)

    def test_find_aggregate_pareto_candidates_drops_dominated_variant(self):
        episode_results = [
            _make_episode_result(
                episode_name="00000",
                baseline_total=11.0,
                support_r040_total=8.0,
                support_r000_total=9.5,
                support_r040_jaccard=0.985,
                support_r000_jaccard=0.900,
                support_r040_worst={"12": 0.93},
                support_r000_worst={"12": 0.70},
            ),
            _make_episode_result(
                episode_name="00001",
                baseline_total=13.0,
                support_r040_total=9.0,
                support_r000_total=10.5,
                support_r040_jaccard=0.980,
                support_r000_jaccard=0.890,
                support_r040_worst={"18": 0.83},
                support_r000_worst={"18": 0.62},
            ),
        ]

        summary = {
            "camera_names": ["varied_camera_1"],
            "aggregate_variant_rows": build_aggregate_variant_rows(episode_results),
            "aggregate_pairwise_comparisons": build_aggregate_pairwise_comparisons(episode_results),
        }
        candidates = find_aggregate_pareto_candidates(summary)
        candidate_names = [item["variant_name"] for item in candidates["varied_camera_1"]]

        self.assertIn("baseline", candidate_names)
        self.assertIn("support_r040", candidate_names)
        self.assertNotIn("support_r000", candidate_names)


if __name__ == "__main__":
    unittest.main()
