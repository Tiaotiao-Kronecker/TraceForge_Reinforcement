#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


CURRENT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(CURRENT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPO_ROOT))

from scripts.data_analysis.benchmark_inference_variants import (  # noqa: E402
    BASELINE_SUPPORT_GRID_RATIO,
    RESULT_JSON_BASENAME,
    SUMMARY_MD_BASENAME,
)
from scripts.data_analysis.benchmark_num_iters_manifest import (  # noqa: E402
    load_benchmark_manifest,
)


def parse_args() -> argparse.Namespace:
    default_output_root = (
        CURRENT_REPO_ROOT
        / "data_tmp"
        / "inference_variant_manifest"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-episode benchmark_inference_variants.py summaries into one "
            "manifest-level runtime and quality report."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--episode-output-root",
        type=Path,
        required=True,
        help=(
            "Directory containing one subdirectory per episode, each with "
            f"`{RESULT_JSON_BASENAME}`."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument(
        "--episode-summary-name",
        type=str,
        default=RESULT_JSON_BASENAME,
        help="Per-episode summary filename relative to each episode directory.",
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _stdev(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    return float(statistics.stdev(values))


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _aggregate_numeric_mappings(records: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    keys = sorted({key for record in records for key in record.keys()})
    aggregated: dict[str, dict[str, float | None]] = {}
    for key in keys:
        values = [float(record[key]) for record in records if record.get(key) is not None]
        aggregated[key] = {
            "mean": _mean(values),
            "stdev": _stdev(values),
        }
    return aggregated


def _aggregate_profile_mean_mappings(
    records: list[dict[str, dict[str, float | None]]],
) -> dict[str, dict[str, float | None]]:
    keys = sorted({key for record in records for key in record.keys()})
    aggregated: dict[str, dict[str, float | None]] = {}
    for key in keys:
        values = [
            float(record[key]["mean"])
            for record in records
            if record.get(key) is not None and record[key].get("mean") is not None
        ]
        aggregated[key] = {
            "mean": _mean(values),
            "stdev": _stdev(values),
        }
    return aggregated


def _find_worst_query(
    *,
    episode_name: str,
    per_sample: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    worst: dict[str, Any] | None = None
    for query_frame_str, metrics in per_sample.items():
        jaccard = _to_float_or_none(metrics.get("traj_valid_mask_jaccard"))
        if jaccard is None:
            continue
        candidate = {
            "episode_name": episode_name,
            "query_frame": int(query_frame_str),
            "traj_valid_mask_jaccard": jaccard,
            "valid_track_count_delta": _to_float_or_none(metrics.get("valid_track_count_delta")),
            "traj_world_l2_mean": _to_float_or_none(metrics.get("traj_world_l2_mean")),
        }
        if worst is None or candidate["traj_valid_mask_jaccard"] < worst["traj_valid_mask_jaccard"]:
            worst = candidate
    return worst


def build_aggregate_variant_rows(
    episode_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for episode_result in episode_results:
        episode_name = str(episode_result["episode_name"])
        for row in episode_result["summary"]["variant_rows"]:
            key = (row["camera_name"], row["variant_name"])
            grouped.setdefault(key, []).append((episode_name, row))

    aggregate_rows: list[dict[str, Any]] = []
    for (camera_name, variant_name), items in sorted(grouped.items()):
        episodes = [episode_name for episode_name, _ in items]
        rows = [row for _, row in items]
        aggregate_rows.append(
            {
                "camera_name": camera_name,
                "variant_name": variant_name,
                "variant_config": dict(rows[0]["variant_config"]),
                "traj_filter_profile": rows[0]["traj_filter_profile"],
                "episode_count": len(rows),
                "episodes": episodes,
                "aggregates": _aggregate_numeric_mappings([row["aggregates"] for row in rows]),
                "profile_aggregates": _aggregate_profile_mean_mappings(
                    [row["profile_aggregates"] for row in rows]
                ),
            }
        )
    return aggregate_rows


def build_aggregate_pairwise_comparisons(
    episode_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for episode_result in episode_results:
        episode_name = str(episode_result["episode_name"])
        for row in episode_result["summary"]["pairwise_comparisons"]:
            row_copy = dict(row)
            row_copy["_episode_name"] = episode_name
            key = (row["camera_name"], row["variant_name"])
            grouped.setdefault(key, []).append((episode_name, row_copy))

    aggregate_rows: list[dict[str, Any]] = []
    for (camera_name, variant_name), items in sorted(grouped.items()):
        episodes = [episode_name for episode_name, _ in items]
        rows = [row for _, row in items]
        worst_candidates = [
            _find_worst_query(
                episode_name=row["_episode_name"],
                per_sample=row["sample_diff"].get("per_sample", {}),
            )
            for row in rows
        ]
        worst_candidates = [item for item in worst_candidates if item is not None]
        worst_query = None
        if worst_candidates:
            worst_query = min(
                worst_candidates,
                key=lambda item: item["traj_valid_mask_jaccard"],
            )
        aggregate_rows.append(
            {
                "camera_name": camera_name,
                "variant_name": variant_name,
                "variant_config": dict(rows[0]["variant_config"]),
                "traj_filter_profile": rows[0]["traj_filter_profile"],
                "episode_count": len(rows),
                "episodes": episodes,
                "speedup_aggregates": _aggregate_numeric_mappings(
                    [
                        {
                            "process_speedup_vs_baseline": row["process_speedup_vs_baseline"],
                            "save_speedup_vs_baseline": row["save_speedup_vs_baseline"],
                            "total_speedup_vs_baseline": row["total_speedup_vs_baseline"],
                            "prepare_inputs_speedup_vs_baseline": row["prepare_inputs_speedup_vs_baseline"],
                            "tracker_inference_speedup_vs_baseline": row["tracker_inference_speedup_vs_baseline"],
                            "tracker_forward_speedup_vs_baseline": row["tracker_forward_speedup_vs_baseline"],
                        }
                        for row in rows
                    ]
                ),
                "sample_diff_aggregates": _aggregate_numeric_mappings(
                    [row["sample_diff"]["aggregates"] for row in rows]
                ),
                "worst_query": worst_query,
            }
        )
    return aggregate_rows


def find_aggregate_pareto_candidates(summary: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    pairwise_by_key = {
        (item["camera_name"], item["variant_name"]): item
        for item in summary["aggregate_pairwise_comparisons"]
    }
    candidates_by_camera: dict[str, list[dict[str, Any]]] = {}
    for camera_name in summary["camera_names"]:
        camera_rows = [row for row in summary["aggregate_variant_rows"] if row["camera_name"] == camera_name]
        comparable_rows: list[dict[str, Any]] = []
        for row in camera_rows:
            variant_name = row["variant_name"]
            total_seconds = _to_float_or_none(row["aggregates"].get("total_seconds_mean", {}).get("mean"))
            if variant_name == "baseline":
                jaccard = 1.0
                world_l2 = 0.0
            else:
                comparison = pairwise_by_key[(camera_name, variant_name)]
                jaccard = _to_float_or_none(
                    comparison["sample_diff_aggregates"].get("traj_valid_mask_jaccard_mean", {}).get("mean")
                )
                world_l2 = _to_float_or_none(
                    comparison["sample_diff_aggregates"].get("traj_world_l2_mean", {}).get("mean")
                )
            comparable_rows.append(
                {
                    "variant_name": variant_name,
                    "support_grid_ratio": row["variant_config"]["support_grid_ratio"],
                    "total_seconds_mean": total_seconds,
                    "traj_valid_mask_jaccard_mean": jaccard,
                    "traj_world_l2_mean": world_l2,
                }
            )

        candidate_rows: list[dict[str, Any]] = []
        for row in comparable_rows:
            row_total = row["total_seconds_mean"]
            row_jaccard = row["traj_valid_mask_jaccard_mean"]
            row_world = row["traj_world_l2_mean"]
            if row_total is None or row_jaccard is None or row_world is None:
                candidate_rows.append(row)
                continue

            dominated = False
            for other in comparable_rows:
                if other["variant_name"] == row["variant_name"]:
                    continue
                other_total = other["total_seconds_mean"]
                other_jaccard = other["traj_valid_mask_jaccard_mean"]
                other_world = other["traj_world_l2_mean"]
                if other_total is None or other_jaccard is None or other_world is None:
                    continue
                dominates = (
                    other_total <= row_total
                    and other_jaccard >= row_jaccard
                    and other_world <= row_world
                    and (
                        other_total < row_total
                        or other_jaccard > row_jaccard
                        or other_world < row_world
                    )
                )
                if dominates:
                    dominated = True
                    break
            if not dominated:
                candidate_rows.append(row)
        candidates_by_camera[camera_name] = candidate_rows
    return candidates_by_camera


def _validate_summary_consistency(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    summary_path: Path,
) -> None:
    keys_to_match = (
        "camera_names",
        "support_grid_ratios",
        "variant_specs",
        "grid_size",
        "benchmark_runs",
        "warmup_runs",
        "traj_filter_profile",
    )
    for key in keys_to_match:
        if candidate.get(key) != baseline.get(key):
            raise ValueError(
                f"Inconsistent field {key!r} in {summary_path}: "
                f"{candidate.get(key)!r} != {baseline.get(key)!r}"
            )


def _load_episode_results(
    *,
    manifest_path: Path,
    episode_output_root: Path,
    episode_summary_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_benchmark_manifest(manifest_path)
    episode_results: list[dict[str, Any]] = []
    baseline_summary: dict[str, Any] | None = None
    for episode_name, episode_dir in zip(manifest["episodes"], manifest["episode_dirs"]):
        summary_path = (episode_output_root / episode_name / episode_summary_name).resolve()
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing episode summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if baseline_summary is None:
            baseline_summary = summary
        else:
            _validate_summary_consistency(
                baseline=baseline_summary,
                candidate=summary,
                summary_path=summary_path,
            )
        summary_md_path = summary_path.with_name(SUMMARY_MD_BASENAME)
        episode_results.append(
            {
                "episode_name": episode_name,
                "episode_dir": str(episode_dir),
                "summary_json_path": str(summary_path),
                "summary_md_path": str(summary_md_path),
                "summary": summary,
            }
        )
    return manifest, episode_results


def _build_device_summary(episode_results: list[dict[str, Any]]) -> str:
    devices = sorted({str(item["summary"]["device"]) for item in episode_results})
    if not devices:
        return "n/a"
    if len(devices) == 1:
        return devices[0]
    return "parallel({devices})".format(devices=",".join(devices))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_manifest_summary_markdown(summary: dict[str, Any], summary_path: Path) -> None:
    lines = [
        "# Inference Support Sweep Manifest Summary",
        "",
        f"- Manifest: `{summary['manifest_path']}`",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Episodes: `{','.join(summary['episodes'])}`",
        f"- Episode output root: `{summary['episode_output_root']}`",
        f"- Cameras: `{','.join(summary['camera_names'])}`",
        f"- Support ratios: `{','.join(str(item) for item in summary['support_grid_ratios'])}`",
        f"- Baseline support ratio: `{BASELINE_SUPPORT_GRID_RATIO}`",
        f"- Device: `{summary['device']}`",
        f"- Benchmark runs: `{summary['benchmark_runs']}`",
        f"- Warmup runs: `{summary['warmup_runs']}`",
        "",
        "## Episode Runtime",
        "",
        "| Episode | Camera | Variant | Ratio | Effective Support Count | Process (s) | Save (s) | Total (s) | Tracker Forward (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for episode_result in summary["episode_results"]:
        episode_name = episode_result["episode_name"]
        for row in episode_result["summary"]["variant_rows"]:
            aggregates = row["aggregates"]
            profile_aggregates = row["profile_aggregates"]
            lines.append(
                "| {episode} | {camera} | {variant} | {ratio} | {effective_support} | {process} | {save} | {total} | {tracker_forward} |".format(
                    episode=episode_name,
                    camera=row["camera_name"],
                    variant=row["variant_name"],
                    ratio=_format_float(row["variant_config"]["support_grid_ratio"], digits=2),
                    effective_support=_format_float(aggregates["effective_support_query_count_mean"], digits=1),
                    process=_format_float(aggregates["process_seconds_mean"]),
                    save=_format_float(aggregates["save_seconds_mean"]),
                    total=_format_float(aggregates["total_seconds_mean"]),
                    tracker_forward=_format_float(
                        _to_float_or_none(profile_aggregates.get("tracker_model_forward_seconds", {}).get("mean"))
                    ),
                )
            )

    lines.extend(
        [
            "",
            "## Aggregate Runtime",
            "",
            "| Camera | Variant | Episodes | Ratio | Effective Support Count | Process (s) | Save (s) | Total (s) | Prepare Inputs (s) | Tracker Forward (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["aggregate_variant_rows"]:
        aggregates = row["aggregates"]
        profile_aggregates = row["profile_aggregates"]
        lines.append(
            "| {camera} | {variant} | {episodes} | {ratio} | {effective_support} | {process} | {save} | {total} | {prepare_inputs} | {tracker_forward} |".format(
                camera=row["camera_name"],
                variant=row["variant_name"],
                episodes=row["episode_count"],
                ratio=_format_float(row["variant_config"]["support_grid_ratio"], digits=2),
                effective_support=_format_float(
                    _to_float_or_none(aggregates.get("effective_support_query_count_mean", {}).get("mean")),
                    digits=1,
                ),
                process=_format_float(_to_float_or_none(aggregates.get("process_seconds_mean", {}).get("mean"))),
                save=_format_float(_to_float_or_none(aggregates.get("save_seconds_mean", {}).get("mean"))),
                total=_format_float(_to_float_or_none(aggregates.get("total_seconds_mean", {}).get("mean"))),
                prepare_inputs=_format_float(
                    _to_float_or_none(profile_aggregates.get("prepare_inputs_seconds", {}).get("mean"))
                ),
                tracker_forward=_format_float(
                    _to_float_or_none(profile_aggregates.get("tracker_model_forward_seconds", {}).get("mean"))
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Aggregate Quality vs Baseline",
            "",
            "| Camera | Variant | Episodes | Ratio | Total Speedup | Forward Speedup | Valid Delta | Mask Jaccard | World L2 Mean | Step Delta Mean | Endpoint Error | Worst Query Jaccard | Worst Episode/QF |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["aggregate_pairwise_comparisons"]:
        speedups = row["speedup_aggregates"]
        diff = row["sample_diff_aggregates"]
        worst_query = row["worst_query"]
        worst_episode_qf = "n/a"
        worst_jaccard = None
        if worst_query is not None:
            worst_episode_qf = f"{worst_query['episode_name']}/{worst_query['query_frame']}"
            worst_jaccard = worst_query["traj_valid_mask_jaccard"]
        lines.append(
            "| {camera} | {variant} | {episodes} | {ratio} | {total_speedup}x | {forward_speedup}x | {valid_delta} | {jaccard} | {world_l2} | {step_delta} | {endpoint_error} | {worst_jaccard} | {worst_episode_qf} |".format(
                camera=row["camera_name"],
                variant=row["variant_name"],
                episodes=row["episode_count"],
                ratio=_format_float(row["variant_config"]["support_grid_ratio"], digits=2),
                total_speedup=_format_float(
                    _to_float_or_none(speedups.get("total_speedup_vs_baseline", {}).get("mean"))
                ),
                forward_speedup=_format_float(
                    _to_float_or_none(speedups.get("tracker_forward_speedup_vs_baseline", {}).get("mean"))
                ),
                valid_delta=_format_float(
                    _to_float_or_none(diff.get("valid_track_count_delta_mean", {}).get("mean"))
                ),
                jaccard=_format_float(
                    _to_float_or_none(diff.get("traj_valid_mask_jaccard_mean", {}).get("mean"))
                ),
                world_l2=_format_float(
                    _to_float_or_none(diff.get("traj_world_l2_mean", {}).get("mean")),
                    digits=5,
                ),
                step_delta=_format_float(
                    _to_float_or_none(diff.get("traj_world_step_delta_l2_mean", {}).get("mean")),
                    digits=5,
                ),
                endpoint_error=_format_float(
                    _to_float_or_none(diff.get("traj_world_endpoint_l2_mean", {}).get("mean")),
                    digits=5,
                ),
                worst_jaccard=_format_float(worst_jaccard, digits=5),
                worst_episode_qf=worst_episode_qf,
            )
        )

    lines.extend(["", "## Pareto Candidates", ""])
    for camera_name in summary["camera_names"]:
        candidates = summary["pareto_candidates"].get(camera_name, [])
        if not candidates:
            lines.append(f"- `{camera_name}`: none")
            continue
        formatted = ", ".join(
            "`{variant}`(ratio={ratio}, total={total}s, jaccard={jaccard}, world_l2={world_l2})".format(
                variant=item["variant_name"],
                ratio=_format_float(item["support_grid_ratio"], digits=2),
                total=_format_float(item["total_seconds_mean"]),
                jaccard=_format_float(item["traj_valid_mask_jaccard_mean"]),
                world_l2=_format_float(item["traj_world_l2_mean"], digits=5),
            )
            for item in candidates
        )
        lines.append(f"- `{camera_name}`: {formatted}")

    lines.extend(
        [
            "",
            "## Episode Summaries",
            "",
            "| Episode | JSON | Markdown |",
            "| --- | --- | --- |",
        ]
    )
    for item in summary["episode_results"]:
        lines.append(
            "| {episode} | `{json_path}` | `{md_path}` |".format(
                episode=item["episode_name"],
                json_path=item["summary_json_path"],
                md_path=item["summary_md_path"],
            )
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest, episode_results = _load_episode_results(
        manifest_path=args.manifest,
        episode_output_root=args.episode_output_root.resolve(),
        episode_summary_name=args.episode_summary_name,
    )
    baseline_summary = episode_results[0]["summary"]
    summary = {
        "manifest_path": str(manifest["manifest_path"]),
        "dataset_root": str(manifest["dataset_root"]),
        "episodes": manifest["episodes"],
        "camera_names": baseline_summary["camera_names"],
        "support_grid_ratios": baseline_summary["support_grid_ratios"],
        "variant_specs": baseline_summary["variant_specs"],
        "checkpoint": baseline_summary["checkpoint"],
        "device": _build_device_summary(episode_results),
        "benchmark_runs": int(baseline_summary["benchmark_runs"]),
        "warmup_runs": int(baseline_summary["warmup_runs"]),
        "grid_size": int(baseline_summary["grid_size"]),
        "traj_filter_profile": baseline_summary["traj_filter_profile"],
        "current_repo_root": str(CURRENT_REPO_ROOT.resolve()),
        "episode_output_root": str(args.episode_output_root.resolve()),
        "episode_results": episode_results,
    }
    summary["aggregate_variant_rows"] = build_aggregate_variant_rows(episode_results)
    summary["aggregate_pairwise_comparisons"] = build_aggregate_pairwise_comparisons(episode_results)
    summary["pareto_candidates"] = find_aggregate_pareto_candidates(summary)

    summary_json_path = output_root / RESULT_JSON_BASENAME
    summary_md_path = output_root / SUMMARY_MD_BASENAME
    _write_json(summary_json_path, summary)
    write_manifest_summary_markdown(summary, summary_md_path)
    print(f"JSON summary: {summary_json_path}")
    print(f"Markdown summary: {summary_md_path}")


if __name__ == "__main__":
    main()
