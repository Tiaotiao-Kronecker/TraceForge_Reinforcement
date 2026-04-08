#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


CURRENT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(CURRENT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPO_ROOT))

from scripts.data_analysis.benchmark_inference_variants import compare_camera_outputs  # noqa: E402
from scripts.data_analysis.benchmark_num_iters_manifest import load_benchmark_manifest  # noqa: E402


BATCH_RUN_SUMMARY_BASENAME = "_batch_run_summary.json"
CAMERA_TASK_METRICS_BASENAME = "_camera_task_metrics.jsonl"
QUERY_TASK_METRICS_BASENAME = "_query_task_metrics.jsonl"
RESULT_JSON_BASENAME = "comparison_results.json"
SUMMARY_MD_BASENAME = "comparison_summary.md"

QUALITY_KEYS = (
    "valid_track_count_delta_mean",
    "traj_valid_mask_jaccard_mean",
    "traj_world_l2_mean",
    "traj_world_step_delta_l2_p95",
    "traj_world_l2_p95",
    "traj_world_error_var_mean",
    "traj_world_endpoint_l2_mean",
    "traj_2d_l2_mean",
    "traj_depth_abs_mean",
    "traj_uvz_mae_mean",
    "common_valid_track_count_mean",
    "common_valid_step_count_mean",
)


def parse_args() -> argparse.Namespace:
    default_output_root = (
        CURRENT_REPO_ROOT
        / "data_tmp"
        / "v5_test_num_iters_eval"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    parser = argparse.ArgumentParser(
        description=(
            "Compare two TraceForge output roots produced from the same episode manifest, "
            "then summarize timing and trajectory-quality deltas."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument(
        "--camera-names",
        type=str,
        default=None,
        help=(
            "Comma-separated camera names. Required. Only these cameras are compared "
            "across the two output roots."
        ),
    )
    parser.add_argument("--baseline-label", type=str, default="iters_6")
    parser.add_argument("--variant-label", type=str, default="iters_5")
    parser.add_argument("--baseline-num-iters", type=int, default=6)
    parser.add_argument("--variant-num-iters", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument(
        "--allow-query-frame-mismatch",
        action="store_true",
        help="Do not fail when the two roots contain different query-frame sets for the same episode/camera.",
    )
    return parser.parse_args()


def parse_camera_names(
    raw: str | None,
    *,
    option_name: str = "--camera-names",
) -> list[str]:
    if raw is None:
        raise ValueError(f"{option_name} is required and must contain at least one value")
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError(f"{option_name} must contain at least one value")
    return values


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _variance(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.variance(values))


def _stdev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values))


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{time.time_ns()}")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}:{line_no}")
        records.append(payload)
    return records


def aggregate_metric(values: list[float]) -> dict[str, float | None]:
    return {
        "count": int(len(values)),
        "mean": _mean(values),
        "variance": _variance(values),
        "stdev": _stdev(values),
        "min": float(min(values)) if values else None,
        "max": float(max(values)) if values else None,
    }


def aggregate_timing_records(
    records: list[dict[str, Any]],
    *,
    episodes: list[str],
    camera_names: list[str],
) -> dict[str, Any]:
    episode_set = set(episodes)
    camera_set = set(camera_names)
    success_records = [
        record
        for record in records
        if record.get("status") == "success"
        and record.get("episode_name") in episode_set
        and record.get("camera_name") in camera_set
    ]
    metric_keys = (
        "query_frame_count",
        "process_seconds",
        "save_seconds",
        "total_seconds",
        "process_seconds_per_query",
        "save_seconds_per_query",
        "total_seconds_per_query",
    )
    by_camera: list[dict[str, Any]] = []
    for camera_name in camera_names:
        camera_records = [record for record in success_records if record["camera_name"] == camera_name]
        metric_summaries = {
            metric_key: aggregate_metric(
                [float(record[metric_key]) for record in camera_records if record.get(metric_key) is not None]
            )
            for metric_key in metric_keys
        }
        by_camera.append(
            {
                "camera_name": camera_name,
                "task_count": len(camera_records),
                "metric_summaries": metric_summaries,
            }
        )

    overall_metric_summaries = {
        metric_key: aggregate_metric(
            [float(record[metric_key]) for record in success_records if record.get(metric_key) is not None]
        )
        for metric_key in metric_keys
    }
    detail_rows = sorted(
        success_records,
        key=lambda item: (str(item.get("episode_name")), str(item.get("camera_name"))),
    )
    return {
        "task_count": len(success_records),
        "by_camera": by_camera,
        "overall_metric_summaries": overall_metric_summaries,
        "detail_rows": detail_rows,
    }


def aggregate_quality_rows(rows: list[dict[str, Any]], *, camera_names: list[str]) -> list[dict[str, Any]]:
    by_camera_values: dict[str, dict[str, list[float]]] = {
        camera_name: {key: [] for key in QUALITY_KEYS}
        for camera_name in camera_names
    }
    for row in rows:
        metric_values = by_camera_values[row["camera_name"]]
        for key in QUALITY_KEYS:
            value = row.get(key)
            if value is not None:
                metric_values[key].append(float(value))

    aggregates: list[dict[str, Any]] = []
    for camera_name in camera_names:
        aggregates.append(
            {
                "camera_name": camera_name,
                "metric_summaries": {
                    key: aggregate_metric(values)
                    for key, values in by_camera_values[camera_name].items()
                },
            }
        )
    return aggregates


def validate_identical_query_frames(
    comparison: dict[str, Any],
    *,
    episode_name: str,
    camera_name: str,
) -> None:
    if comparison["baseline_query_frames"] != comparison["variant_query_frames"]:
        raise ValueError(
            f"Query-frame mismatch for {episode_name}/{camera_name}: "
            f"baseline={comparison['baseline_query_frames']} variant={comparison['variant_query_frames']}"
        )


def validate_identical_manifest_hashes(
    baseline_batch_summary: dict[str, Any] | None,
    variant_batch_summary: dict[str, Any] | None,
) -> None:
    if baseline_batch_summary is None or variant_batch_summary is None:
        return
    baseline_hash = baseline_batch_summary.get("query_task_manifest_sha256")
    variant_hash = variant_batch_summary.get("query_task_manifest_sha256")
    if baseline_hash is None or variant_hash is None:
        return
    if str(baseline_hash) != str(variant_hash):
        raise ValueError(
            f"Query-task manifest hash mismatch: baseline={baseline_hash} variant={variant_hash}"
        )


def build_worst_quality_cases(per_sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selectors = [
        ("min_jaccard", "traj_valid_mask_jaccard", min),
        ("max_world_l2", "traj_world_l2_mean", max),
        ("max_step_delta_p95", "traj_world_step_delta_l2_p95", max),
    ]
    worst_cases: list[dict[str, Any]] = []
    for label, metric_key, chooser in selectors:
        candidates = [row for row in per_sample_rows if row.get(metric_key) is not None]
        if not candidates:
            continue
        worst_row = chooser(candidates, key=lambda row: float(row[metric_key]))
        payload = {"selector": label, **worst_row}
        if payload not in worst_cases:
            worst_cases.append(payload)
    return worst_cases


def summarize_quality_for_roots(
    *,
    episodes: list[str],
    camera_names: list[str],
    baseline_root: Path,
    variant_root: Path,
    require_identical_query_frames: bool,
) -> dict[str, Any]:
    per_episode_camera_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    for episode_name in episodes:
        baseline_episode_root = baseline_root / episode_name
        variant_episode_root = variant_root / episode_name
        if not baseline_episode_root.is_dir():
            raise FileNotFoundError(f"Missing baseline episode output: {baseline_episode_root}")
        if not variant_episode_root.is_dir():
            raise FileNotFoundError(f"Missing variant episode output: {variant_episode_root}")
        for camera_name in camera_names:
            comparison = compare_camera_outputs(
                baseline_episode_root,
                variant_episode_root,
                camera_name=camera_name,
            )
            if require_identical_query_frames:
                validate_identical_query_frames(
                    comparison,
                    episode_name=episode_name,
                    camera_name=camera_name,
                )
            aggregates = comparison["aggregates"]
            per_episode_camera_rows.append(
                {
                    "episode_name": episode_name,
                    "camera_name": camera_name,
                    "common_query_frame_count": int(aggregates["common_query_frame_count"]),
                    "valid_track_count_delta_mean": aggregates["valid_track_count_delta_mean"],
                    "traj_valid_mask_jaccard_mean": aggregates["traj_valid_mask_jaccard_mean"],
                    "traj_world_l2_mean": aggregates["traj_world_l2_mean"],
                    "traj_world_step_delta_l2_p95": aggregates["traj_world_step_delta_l2_p95"],
                    "traj_world_l2_p95": aggregates["traj_world_l2_p95"],
                    "traj_world_error_var_mean": aggregates["traj_world_error_var_mean"],
                    "traj_world_endpoint_l2_mean": aggregates["traj_world_endpoint_l2_mean"],
                    "traj_2d_l2_mean": aggregates["traj_2d_l2_mean"],
                    "traj_depth_abs_mean": aggregates["traj_depth_abs_mean"],
                    "traj_uvz_mae_mean": aggregates["traj_uvz_mae_mean"],
                    "common_valid_track_count_mean": aggregates["common_valid_track_count_mean"],
                    "common_valid_step_count_mean": aggregates["common_valid_step_count_mean"],
                    "baseline_query_frames": comparison["baseline_query_frames"],
                    "variant_query_frames": comparison["variant_query_frames"],
                }
            )
            for query_frame, per_sample in sorted(comparison["per_sample"].items(), key=lambda item: int(item[0])):
                per_sample_rows.append(
                    {
                        "episode_name": episode_name,
                        "camera_name": camera_name,
                        "query_frame": int(query_frame),
                        "traj_valid_mask_jaccard": per_sample.get("traj_valid_mask_jaccard"),
                        "traj_world_l2_mean": per_sample.get("traj_world_l2_mean"),
                        "traj_world_step_delta_l2_p95": per_sample.get("traj_world_step_delta_l2_p95"),
                        "valid_track_count_delta": per_sample.get("valid_track_count_delta"),
                    }
                )
    return {
        "per_episode_camera_rows": per_episode_camera_rows,
        "aggregate_by_camera": aggregate_quality_rows(per_episode_camera_rows, camera_names=camera_names),
        "worst_cases": build_worst_quality_cases(per_sample_rows),
        "per_sample_rows": per_sample_rows,
    }


def write_summary_markdown(summary: dict[str, Any], summary_path: Path) -> None:
    lines = [
        "# Output Root Comparison Summary",
        "",
        f"- Manifest: `{summary['manifest_path']}`",
        f"- Baseline: `{summary['baseline']['label']}` root=`{summary['baseline']['root']}` support_grid_ratio=`{summary['baseline']['support_grid_ratio']}`",
        f"- Variant: `{summary['variant']['label']}` root=`{summary['variant']['root']}` support_grid_ratio=`{summary['variant']['support_grid_ratio']}`",
        f"- Cameras: `{','.join(summary['camera_names'])}`",
        f"- Episodes: `{','.join(summary['episodes'])}`",
        "",
        "## Batch Wall Clock",
        "",
        "| Variant | num_iters | Wall Clock (s) | Speedup Vs Baseline |",
        "| --- | ---: | ---: | ---: |",
    ]
    wall_clock = summary["wall_clock_comparison"]
    lines.append(
        "| {label} | {num_iters} | {wall_clock} | {speedup} |".format(
            label=summary["baseline"]["label"],
            num_iters=summary["baseline"]["num_iters"],
            wall_clock=_format_float(wall_clock["baseline_wall_clock_seconds"]),
            speedup="1.000x",
        )
    )
    lines.append(
        "| {label} | {num_iters} | {wall_clock} | {speedup}x |".format(
            label=summary["variant"]["label"],
            num_iters=summary["variant"]["num_iters"],
            wall_clock=_format_float(wall_clock["variant_wall_clock_seconds"]),
            speedup=_format_float(wall_clock["variant_speedup_vs_baseline"]),
        )
    )

    lines.extend(
        [
            "",
            "## Single-GPU Per-Query Timing",
            "",
            "| Variant | Camera | Query Task Count | Process / Query Mean (s) | Variance | Save / Query Mean (s) | Variance | Total / Query Mean (s) | Variance |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant_key in ("baseline", "variant"):
        timing_summary = summary[variant_key]["per_query_timing_summary"] or summary[variant_key]["timing_summary"]
        for row in timing_summary["by_camera"]:
            metric_summaries = row["metric_summaries"]
            lines.append(
                "| {label} | {camera} | {task_count} | {process_mean} | {process_var} | {save_mean} | {save_var} | {total_mean} | {total_var} |".format(
                    label=summary[variant_key]["label"],
                    camera=row["camera_name"],
                    task_count=row["task_count"],
                    process_mean=_format_float(metric_summaries["process_seconds_per_query"]["mean"], digits=4),
                    process_var=_format_float(metric_summaries["process_seconds_per_query"]["variance"], digits=4),
                    save_mean=_format_float(metric_summaries["save_seconds_per_query"]["mean"], digits=4),
                    save_var=_format_float(metric_summaries["save_seconds_per_query"]["variance"], digits=4),
                    total_mean=_format_float(metric_summaries["total_seconds_per_query"]["mean"], digits=4),
                    total_var=_format_float(metric_summaries["total_seconds_per_query"]["variance"], digits=4),
                )
            )

    lines.extend(
        [
            "",
            "## Quality Aggregate",
            "",
            "| Camera | Valid Jaccard | Valid Delta | World L2 Mean | Step Delta P95 | World L2 P95 | Endpoint Error |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["quality_summary"]["aggregate_by_camera"]:
        metric_summaries = row["metric_summaries"]
        lines.append(
            "| {camera} | {jaccard} | {valid_delta} | {world_l2} | {step_delta_p95} | {world_l2_p95} | {endpoint_error} |".format(
                camera=row["camera_name"],
                jaccard=_format_float(metric_summaries["traj_valid_mask_jaccard_mean"]["mean"], digits=4),
                valid_delta=_format_float(metric_summaries["valid_track_count_delta_mean"]["mean"], digits=4),
                world_l2=_format_float(metric_summaries["traj_world_l2_mean"]["mean"], digits=6),
                step_delta_p95=_format_float(metric_summaries["traj_world_step_delta_l2_p95"]["mean"], digits=6),
                world_l2_p95=_format_float(metric_summaries["traj_world_l2_p95"]["mean"], digits=6),
                endpoint_error=_format_float(metric_summaries["traj_world_endpoint_l2_mean"]["mean"], digits=6),
            )
        )

    lines.extend(
        [
            "",
            "## Episode Quality",
            "",
            "| Episode | Camera | Common Query Frames | Valid Jaccard | Valid Delta | World L2 Mean | Step Delta P95 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["quality_summary"]["per_episode_camera_rows"]:
        lines.append(
            "| {episode} | {camera} | {common_query_frames} | {jaccard} | {valid_delta} | {world_l2} | {step_delta_p95} |".format(
                episode=row["episode_name"],
                camera=row["camera_name"],
                common_query_frames=row["common_query_frame_count"],
                jaccard=_format_float(row["traj_valid_mask_jaccard_mean"], digits=4),
                valid_delta=_format_float(row["valid_track_count_delta_mean"], digits=4),
                world_l2=_format_float(row["traj_world_l2_mean"], digits=6),
                step_delta_p95=_format_float(row["traj_world_step_delta_l2_p95"], digits=6),
            )
        )

    if summary["quality_summary"]["worst_cases"]:
        lines.extend(["", "## Worst Cases", ""])
        for row in summary["quality_summary"]["worst_cases"]:
            lines.append(
                "- `{selector}`: `{episode}` / `{camera}` / query_frame=`{query_frame}` / "
                "jaccard=`{jaccard}` / world_l2=`{world_l2}` / step_delta_p95=`{step_delta}`".format(
                    selector=row["selector"],
                    episode=row["episode_name"],
                    camera=row["camera_name"],
                    query_frame=row["query_frame"],
                    jaccard=_format_float(row.get("traj_valid_mask_jaccard"), digits=4),
                    world_l2=_format_float(row.get("traj_world_l2_mean"), digits=6),
                    step_delta=_format_float(row.get("traj_world_step_delta_l2_p95"), digits=6),
                )
            )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.baseline_root = args.baseline_root.resolve()
    args.variant_root = args.variant_root.resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_benchmark_manifest(args.manifest)
    camera_names = parse_camera_names(args.camera_names)

    baseline_batch_summary = load_optional_json(args.baseline_root / BATCH_RUN_SUMMARY_BASENAME)
    variant_batch_summary = load_optional_json(args.variant_root / BATCH_RUN_SUMMARY_BASENAME)
    validate_identical_manifest_hashes(baseline_batch_summary, variant_batch_summary)
    baseline_timing_summary = aggregate_timing_records(
        load_jsonl(args.baseline_root / CAMERA_TASK_METRICS_BASENAME),
        episodes=manifest["episodes"],
        camera_names=camera_names,
    )
    variant_timing_summary = aggregate_timing_records(
        load_jsonl(args.variant_root / CAMERA_TASK_METRICS_BASENAME),
        episodes=manifest["episodes"],
        camera_names=camera_names,
    )
    baseline_query_task_path = args.baseline_root / QUERY_TASK_METRICS_BASENAME
    variant_query_task_path = args.variant_root / QUERY_TASK_METRICS_BASENAME
    baseline_query_task_rows = (
        load_jsonl(baseline_query_task_path)
        if baseline_query_task_path.is_file()
        else []
    )
    variant_query_task_rows = (
        load_jsonl(variant_query_task_path)
        if variant_query_task_path.is_file()
        else []
    )
    baseline_per_query_timing_summary = (
        aggregate_timing_records(
            baseline_query_task_rows,
            episodes=manifest["episodes"],
            camera_names=camera_names,
        )
        if baseline_query_task_rows
        else None
    )
    variant_per_query_timing_summary = (
        aggregate_timing_records(
            variant_query_task_rows,
            episodes=manifest["episodes"],
            camera_names=camera_names,
        )
        if variant_query_task_rows
        else None
    )
    quality_summary = summarize_quality_for_roots(
        episodes=manifest["episodes"],
        camera_names=camera_names,
        baseline_root=args.baseline_root,
        variant_root=args.variant_root,
        require_identical_query_frames=not bool(args.allow_query_frame_mismatch),
    )

    baseline_wall_clock_seconds = (
        float(baseline_batch_summary["wall_clock_seconds"])
        if baseline_batch_summary is not None and baseline_batch_summary.get("wall_clock_seconds") is not None
        else None
    )
    variant_wall_clock_seconds = (
        float(variant_batch_summary["wall_clock_seconds"])
        if variant_batch_summary is not None and variant_batch_summary.get("wall_clock_seconds") is not None
        else None
    )
    variant_speedup_vs_baseline = (
        None
        if baseline_wall_clock_seconds is None
        or variant_wall_clock_seconds is None
        or variant_wall_clock_seconds == 0
        else float(baseline_wall_clock_seconds / variant_wall_clock_seconds)
    )

    summary = {
        "manifest_path": str(manifest["manifest_path"]),
        "dataset_root": str(manifest["dataset_root"]),
        "episodes": manifest["episodes"],
        "camera_names": camera_names,
        "baseline": {
            "label": args.baseline_label,
            "num_iters": int(args.baseline_num_iters),
            "root": str(args.baseline_root),
            "batch_summary": baseline_batch_summary,
            "timing_summary": baseline_timing_summary,
            "per_query_timing_summary": baseline_per_query_timing_summary,
            "support_grid_ratio": (
                None if baseline_batch_summary is None else baseline_batch_summary.get("support_grid_ratio")
            ),
        },
        "variant": {
            "label": args.variant_label,
            "num_iters": int(args.variant_num_iters),
            "root": str(args.variant_root),
            "batch_summary": variant_batch_summary,
            "timing_summary": variant_timing_summary,
            "per_query_timing_summary": variant_per_query_timing_summary,
            "support_grid_ratio": (
                None if variant_batch_summary is None else variant_batch_summary.get("support_grid_ratio")
            ),
        },
        "wall_clock_comparison": {
            "baseline_wall_clock_seconds": baseline_wall_clock_seconds,
            "variant_wall_clock_seconds": variant_wall_clock_seconds,
            "variant_speedup_vs_baseline": variant_speedup_vs_baseline,
        },
        "quality_summary": quality_summary,
    }

    summary_json_path = args.output_root / RESULT_JSON_BASENAME
    summary_md_path = args.output_root / SUMMARY_MD_BASENAME
    _atomic_write_json(summary_json_path, summary)
    write_summary_markdown(summary, summary_md_path)
    print(f"JSON summary: {summary_json_path}")
    print(f"Markdown summary: {summary_md_path}")


if __name__ == "__main__":
    main()
