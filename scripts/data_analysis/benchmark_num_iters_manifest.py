#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


CURRENT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(CURRENT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPO_ROOT))

from scripts.data_analysis.benchmark_num_iters_sweep import (  # noqa: E402
    RESULT_JSON_BASENAME,
    SUMMARY_MD_BASENAME,
    _atomic_write_json,
    _format_float,
    _mean,
    _stdev,
    load_benchmark_runtime,
    release_benchmark_runtime,
    run_num_iters_sweep,
    write_num_iters_summary,
)


DEFAULT_MANIFEST_NUM_ITERS_VALUES = (5, 4, 3, 2, 1)
DEFAULT_VOLATILITY_METRIC_KEYS = (
    "process_total_seconds",
    "prepare_depth_filter_seconds",
    "prepare_depth_filter_worker_total_seconds",
    "prepare_depth_filter_points_to_normals_seconds",
    "prepare_depth_filter_edge_mask_seconds",
)


def parse_args() -> argparse.Namespace:
    default_output_root = (
        CURRENT_REPO_ROOT
        / "data_tmp"
        / "num_iters_manifest_sweeps"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    parser = argparse.ArgumentParser(
        description=(
            "Run the single-episode num_iters sweep over a manifest-defined episode set, "
            "then summarize cross-episode runtime and volatility."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a JSON manifest with dataset_root and episodes.",
    )
    parser.add_argument(
        "--camera-names",
        type=str,
        default="varied_camera_1,varied_camera_3",
        help="Comma-separated camera names. Defaults to one external and one wrist-like camera.",
    )
    parser.add_argument(
        "--num-iters-values",
        type=str,
        default=",".join(str(value) for value in DEFAULT_MANIFEST_NUM_ITERS_VALUES),
        help="Comma-separated num_iters sweep, for example 5,4,3,2,1.",
    )
    parser.add_argument(
        "--baseline-num-iters",
        type=int,
        default=5,
        help="Baseline num_iters used for pairwise quantitative comparison.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Optional comma-separated subset of generated variant names, such as iters_5,iters_3.",
    )
    parser.add_argument(
        "--traj-filter-profile",
        type=str,
        default="auto",
        help=(
            "Trajectory filter profile. 'auto' maps wrist-like camera names to "
            "wrist_manipulator_top95 and others to external."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CURRENT_REPO_ROOT / "checkpoints" / "tapip3d_final.pth",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-num-frames", type=int, default=512)
    parser.add_argument("--future-len", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument(
        "--support-grid-ratio",
        type=float,
        default=0.0,
        help="Shared support_grid_ratio for all num_iters variants. Defaults to 0.0.",
    )
    parser.add_argument(
        "--filter-level",
        type=str,
        default="standard",
        choices=["none", "basic", "standard", "strict"],
    )
    parser.add_argument("--keyframes-per-sec-min", type=int, default=2)
    parser.add_argument("--keyframes-per-sec-max", type=int, default=3)
    parser.add_argument("--keyframe-seed", type=int, default=0)
    parser.add_argument("--fallback-episode-fps", type=float, default=0.0)
    parser.add_argument("--external-geom-name", type=str, default="trajectory_valid.h5")
    parser.add_argument(
        "--external-extr-mode",
        type=str,
        default="w2c",
        choices=["w2c", "c2w"],
    )
    parser.add_argument("--benchmark-runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument("--keep-outputs", action="store_true")
    parser.add_argument("--run-visual-verification", action="store_true")
    return parser.parse_args()


def _resolve_manifest_root(manifest_path: Path, dataset_root_raw: str) -> Path:
    dataset_root = Path(dataset_root_raw)
    if not dataset_root.is_absolute():
        dataset_root = (manifest_path.parent / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()
    return dataset_root


def load_benchmark_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {manifest_path}")

    dataset_root_raw = payload.get("dataset_root")
    if not isinstance(dataset_root_raw, str) or not dataset_root_raw.strip():
        raise ValueError("Manifest must contain non-empty string field 'dataset_root'")

    episodes_raw = payload.get("episodes")
    if not isinstance(episodes_raw, list) or not episodes_raw:
        raise ValueError("Manifest must contain non-empty list field 'episodes'")

    dataset_root = _resolve_manifest_root(manifest_path, dataset_root_raw)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Manifest dataset_root does not exist: {dataset_root}")

    episodes: list[str] = []
    episode_dirs: list[Path] = []
    for item in episodes_raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Manifest episodes must be non-empty strings")
        episode_name = item.strip()
        episode_dir = (dataset_root / episode_name).resolve()
        if not episode_dir.is_dir():
            raise FileNotFoundError(f"Manifest episode does not exist: {episode_dir}")
        episodes.append(episode_name)
        episode_dirs.append(episode_dir)

    return {
        "manifest_path": manifest_path,
        "dataset_root": dataset_root,
        "episodes": episodes,
        "episode_dirs": episode_dirs,
    }


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


def build_aggregate_case_results(
    episode_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
    for episode_result in episode_results:
        episode_name = str(episode_result["episode_name"])
        summary = episode_result["summary"]
        for case in summary["case_results"]:
            key = (case["camera_name"], case["variant_name"])
            grouped.setdefault(key, []).append((episode_name, case))

    aggregate_rows: list[dict[str, Any]] = []
    for (camera_name, variant_name), items in sorted(grouped.items()):
        episodes = [episode_name for episode_name, _ in items]
        cases = [case for _, case in items]
        aggregate_rows.append(
            {
                "camera_name": camera_name,
                "variant_name": variant_name,
                "variant_config": dict(cases[0]["variant_config"]),
                "traj_filter_profile": cases[0]["traj_filter_profile"],
                "episode_count": len(cases),
                "episodes": episodes,
                "aggregates": _aggregate_numeric_mappings([case["aggregates"] for case in cases]),
                "process_profile_aggregates": _aggregate_profile_mean_mappings(
                    [case["process_profile_aggregates"] for case in cases]
                ),
                "save_profile_aggregates": _aggregate_profile_mean_mappings(
                    [case["save_profile_aggregates"] for case in cases]
                ),
            }
        )
    return aggregate_rows


def _build_metric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "sample_count": 0,
            "mean": None,
            "stdev": None,
            "cv": None,
            "min": None,
            "max": None,
        }
    mean_value = _mean(values)
    stdev_value = _stdev(values)
    return {
        "sample_count": len(values),
        "mean": mean_value,
        "stdev": stdev_value,
        "cv": None if mean_value in (None, 0.0) or stdev_value is None else float(stdev_value / mean_value),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def build_volatility_summary(
    episode_results: list[dict[str, Any]],
    *,
    metric_keys: tuple[str, ...] = DEFAULT_VOLATILITY_METRIC_KEYS,
) -> dict[str, Any]:
    by_episode_case: list[dict[str, Any]] = []
    cross_episode_values: dict[tuple[str, str], dict[str, list[float]]] = {}
    cross_episode_names: dict[tuple[str, str], list[str]] = {}

    for episode_result in episode_results:
        episode_name = str(episode_result["episode_name"])
        summary = episode_result["summary"]
        for case in summary["case_results"]:
            metric_values_by_key: dict[str, list[float]] = {metric_key: [] for metric_key in metric_keys}
            for run in case["measured_runs"]:
                profile_stats = run.get("process_profile_stats", {})
                for metric_key in metric_keys:
                    if metric_key in profile_stats and profile_stats[metric_key] is not None:
                        metric_values_by_key[metric_key].append(float(profile_stats[metric_key]))

            by_episode_case.append(
                {
                    "episode_name": episode_name,
                    "camera_name": case["camera_name"],
                    "variant_name": case["variant_name"],
                    "variant_config": dict(case["variant_config"]),
                    "metric_summaries": {
                        metric_key: _build_metric_summary(metric_values_by_key[metric_key])
                        for metric_key in metric_keys
                    },
                }
            )

            cross_key = (case["camera_name"], case["variant_name"])
            cross_episode_values.setdefault(
                cross_key,
                {metric_key: [] for metric_key in metric_keys},
            )
            cross_episode_names.setdefault(cross_key, []).append(episode_name)
            for metric_key in metric_keys:
                cross_episode_values[cross_key][metric_key].extend(metric_values_by_key[metric_key])

    by_camera_variant: list[dict[str, Any]] = []
    for (camera_name, variant_name), metric_values in sorted(cross_episode_values.items()):
        by_camera_variant.append(
            {
                "camera_name": camera_name,
                "variant_name": variant_name,
                "episode_count": len(cross_episode_names[(camera_name, variant_name)]),
                "episodes": cross_episode_names[(camera_name, variant_name)],
                "metric_summaries": {
                    metric_key: _build_metric_summary(metric_values[metric_key])
                    for metric_key in metric_keys
                },
            }
        )

    return {
        "metric_keys": list(metric_keys),
        "by_episode_case": by_episode_case,
        "by_camera_variant": by_camera_variant,
    }


def flatten_animation_commands(episode_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for episode_result in episode_results:
        episode_name = str(episode_result["episode_name"])
        for item in episode_result["summary"].get("animation_commands", []):
            flattened.append(
                {
                    "episode_name": episode_name,
                    "camera_name": item["camera_name"],
                    "variant_name": item["variant_name"],
                    "query_frame": int(item["query_frame"]),
                    "episode_dir": item["episode_dir"],
                    "command": item["command"],
                }
            )
    return flattened


def write_manifest_summary_markdown(summary: dict[str, Any], summary_path: Path) -> None:
    lines = [
        "# Num Iters Manifest Sweep Summary",
        "",
        f"- Manifest: `{summary['manifest_path']}`",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Episodes: `{','.join(summary['episodes'])}`",
        f"- Cameras: `{','.join(summary['camera_names'])}`",
        f"- Num iters values: `{','.join(str(item) for item in summary['num_iters_values'])}`",
        f"- Baseline num_iters: `{summary['baseline_num_iters']}`",
        f"- Support grid ratio: `{summary['support_grid_ratio']}`",
        f"- Current repo: `{summary['current_repo_root']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Device: `{summary['device']}`",
        f"- Benchmark runs: `{summary['benchmark_runs']}`",
        f"- Warmup runs: `{summary['warmup_runs']}`",
        "",
        "## Episode Runtime",
        "",
        "| Episode | Camera | Variant | num_iters | Process (s) | Save (s) | Total (s) | Depth Filter (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for episode_result in summary["episode_results"]:
        episode_name = episode_result["episode_name"]
        for case in episode_result["summary"]["case_results"]:
            process_profile = case["process_profile_aggregates"]
            lines.append(
                "| {episode} | {camera} | {variant} | {num_iters} | {process} | {save} | {total} | {depth_filter} |".format(
                    episode=episode_name,
                    camera=case["camera_name"],
                    variant=case["variant_name"],
                    num_iters=case["variant_config"]["num_iters"],
                    process=_format_float(case["aggregates"]["process_seconds_mean"]),
                    save=_format_float(case["aggregates"]["save_seconds_mean"]),
                    total=_format_float(case["aggregates"]["total_seconds_mean"]),
                    depth_filter=_format_float(process_profile.get("prepare_depth_filter_seconds", {}).get("mean")),
                )
            )

    lines.extend(
        [
            "",
            "## Aggregate Runtime",
            "",
            "| Camera | Variant | Episodes | Process (s) | Save (s) | Total (s) | Depth Filter (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["aggregate_case_results"]:
        lines.append(
            "| {camera} | {variant} | {episodes} | {process} | {save} | {total} | {depth_filter} |".format(
                camera=row["camera_name"],
                variant=row["variant_name"],
                episodes=row["episode_count"],
                process=_format_float(row["aggregates"].get("process_seconds_mean", {}).get("mean")),
                save=_format_float(row["aggregates"].get("save_seconds_mean", {}).get("mean")),
                total=_format_float(row["aggregates"].get("total_seconds_mean", {}).get("mean")),
                depth_filter=_format_float(
                    row["process_profile_aggregates"].get("prepare_depth_filter_seconds", {}).get("mean")
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Volatility By Episode Case",
            "",
            "| Episode | Camera | Variant | Runs | PointsToNormals Mean (s) | Std (s) | CV | Depth Filter Mean (s) | Edge Mask Mean (s) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["volatility_summary"]["by_episode_case"]:
        ptn = item["metric_summaries"]["prepare_depth_filter_points_to_normals_seconds"]
        depth_filter = item["metric_summaries"]["prepare_depth_filter_seconds"]
        edge_mask = item["metric_summaries"]["prepare_depth_filter_edge_mask_seconds"]
        lines.append(
            "| {episode} | {camera} | {variant} | {runs} | {ptn_mean} | {ptn_stdev} | {ptn_cv} | {depth_mean} | {edge_mean} |".format(
                episode=item["episode_name"],
                camera=item["camera_name"],
                variant=item["variant_name"],
                runs=ptn["sample_count"],
                ptn_mean=_format_float(ptn["mean"]),
                ptn_stdev=_format_float(ptn["stdev"]),
                ptn_cv=_format_float(ptn["cv"]),
                depth_mean=_format_float(depth_filter["mean"]),
                edge_mean=_format_float(edge_mask["mean"]),
            )
        )

    lines.extend(
        [
            "",
            "## Volatility Aggregate",
            "",
            "| Camera | Variant | Episodes | Samples | PointsToNormals Mean (s) | Std (s) | CV | Min (s) | Max (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["volatility_summary"]["by_camera_variant"]:
        ptn = item["metric_summaries"]["prepare_depth_filter_points_to_normals_seconds"]
        lines.append(
            "| {camera} | {variant} | {episodes} | {samples} | {mean} | {stdev} | {cv} | {min_v} | {max_v} |".format(
                camera=item["camera_name"],
                variant=item["variant_name"],
                episodes=item["episode_count"],
                samples=ptn["sample_count"],
                mean=_format_float(ptn["mean"]),
                stdev=_format_float(ptn["stdev"]),
                cv=_format_float(ptn["cv"]),
                min_v=_format_float(ptn["min"]),
                max_v=_format_float(ptn["max"]),
            )
        )

    if summary["animation_commands"]:
        lines.extend(["", "## 3D Animation Commands", ""])
        for item in summary["animation_commands"]:
            lines.append(
                f"- `{item['episode_name']}` / `{item['camera_name']}` / `{item['variant_name']}` / query_frame=`{item['query_frame']}`"
            )
            lines.append("```bash")
            lines.append(item["command"])
            lines.append("```")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.output_root = args.output_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_benchmark_manifest(args.manifest)
    runtime = load_benchmark_runtime(
        checkpoint=args.checkpoint,
        device=args.device,
    )

    episode_results: list[dict[str, Any]] = []
    try:
        for episode_name, episode_dir in zip(manifest["episodes"], manifest["episode_dirs"]):
            episode_output_root = args.output_root / "episodes" / episode_name
            episode_args = argparse.Namespace(
                episode_dir=episode_dir,
                camera_names=args.camera_names,
                num_iters_values=args.num_iters_values,
                baseline_num_iters=args.baseline_num_iters,
                variants=args.variants,
                traj_filter_profile=args.traj_filter_profile,
                checkpoint=args.checkpoint,
                device=args.device,
                fps=args.fps,
                max_num_frames=args.max_num_frames,
                future_len=args.future_len,
                grid_size=args.grid_size,
                support_grid_ratio=args.support_grid_ratio,
                filter_level=args.filter_level,
                keyframes_per_sec_min=args.keyframes_per_sec_min,
                keyframes_per_sec_max=args.keyframes_per_sec_max,
                keyframe_seed=args.keyframe_seed,
                fallback_episode_fps=args.fallback_episode_fps,
                external_geom_name=args.external_geom_name,
                external_extr_mode=args.external_extr_mode,
                benchmark_runs=args.benchmark_runs,
                warmup_runs=args.warmup_runs,
                output_root=episode_output_root,
                keep_outputs=args.keep_outputs,
                run_visual_verification=args.run_visual_verification,
            )
            episode_summary = run_num_iters_sweep(
                episode_args,
                runtime=runtime,
            )
            summary_json_path, summary_md_path = write_num_iters_summary(
                episode_summary,
                output_root=episode_output_root,
            )
            episode_results.append(
                {
                    "episode_name": episode_name,
                    "episode_dir": str(episode_dir),
                    "output_root": str(episode_output_root),
                    "summary_json_path": str(summary_json_path),
                    "summary_md_path": str(summary_md_path),
                    "summary": episode_summary,
                }
            )
    finally:
        release_benchmark_runtime(runtime)

    aggregate_case_results = build_aggregate_case_results(episode_results)
    volatility_summary = build_volatility_summary(episode_results)
    animation_commands = flatten_animation_commands(episode_results)

    summary = {
        "manifest_path": str(manifest["manifest_path"]),
        "dataset_root": str(manifest["dataset_root"]),
        "episodes": manifest["episodes"],
        "camera_names": episode_results[0]["summary"]["camera_names"] if episode_results else [],
        "num_iters_values": episode_results[0]["summary"]["num_iters_values"] if episode_results else [],
        "baseline_num_iters": int(args.baseline_num_iters),
        "baseline_variant_name": episode_results[0]["summary"]["baseline_variant_name"] if episode_results else None,
        "support_grid_ratio": float(args.support_grid_ratio),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "benchmark_runs": int(args.benchmark_runs),
        "warmup_runs": int(args.warmup_runs),
        "current_repo_root": str(CURRENT_REPO_ROOT.resolve()),
        "episode_results": episode_results,
        "aggregate_case_results": aggregate_case_results,
        "volatility_summary": volatility_summary,
        "animation_commands": animation_commands,
    }

    summary_json_path = args.output_root / RESULT_JSON_BASENAME
    summary_md_path = args.output_root / SUMMARY_MD_BASENAME
    _atomic_write_json(summary_json_path, summary)
    write_manifest_summary_markdown(summary, summary_md_path)
    print(f"JSON summary: {summary_json_path}")
    print(f"Markdown summary: {summary_md_path}")


if __name__ == "__main__":
    main()
