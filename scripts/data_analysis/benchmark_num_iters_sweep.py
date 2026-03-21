#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import shutil
import shlex
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


CURRENT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(CURRENT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPO_ROOT))

from scripts.data_analysis.benchmark_depth_volatility_optimization import (  # noqa: E402
    ensure_query_frame_schedule,
    parse_camera_names,
    resolve_traj_filter_profile,
)
from scripts.data_analysis.benchmark_inference_variants import (  # noqa: E402
    collect_sample_summaries,
    compare_camera_outputs,
    run_visual_verification,
    summarize_case_samples,
)


DEFAULT_NUM_ITERS_VALUES = (6, 5, 4)
DEFAULT_SUPPORT_GRID_RATIO = 0.0
QUERY_PREFILTER_MODE_OFF = "off"
RESULT_JSON_BASENAME = "benchmark_results.json"
SUMMARY_MD_BASENAME = "benchmark_summary.md"


def parse_args() -> argparse.Namespace:
    default_output_root = (
        CURRENT_REPO_ROOT
        / "data_tmp"
        / "num_iters_sweeps"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark current-repo inference variants that sweep tracker num_iters, "
            "then compare runtime, quantitative trajectory deltas, and visual verification artifacts."
        )
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        required=True,
        help="Episode directory, e.g. /data1/yaoxuran/press_one_button_demo_v1/episode_00105",
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
        default=",".join(str(value) for value in DEFAULT_NUM_ITERS_VALUES),
        help="Comma-separated num_iters sweep, for example 6,5,4.",
    )
    parser.add_argument(
        "--baseline-num-iters",
        type=int,
        default=6,
        help="Baseline num_iters used for pairwise quantitative comparison.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help=(
            "Optional comma-separated subset of generated variant names. "
            "Generated names are iters_N, for example iters_6,iters_5,iters_4."
        ),
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
        default=DEFAULT_SUPPORT_GRID_RATIO,
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


def parse_variant_names(raw: str | None) -> list[str]:
    if raw is None:
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("variants must contain at least one entry")
    return values


def parse_num_iters_values(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("num-iters-values must contain positive integers")
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("num-iters-values must contain at least one value")
    return values


def format_num_iters_variant_name(num_iters: int) -> str:
    return f"iters_{int(num_iters)}"


def build_variant_specs(
    *,
    num_iters_values: list[int],
    baseline_num_iters: int,
    support_grid_ratio: float,
    selected_variant_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    if baseline_num_iters not in num_iters_values:
        raise ValueError(
            f"baseline_num_iters={baseline_num_iters} must be included in num_iters_values={num_iters_values}"
        )
    if support_grid_ratio < 0.0:
        raise ValueError("support_grid_ratio must be >= 0")

    selected_variant_names = selected_variant_names or []
    supported_by_name: dict[str, dict[str, Any]] = {}
    for num_iters in num_iters_values:
        name = format_num_iters_variant_name(num_iters)
        supported_by_name[name] = {
            "name": name,
            "num_iters": int(num_iters),
            "query_prefilter_mode": QUERY_PREFILTER_MODE_OFF,
            "support_grid_ratio": float(support_grid_ratio),
            "is_baseline": bool(num_iters == baseline_num_iters),
        }

    if selected_variant_names:
        missing = [name for name in selected_variant_names if name not in supported_by_name]
        if missing:
            raise ValueError(
                f"Unsupported variants {missing}. Expected a subset of {sorted(supported_by_name.keys())}"
            )
        specs = [dict(supported_by_name[name]) for name in selected_variant_names]
    else:
        ordered_values = [baseline_num_iters] + [value for value in num_iters_values if value != baseline_num_iters]
        specs = [dict(supported_by_name[format_num_iters_variant_name(value)]) for value in ordered_values]

    if not any(bool(spec["is_baseline"]) for spec in specs):
        raise ValueError("Selected variants must include the baseline num_iters variant")
    return specs


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


def resolve_num_iters_sweep_args(args: argparse.Namespace) -> argparse.Namespace:
    resolved = argparse.Namespace(**vars(args))
    resolved.episode_dir = Path(resolved.episode_dir).resolve()
    resolved.output_root = Path(resolved.output_root).resolve()
    resolved.checkpoint = Path(resolved.checkpoint).resolve()
    resolved.output_root.mkdir(parents=True, exist_ok=True)
    return resolved


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _stdev(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.stdev(values))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator / denominator)


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _finite_percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, q))


def aggregate_profile_stats(
    run_records: list[dict[str, Any]],
    *,
    key: str,
) -> dict[str, dict[str, float | None]]:
    keys = sorted(
        {
            profile_key
            for record in run_records
            for profile_key in record.get(key, {}).keys()
        }
    )
    aggregated: dict[str, dict[str, float | None]] = {}
    for profile_key in keys:
        values = [
            float(record[key][profile_key])
            for record in run_records
            if profile_key in record.get(key, {})
        ]
        aggregated[profile_key] = {
            "mean": _mean(values),
            "stdev": _stdev(values),
        }
    return aggregated


def _sync_cuda_if_needed(torch_module, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(device)


def load_benchmark_runtime(
    *,
    checkpoint: Path,
    device: str,
) -> dict[str, Any]:
    import importlib

    infer_module = importlib.import_module("scripts.batch_inference.infer")
    torch_module = importlib.import_module("torch")
    model_3dtracker = infer_module.load_model(str(checkpoint)).to(device)
    return {
        "infer_module": infer_module,
        "torch_module": torch_module,
        "model_3dtracker": model_3dtracker,
    }


def release_benchmark_runtime(runtime: dict[str, Any]) -> None:
    torch_module = runtime["torch_module"]
    del runtime["model_3dtracker"]
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def build_infer_args(
    *,
    args: argparse.Namespace,
    camera_name: str,
    query_frame_schedule_path: Path,
    traj_filter_profile: str,
    variant_spec: dict[str, Any],
) -> argparse.Namespace:
    return argparse.Namespace(
        video_path=str(args.episode_dir / "rgb" / camera_name),
        depth_path=str(args.episode_dir / "depth" / camera_name),
        mask_dir=None,
        checkpoint=str(args.checkpoint),
        depth_pose_method="external",
        external_extr_mode=args.external_extr_mode,
        external_geom_npz=str(args.episode_dir / args.external_geom_name),
        camera_name=camera_name,
        query_frame_schedule_path=str(query_frame_schedule_path),
        device=args.device,
        num_iters=int(variant_spec["num_iters"]),
        fps=args.fps,
        out_dir=str(args.output_root),
        video_name=camera_name,
        max_num_frames=args.max_num_frames,
        save_video=False,
        output_layout="v2",
        scene_storage_mode="source_ref",
        save_visibility=False,
        horizon=16,
        batch_process=False,
        skip_existing=False,
        use_all_trajectories=True,
        frame_drop_rate=1,
        scan_depth=2,
        future_len=args.future_len,
        max_frames_per_video=args.max_num_frames,
        grid_size=args.grid_size,
        filter_level=args.filter_level,
        traj_filter_profile=traj_filter_profile,
        traj_filter_ablation_mode="none",
        min_valid_frames=None,
        visibility_threshold=None,
        min_depth=0.01,
        max_depth=10.0,
        boundary_margin=None,
        depth_change_threshold=None,
        query_prefilter_mode=QUERY_PREFILTER_MODE_OFF,
        query_prefilter_wrist_rank_keep_ratio=0.30,
        support_grid_ratio=float(variant_spec["support_grid_ratio"]),
        collect_profile_stats=True,
    )


def benchmark_variant_case(
    *,
    infer_module,
    torch_module,
    model_3dtracker,
    args: argparse.Namespace,
    camera_name: str,
    traj_filter_profile: str,
    query_frame_schedule_path: Path,
    variant_spec: dict[str, Any],
) -> dict[str, Any]:
    infer_args = build_infer_args(
        args=args,
        camera_name=camera_name,
        query_frame_schedule_path=query_frame_schedule_path,
        traj_filter_profile=traj_filter_profile,
        variant_spec=variant_spec,
    )
    model_depth_pose = infer_module.video_depth_pose_dict[infer_args.depth_pose_method](infer_args)

    raw_runs: list[dict[str, Any]] = []
    measured_runs: list[dict[str, Any]] = []
    representative_output_dir: str | None = None
    representative_sample_summaries: dict[str, dict[str, Any]] | None = None

    total_runs = int(args.warmup_runs) + int(args.benchmark_runs)
    artifacts_root = args.output_root / "artifacts" / camera_name / variant_spec["name"]
    try:
        for run_idx in range(total_runs):
            run_name = f"run_{run_idx:02d}"
            run_output_root = artifacts_root / run_name
            if run_output_root.exists():
                shutil.rmtree(run_output_root)
            run_output_root.mkdir(parents=True, exist_ok=True)

            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            _sync_cuda_if_needed(torch_module, args.device)
            process_start = time.perf_counter()
            result = infer_module.process_single_video(
                str(args.episode_dir / "rgb" / camera_name),
                str(args.episode_dir / "depth" / camera_name),
                infer_args,
                model_3dtracker,
                model_depth_pose,
                camera_name,
                str(run_output_root),
            )
            _sync_cuda_if_needed(torch_module, args.device)
            process_seconds = time.perf_counter() - process_start

            save_start = time.perf_counter()
            save_result = infer_module.save_structured_data(
                video_name=camera_name,
                output_dir=str(run_output_root),
                video_tensor=result["video_tensor"],
                depths=result["depths"],
                coords=result["coords"],
                visibs=result["visibs"],
                intrinsics=result["intrinsics"],
                extrinsics=result["extrinsics"],
                query_points_per_frame=result["query_points_per_frame"],
                original_filenames=result["original_filenames"],
                query_frame_results=result.get("query_frame_results"),
                future_len=infer_args.future_len,
                grid_size=infer_args.grid_size,
                filter_args=infer_args,
                full_video_tensor=result["full_video_tensor"],
                full_depths=result["full_depths"],
                full_intrinsics=result["full_intrinsics"],
                full_extrinsics=result["full_extrinsics"],
                depth_conf=result["depth_conf"],
                video_source_path=str(args.episode_dir / "rgb" / camera_name),
                depth_source_path=str(args.episode_dir / "depth" / camera_name),
                source_frame_indices=result["source_frame_indices"],
                query_frame_metadata=result.get("query_frame_metadata"),
            )
            _sync_cuda_if_needed(torch_module, args.device)
            save_seconds = time.perf_counter() - save_start

            sample_summaries = collect_sample_summaries(run_output_root, camera_name=camera_name)
            query_frame_results = result.get("query_frame_results") or {}
            tracked_query_counts = [
                int(frame_data["tracked_query_count"])
                for frame_data in query_frame_results.values()
            ]
            dense_query_counts = [
                int(frame_data["dense_query_count"])
                for frame_data in query_frame_results.values()
            ]
            support_grid_sizes = [
                int(frame_data["support_grid_size"])
                for frame_data in query_frame_results.values()
                if frame_data.get("support_grid_size") is not None
            ]
            effective_support_query_counts = [
                int(frame_data.get("effective_support_query_count", 0))
                for frame_data in query_frame_results.values()
            ]
            process_profile_stats = {
                key: float(value)
                for key, value in (result.get("profile_stats") or {}).items()
                if value is not None
            }
            save_profile_stats = {
                key: float(value)
                for key, value in (save_result.get("save_profile_stats") or {}).items()
                if value is not None
            }
            run_record = {
                "run_index": int(run_idx),
                "warmup": bool(run_idx < args.warmup_runs),
                "process_seconds": float(process_seconds),
                "save_seconds": float(save_seconds),
                "total_seconds": float(process_seconds + save_seconds),
                "query_frame_count": int(len(query_frame_results)),
                "saved_sample_count": int(len(sample_summaries)),
                "frame_count": int(result["full_video_tensor"].shape[0]),
                "dense_query_count_mean": _mean([float(value) for value in dense_query_counts]),
                "tracked_query_count_mean": _mean([float(value) for value in tracked_query_counts]),
                "support_grid_sizes": support_grid_sizes,
                "support_grid_size_mean": _mean([float(value) for value in support_grid_sizes]),
                "effective_support_query_counts": effective_support_query_counts,
                "effective_support_query_count_mean": _mean(
                    [float(value) for value in effective_support_query_counts]
                ),
                "process_profile_stats": process_profile_stats,
                "save_profile_stats": save_profile_stats,
                "output_dir": str(run_output_root),
            }
            raw_runs.append(run_record)
            if run_idx >= args.warmup_runs:
                measured_runs.append(run_record)

            keep_run_outputs = bool(args.keep_outputs)
            if run_idx >= args.warmup_runs and representative_output_dir is None:
                representative_output_dir = str(run_output_root)
                representative_sample_summaries = sample_summaries
                keep_run_outputs = True

            del save_result
            del result
            gc.collect()
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            if not keep_run_outputs:
                shutil.rmtree(run_output_root)
    finally:
        del model_depth_pose
        gc.collect()
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

    process_values = [run["process_seconds"] for run in measured_runs]
    save_values = [run["save_seconds"] for run in measured_runs]
    total_values = [run["total_seconds"] for run in measured_runs]
    query_frame_count_values = [float(run["query_frame_count"]) for run in measured_runs]
    saved_sample_count_values = [float(run["saved_sample_count"]) for run in measured_runs]
    tracked_query_count_values = [
        float(run["tracked_query_count_mean"])
        for run in measured_runs
        if run["tracked_query_count_mean"] is not None
    ]
    dense_query_count_values = [
        float(run["dense_query_count_mean"])
        for run in measured_runs
        if run["dense_query_count_mean"] is not None
    ]
    support_grid_size_values = [
        float(value)
        for run in measured_runs
        for value in run.get("support_grid_sizes", [])
    ]
    effective_support_query_count_values = [
        float(value)
        for run in measured_runs
        for value in run.get("effective_support_query_counts", [])
    ]
    representative_sample_summaries = representative_sample_summaries or {}

    return {
        "variant_name": variant_spec["name"],
        "variant_config": dict(variant_spec),
        "camera_name": camera_name,
        "traj_filter_profile": traj_filter_profile,
        "schedule_path": str(query_frame_schedule_path),
        "benchmark_runs": int(args.benchmark_runs),
        "warmup_runs": int(args.warmup_runs),
        "raw_runs": raw_runs,
        "measured_runs": measured_runs,
        "aggregates": {
            "process_seconds_mean": _mean(process_values),
            "process_seconds_stdev": _stdev(process_values),
            "save_seconds_mean": _mean(save_values),
            "save_seconds_stdev": _stdev(save_values),
            "total_seconds_mean": _mean(total_values),
            "total_seconds_stdev": _stdev(total_values),
            "query_frame_count_mean": _mean(query_frame_count_values),
            "saved_sample_count_mean": _mean(saved_sample_count_values),
            "tracked_query_count_mean": _mean(tracked_query_count_values),
            "dense_query_count_mean": _mean(dense_query_count_values),
            "support_grid_size_mean": _mean(support_grid_size_values),
            "support_grid_size_p50": _finite_percentile(support_grid_size_values, 50.0),
            "support_grid_size_p90": _finite_percentile(support_grid_size_values, 90.0),
            "effective_support_query_count_mean": _mean(effective_support_query_count_values),
            "effective_support_query_count_p50": _finite_percentile(
                effective_support_query_count_values,
                50.0,
            ),
            "effective_support_query_count_p90": _finite_percentile(
                effective_support_query_count_values,
                90.0,
            ),
        },
        "process_profile_aggregates": aggregate_profile_stats(
            measured_runs,
            key="process_profile_stats",
        ),
        "save_profile_aggregates": aggregate_profile_stats(
            measured_runs,
            key="save_profile_stats",
        ),
        "representative_output_dir": representative_output_dir,
        "representative_sample_overview": summarize_case_samples(representative_sample_summaries),
        "representative_sample_summaries": representative_sample_summaries,
    }


def build_pairwise_comparisons(
    *,
    summary_by_key: dict[tuple[str, str], dict[str, Any]],
    variant_specs: list[dict[str, Any]],
    camera_names: list[str],
    baseline_variant_name: str,
) -> list[dict[str, Any]]:
    pairwise_comparisons: list[dict[str, Any]] = []
    for camera_name in camera_names:
        baseline_case = summary_by_key[(baseline_variant_name, camera_name)]
        for variant_spec in variant_specs:
            if variant_spec["name"] == baseline_variant_name:
                continue
            case = summary_by_key[(variant_spec["name"], camera_name)]
            sample_diff = compare_camera_outputs(
                Path(baseline_case["representative_output_dir"]),
                Path(case["representative_output_dir"]),
                camera_name=camera_name,
            )
            pairwise_comparisons.append(
                {
                    "camera_name": camera_name,
                    "variant_name": variant_spec["name"],
                    "traj_filter_profile": case["traj_filter_profile"],
                    "variant_config": case["variant_config"],
                    "process_speedup_vs_baseline": _safe_ratio(
                        baseline_case["aggregates"]["process_seconds_mean"],
                        case["aggregates"]["process_seconds_mean"],
                    ),
                    "save_speedup_vs_baseline": _safe_ratio(
                        baseline_case["aggregates"]["save_seconds_mean"],
                        case["aggregates"]["save_seconds_mean"],
                    ),
                    "total_speedup_vs_baseline": _safe_ratio(
                        baseline_case["aggregates"]["total_seconds_mean"],
                        case["aggregates"]["total_seconds_mean"],
                    ),
                    "baseline": baseline_case["aggregates"],
                    "variant": case["aggregates"],
                    "sample_diff": sample_diff,
                }
            )
    return pairwise_comparisons


def build_animation_commands(
    *,
    summary_by_key: dict[tuple[str, str], dict[str, Any]],
    variant_specs: list[dict[str, Any]],
    camera_names: list[str],
) -> list[dict[str, Any]]:
    animation_commands: list[dict[str, Any]] = []
    for camera_index, camera_name in enumerate(camera_names):
        representative_cases = [
            summary_by_key[(variant_spec["name"], camera_name)]
            for variant_spec in variant_specs
            if summary_by_key[(variant_spec["name"], camera_name)].get("representative_output_dir") is not None
        ]
        if len(representative_cases) != len(variant_specs):
            continue
        query_frame_sets = [
            set(map(int, case["representative_sample_summaries"].keys()))
            for case in representative_cases
        ]
        if not query_frame_sets:
            continue
        common_query_frames = sorted(set.intersection(*query_frame_sets))
        if not common_query_frames:
            continue
        query_frame = int(common_query_frames[0])
        for variant_index, variant_spec in enumerate(variant_specs):
            case = summary_by_key[(variant_spec["name"], camera_name)]
            representative_output_dir = Path(case["representative_output_dir"])
            episode_dir = representative_output_dir / camera_name
            cmd = [
                sys.executable,
                str(CURRENT_REPO_ROOT / "scripts" / "visualization" / "visualize_3d_keypoint_animation.py"),
                "--episode_dir",
                str(episode_dir),
                "--query_frame",
                str(query_frame),
                "--dense_pointcloud",
                "--port",
                str(8080 + camera_index * 100 + variant_index),
            ]
            animation_commands.append(
                {
                    "camera_name": camera_name,
                    "variant_name": variant_spec["name"],
                    "query_frame": query_frame,
                    "episode_dir": str(episode_dir),
                    "command": shlex.join(cmd),
                }
            )
    return animation_commands


def write_summary_markdown(summary: dict[str, Any], summary_path: Path) -> None:
    lines = [
        "# Num Iters Sweep Summary",
        "",
        f"- Episode: `{summary['episode_dir']}`",
        f"- Cameras: `{','.join(summary['camera_names'])}`",
        f"- Num iters values: `{','.join(str(item) for item in summary['num_iters_values'])}`",
        f"- Baseline num_iters: `{summary['baseline_num_iters']}`",
        f"- Support grid ratio: `{summary['support_grid_ratio']}`",
        f"- Current repo: `{summary['current_repo_root']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Device: `{summary['device']}`",
        f"- Schedule: `{summary['schedule_path']}`",
        f"- Benchmark runs: `{summary['benchmark_runs']}`",
        f"- Warmup runs: `{summary['warmup_runs']}`",
        "",
        "## Runtime",
        "",
        "| Camera | Variant | num_iters | Process (s) | Save (s) | Total (s) | Tracker Forward (s) | Depth Filter (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["case_results"]:
        process_profile = case["process_profile_aggregates"]
        lines.append(
            "| {camera} | {variant} | {num_iters} | {process} | {save} | {total} | {forward} | {depth_filter} |".format(
                camera=case["camera_name"],
                variant=case["variant_name"],
                num_iters=case["variant_config"]["num_iters"],
                process=_format_float(case["aggregates"]["process_seconds_mean"]),
                save=_format_float(case["aggregates"]["save_seconds_mean"]),
                total=_format_float(case["aggregates"]["total_seconds_mean"]),
                forward=_format_float(process_profile.get("tracker_model_forward_seconds", {}).get("mean")),
                depth_filter=_format_float(process_profile.get("prepare_depth_filter_seconds", {}).get("mean")),
            )
        )

    lines.extend(
        [
            "",
            "## Quantitative Diff Vs Baseline",
            "",
            "| Camera | Variant | Process Speedup | Total Speedup | Valid Jaccard | Valid Delta | World L2 Mean | Step Delta P95 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison in summary["pairwise_comparisons"]:
        diff_aggregates = comparison["sample_diff"]["aggregates"]
        lines.append(
            "| {camera} | {variant} | {process_speedup}x | {total_speedup}x | {jaccard} | {valid_delta} | {world_l2} | {step_delta_p95} |".format(
                camera=comparison["camera_name"],
                variant=comparison["variant_name"],
                process_speedup=_format_float(comparison["process_speedup_vs_baseline"]),
                total_speedup=_format_float(comparison["total_speedup_vs_baseline"]),
                jaccard=_format_float(diff_aggregates["traj_valid_mask_jaccard_mean"]),
                valid_delta=_format_float(diff_aggregates["valid_track_count_delta_mean"]),
                world_l2=_format_float(diff_aggregates["traj_world_l2_mean"]),
                step_delta_p95=_format_float(diff_aggregates["traj_world_step_delta_l2_p95"]),
            )
        )

    if summary["visual_verification"]:
        lines.extend(["", "## Visual Verification", ""])
        for item in summary["visual_verification"]:
            lines.append(
                f"- `{item['camera_name']}` query_frame=`{item['query_frame']}`"
            )
            for variant in item["variants"]:
                lines.append(
                    f"  - `{variant['variant_name']}`: `{variant['output_dir']}`"
                )

    if summary["animation_commands"]:
        lines.extend(["", "## 3D Animation Commands", ""])
        for item in summary["animation_commands"]:
            lines.append(
                f"- `{item['camera_name']}` / `{item['variant_name']}` / query_frame=`{item['query_frame']}`"
            )
            lines.append("```bash")
            lines.append(item["command"])
            lines.append("```")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_num_iters_sweep(
    args: argparse.Namespace,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args = resolve_num_iters_sweep_args(args)
    camera_names = parse_camera_names(args.camera_names)
    num_iters_values = parse_num_iters_values(args.num_iters_values)
    selected_variant_names = parse_variant_names(args.variants)
    variant_specs = build_variant_specs(
        num_iters_values=num_iters_values,
        baseline_num_iters=int(args.baseline_num_iters),
        support_grid_ratio=float(args.support_grid_ratio),
        selected_variant_names=selected_variant_names,
    )
    baseline_variant_name = format_num_iters_variant_name(int(args.baseline_num_iters))

    schedule_path = ensure_query_frame_schedule(
        episode_dir=args.episode_dir,
        camera_names=camera_names,
        external_geom_name=args.external_geom_name,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        keyframes_per_sec_min=args.keyframes_per_sec_min,
        keyframes_per_sec_max=args.keyframes_per_sec_max,
        keyframe_seed=args.keyframe_seed,
        fallback_episode_fps=args.fallback_episode_fps,
        output_root=args.output_root,
    )

    own_runtime = runtime is None
    active_runtime = runtime or load_benchmark_runtime(
        checkpoint=args.checkpoint,
        device=args.device,
    )
    infer_module = active_runtime["infer_module"]
    torch_module = active_runtime["torch_module"]
    model_3dtracker = active_runtime["model_3dtracker"]

    case_results: list[dict[str, Any]] = []
    try:
        for camera_name in camera_names:
            traj_filter_profile = resolve_traj_filter_profile(camera_name, args.traj_filter_profile)
            for variant_spec in variant_specs:
                case_results.append(
                    benchmark_variant_case(
                        infer_module=infer_module,
                        torch_module=torch_module,
                        model_3dtracker=model_3dtracker,
                        args=args,
                        camera_name=camera_name,
                        traj_filter_profile=traj_filter_profile,
                        query_frame_schedule_path=schedule_path,
                        variant_spec=variant_spec,
                    )
                )
    finally:
        if own_runtime:
            release_benchmark_runtime(active_runtime)

    summary_by_key = {
        (case["variant_name"], case["camera_name"]): case
        for case in case_results
    }
    pairwise_comparisons = build_pairwise_comparisons(
        summary_by_key=summary_by_key,
        variant_specs=variant_specs,
        camera_names=camera_names,
        baseline_variant_name=baseline_variant_name,
    )
    visual_verification: list[dict[str, Any]] = []
    if args.run_visual_verification:
        visual_verification = run_visual_verification(
            summary_by_key=summary_by_key,
            variant_specs=variant_specs,
            camera_names=camera_names,
            output_root=args.output_root,
        )
    animation_commands = build_animation_commands(
        summary_by_key=summary_by_key,
        variant_specs=variant_specs,
        camera_names=camera_names,
    )

    return {
        "episode_dir": str(args.episode_dir),
        "camera_names": camera_names,
        "num_iters_values": num_iters_values,
        "baseline_num_iters": int(args.baseline_num_iters),
        "baseline_variant_name": baseline_variant_name,
        "support_grid_ratio": float(args.support_grid_ratio),
        "schedule_path": str(schedule_path),
        "current_repo_root": str(CURRENT_REPO_ROOT.resolve()),
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "benchmark_runs": int(args.benchmark_runs),
        "warmup_runs": int(args.warmup_runs),
        "fps": int(args.fps),
        "max_num_frames": int(args.max_num_frames),
        "future_len": int(args.future_len),
        "grid_size": int(args.grid_size),
        "filter_level": args.filter_level,
        "traj_filter_profile": args.traj_filter_profile,
        "keyframes_per_sec_min": int(args.keyframes_per_sec_min),
        "keyframes_per_sec_max": int(args.keyframes_per_sec_max),
        "keep_outputs": bool(args.keep_outputs),
        "case_results": case_results,
        "pairwise_comparisons": pairwise_comparisons,
        "visual_verification": visual_verification,
        "animation_commands": animation_commands,
    }


def write_num_iters_summary(
    summary: dict[str, Any],
    *,
    output_root: Path,
) -> tuple[Path, Path]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_root / RESULT_JSON_BASENAME
    summary_md_path = output_root / SUMMARY_MD_BASENAME
    _atomic_write_json(summary_json_path, summary)
    write_summary_markdown(summary, summary_md_path)
    return summary_json_path, summary_md_path


def main() -> None:
    args = resolve_num_iters_sweep_args(parse_args())
    summary = run_num_iters_sweep(args)
    summary_json_path, summary_md_path = write_num_iters_summary(
        summary,
        output_root=args.output_root,
    )
    print(f"JSON summary: {summary_json_path}")
    print(f"Markdown summary: {summary_md_path}")


if __name__ == "__main__":
    main()
