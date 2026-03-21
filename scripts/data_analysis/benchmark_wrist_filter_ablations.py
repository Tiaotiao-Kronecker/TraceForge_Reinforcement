#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
import statistics
import subprocess
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
)
from scripts.data_analysis.benchmark_inference_variants import (  # noqa: E402
    collect_sample_summaries,
    compare_camera_outputs,
    summarize_case_samples,
)


DEFAULT_BASE_PATH = Path("/data1/yaoxuran/press_one_button_demo_v1")
DEFAULT_EPISODE_NAMES = (
    "episode_00089",
    "episode_00097",
    "episode_00105",
    "episode_00113",
)
DEFAULT_EXTERNAL_CAMERA_NAME = "varied_camera_1"
DEFAULT_WRIST_CAMERA_NAME = "varied_camera_3"
DEFAULT_OUTPUT_ROOT = (
    CURRENT_REPO_ROOT / "data_tmp" / "wrist_filter_ablations" / time.strftime("%Y%m%d_%H%M%S")
)
RESULT_JSON_BASENAME = "benchmark_results.json"
SUMMARY_MD_BASENAME = "benchmark_summary.md"

EXTERNAL_STAGE_ORDER = (
    "base_mask",
    "query_depth_quality",
    "query_depth_keep",
    "supervision_support",
    "final",
)
WRIST_STAGE_ORDER = (
    "base_mask",
    "query_depth_quality",
    "query_depth_keep",
    "supervision_support",
    "wrist_seed",
    "final",
)
WRIST_MANIPULATOR_STAGE_ORDER = (
    "base_mask",
    "query_depth_quality",
    "query_depth_keep",
    "supervision_support",
    "wrist_seed",
    "near_depth",
    "motion",
    "cluster",
    "pre_top95",
    "final",
)
RECOMMENDATION_PRIORITY = (
    "wrist_external",
    "wrist",
    "wrist_seed_top95",
    "wrist_manipulator",
    "wrist_manipulator_top95",
)
SAVE_ALIGNMENT_TOLERANCE_SECONDS = 0.5
MASK_JACCARD_THRESHOLD = 0.90
VALID_DELTA_RATIO_THRESHOLD = 0.05

STAGE_LABELS = {
    "base_mask": "base",
    "query_depth_quality": "query_quality",
    "query_depth_keep": "query_keep",
    "supervision_support": "support",
    "wrist_seed": "wrist_seed",
    "near_depth": "near_depth",
    "motion": "motion",
    "cluster": "cluster",
    "pre_top95": "pre_top95",
    "final": "final",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark save-time wrist filter ablations by replaying multiple save variants "
            "from one shared tracking result per episode/camera."
        )
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=DEFAULT_BASE_PATH,
        help="Dataset root, e.g. /data1/yaoxuran/press_one_button_demo_v1",
    )
    parser.add_argument(
        "--episode-names",
        type=str,
        default=",".join(DEFAULT_EPISODE_NAMES),
        help="Comma-separated episode names. Defaults to the four representative cases used in the ablation plan.",
    )
    parser.add_argument(
        "--external-camera-name",
        type=str,
        default=DEFAULT_EXTERNAL_CAMERA_NAME,
    )
    parser.add_argument(
        "--wrist-camera-name",
        type=str,
        default=DEFAULT_WRIST_CAMERA_NAME,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CURRENT_REPO_ROOT / "checkpoints" / "tapip3d_final.pth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-num-frames", type=int, default=512)
    parser.add_argument("--future-len", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument(
        "--filter-level",
        type=str,
        default="standard",
        choices=["none", "basic", "standard", "strict"],
    )
    parser.add_argument(
        "--keyframes-per-sec-min",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--keyframes-per-sec-max",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--keyframe-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--fallback-episode-fps",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--external-geom-name",
        type=str,
        default="trajectory_valid.h5",
    )
    parser.add_argument(
        "--external-extr-mode",
        type=str,
        default="w2c",
        choices=["w2c", "c2w"],
    )
    parser.add_argument(
        "--support-grid-ratio",
        type=float,
        default=0.0,
        help=(
            "Shared process-time support_grid_ratio. Defaults to 0.0 so support points do not "
            "confound wrist filter ablation results."
        ),
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="Measured save runs per variant after warmup.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup save runs per variant before measurements.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep all run outputs under output-root/artifacts.",
    )
    parser.add_argument(
        "--run-visual-verification",
        action="store_true",
        help="Export representative verification images for the wrist variants of each episode.",
    )
    return parser.parse_args()


def parse_episode_names(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("episode-names must contain at least one entry")
    return values


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


def aggregate_profile_stats(
    run_records: list[dict[str, Any]],
    *,
    key: str,
) -> dict[str, dict[str, float | None]]:
    stat_keys = sorted(
        {
            stat_key
            for record in run_records
            for stat_key in record.get(key, {}).keys()
        }
    )
    aggregated: dict[str, dict[str, float | None]] = {}
    for stat_key in stat_keys:
        values = [
            float(record[key][stat_key])
            for record in run_records
            if stat_key in record.get(key, {})
        ]
        aggregated[stat_key] = {
            "mean": _mean(values),
            "stdev": _stdev(values),
        }
    return aggregated


def build_save_variant_specs(
    *,
    external_camera_name: str,
    wrist_camera_name: str,
) -> list[dict[str, Any]]:
    return [
        {
            "name": "external_control",
            "camera_name": external_camera_name,
            "traj_filter_profile": "external",
            "traj_filter_ablation_mode": "none",
            "stage_order": EXTERNAL_STAGE_ORDER,
            "recommendation_priority": None,
        },
        {
            "name": "wrist_external",
            "camera_name": wrist_camera_name,
            "traj_filter_profile": "external",
            "traj_filter_ablation_mode": "none",
            "stage_order": EXTERNAL_STAGE_ORDER,
            "recommendation_priority": 0,
        },
        {
            "name": "wrist",
            "camera_name": wrist_camera_name,
            "traj_filter_profile": "wrist",
            "traj_filter_ablation_mode": "none",
            "stage_order": WRIST_STAGE_ORDER,
            "recommendation_priority": 1,
        },
        {
            "name": "wrist_seed_top95",
            "camera_name": wrist_camera_name,
            "traj_filter_profile": "wrist_manipulator_top95",
            "traj_filter_ablation_mode": "wrist_seed_top95",
            "stage_order": WRIST_MANIPULATOR_STAGE_ORDER,
            "recommendation_priority": 2,
        },
        {
            "name": "wrist_manipulator",
            "camera_name": wrist_camera_name,
            "traj_filter_profile": "wrist_manipulator",
            "traj_filter_ablation_mode": "none",
            "stage_order": WRIST_MANIPULATOR_STAGE_ORDER,
            "recommendation_priority": 3,
        },
        {
            "name": "wrist_manipulator_top95",
            "camera_name": wrist_camera_name,
            "traj_filter_profile": "wrist_manipulator_top95",
            "traj_filter_ablation_mode": "none",
            "stage_order": WRIST_MANIPULATOR_STAGE_ORDER,
            "recommendation_priority": 4,
        },
        {
            "name": "wrist_no_query_edge",
            "camera_name": wrist_camera_name,
            "traj_filter_profile": "wrist_manipulator_top95",
            "traj_filter_ablation_mode": "wrist_no_query_edge",
            "stage_order": WRIST_MANIPULATOR_STAGE_ORDER,
            "recommendation_priority": None,
        },
        {
            "name": "wrist_no_manipulator_depth",
            "camera_name": wrist_camera_name,
            "traj_filter_profile": "wrist_manipulator_top95",
            "traj_filter_ablation_mode": "wrist_no_manipulator_depth",
            "stage_order": WRIST_MANIPULATOR_STAGE_ORDER,
            "recommendation_priority": None,
        },
        {
            "name": "wrist_no_manipulator_motion",
            "camera_name": wrist_camera_name,
            "traj_filter_profile": "wrist_manipulator_top95",
            "traj_filter_ablation_mode": "wrist_no_manipulator_motion",
            "stage_order": WRIST_MANIPULATOR_STAGE_ORDER,
            "recommendation_priority": None,
        },
        {
            "name": "wrist_no_manipulator_cluster",
            "camera_name": wrist_camera_name,
            "traj_filter_profile": "wrist_manipulator_top95",
            "traj_filter_ablation_mode": "wrist_no_manipulator_cluster",
            "stage_order": WRIST_MANIPULATOR_STAGE_ORDER,
            "recommendation_priority": None,
        },
    ]


def _compute_stage_drop_counts(
    *,
    dense_query_count: int,
    stage_counts: dict[str, int],
    stage_order: tuple[str, ...],
) -> dict[str, int]:
    prev_count = int(dense_query_count)
    stage_drop_counts: dict[str, int] = {}
    for stage_name in stage_order:
        stage_count = int(stage_counts.get(stage_name, 0))
        stage_drop_counts[stage_name] = max(0, int(prev_count - stage_count))
        prev_count = stage_count
    return stage_drop_counts


def summarize_stage_debug_records(
    records: list[dict[str, Any]],
    *,
    stage_order: tuple[str, ...],
) -> dict[str, Any]:
    if not records:
        return {
            "sample_count": 0,
            "stage_order": list(stage_order),
            "stage_count_means": {},
            "stage_drop_means": {},
            "per_sample": {},
        }

    per_sample: dict[str, dict[str, Any]] = {}
    stage_count_values: dict[str, list[float]] = {stage_name: [] for stage_name in stage_order}
    stage_drop_values: dict[str, list[float]] = {stage_name: [] for stage_name in stage_order}
    dense_counts: list[float] = []
    tracked_counts: list[float] = []
    valid_counts: list[float] = []

    for record in records:
        query_frame_index = int(record["query_frame_index"])
        stage_counts = {
            stage_name: int(record["stage_counts"].get(stage_name, 0))
            for stage_name in stage_order
        }
        stage_drop_counts = _compute_stage_drop_counts(
            dense_query_count=int(record["dense_query_count"]),
            stage_counts=stage_counts,
            stage_order=stage_order,
        )
        dense_counts.append(float(record["dense_query_count"]))
        tracked_counts.append(float(record["tracked_query_count"]))
        valid_counts.append(float(record["valid_track_count"]))
        for stage_name in stage_order:
            stage_count_values[stage_name].append(float(stage_counts[stage_name]))
            stage_drop_values[stage_name].append(float(stage_drop_counts[stage_name]))
        per_sample[str(query_frame_index)] = {
            "query_frame_index": query_frame_index,
            "dense_query_count": int(record["dense_query_count"]),
            "tracked_query_count": int(record["tracked_query_count"]),
            "valid_track_count": int(record["valid_track_count"]),
            "stage_counts": stage_counts,
            "stage_drop_counts": stage_drop_counts,
        }

    return {
        "sample_count": int(len(records)),
        "stage_order": list(stage_order),
        "dense_query_count_mean": _mean(dense_counts),
        "tracked_query_count_mean": _mean(tracked_counts),
        "valid_track_count_mean": _mean(valid_counts),
        "stage_count_means": {
            stage_name: _mean(stage_count_values[stage_name])
            for stage_name in stage_order
        },
        "stage_drop_means": {
            stage_name: _mean(stage_drop_values[stage_name])
            for stage_name in stage_order
        },
        "per_sample": per_sample,
    }


def build_infer_args(
    *,
    args: argparse.Namespace,
    episode_dir: Path,
    camera_name: str,
    query_frame_schedule_path: Path,
    traj_filter_profile: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        video_path=str(episode_dir / "rgb" / camera_name),
        depth_path=str(episode_dir / "depth" / camera_name),
        mask_dir=None,
        checkpoint=str(args.checkpoint),
        depth_pose_method="external",
        external_extr_mode=args.external_extr_mode,
        external_geom_npz=str(episode_dir / args.external_geom_name),
        camera_name=camera_name,
        query_frame_schedule_path=str(query_frame_schedule_path),
        device=args.device,
        num_iters=args.num_iters,
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
        query_prefilter_mode="off",
        query_prefilter_wrist_rank_keep_ratio=0.30,
        support_grid_ratio=float(args.support_grid_ratio),
        collect_profile_stats=True,
        collect_filter_stage_diagnostics=False,
    )


def _sync_cuda_if_needed(torch_module, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(device)


def process_single_camera(
    *,
    infer_module,
    torch_module,
    model_3dtracker,
    episode_dir: Path,
    camera_name: str,
    infer_args: argparse.Namespace,
) -> dict[str, Any]:
    model_depth_pose = infer_module.video_depth_pose_dict[infer_args.depth_pose_method](infer_args)
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    _sync_cuda_if_needed(torch_module, infer_args.device)
    process_start = time.perf_counter()
    result = infer_module.process_single_video(
        str(episode_dir / "rgb" / camera_name),
        str(episode_dir / "depth" / camera_name),
        infer_args,
        model_3dtracker,
        model_depth_pose,
        camera_name,
        str(infer_args.out_dir),
    )
    _sync_cuda_if_needed(torch_module, infer_args.device)
    process_seconds = time.perf_counter() - process_start
    process_profile_stats = {
        key: float(value)
        for key, value in (result.get("profile_stats") or {}).items()
        if value is not None
    }
    del model_depth_pose
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    return {
        "result": result,
        "process_seconds": float(process_seconds),
        "process_profile_stats": process_profile_stats,
    }


def benchmark_save_variant(
    *,
    infer_module,
    torch_module,
    episode_name: str,
    camera_name: str,
    base_infer_args: argparse.Namespace,
    variant_spec: dict[str, Any],
    shared_result: dict[str, Any],
    output_root: Path,
    warmup_runs: int,
    benchmark_runs: int,
    keep_outputs: bool,
) -> dict[str, Any]:
    total_runs = int(warmup_runs) + int(benchmark_runs)
    raw_runs: list[dict[str, Any]] = []
    measured_runs: list[dict[str, Any]] = []
    representative_output_dir: str | None = None
    representative_sample_summaries: dict[str, dict[str, Any]] | None = None
    representative_stage_overview: dict[str, Any] | None = None

    run_root = output_root / "artifacts" / episode_name / variant_spec["name"]
    for run_idx in range(total_runs):
        run_name = f"run_{run_idx:02d}"
        run_output_root = run_root / run_name
        if run_output_root.exists():
            shutil.rmtree(run_output_root)
        run_output_root.mkdir(parents=True, exist_ok=True)

        variant_args = copy.deepcopy(base_infer_args)
        variant_args.traj_filter_profile = variant_spec["traj_filter_profile"]
        variant_args.traj_filter_ablation_mode = variant_spec["traj_filter_ablation_mode"]
        variant_args.collect_profile_stats = True
        variant_args.collect_filter_stage_diagnostics = True

        _sync_cuda_if_needed(torch_module, variant_args.device)
        save_start = time.perf_counter()
        save_result = infer_module.save_structured_data(
            video_name=camera_name,
            output_dir=str(run_output_root),
            video_tensor=shared_result["video_tensor"],
            depths=shared_result["depths"],
            coords=shared_result["coords"],
            visibs=shared_result["visibs"],
            intrinsics=shared_result["intrinsics"],
            extrinsics=shared_result["extrinsics"],
            query_points_per_frame=shared_result["query_points_per_frame"],
            original_filenames=shared_result["original_filenames"],
            query_frame_results=shared_result.get("query_frame_results"),
            future_len=variant_args.future_len,
            grid_size=variant_args.grid_size,
            filter_args=variant_args,
            full_video_tensor=shared_result["full_video_tensor"],
            full_depths=shared_result["full_depths"],
            full_intrinsics=shared_result["full_intrinsics"],
            full_extrinsics=shared_result["full_extrinsics"],
            depth_conf=shared_result["depth_conf"],
            video_source_path=str(Path(base_infer_args.video_path)),
            depth_source_path=str(Path(base_infer_args.depth_path)),
            source_frame_indices=shared_result["source_frame_indices"],
            query_frame_metadata=shared_result.get("query_frame_metadata"),
        )
        _sync_cuda_if_needed(torch_module, variant_args.device)
        save_seconds = time.perf_counter() - save_start

        sample_summaries = collect_sample_summaries(run_output_root, camera_name=camera_name)
        stage_overview = summarize_stage_debug_records(
            save_result.get("sample_debug_records", []),
            stage_order=tuple(variant_spec["stage_order"]),
        )
        save_profile_stats = {
            key: float(value)
            for key, value in save_result.get("save_profile_stats", {}).items()
            if value is not None
        }
        run_record = {
            "run_index": int(run_idx),
            "warmup": bool(run_idx < warmup_runs),
            "save_seconds": float(save_seconds),
            "save_profile_stats": save_profile_stats,
            "output_dir": str(run_output_root),
            "stage_overview": stage_overview,
            "saved_sample_count": int(len(sample_summaries)),
        }
        raw_runs.append(run_record)
        if run_idx >= warmup_runs:
            measured_runs.append(run_record)

        keep_run_outputs = bool(keep_outputs)
        if run_idx >= warmup_runs and representative_output_dir is None:
            representative_output_dir = str(run_output_root)
            representative_sample_summaries = sample_summaries
            representative_stage_overview = stage_overview
            keep_run_outputs = True

        if not keep_run_outputs:
            shutil.rmtree(run_output_root)

    representative_sample_summaries = representative_sample_summaries or {}
    representative_stage_overview = representative_stage_overview or summarize_stage_debug_records(
        [],
        stage_order=tuple(variant_spec["stage_order"]),
    )
    save_values = [run["save_seconds"] for run in measured_runs]
    return {
        "episode_name": episode_name,
        "camera_name": camera_name,
        "variant_name": variant_spec["name"],
        "traj_filter_profile": variant_spec["traj_filter_profile"],
        "traj_filter_ablation_mode": variant_spec["traj_filter_ablation_mode"],
        "stage_order": list(variant_spec["stage_order"]),
        "recommendation_priority": variant_spec["recommendation_priority"],
        "raw_runs": raw_runs,
        "measured_runs": measured_runs,
        "aggregates": {
            "save_seconds_mean": _mean(save_values),
            "save_seconds_stdev": _stdev(save_values),
        },
        "save_profile_aggregates": aggregate_profile_stats(measured_runs, key="save_profile_stats"),
        "representative_output_dir": representative_output_dir,
        "representative_sample_overview": summarize_case_samples(representative_sample_summaries),
        "representative_sample_summaries": representative_sample_summaries,
        "representative_stage_overview": representative_stage_overview,
    }


def build_pairwise_case_results(
    *,
    case_results: list[dict[str, Any]],
    wrist_camera_name: str,
) -> list[dict[str, Any]]:
    summary_by_key = {
        (case["episode_name"], case["variant_name"]): case
        for case in case_results
    }
    pairwise_results: list[dict[str, Any]] = []
    wrist_variant_names = [
        case["variant_name"]
        for case in case_results
        if case["camera_name"] == wrist_camera_name
    ]
    wrist_variant_names = sorted(set(wrist_variant_names))
    episode_names = sorted({case["episode_name"] for case in case_results})
    for episode_name in episode_names:
        baseline_key = (episode_name, "wrist_manipulator_top95")
        external_key = (episode_name, "external_control")
        if baseline_key not in summary_by_key or external_key not in summary_by_key:
            continue
        baseline_case = summary_by_key[baseline_key]
        external_case = summary_by_key[external_key]
        for variant_name in wrist_variant_names:
            if variant_name == "wrist_manipulator_top95":
                continue
            case_key = (episode_name, variant_name)
            if case_key not in summary_by_key:
                continue
            case = summary_by_key[case_key]
            sample_diff = compare_camera_outputs(
                Path(baseline_case["representative_output_dir"]),
                Path(case["representative_output_dir"]),
                camera_name=wrist_camera_name,
            )
            pairwise_results.append(
                {
                    "episode_name": episode_name,
                    "variant_name": variant_name,
                    "camera_name": wrist_camera_name,
                    "recommendation_priority": case["recommendation_priority"],
                    "save_seconds_mean": case["aggregates"]["save_seconds_mean"],
                    "save_gap_vs_baseline": (
                        None
                        if case["aggregates"]["save_seconds_mean"] is None
                        or baseline_case["aggregates"]["save_seconds_mean"] is None
                        else float(
                            case["aggregates"]["save_seconds_mean"]
                            - baseline_case["aggregates"]["save_seconds_mean"]
                        )
                    ),
                    "save_gap_vs_external": (
                        None
                        if case["aggregates"]["save_seconds_mean"] is None
                        or external_case["aggregates"]["save_seconds_mean"] is None
                        else float(
                            case["aggregates"]["save_seconds_mean"]
                            - external_case["aggregates"]["save_seconds_mean"]
                        )
                    ),
                    "baseline_save_seconds_mean": baseline_case["aggregates"]["save_seconds_mean"],
                    "external_save_seconds_mean": external_case["aggregates"]["save_seconds_mean"],
                    "variant_sample_overview": case["representative_sample_overview"],
                    "sample_diff": sample_diff,
                }
            )
    return pairwise_results


def aggregate_variant_case_rows(case_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in case_results:
        grouped.setdefault(case["variant_name"], []).append(case)

    rows: list[dict[str, Any]] = []
    for variant_name, variant_cases in sorted(grouped.items()):
        save_values = [
            float(case["aggregates"]["save_seconds_mean"])
            for case in variant_cases
            if case["aggregates"]["save_seconds_mean"] is not None
        ]
        dense_values = [
            float(case["representative_sample_overview"]["dense_query_count_mean"])
            for case in variant_cases
            if case["representative_sample_overview"]["dense_query_count_mean"] is not None
        ]
        tracked_values = [
            float(case["representative_sample_overview"]["tracked_query_count_mean"])
            for case in variant_cases
            if case["representative_sample_overview"]["tracked_query_count_mean"] is not None
        ]
        valid_values = [
            float(case["representative_sample_overview"]["valid_track_count_mean"])
            for case in variant_cases
            if case["representative_sample_overview"]["valid_track_count_mean"] is not None
        ]
        save_profile_stats = aggregate_profile_stats(
            [
                {
                    "save_profile_stats": {
                        stat_key: case["save_profile_aggregates"][stat_key]["mean"]
                        for stat_key in case["save_profile_aggregates"]
                        if case["save_profile_aggregates"][stat_key]["mean"] is not None
                    }
                }
                for case in variant_cases
            ],
            key="save_profile_stats",
        )
        first_stage_order = tuple(variant_cases[0]["representative_stage_overview"]["stage_order"])
        stage_count_means: dict[str, float | None] = {}
        stage_drop_means: dict[str, float | None] = {}
        for stage_name in first_stage_order:
            count_values = [
                float(case["representative_stage_overview"]["stage_count_means"][stage_name])
                for case in variant_cases
                if case["representative_stage_overview"]["stage_count_means"].get(stage_name) is not None
            ]
            drop_values = [
                float(case["representative_stage_overview"]["stage_drop_means"][stage_name])
                for case in variant_cases
                if case["representative_stage_overview"]["stage_drop_means"].get(stage_name) is not None
            ]
            stage_count_means[stage_name] = _mean(count_values)
            stage_drop_means[stage_name] = _mean(drop_values)
        rows.append(
            {
                "variant_name": variant_name,
                "camera_name": variant_cases[0]["camera_name"],
                "episode_count": int(len(variant_cases)),
                "recommendation_priority": variant_cases[0]["recommendation_priority"],
                "save_seconds_mean": _mean(save_values),
                "save_seconds_stdev": _stdev(save_values),
                "dense_query_count_mean": _mean(dense_values),
                "tracked_query_count_mean": _mean(tracked_values),
                "valid_track_count_mean": _mean(valid_values),
                "save_profile_aggregates": save_profile_stats,
                "stage_order": list(first_stage_order),
                "stage_count_means": stage_count_means,
                "stage_drop_means": stage_drop_means,
            }
        )
    return rows


def aggregate_pairwise_results(
    *,
    pairwise_results: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    variant_row_by_name = {row["variant_name"]: row for row in variant_rows}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in pairwise_results:
        grouped.setdefault(item["variant_name"], []).append(item)

    aggregated: list[dict[str, Any]] = []
    for variant_name, items in sorted(grouped.items()):
        diff_aggregates = [item["sample_diff"]["aggregates"] for item in items]
        save_gap_vs_baseline_values = [
            float(item["save_gap_vs_baseline"])
            for item in items
            if item["save_gap_vs_baseline"] is not None
        ]
        save_gap_vs_external_values = [
            float(item["save_gap_vs_external"])
            for item in items
            if item["save_gap_vs_external"] is not None
        ]
        jaccard_values = [
            float(diff["traj_valid_mask_jaccard_mean"])
            for diff in diff_aggregates
            if diff["traj_valid_mask_jaccard_mean"] is not None
        ]
        valid_delta_values = [
            float(diff["valid_track_count_delta_mean"])
            for diff in diff_aggregates
            if diff["valid_track_count_delta_mean"] is not None
        ]
        aggregated.append(
            {
                "variant_name": variant_name,
                "camera_name": items[0]["camera_name"],
                "episode_count": int(len(items)),
                "recommendation_priority": items[0]["recommendation_priority"],
                "save_seconds_mean": variant_row_by_name[variant_name]["save_seconds_mean"],
                "save_gap_vs_baseline_mean": _mean(save_gap_vs_baseline_values),
                "save_gap_vs_external_mean": _mean(save_gap_vs_external_values),
                "traj_valid_mask_jaccard_mean": _mean(jaccard_values),
                "valid_track_count_delta_mean": _mean(valid_delta_values),
                "dense_query_count_mean": variant_row_by_name[variant_name]["dense_query_count_mean"],
            }
        )
    return aggregated


def choose_recommended_variant(
    *,
    pairwise_aggregates: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    variant_row_by_name = {row["variant_name"]: row for row in variant_rows}
    external_control_row = variant_row_by_name.get("external_control")
    if external_control_row is None or external_control_row["save_seconds_mean"] is None:
        return None
    external_save_limit = float(external_control_row["save_seconds_mean"]) + SAVE_ALIGNMENT_TOLERANCE_SECONDS
    pairwise_by_name = {item["variant_name"]: item for item in pairwise_aggregates}

    for variant_name in RECOMMENDATION_PRIORITY:
        item = pairwise_by_name.get(variant_name)
        if item is None:
            continue
        save_seconds_mean = item["save_seconds_mean"]
        jaccard_mean = item["traj_valid_mask_jaccard_mean"]
        valid_delta_mean = item["valid_track_count_delta_mean"]
        dense_query_count_mean = item["dense_query_count_mean"]
        if (
            save_seconds_mean is None
            or jaccard_mean is None
            or valid_delta_mean is None
            or dense_query_count_mean is None
        ):
            continue
        valid_delta_limit = VALID_DELTA_RATIO_THRESHOLD * float(dense_query_count_mean)
        if (
            float(save_seconds_mean) <= external_save_limit
            and float(jaccard_mean) >= MASK_JACCARD_THRESHOLD
            and abs(float(valid_delta_mean)) <= valid_delta_limit
        ):
            return {
                "variant_name": variant_name,
                "save_seconds_mean": float(save_seconds_mean),
                "external_save_limit": float(external_save_limit),
                "traj_valid_mask_jaccard_mean": float(jaccard_mean),
                "valid_track_count_delta_mean": float(valid_delta_mean),
                "valid_track_count_delta_limit": float(valid_delta_limit),
            }
    return None


def run_visual_verification(
    *,
    case_results: list[dict[str, Any]],
    output_root: Path,
    wrist_camera_name: str,
) -> list[dict[str, Any]]:
    verify_script_path = CURRENT_REPO_ROOT / "scripts" / "visualization" / "verify_episode_trajectory_outputs.py"
    summary_by_key = {
        (case["episode_name"], case["variant_name"]): case
        for case in case_results
        if case["camera_name"] == wrist_camera_name
    }
    wrist_variant_names = sorted(
        {
            case["variant_name"]
            for case in case_results
            if case["camera_name"] == wrist_camera_name
        }
    )
    verification_results: list[dict[str, Any]] = []
    for episode_name in sorted({case["episode_name"] for case in case_results}):
        variant_query_frame_sets: list[set[int]] = []
        available_variant_names: list[str] = []
        for variant_name in wrist_variant_names:
            case = summary_by_key.get((episode_name, variant_name))
            if case is None or case.get("representative_output_dir") is None:
                continue
            samples = case["representative_sample_summaries"]
            variant_query_frame_sets.append({int(item) for item in samples.keys()})
            available_variant_names.append(variant_name)
        if not variant_query_frame_sets:
            continue
        common_query_frames = sorted(set.intersection(*variant_query_frame_sets))
        if not common_query_frames:
            verification_results.append(
                {
                    "episode_name": episode_name,
                    "camera_name": wrist_camera_name,
                    "query_frame": None,
                    "variants": [],
                }
            )
            continue
        query_frame = int(common_query_frames[0])
        variant_outputs: list[dict[str, Any]] = []
        for variant_name in available_variant_names:
            case = summary_by_key[(episode_name, variant_name)]
            representative_output_dir = Path(case["representative_output_dir"])
            variant_output_dir = output_root / "visual_verification" / episode_name / variant_name
            cmd = [
                sys.executable,
                str(verify_script_path),
                "--episode_dir",
                str(representative_output_dir),
                "--trajectory_dirname",
                ".",
                "--camera_names",
                wrist_camera_name,
                "--query_frames",
                str(query_frame),
                "--output_dir",
                str(variant_output_dir),
            ]
            subprocess.run(cmd, check=True)
            variant_outputs.append(
                {
                    "variant_name": variant_name,
                    "output_dir": str(variant_output_dir),
                    "summary_path": str(variant_output_dir / "summary.json"),
                }
            )
        verification_results.append(
            {
                "episode_name": episode_name,
                "camera_name": wrist_camera_name,
                "query_frame": query_frame,
                "variants": variant_outputs,
            }
        )
    return verification_results


def write_summary_markdown(summary: dict[str, Any], summary_path: Path) -> None:
    lines = [
        "# Wrist Filter Ablation Benchmark Summary",
        "",
        f"- Base path: `{summary['base_path']}`",
        f"- Episodes: `{','.join(summary['episode_names'])}`",
        f"- External camera: `{summary['external_camera_name']}`",
        f"- Wrist camera: `{summary['wrist_camera_name']}`",
        f"- Device: `{summary['device']}`",
        f"- Warmup runs: `{summary['warmup_runs']}`",
        f"- Measured runs: `{summary['benchmark_runs']}`",
        "",
        "## Save Timing by Episode",
        "",
        "| Episode | Variant | Camera | Save (s) | Filter Eval (s) | Sample Write (s) | Prepare Bundles (s) | High Volatility (s) | Scene Meta (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["case_results"]:
        profile_stats = case["save_profile_aggregates"]
        lines.append(
            "| {episode} | {variant} | {camera} | {save} | {filter_eval} | {sample_write} | {prepare_bundles} | {high_volatility} | {scene_meta} |".format(
                episode=case["episode_name"],
                variant=case["variant_name"],
                camera=case["camera_name"],
                save=_format_float(case["aggregates"]["save_seconds_mean"]),
                filter_eval=_format_float(profile_stats.get("filter_eval_seconds", {}).get("mean")),
                sample_write=_format_float(profile_stats.get("sample_write_seconds", {}).get("mean")),
                prepare_bundles=_format_float(profile_stats.get("prepare_bundles_seconds", {}).get("mean")),
                high_volatility=_format_float(profile_stats.get("high_volatility_mask_seconds", {}).get("mean")),
                scene_meta=_format_float(profile_stats.get("scene_meta_write_seconds", {}).get("mean")),
            )
        )

    lines.extend(
        [
            "",
            "## Aggregated Variants",
            "",
            "| Variant | Camera | Save (s) | Dense | Tracked | Valid |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["variant_rows"]:
        lines.append(
            "| {variant} | {camera} | {save} | {dense} | {tracked} | {valid} |".format(
                variant=row["variant_name"],
                camera=row["camera_name"],
                save=_format_float(row["save_seconds_mean"]),
                dense=_format_float(row["dense_query_count_mean"], digits=1),
                tracked=_format_float(row["tracked_query_count_mean"], digits=1),
                valid=_format_float(row["valid_track_count_mean"], digits=1),
            )
        )

    lines.extend(
        [
            "",
            "## Wrist vs Baseline",
            "",
            "| Variant | Save Gap vs Baseline (s) | Save Gap vs External (s) | Mask Jaccard | Valid Delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["pairwise_aggregates"]:
        lines.append(
            "| {variant} | {save_gap_baseline} | {save_gap_external} | {jaccard} | {valid_delta} |".format(
                variant=item["variant_name"],
                save_gap_baseline=_format_float(item["save_gap_vs_baseline_mean"]),
                save_gap_external=_format_float(item["save_gap_vs_external_mean"]),
                jaccard=_format_float(item["traj_valid_mask_jaccard_mean"]),
                valid_delta=_format_float(item["valid_track_count_delta_mean"], digits=1),
            )
        )

    lines.extend(["", "## Stage Drops", ""])
    for row in summary["variant_rows"]:
        stage_bits = ", ".join(
            "{label}={value}".format(
                label=STAGE_LABELS.get(stage_name, stage_name),
                value=_format_float(row["stage_drop_means"].get(stage_name), digits=1),
            )
            for stage_name in row["stage_order"]
        )
        lines.append(f"- `{row['variant_name']}`: {stage_bits}")

    recommendation = summary.get("recommended_variant")
    lines.extend(["", "## Recommendation", ""])
    if recommendation is None:
        lines.append("- No wrist variant satisfied the configured timing and quality thresholds.")
    else:
        lines.append(
            "- Recommended `{variant}`: save={save}s, external_limit={limit}s, jaccard={jaccard}, valid_delta={delta}/{delta_limit}".format(
                variant=recommendation["variant_name"],
                save=_format_float(recommendation["save_seconds_mean"]),
                limit=_format_float(recommendation["external_save_limit"]),
                jaccard=_format_float(recommendation["traj_valid_mask_jaccard_mean"]),
                delta=_format_float(recommendation["valid_track_count_delta_mean"], digits=1),
                delta_limit=_format_float(recommendation["valid_track_count_delta_limit"], digits=1),
            )
        )

    if summary["visual_verification"]:
        lines.extend(["", "## Verification", ""])
        for item in summary["visual_verification"]:
            if item["query_frame"] is None:
                lines.append(f"- `{item['episode_name']}`: no common wrist query frame across variants")
                continue
            lines.append(f"- `{item['episode_name']}`: query frame `{item['query_frame']}`")
            for variant in item["variants"]:
                lines.append(f"  {variant['variant_name']}: `{variant['output_dir']}`")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_orchestrator(args: argparse.Namespace) -> dict[str, Any]:
    import importlib

    infer_module = importlib.import_module("scripts.batch_inference.infer")
    torch_module = importlib.import_module("torch")

    args.base_path = args.base_path.resolve()
    args.output_root = args.output_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    episode_names = parse_episode_names(args.episode_names)

    variant_specs = build_save_variant_specs(
        external_camera_name=args.external_camera_name,
        wrist_camera_name=args.wrist_camera_name,
    )
    external_variant_specs = [spec for spec in variant_specs if spec["camera_name"] == args.external_camera_name]
    wrist_variant_specs = [spec for spec in variant_specs if spec["camera_name"] == args.wrist_camera_name]

    model_3dtracker = infer_module.load_model(str(args.checkpoint)).to(args.device)
    case_results: list[dict[str, Any]] = []
    process_results: list[dict[str, Any]] = []
    try:
        for episode_name in episode_names:
            episode_dir = args.base_path / episode_name
            schedule_path = ensure_query_frame_schedule(
                episode_dir=episode_dir,
                camera_names=[args.external_camera_name, args.wrist_camera_name],
                external_geom_name=args.external_geom_name,
                fps=args.fps,
                max_num_frames=args.max_num_frames,
                keyframes_per_sec_min=args.keyframes_per_sec_min,
                keyframes_per_sec_max=args.keyframes_per_sec_max,
                keyframe_seed=args.keyframe_seed,
                fallback_episode_fps=args.fallback_episode_fps,
                output_root=args.output_root / "query_schedules" / episode_name,
            )

            external_args = build_infer_args(
                args=args,
                episode_dir=episode_dir,
                camera_name=args.external_camera_name,
                query_frame_schedule_path=schedule_path,
                traj_filter_profile="external",
            )
            external_shared = process_single_camera(
                infer_module=infer_module,
                torch_module=torch_module,
                model_3dtracker=model_3dtracker,
                episode_dir=episode_dir,
                camera_name=args.external_camera_name,
                infer_args=external_args,
            )
            process_results.append(
                {
                    "episode_name": episode_name,
                    "camera_name": args.external_camera_name,
                    "process_seconds": external_shared["process_seconds"],
                    "process_profile_stats": external_shared["process_profile_stats"],
                }
            )
            for variant_spec in external_variant_specs:
                case_result = benchmark_save_variant(
                    infer_module=infer_module,
                    torch_module=torch_module,
                    episode_name=episode_name,
                    camera_name=args.external_camera_name,
                    base_infer_args=external_args,
                    variant_spec=variant_spec,
                    shared_result=external_shared["result"],
                    output_root=args.output_root,
                    warmup_runs=args.warmup_runs,
                    benchmark_runs=args.benchmark_runs,
                    keep_outputs=args.keep_outputs,
                )
                case_results.append(case_result)
            del external_shared
            gc.collect()
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()

            wrist_args = build_infer_args(
                args=args,
                episode_dir=episode_dir,
                camera_name=args.wrist_camera_name,
                query_frame_schedule_path=schedule_path,
                traj_filter_profile="wrist_manipulator_top95",
            )
            wrist_shared = process_single_camera(
                infer_module=infer_module,
                torch_module=torch_module,
                model_3dtracker=model_3dtracker,
                episode_dir=episode_dir,
                camera_name=args.wrist_camera_name,
                infer_args=wrist_args,
            )
            process_results.append(
                {
                    "episode_name": episode_name,
                    "camera_name": args.wrist_camera_name,
                    "process_seconds": wrist_shared["process_seconds"],
                    "process_profile_stats": wrist_shared["process_profile_stats"],
                }
            )
            for variant_spec in wrist_variant_specs:
                case_result = benchmark_save_variant(
                    infer_module=infer_module,
                    torch_module=torch_module,
                    episode_name=episode_name,
                    camera_name=args.wrist_camera_name,
                    base_infer_args=wrist_args,
                    variant_spec=variant_spec,
                    shared_result=wrist_shared["result"],
                    output_root=args.output_root,
                    warmup_runs=args.warmup_runs,
                    benchmark_runs=args.benchmark_runs,
                    keep_outputs=args.keep_outputs,
                )
                case_results.append(case_result)
            del wrist_shared
            gc.collect()
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
    finally:
        del model_3dtracker
        gc.collect()
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

    variant_rows = aggregate_variant_case_rows(case_results)
    pairwise_case_results = build_pairwise_case_results(
        case_results=case_results,
        wrist_camera_name=args.wrist_camera_name,
    )
    pairwise_aggregates = aggregate_pairwise_results(
        pairwise_results=pairwise_case_results,
        variant_rows=variant_rows,
    )
    recommended_variant = choose_recommended_variant(
        pairwise_aggregates=pairwise_aggregates,
        variant_rows=variant_rows,
    )
    visual_verification: list[dict[str, Any]] = []
    if args.run_visual_verification:
        visual_verification = run_visual_verification(
            case_results=case_results,
            output_root=args.output_root,
            wrist_camera_name=args.wrist_camera_name,
        )

    summary = {
        "base_path": str(args.base_path),
        "episode_names": episode_names,
        "external_camera_name": args.external_camera_name,
        "wrist_camera_name": args.wrist_camera_name,
        "device": args.device,
        "benchmark_runs": int(args.benchmark_runs),
        "warmup_runs": int(args.warmup_runs),
        "keyframes_per_sec_min": int(args.keyframes_per_sec_min),
        "keyframes_per_sec_max": int(args.keyframes_per_sec_max),
        "support_grid_ratio": float(args.support_grid_ratio),
        "case_results": case_results,
        "process_results": process_results,
        "variant_rows": variant_rows,
        "pairwise_case_results": pairwise_case_results,
        "pairwise_aggregates": pairwise_aggregates,
        "recommended_variant": recommended_variant,
        "visual_verification": visual_verification,
    }
    summary_json_path = args.output_root / RESULT_JSON_BASENAME
    summary_md_path = args.output_root / SUMMARY_MD_BASENAME
    _atomic_write_json(summary_json_path, summary)
    write_summary_markdown(summary, summary_md_path)
    print(f"JSON summary: {summary_json_path}")
    print(f"Markdown summary: {summary_md_path}")
    return summary


def main() -> None:
    args = parse_args()
    run_orchestrator(args)


if __name__ == "__main__":
    main()
