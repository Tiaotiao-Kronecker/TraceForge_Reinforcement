#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


CURRENT_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(CURRENT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPO_ROOT))

from scripts.data_analysis.benchmark_depth_volatility_optimization import (  # noqa: E402
    ensure_query_frame_schedule,
    parse_camera_names,
    resolve_traj_filter_profile,
)
from scripts.data_analysis.benchmark_num_iters_manifest import (  # noqa: E402
    build_aggregate_case_results,
    build_volatility_summary,
    flatten_animation_commands,
    load_benchmark_manifest,
    write_manifest_summary_markdown,
)
from scripts.data_analysis.benchmark_num_iters_sweep import (  # noqa: E402
    RESULT_JSON_BASENAME,
    SUMMARY_MD_BASENAME,
    _atomic_write_json,
    benchmark_variant_case,
    build_animation_commands,
    build_pairwise_comparisons,
    build_variant_specs,
    format_num_iters_variant_name,
    load_benchmark_runtime,
    parse_num_iters_values,
    parse_variant_names,
    release_benchmark_runtime,
    write_num_iters_summary,
)


def parse_args() -> argparse.Namespace:
    default_output_root = (
        CURRENT_REPO_ROOT
        / "data_tmp"
        / "num_iters_manifest_parallel"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    parser = argparse.ArgumentParser(
        description=(
            "Run a manifest-defined num_iters sweep with shared per-episode query-frame "
            "schedules, while parallelizing episode/camera workers across multiple devices."
        )
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--camera-names",
        type=str,
        default=None,
        help=(
            "Comma-separated camera names. Required in orchestrator mode. The same set is "
            "used to derive each episode's shared schedule."
        ),
    )
    parser.add_argument(
        "--num-iters-values",
        type=str,
        default="5,4,3",
        help="Comma-separated num_iters sweep, for example 5,4,3.",
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
        default="external",
        help=(
            "Trajectory filter profile. The maintained external-only default is external "
            "for all cameras; auto is retained as a compatibility alias and currently "
            "resolves to external."
        ),
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
        help="Fallback single device used when --devices is omitted.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="Optional comma-separated worker device pool, for example cuda:0,cuda:1,cuda:2.",
    )
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max-num-frames", type=int, default=512)
    parser.add_argument("--future-len", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument(
        "--support-grid-ratio",
        type=float,
        default=0.8,
        help="Shared support_grid_ratio for all num_iters variants.",
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
    parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=5.0,
        help="How often the orchestrator polls background workers.",
    )
    parser.add_argument("--episode-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--camera-name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--query-frame-schedule-path", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_device_pool(raw_devices: str | None, *, fallback_device: str) -> list[str]:
    source = raw_devices if raw_devices is not None else fallback_device
    values = [item.strip() for item in source.split(",") if item.strip()]
    if not values:
        raise ValueError("device pool must contain at least one device")
    return values


def _resolve_common_args(args: argparse.Namespace) -> argparse.Namespace:
    resolved = argparse.Namespace(**vars(args))
    if resolved.manifest is not None:
        resolved.manifest = Path(resolved.manifest).resolve()
    if resolved.episode_dir is not None:
        resolved.episode_dir = Path(resolved.episode_dir).resolve()
    if resolved.query_frame_schedule_path is not None:
        resolved.query_frame_schedule_path = Path(resolved.query_frame_schedule_path).resolve()
    resolved.output_root = Path(resolved.output_root).resolve()
    resolved.checkpoint = Path(resolved.checkpoint).resolve()
    resolved.output_root.mkdir(parents=True, exist_ok=True)
    return resolved


def _build_worker_summary(
    *,
    args: argparse.Namespace,
    schedule_path: Path,
    camera_name: str,
) -> dict[str, Any]:
    num_iters_values = parse_num_iters_values(args.num_iters_values)
    selected_variant_names = parse_variant_names(args.variants)
    variant_specs = build_variant_specs(
        num_iters_values=num_iters_values,
        baseline_num_iters=int(args.baseline_num_iters),
        support_grid_ratio=float(args.support_grid_ratio),
        selected_variant_names=selected_variant_names or None,
    )
    baseline_variant_name = format_num_iters_variant_name(int(args.baseline_num_iters))
    traj_filter_profile = resolve_traj_filter_profile(camera_name, args.traj_filter_profile)

    runtime = load_benchmark_runtime(
        checkpoint=args.checkpoint,
        device=args.device,
    )
    try:
        case_results = [
            benchmark_variant_case(
                infer_module=runtime["infer_module"],
                torch_module=runtime["torch_module"],
                model_3dtracker=runtime["model_3dtracker"],
                args=args,
                camera_name=camera_name,
                traj_filter_profile=traj_filter_profile,
                query_frame_schedule_path=schedule_path,
                variant_spec=variant_spec,
            )
            for variant_spec in variant_specs
        ]
    finally:
        release_benchmark_runtime(runtime)

    summary_by_key = {
        (case["variant_name"], case["camera_name"]): case
        for case in case_results
    }
    pairwise_comparisons = build_pairwise_comparisons(
        summary_by_key=summary_by_key,
        variant_specs=variant_specs,
        camera_names=[camera_name],
        baseline_variant_name=baseline_variant_name,
    )
    animation_commands = build_animation_commands(
        summary_by_key=summary_by_key,
        variant_specs=variant_specs,
        camera_names=[camera_name],
    )
    return {
        "episode_dir": str(args.episode_dir),
        "camera_names": [camera_name],
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
        "visual_verification": [],
        "animation_commands": animation_commands,
    }


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.episode_dir is None:
        raise ValueError("--episode-dir is required in worker mode")
    if args.camera_name is None:
        raise ValueError("--camera-name is required in worker mode")
    if args.query_frame_schedule_path is None:
        raise ValueError("--query-frame-schedule-path is required in worker mode")

    args = _resolve_common_args(args)
    summary = _build_worker_summary(
        args=args,
        schedule_path=args.query_frame_schedule_path,
        camera_name=args.camera_name,
    )
    summary_json_path, summary_md_path = write_num_iters_summary(
        summary,
        output_root=args.output_root,
    )
    print(
        "{episode} / {camera}: JSON summary {json_path}".format(
            episode=args.episode_dir.name,
            camera=args.camera_name,
            json_path=summary_json_path,
        )
    )
    print(f"Markdown summary: {summary_md_path}")
    return summary


def _build_worker_command(
    *,
    script_path: Path,
    task: dict[str, Any],
    device: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--episode-dir",
        str(task["episode_dir"]),
        "--camera-name",
        str(task["camera_name"]),
        "--query-frame-schedule-path",
        str(task["schedule_path"]),
        "--num-iters-values",
        args.num_iters_values,
        "--baseline-num-iters",
        str(args.baseline_num_iters),
        "--traj-filter-profile",
        args.traj_filter_profile,
        "--checkpoint",
        str(args.checkpoint),
        "--device",
        device,
        "--fps",
        str(args.fps),
        "--max-num-frames",
        str(args.max_num_frames),
        "--future-len",
        str(args.future_len),
        "--grid-size",
        str(args.grid_size),
        "--support-grid-ratio",
        str(args.support_grid_ratio),
        "--filter-level",
        args.filter_level,
        "--keyframes-per-sec-min",
        str(args.keyframes_per_sec_min),
        "--keyframes-per-sec-max",
        str(args.keyframes_per_sec_max),
        "--keyframe-seed",
        str(args.keyframe_seed),
        "--fallback-episode-fps",
        str(args.fallback_episode_fps),
        "--external-geom-name",
        args.external_geom_name,
        "--external-extr-mode",
        args.external_extr_mode,
        "--benchmark-runs",
        str(args.benchmark_runs),
        "--warmup-runs",
        str(args.warmup_runs),
        "--output-root",
        str(task["worker_output_root"]),
    ]
    if args.variants:
        cmd.extend(["--variants", args.variants])
    if args.keep_outputs:
        cmd.append("--keep-outputs")
    return cmd


def _terminate_running_workers(running_workers: list[dict[str, Any]]) -> None:
    for item in running_workers:
        proc = item["proc"]
        if proc.poll() is None:
            proc.terminate()
    for item in running_workers:
        proc = item["proc"]
        if proc.poll() is None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        item["log_handle"].close()


def run_orchestrator(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest is None:
        raise ValueError("--manifest is required")
    if args.benchmark_runs <= 0:
        raise ValueError("--benchmark-runs must be >= 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.keyframes_per_sec_min <= 0 or args.keyframes_per_sec_max <= 0:
        raise ValueError("--keyframes-per-sec-min/max must both be >= 1")
    if args.keyframes_per_sec_min > args.keyframes_per_sec_max:
        raise ValueError("--keyframes-per-sec-min must be <= --keyframes-per-sec-max")
    if args.poll_interval_sec <= 0:
        raise ValueError("--poll-interval-sec must be > 0")

    args = _resolve_common_args(args)
    manifest = load_benchmark_manifest(args.manifest)
    camera_names = parse_camera_names(args.camera_names)
    device_pool = parse_device_pool(args.devices, fallback_device=args.device)
    script_path = Path(__file__).resolve()

    tasks: list[dict[str, Any]] = []
    schedule_paths_by_episode: dict[str, str] = {}
    for episode_name, episode_dir in zip(manifest["episodes"], manifest["episode_dirs"]):
        episode_output_root = args.output_root / "episodes" / episode_name
        episode_output_root.mkdir(parents=True, exist_ok=True)
        schedule_path = ensure_query_frame_schedule(
            episode_dir=episode_dir,
            camera_names=camera_names,
            external_geom_name=args.external_geom_name,
            fps=args.fps,
            max_num_frames=args.max_num_frames,
            keyframes_per_sec_min=args.keyframes_per_sec_min,
            keyframes_per_sec_max=args.keyframes_per_sec_max,
            keyframe_seed=args.keyframe_seed,
            fallback_episode_fps=args.fallback_episode_fps,
            output_root=episode_output_root,
        )
        schedule_paths_by_episode[episode_name] = str(schedule_path)
        for camera_name in camera_names:
            worker_output_root = episode_output_root / camera_name
            tasks.append(
                {
                    "episode_name": episode_name,
                    "episode_dir": episode_dir,
                    "camera_name": camera_name,
                    "schedule_path": schedule_path,
                    "worker_output_root": worker_output_root,
                    "result_json": worker_output_root / RESULT_JSON_BASENAME,
                    "summary_md": worker_output_root / SUMMARY_MD_BASENAME,
                    "log_path": worker_output_root / "worker.log",
                }
            )

    pending_tasks = list(tasks)
    available_devices = list(device_pool)
    running_workers: list[dict[str, Any]] = []

    try:
        while pending_tasks or running_workers:
            while pending_tasks and available_devices:
                task = pending_tasks.pop(0)
                device = available_devices.pop(0)
                task["worker_output_root"].mkdir(parents=True, exist_ok=True)
                log_handle = task["log_path"].open("w", encoding="utf-8")
                cmd = _build_worker_command(
                    script_path=script_path,
                    task=task,
                    device=device,
                    args=args,
                )
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
                running_workers.append(
                    {
                        "task": task,
                        "device": device,
                        "cmd": cmd,
                        "proc": proc,
                        "log_handle": log_handle,
                        "started_at": time.time(),
                    }
                )
                print(
                    "[start] {episode} / {camera} on {device} -> {log_path}".format(
                        episode=task["episode_name"],
                        camera=task["camera_name"],
                        device=device,
                        log_path=task["log_path"],
                    ),
                    flush=True,
                )

            if not running_workers:
                continue

            time.sleep(args.poll_interval_sec)
            next_running: list[dict[str, Any]] = []
            failed_worker: dict[str, Any] | None = None
            for item in running_workers:
                returncode = item["proc"].poll()
                if returncode is None:
                    next_running.append(item)
                    continue
                item["log_handle"].close()
                available_devices.append(item["device"])
                duration = time.time() - item["started_at"]
                if returncode != 0:
                    failed_worker = item
                    break
                print(
                    "[done] {episode} / {camera} on {device} in {seconds:.1f}s".format(
                        episode=item["task"]["episode_name"],
                        camera=item["task"]["camera_name"],
                        device=item["device"],
                        seconds=duration,
                    ),
                    flush=True,
                )

            if failed_worker is not None:
                _terminate_running_workers(next_running)
                raise subprocess.CalledProcessError(
                    failed_worker["proc"].returncode,
                    failed_worker["cmd"],
                    output=(
                        "worker failed for {episode} / {camera}; inspect {log_path}".format(
                            episode=failed_worker["task"]["episode_name"],
                            camera=failed_worker["task"]["camera_name"],
                            log_path=failed_worker["task"]["log_path"],
                        )
                    ),
                )
            running_workers = next_running
    finally:
        if running_workers:
            _terminate_running_workers(running_workers)

    episode_results: list[dict[str, Any]] = []
    for task in tasks:
        if not task["result_json"].is_file():
            raise FileNotFoundError(f"Missing worker result json: {task['result_json']}")
        summary = json.loads(task["result_json"].read_text(encoding="utf-8"))
        episode_results.append(
            {
                "episode_name": task["episode_name"],
                "episode_dir": str(task["episode_dir"]),
                "output_root": str(task["worker_output_root"]),
                "summary_json_path": str(task["result_json"]),
                "summary_md_path": str(task["summary_md"]),
                "summary": summary,
            }
        )

    aggregate_case_results = build_aggregate_case_results(episode_results)
    volatility_summary = build_volatility_summary(episode_results)
    animation_commands = flatten_animation_commands(episode_results)
    num_iters_values = parse_num_iters_values(args.num_iters_values)
    baseline_variant_name = format_num_iters_variant_name(int(args.baseline_num_iters))
    device_summary = (
        device_pool[0]
        if len(device_pool) == 1
        else "parallel({devices})".format(devices=",".join(device_pool))
    )
    summary = {
        "manifest_path": str(manifest["manifest_path"]),
        "dataset_root": str(manifest["dataset_root"]),
        "episodes": manifest["episodes"],
        "camera_names": camera_names,
        "num_iters_values": num_iters_values,
        "baseline_num_iters": int(args.baseline_num_iters),
        "baseline_variant_name": baseline_variant_name,
        "support_grid_ratio": float(args.support_grid_ratio),
        "checkpoint": str(args.checkpoint),
        "device": device_summary,
        "benchmark_runs": int(args.benchmark_runs),
        "warmup_runs": int(args.warmup_runs),
        "current_repo_root": str(CURRENT_REPO_ROOT.resolve()),
        "schedule_paths_by_episode": schedule_paths_by_episode,
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
    return summary


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return
    run_orchestrator(args)


if __name__ == "__main__":
    main()
