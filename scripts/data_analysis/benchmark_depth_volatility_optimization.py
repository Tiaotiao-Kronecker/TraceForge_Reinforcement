#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CURRENT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_REF = "8f9060d"
DEFAULT_CAMERAS = ("varied_camera_1", "varied_camera_3")
QUERY_FRAME_SCHEDULE_VERSION = 1
QUERY_FRAME_SHARED_DIRNAME = "_shared"
RESULT_JSON_BASENAME = "benchmark_results.json"
SUMMARY_MD_BASENAME = "benchmark_summary.md"


def parse_args() -> argparse.Namespace:
    default_output_root = (
        CURRENT_REPO_ROOT
        / "data_tmp"
        / "depth_volatility_benchmarks"
        / time.strftime("%Y%m%d_%H%M%S")
    )

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the depth-volatility optimization on one external camera and one "
            "wrist-like camera, and compare the current repo against a baseline ref."
        )
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=None,
        help="Episode directory, e.g. /data1/yaoxuran/press_one_button_demo_v1/episode_00000",
    )
    parser.add_argument(
        "--camera-names",
        type=str,
        default=",".join(DEFAULT_CAMERAS),
        help="Comma-separated camera names. Defaults to one external and one wrist-like camera.",
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
        "--baseline-ref",
        type=str,
        default=DEFAULT_BASELINE_REF,
        help="Git ref used as the pre-optimization baseline.",
    )
    parser.add_argument(
        "--baseline-repo-root",
        type=Path,
        default=None,
        help="Optional existing baseline repo/worktree root. If omitted, a temporary worktree is created.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CURRENT_REPO_ROOT / "checkpoints" / "tapip3d_final.pth",
        help="Checkpoint used for both baseline and current runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device passed to infer.py, e.g. cuda:1",
    )
    parser.add_argument("--num-iters", type=int, default=6)
    parser.add_argument("--fps", type=int, default=1, help="Load stride passed to infer.py")
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
        help="Shared schedule minimum query frames per second.",
    )
    parser.add_argument(
        "--keyframes-per-sec-max",
        type=int,
        default=3,
        help="Shared schedule maximum query frames per second.",
    )
    parser.add_argument(
        "--keyframe-seed",
        type=int,
        default=0,
        help="Base seed for deterministic shared schedule generation.",
    )
    parser.add_argument(
        "--fallback-episode-fps",
        type=float,
        default=0.0,
        help="Used only if trajectory_valid.h5 does not carry a root attr 'fps'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="Directory used for manifests, result json, and optional kept artifacts.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="Measured runs per repo/camera after warmup.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs per repo/camera before measurements.",
    )
    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep generated TraceForge outputs for each run under output-root/artifacts.",
    )
    parser.add_argument(
        "--external-geom-name",
        type=str,
        default="trajectory_valid.h5",
        help="Per-episode geometry filename.",
    )
    parser.add_argument(
        "--external-extr-mode",
        type=str,
        default="w2c",
        choices=["w2c", "c2w"],
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--query-frame-schedule-path",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def parse_camera_names(camera_names: str) -> list[str]:
    values = [item.strip() for item in camera_names.split(",") if item.strip()]
    if not values:
        raise ValueError("camera_names must contain at least one camera name")
    return values


def resolve_traj_filter_profile(camera_name: str, requested_profile: str) -> str:
    if requested_profile != "auto":
        return requested_profile
    camera_name = camera_name.lower()
    if camera_name.endswith("camera_3") or "wrist" in camera_name or "hand" in camera_name:
        return "wrist_manipulator_top95"
    return "external"


def _count_files(dir_path: Path, suffixes: tuple[str, ...]) -> int:
    if not dir_path.is_dir():
        return 0
    return sum(
        1
        for path in dir_path.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _read_episode_fps(geom_path: Path, fallback_episode_fps: float) -> float:
    if geom_path.suffix.lower() != ".h5":
        raise ValueError(
            f"Shared per-second keyframe sampling requires H5 geometry with root attr 'fps', got: {geom_path}"
        )

    with h5py.File(geom_path, "r") as h5_file:
        fps_attr = h5_file.attrs.get("fps")
    if fps_attr is None:
        if fallback_episode_fps > 0:
            return float(fallback_episode_fps)
        raise ValueError(f"{geom_path} missing root attr 'fps'")
    return float(fps_attr)


def _read_geom_frame_count(geom_path: Path, camera_name: str) -> int:
    if geom_path.suffix.lower() == ".h5":
        with h5py.File(geom_path, "r") as h5_file:
            intr_key_with_suffix = f"observation/camera/intrinsics/{camera_name}_left"
            extr_key_with_suffix = f"observation/camera/extrinsics/{camera_name}_left"
            intr_key_no_suffix = f"observation/camera/intrinsics/{camera_name}"
            extr_key_no_suffix = f"observation/camera/extrinsics/{camera_name}"
            if intr_key_with_suffix in h5_file and extr_key_with_suffix in h5_file:
                intr_count = int(h5_file[intr_key_with_suffix].shape[0])
                extr_count = int(h5_file[extr_key_with_suffix].shape[0])
                return min(intr_count, extr_count)
            if intr_key_no_suffix in h5_file and extr_key_no_suffix in h5_file:
                intr_count = int(h5_file[intr_key_no_suffix].shape[0])
                extr_count = int(h5_file[extr_key_no_suffix].shape[0])
                return min(intr_count, extr_count)
        raise KeyError(
            f"{geom_path} missing intrinsics/extrinsics datasets for camera '{camera_name}'"
        )

    with np.load(geom_path) as data:
        if "intrinsics" not in data or "extrinsics" not in data:
            raise KeyError(
                f"NPZ geometry must contain 'intrinsics' and 'extrinsics': {geom_path}"
            )
        return min(int(data["intrinsics"].shape[0]), int(data["extrinsics"].shape[0]))


def _schedule_spec_hash(spec: dict[str, object]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _derive_episode_schedule_seed(
    *,
    base_seed: int,
    episode_name: str,
    spec_hash: str,
) -> int:
    material = f"{base_seed}:{episode_name}:{spec_hash}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def ensure_query_frame_schedule(
    *,
    episode_dir: Path,
    camera_names: list[str],
    external_geom_name: str,
    fps: int,
    max_num_frames: int,
    keyframes_per_sec_min: int,
    keyframes_per_sec_max: int,
    keyframe_seed: int,
    fallback_episode_fps: float,
    output_root: Path,
) -> Path:
    if str(CURRENT_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(CURRENT_REPO_ROOT))
    from utils.keyframe_schedule_utils import (
        build_candidate_source_frame_indices,
        sample_query_source_indices_per_second,
    )

    geom_path = episode_dir / external_geom_name
    episode_fps = _read_episode_fps(geom_path, fallback_episode_fps)

    camera_raw_frame_counts: dict[str, int] = {}
    for camera_name in camera_names:
        rgb_dir = episode_dir / "rgb" / camera_name
        depth_dir = episode_dir / "depth" / camera_name
        rgb_count = _count_files(rgb_dir, (".png", ".jpg", ".jpeg"))
        depth_count = _count_files(depth_dir, (".npy", ".png"))
        geom_count = _read_geom_frame_count(geom_path, camera_name)
        common_count = min(rgb_count, depth_count, geom_count)
        if common_count <= 0:
            raise ValueError(
                f"Camera '{camera_name}' has no usable shared frames under {episode_dir}"
            )
        camera_raw_frame_counts[camera_name] = common_count

    common_raw_frame_count = min(camera_raw_frame_counts.values())
    candidate_source_frame_indices = build_candidate_source_frame_indices(
        common_raw_frame_count,
        stride=int(fps),
        max_num_frames=max_num_frames,
    )
    if candidate_source_frame_indices.size == 0:
        raise ValueError(
            f"{episode_dir.name}: no candidate frames remain after stride/max_num_frames filtering"
        )

    schedule_spec = {
        "version": QUERY_FRAME_SCHEDULE_VERSION,
        "external_geom_name": external_geom_name,
        "camera_names": list(camera_names),
        "episode_fps": float(episode_fps),
        "keyframes_per_sec_min": int(keyframes_per_sec_min),
        "keyframes_per_sec_max": int(keyframes_per_sec_max),
        "base_seed": int(keyframe_seed),
        "load_stride": int(fps),
        "max_num_frames": int(max_num_frames),
        "common_raw_frame_count": int(common_raw_frame_count),
    }
    spec_hash = _schedule_spec_hash(schedule_spec)
    schedule_dir = output_root / QUERY_FRAME_SHARED_DIRNAME
    schedule_path = schedule_dir / (
        f"query_frame_schedule_v{QUERY_FRAME_SCHEDULE_VERSION}_{spec_hash}.json"
    )
    if schedule_path.is_file():
        return schedule_path

    derived_seed = _derive_episode_schedule_seed(
        base_seed=int(keyframe_seed),
        episode_name=episode_dir.name,
        spec_hash=spec_hash,
    )
    query_frame_source_indices = sample_query_source_indices_per_second(
        candidate_source_frame_indices,
        episode_fps=episode_fps,
        keyframes_per_sec_min=int(keyframes_per_sec_min),
        keyframes_per_sec_max=int(keyframes_per_sec_max),
        seed=derived_seed,
    )
    if query_frame_source_indices.size == 0:
        raise ValueError(f"{episode_dir.name}: sampled zero query frames")

    _atomic_write_json(
        schedule_path,
        {
            **schedule_spec,
            "derived_seed": int(derived_seed),
            "camera_raw_frame_counts": camera_raw_frame_counts,
            "candidate_source_frame_indices": candidate_source_frame_indices.tolist(),
            "query_frame_source_indices": query_frame_source_indices.tolist(),
        },
    )
    return schedule_path


def ensure_baseline_repo_root(
    *,
    current_repo_root: Path,
    baseline_ref: str,
    baseline_repo_root: Path | None,
) -> Path:
    if baseline_repo_root is None:
        baseline_repo_root = Path(tempfile.gettempdir()) / f"traceforge_bench_{baseline_ref}"
    baseline_repo_root = baseline_repo_root.resolve()

    if baseline_repo_root.exists():
        git_dir = baseline_repo_root / ".git"
        if not git_dir.exists():
            raise ValueError(
                f"Existing baseline_repo_root is not a git worktree: {baseline_repo_root}"
            )
        completed = subprocess.run(
            ["git", "-C", str(baseline_repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        head = completed.stdout.strip()
        expected = subprocess.run(
            ["git", "-C", str(current_repo_root), "rev-parse", baseline_ref],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if head != expected:
            raise ValueError(
                f"Existing baseline_repo_root points to {head}, expected {expected}: {baseline_repo_root}"
            )
        return baseline_repo_root

    subprocess.run(
        [
            "git",
            "-C",
            str(current_repo_root),
            "worktree",
            "add",
            "--detach",
            str(baseline_repo_root),
            baseline_ref,
        ],
        check=True,
    )
    return baseline_repo_root


def _sync_cuda_if_needed(torch_module, device: str) -> None:
    if device.startswith("cuda") and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(device)


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _stdev(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    return float(statistics.stdev(values))


def _purge_repo_modules() -> None:
    prefixes = (
        "scripts",
        "utils",
        "datasets",
        "models",
        "checker",
        "third_party",
    )
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[name]


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.repo_root is None:
        raise ValueError("--repo-root is required in worker mode")
    if args.result_json is None:
        raise ValueError("--result-json is required in worker mode")
    if args.episode_dir is None:
        raise ValueError("--episode-dir is required in worker mode")
    if args.camera_name is None:
        raise ValueError("--camera-name is required in worker mode")
    if args.run_label is None:
        raise ValueError("--run-label is required in worker mode")
    if args.query_frame_schedule_path is None:
        raise ValueError("--query-frame-schedule-path is required in worker mode")

    args.episode_dir = args.episode_dir.resolve()
    args.repo_root = args.repo_root.resolve()
    args.result_json = args.result_json.resolve()
    args.output_root = args.output_root.resolve()
    args.query_frame_schedule_path = args.query_frame_schedule_path.resolve()

    mpl_config_dir = args.output_root / "_mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    _purge_repo_modules()
    sys.path.insert(0, str(args.repo_root))

    import importlib

    infer_module = importlib.import_module("scripts.batch_inference.infer")
    torch = importlib.import_module("torch")

    profile = resolve_traj_filter_profile(args.camera_name, args.traj_filter_profile)

    infer_args = argparse.Namespace(
        video_path=str(args.episode_dir / "rgb" / args.camera_name),
        depth_path=str(args.episode_dir / "depth" / args.camera_name),
        mask_dir=None,
        checkpoint=str(args.checkpoint),
        depth_pose_method="external",
        external_extr_mode=args.external_extr_mode,
        external_geom_npz=str(args.episode_dir / args.external_geom_name),
        camera_name=args.camera_name,
        query_frame_schedule_path=str(args.query_frame_schedule_path),
        device=args.device,
        num_iters=args.num_iters,
        fps=args.fps,
        out_dir=str(args.output_root),
        video_name=args.camera_name,
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
        traj_filter_profile=profile,
        min_valid_frames=None,
        visibility_threshold=None,
        min_depth=0.01,
        max_depth=10.0,
        boundary_margin=None,
        depth_change_threshold=None,
    )

    model_depth_pose = infer_module.video_depth_pose_dict[infer_args.depth_pose_method](infer_args)
    model_3dtracker = infer_module.load_model(str(args.checkpoint)).to(args.device)

    raw_runs: list[dict[str, Any]] = []
    measured_runs: list[dict[str, Any]] = []
    total_runs = int(args.warmup_runs) + int(args.benchmark_runs)
    artifacts_root = args.output_root / "artifacts" / args.run_label
    for run_idx in range(total_runs):
        run_name = f"run_{run_idx:02d}"
        run_output_root = artifacts_root / run_name
        if run_output_root.exists():
            shutil.rmtree(run_output_root)
        run_output_root.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _sync_cuda_if_needed(torch, args.device)
        process_start = time.perf_counter()
        result = infer_module.process_single_video(
            str(args.episode_dir / "rgb" / args.camera_name),
            str(args.episode_dir / "depth" / args.camera_name),
            infer_args,
            model_3dtracker,
            model_depth_pose,
            args.camera_name,
            str(run_output_root),
        )
        _sync_cuda_if_needed(torch, args.device)
        process_seconds = time.perf_counter() - process_start

        save_start = time.perf_counter()
        infer_module.save_structured_data(
            video_name=args.camera_name,
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
            video_source_path=str(args.episode_dir / "rgb" / args.camera_name),
            depth_source_path=str(args.episode_dir / "depth" / args.camera_name),
            source_frame_indices=result["source_frame_indices"],
            query_frame_metadata=result.get("query_frame_metadata"),
        )
        _sync_cuda_if_needed(torch, args.device)
        save_seconds = time.perf_counter() - save_start

        query_frame_count = len(result.get("query_frame_results") or {})
        sample_dir = run_output_root / args.camera_name / "samples"
        saved_sample_count = (
            len(list(sample_dir.glob("*.npz")))
            if sample_dir.is_dir()
            else 0
        )
        run_record = {
            "run_index": run_idx,
            "warmup": run_idx < args.warmup_runs,
            "process_seconds": float(process_seconds),
            "save_seconds": float(save_seconds),
            "total_seconds": float(process_seconds + save_seconds),
            "query_frame_count": int(query_frame_count),
            "saved_sample_count": int(saved_sample_count),
            "frame_count": int(result["full_video_tensor"].shape[0]),
            "output_dir": str(run_output_root),
        }
        raw_runs.append(run_record)
        if run_idx >= args.warmup_runs:
            measured_runs.append(run_record)

        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not args.keep_outputs:
            shutil.rmtree(run_output_root)

    del model_3dtracker
    del model_depth_pose
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    process_values = [run["process_seconds"] for run in measured_runs]
    save_values = [run["save_seconds"] for run in measured_runs]
    total_values = [run["total_seconds"] for run in measured_runs]
    summary = {
        "run_label": args.run_label,
        "repo_root": str(args.repo_root),
        "camera_name": args.camera_name,
        "traj_filter_profile": profile,
        "schedule_path": str(args.query_frame_schedule_path),
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
            "query_frame_count_mean": _mean([run["query_frame_count"] for run in measured_runs]),
            "saved_sample_count_mean": _mean([run["saved_sample_count"] for run in measured_runs]),
        },
    }
    _atomic_write_json(args.result_json, summary)
    return summary


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator / denominator)


def _safe_delta_percent(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or baseline == 0:
        return None
    return float((current - baseline) / baseline * 100.0)


def build_summary_payload(
    *,
    args: argparse.Namespace,
    schedule_path: Path,
    baseline_repo_root: Path,
    current_repo_root: Path,
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {
        (case["label"], case["camera_name"]): case
        for case in case_results
    }
    camera_comparisons: list[dict[str, Any]] = []
    for camera_name in parse_camera_names(args.camera_names):
        baseline = by_key[("baseline", camera_name)]
        current = by_key[("current", camera_name)]
        baseline_agg = baseline["aggregates"]
        current_agg = current["aggregates"]
        camera_comparisons.append(
            {
                "camera_name": camera_name,
                "traj_filter_profile": current["traj_filter_profile"],
                "process_speedup_vs_current": _safe_ratio(
                    baseline_agg["process_seconds_mean"],
                    current_agg["process_seconds_mean"],
                ),
                "save_speedup_vs_current": _safe_ratio(
                    baseline_agg["save_seconds_mean"],
                    current_agg["save_seconds_mean"],
                ),
                "total_speedup_vs_current": _safe_ratio(
                    baseline_agg["total_seconds_mean"],
                    current_agg["total_seconds_mean"],
                ),
                "process_delta_percent_vs_baseline": _safe_delta_percent(
                    current_agg["process_seconds_mean"],
                    baseline_agg["process_seconds_mean"],
                ),
                "save_delta_percent_vs_baseline": _safe_delta_percent(
                    current_agg["save_seconds_mean"],
                    baseline_agg["save_seconds_mean"],
                ),
                "total_delta_percent_vs_baseline": _safe_delta_percent(
                    current_agg["total_seconds_mean"],
                    baseline_agg["total_seconds_mean"],
                ),
                "baseline": baseline_agg,
                "current": current_agg,
            }
        )

    return {
        "episode_dir": str(args.episode_dir),
        "camera_names": parse_camera_names(args.camera_names),
        "schedule_path": str(schedule_path),
        "current_repo_root": str(current_repo_root),
        "baseline_repo_root": str(baseline_repo_root),
        "baseline_ref": args.baseline_ref,
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
        "camera_comparisons": camera_comparisons,
    }


def write_summary_markdown(summary: dict[str, Any], summary_path: Path) -> None:
    lines = [
        "# Depth Volatility Benchmark Summary",
        "",
        f"- Episode: `{summary['episode_dir']}`",
        f"- Cameras: `{','.join(summary['camera_names'])}`",
        f"- Baseline ref: `{summary['baseline_ref']}`",
        f"- Current repo: `{summary['current_repo_root']}`",
        f"- Baseline repo: `{summary['baseline_repo_root']}`",
        f"- Device: `{summary['device']}`",
        f"- Schedule: `{summary['schedule_path']}`",
        f"- Measured runs: `{summary['benchmark_runs']}`",
        f"- Warmup runs: `{summary['warmup_runs']}`",
        "",
        "| Camera | Profile | Baseline Save (s) | Current Save (s) | Save Speedup | Baseline Total (s) | Current Total (s) | Total Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for comparison in summary["camera_comparisons"]:
        baseline = comparison["baseline"]
        current = comparison["current"]
        lines.append(
            "| {camera} | {profile} | {baseline_save:.3f} | {current_save:.3f} | {save_speedup:.3f}x | "
            "{baseline_total:.3f} | {current_total:.3f} | {total_speedup:.3f}x |".format(
                camera=comparison["camera_name"],
                profile=comparison["traj_filter_profile"],
                baseline_save=baseline["save_seconds_mean"],
                current_save=current["save_seconds_mean"],
                save_speedup=comparison["save_speedup_vs_current"],
                baseline_total=baseline["total_seconds_mean"],
                current_total=current["total_seconds_mean"],
                total_speedup=comparison["total_speedup_vs_current"],
            )
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summary: dict[str, Any]) -> None:
    print("")
    print("Camera benchmarks")
    for comparison in summary["camera_comparisons"]:
        baseline = comparison["baseline"]
        current = comparison["current"]
        print(
            "  {camera} ({profile}): save {baseline_save:.3f}s -> {current_save:.3f}s "
            "({save_speedup:.3f}x), total {baseline_total:.3f}s -> {current_total:.3f}s "
            "({total_speedup:.3f}x)".format(
                camera=comparison["camera_name"],
                profile=comparison["traj_filter_profile"],
                baseline_save=baseline["save_seconds_mean"],
                current_save=current["save_seconds_mean"],
                save_speedup=comparison["save_speedup_vs_current"],
                baseline_total=baseline["total_seconds_mean"],
                current_total=current["total_seconds_mean"],
                total_speedup=comparison["total_speedup_vs_current"],
            )
        )


def run_orchestrator(args: argparse.Namespace) -> dict[str, Any]:
    if args.episode_dir is None:
        raise ValueError("--episode-dir is required")
    if args.benchmark_runs <= 0:
        raise ValueError("--benchmark-runs must be >= 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if args.keyframes_per_sec_min <= 0 or args.keyframes_per_sec_max <= 0:
        raise ValueError("--keyframes-per-sec-min/max must both be >= 1")
    if args.keyframes_per_sec_min > args.keyframes_per_sec_max:
        raise ValueError("--keyframes-per-sec-min must be <= --keyframes-per-sec-max")

    args.episode_dir = args.episode_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    current_repo_root = CURRENT_REPO_ROOT.resolve()
    baseline_repo_root = ensure_baseline_repo_root(
        current_repo_root=current_repo_root,
        baseline_ref=args.baseline_ref,
        baseline_repo_root=args.baseline_repo_root,
    )
    schedule_path = ensure_query_frame_schedule(
        episode_dir=args.episode_dir,
        camera_names=parse_camera_names(args.camera_names),
        external_geom_name=args.external_geom_name,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        keyframes_per_sec_min=args.keyframes_per_sec_min,
        keyframes_per_sec_max=args.keyframes_per_sec_max,
        keyframe_seed=args.keyframe_seed,
        fallback_episode_fps=args.fallback_episode_fps,
        output_root=args.output_root,
    )

    cases = [
        ("baseline", baseline_repo_root),
        ("current", current_repo_root),
    ]
    case_results: list[dict[str, Any]] = []
    raw_results_dir = args.output_root / "raw_results"
    raw_results_dir.mkdir(parents=True, exist_ok=True)

    for label, repo_root in cases:
        for camera_name in parse_camera_names(args.camera_names):
            run_label = f"{label}_{camera_name}"
            result_json = raw_results_dir / f"{run_label}.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--episode-dir",
                str(args.episode_dir),
                "--camera-name",
                camera_name,
                "--run-label",
                run_label,
                "--traj-filter-profile",
                args.traj_filter_profile,
                "--checkpoint",
                str(args.checkpoint),
                "--device",
                args.device,
                "--num-iters",
                str(args.num_iters),
                "--fps",
                str(args.fps),
                "--max-num-frames",
                str(args.max_num_frames),
                "--future-len",
                str(args.future_len),
                "--grid-size",
                str(args.grid_size),
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
                str(args.output_root),
                "--result-json",
                str(result_json),
                "--repo-root",
                str(repo_root),
                "--query-frame-schedule-path",
                str(schedule_path),
            ]
            if args.keep_outputs:
                cmd.append("--keep-outputs")
            subprocess.run(cmd, check=True)
            worker_result = json.loads(result_json.read_text(encoding="utf-8"))
            case_results.append(
                {
                    "label": label,
                    "camera_name": camera_name,
                    **worker_result,
                }
            )

    summary = build_summary_payload(
        args=args,
        schedule_path=schedule_path,
        baseline_repo_root=baseline_repo_root,
        current_repo_root=current_repo_root,
        case_results=case_results,
    )
    summary_json_path = args.output_root / RESULT_JSON_BASENAME
    summary_md_path = args.output_root / SUMMARY_MD_BASENAME
    _atomic_write_json(summary_json_path, summary)
    write_summary_markdown(summary, summary_md_path)
    print_summary(summary)
    print("")
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
