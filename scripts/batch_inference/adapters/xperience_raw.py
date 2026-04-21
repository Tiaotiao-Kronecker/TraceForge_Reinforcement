from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import queue
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from loguru import logger


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.batch_inference.batch_infer_sim_file_layout import (
    BatchTelemetryWriter,
    GpuMemoryInfo,
    HardwareTelemetrySampler,
    WorkerProcessResult,
    WorkerSlot,
    build_worker_slots,
    collect_gpu_static_info,
    filter_gpu_ids_by_free_memory,
    is_retryable_cuda_error,
    mark_task_completed,
    parse_gpu_ids,
    resolve_telemetry_gpu_ids,
    safe_empty_cuda_cache,
    unload_tracker_model,
    wait_for_gpu_recovery,
    warm_up_cuda_linalg,
)
from scripts.batch_inference import infer
from utils.inference_utils import load_model
from utils.keyframe_schedule_utils import (
    build_candidate_source_frame_indices,
    filter_query_local_indices_by_remaining_frames,
    sample_query_source_indices_per_second,
)
from utils.traceforge_artifact_utils import (
    SCENE_STORAGE_ADAPTER_REF,
    SCENE_STORAGE_CACHE,
    V2_LAYOUT,
    is_traceforge_output_complete,
)
from utils.xperience_adapter_utils import (
    XPERIENCE_SUPPORTED_CAMERAS,
    aligned_video_fps,
    build_stereo_left_extrinsics,
    build_stereo_left_intrinsics,
    build_xperience_source_descriptor,
    load_video_frame,
    load_video_frames,
    open_xperience_episode,
    read_caption_main_task,
)


ADAPTER_NAME = "xperience_raw"
DEFAULT_XPERIENCE_EPISODE_GLOB = "*/*"
DEFAULT_XPERIENCE_WINDOW_SIZE = 512
_QUERY_FRAME_SCHEDULE_VERSION = 3
_QUERY_FRAME_SHARED_DIRNAME = "_shared"


@dataclass(frozen=True)
class WindowTask:
    task_index: int
    total_tasks: int
    episode_dir: Path
    video_name: str
    window_start: int
    window_stop: int
    output_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native Xperience batch inference")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root of the raw Xperience dataset tree")
    parser.add_argument(
        "--episode_glob",
        type=str,
        default=DEFAULT_XPERIENCE_EPISODE_GLOB,
        help="Glob under dataset_root used to discover episodes. Expected matches look like <uuid>/epN.",
    )
    parser.add_argument("--episode_limit", type=int, default=None, help="Optional max number of episodes to process")
    parser.add_argument(
        "--camera_name",
        type=str,
        default="stereo_left",
        choices=list(XPERIENCE_SUPPORTED_CAMERAS),
        help="Raw Xperience camera to process. First maintained release only supports stereo_left.",
    )
    parser.add_argument("--start_index", type=int, default=0, help="Raw start frame within each episode")
    parser.add_argument("--stop_index", type=int, default=None, help="Optional raw stop frame within each episode")
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_XPERIENCE_WINDOW_SIZE,
        help="Raw episode window size before stride/cap sampling. <=0 means use one whole-episode window.",
    )
    parser.add_argument(
        "--window_step",
        type=int,
        default=None,
        help="Raw window step. Defaults to window_size for non-overlapping windows.",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
        help="Comma-separated GPU ids used for dynamic multi-GPU window scheduling.",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="Number of resident workers to launch per GPU when --gpu_id is provided.",
    )
    parser.add_argument(
        "--min_free_gpu_mem_gb",
        type=float,
        default=0.0,
        help="Optional free-memory threshold required before a worker starts or resumes on a GPU.",
    )
    parser.add_argument(
        "--gpu_recovery_poll_sec",
        type=float,
        default=30.0,
        help="Polling interval used while waiting for GPU recovery after retryable CUDA failures.",
    )
    parser.add_argument(
        "--telemetry_out_dir",
        type=str,
        default=None,
        help="Optional output directory for batch telemetry. Defaults to --out_dir.",
    )
    parser.add_argument(
        "--hardware_telemetry_interval_sec",
        type=float,
        default=0.0,
        help="Hardware telemetry sampling interval in seconds. <=0 disables background sampling.",
    )
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--fps", type=int, default=1, help="Raw frame stride applied within each window")
    parser.add_argument("--max_num_frames", type=int, default=512)
    parser.add_argument(
        "--frame_drop_rate",
        type=int,
        default=1,
        help=(
            "Legacy fallback query spacing. Current xperience_raw runs generate per-second "
            "query schedules by default, so this is only used if schedule generation is disabled."
        ),
    )
    parser.add_argument(
        "--keyframes_per_sec_min",
        type=int,
        default=2,
        help="Minimum number of query frames sampled per second within each Xperience window.",
    )
    parser.add_argument(
        "--keyframes_per_sec_max",
        type=int,
        default=3,
        help="Maximum number of query frames sampled per second within each Xperience window.",
    )
    parser.add_argument(
        "--keyframe_seed",
        type=int,
        default=0,
        help="Base random seed for deterministic per-window keyframe schedules.",
    )
    parser.add_argument(
        "--fallback_episode_fps",
        type=float,
        default=0.0,
        help="Fallback FPS used only if the raw Xperience episode cannot provide a valid video rate.",
    )
    parser.add_argument("--future_len", type=int, default=16)
    parser.add_argument("--grid_size", type=int, default=80)
    parser.add_argument("--grid_width", type=int, default=None)
    parser.add_argument("--grid_height", type=int, default=None)
    parser.add_argument("--support_grid_ratio", type=float, default=0.0)
    parser.add_argument("--query_candidate_grid_factor", type=float, default=2.0)
    parser.add_argument("--grid_border_trim_left", type=int, default=30)
    parser.add_argument("--grid_border_trim_right", type=int, default=30)
    parser.add_argument("--grid_border_trim_top", type=int, default=30)
    parser.add_argument("--grid_border_trim_bottom", type=int, default=10)
    parser.add_argument("--query_sampler_mode", type=str, default="grid", choices=["auto", "grid", "relevance_first_v1"])
    parser.add_argument("--query_prefilter_mode", type=str, default="off", choices=["off", "profile_aware_static_v1", "external_depth_static_v1"])
    parser.add_argument("--query_prefilter_wrist_rank_keep_ratio", type=float, default=0.30)
    parser.add_argument("--query_visibility_gate_mode", type=str, default="all_future_v1", choices=["off", "all_future_v1"])
    parser.add_argument("--query_visibility_gate_min_border_dist_px", type=float, default=0.0)
    parser.add_argument("--query_visibility_gate_near_depth_exempt_threshold_m", type=float, default=0.0)
    parser.add_argument("--query_fixed_view_depth_gate_mode", type=str, default="first_frame_uvd_v1", choices=["off", "first_frame_uvd_v1"])
    parser.add_argument("--query_fixed_view_depth_gate_uv_threshold_px", type=float, default=1.0)
    parser.add_argument("--query_fixed_view_depth_gate_depth_threshold_m", type=float, default=0.10)
    parser.add_argument("--traj_uvd_gate_mode", type=str, default="delta_uv_depth_v1", choices=["off", "delta_uv_depth_v1"])
    parser.add_argument("--traj_uvd_gate_uv_mean_threshold_px", type=float, default=3.0)
    parser.add_argument("--traj_uvd_gate_depth_std_threshold_m", type=float, default=0.01)
    parser.add_argument("--traj_uvd_gate_max_depth_threshold_m", type=float, default=1.5)
    parser.add_argument("--traj_uvd_gate_near_depth_threshold_m", type=float, default=0.0)
    parser.add_argument("--traj_uvd_gate_near_depth_relaxed_std_threshold_m", type=float, default=0.0)
    parser.add_argument("--traj_uvd_gate_near_depth_exempt_threshold_m", type=float, default=0.0)
    parser.add_argument("--query_depth_stabilization_mode", type=str, default="off", choices=["off", "temporal_median_world_v1"])
    parser.add_argument("--query_depth_stabilization_reproj_tol_px", type=float, default=3.0)
    parser.add_argument("--query_depth_stabilization_min_support", type=int, default=3)
    parser.add_argument("--query_depth_stabilization_min_query_depth_m", type=float, default=0.01)
    parser.add_argument("--query_depth_stabilization_min_border_dist_px", type=float, default=0.0)
    parser.add_argument("--dense_depth_stabilization_mode", type=str, default="off", choices=["off", "temporal_median_reproject_v1"])
    parser.add_argument("--dense_depth_stabilization_radius", type=int, default=2)
    parser.add_argument("--dense_depth_stabilization_min_support", type=int, default=3)
    parser.add_argument("--tracker_precision_mode", type=str, default="fp32", choices=["fp32", "autocast_bf16", "deep_bf16", "bf16"])
    parser.add_argument("--filter_level", type=str, default="none", choices=["none", "basic", "standard", "strict"])
    parser.add_argument("--traj_filter_profile", type=str, default="external")
    parser.add_argument("--traj_filter_ablation_mode", type=str, default="none")
    parser.add_argument("--collect_profile_stats", action="store_true", default=False)
    parser.add_argument("--collect_filter_stage_diagnostics", action="store_true", default=False)
    parser.add_argument("--depth_filter_workers", type=int, default=8)
    parser.add_argument("--depth_filter_blas_threads", type=int, default=1)
    parser.add_argument("--resize_width", type=int, default=None)
    parser.add_argument("--resize_height", type=int, default=None)
    parser.add_argument("--save_visibility", action="store_true", default=False)
    parser.add_argument("--output_layout", type=str, default=V2_LAYOUT, choices=[V2_LAYOUT])
    parser.add_argument(
        "--scene_storage_mode",
        type=str,
        default=SCENE_STORAGE_ADAPTER_REF,
        choices=[SCENE_STORAGE_ADAPTER_REF, SCENE_STORAGE_CACHE],
    )
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args = copy.copy(args)
    if args.fps <= 0:
        raise ValueError("--fps must be >= 1 for native Xperience loading")
    if args.frame_drop_rate <= 0:
        raise ValueError("--frame_drop_rate must be >= 1")
    if args.keyframes_per_sec_min <= 0 or args.keyframes_per_sec_max <= 0:
        raise ValueError("--keyframes_per_sec_min/max must both be >= 1")
    if args.keyframes_per_sec_min > args.keyframes_per_sec_max:
        raise ValueError("--keyframes_per_sec_min must be <= --keyframes_per_sec_max")
    if args.window_step is not None and args.window_step <= 0:
        raise ValueError("--window_step must be > 0 when provided")
    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0")
    if args.stop_index is not None and args.stop_index <= args.start_index:
        raise ValueError("--stop_index must be greater than --start_index")
    if args.depth_filter_workers <= 0:
        raise ValueError("--depth_filter_workers must be >= 1")
    if args.depth_filter_blas_threads < 0:
        raise ValueError("--depth_filter_blas_threads must be >= 0")
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers_per_gpu must be >= 1")
    if args.min_free_gpu_mem_gb < 0.0:
        raise ValueError("--min_free_gpu_mem_gb must be >= 0")
    if args.gpu_recovery_poll_sec <= 0.0:
        raise ValueError("--gpu_recovery_poll_sec must be > 0")
    if args.hardware_telemetry_interval_sec < 0.0:
        raise ValueError("--hardware_telemetry_interval_sec must be >= 0")
    if args.fallback_episode_fps < 0:
        raise ValueError("--fallback_episode_fps must be >= 0")
    if args.query_prefilter_wrist_rank_keep_ratio < 0.0 or args.query_prefilter_wrist_rank_keep_ratio > 1.0:
        raise ValueError("--query_prefilter_wrist_rank_keep_ratio must be within [0, 1]")
    if args.support_grid_ratio < 0.0:
        raise ValueError("--support_grid_ratio must be >= 0")
    args.processing_resize_hw = infer._resolve_processing_resize_hw(args)
    args.query_grid_hw = infer._resolve_query_grid_hw(args)
    args.depth_pose_method = "external"
    args.external_geom_npz = None
    args.external_extr_mode = "w2c"
    args.mask_dir = None
    args.video_name = None
    args.batch_process = False
    args.scan_depth = 0
    args.query_frame_schedule_path = None
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    parsed_args = parser.parse_args(argv)
    try:
        return finalize_args(parsed_args)
    except ValueError as exc:
        parser.error(str(exc))
    raise AssertionError("unreachable")


def _schedule_spec_hash(spec: dict[str, object]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _derive_window_schedule_seed(
    *,
    base_seed: int,
    video_name: str,
    spec_hash: str,
) -> int:
    material = f"{base_seed}:{video_name}:{spec_hash}".encode("utf-8")
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


def _resolve_episode_fps(handle: h5py.File, fallback_episode_fps: float) -> float:
    try:
        episode_fps = float(aligned_video_fps(handle))
    except Exception:
        episode_fps = 0.0
    if np.isfinite(episode_fps) and episode_fps > 0:
        return episode_fps
    if fallback_episode_fps > 0:
        return float(fallback_episode_fps)
    raise ValueError(
        "Failed to resolve a valid episode FPS from raw Xperience metadata; "
        "pass --fallback_episode_fps to override."
    )


def _sample_window_query_source_indices(
    *,
    source_frame_indices: np.ndarray,
    episode_fps: float,
    keyframes_per_sec_min: int,
    keyframes_per_sec_max: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_frame_indices = np.asarray(source_frame_indices, dtype=np.int32).reshape(-1)
    if source_frame_indices.size == 0:
        raise ValueError("source_frame_indices must not be empty")

    candidate_local_indices = np.arange(source_frame_indices.size, dtype=np.int32)
    candidate_local_indices, short_tail_local_indices = (
        filter_query_local_indices_by_remaining_frames(
            candidate_local_indices,
            video_length=int(source_frame_indices.size),
        )
    )
    candidate_source_frame_indices = source_frame_indices[candidate_local_indices]
    short_tail_source_indices = source_frame_indices[short_tail_local_indices]
    if candidate_source_frame_indices.size == 0:
        raise ValueError(
            "No candidate query frames remain after stride/max_num_frames/tail filtering"
        )

    relative_candidate_source_indices = (
        candidate_source_frame_indices - int(candidate_source_frame_indices[0])
    ).astype(np.int32, copy=False)
    query_frame_relative_indices = sample_query_source_indices_per_second(
        relative_candidate_source_indices,
        episode_fps=episode_fps,
        keyframes_per_sec_min=keyframes_per_sec_min,
        keyframes_per_sec_max=keyframes_per_sec_max,
        seed=seed,
    )
    query_frame_source_indices = (
        query_frame_relative_indices + int(candidate_source_frame_indices[0])
    ).astype(np.int32, copy=False)
    return (
        candidate_source_frame_indices,
        short_tail_source_indices.astype(np.int32, copy=False),
        query_frame_source_indices,
    )


def _prepare_query_frame_schedule(
    *,
    out_dir: Path,
    video_name: str,
    source_frame_indices: np.ndarray,
    episode_fps: float,
    args: argparse.Namespace,
    start_index: int,
    stop_index: int,
) -> Path:
    schedule_spec = {
        "version": _QUERY_FRAME_SCHEDULE_VERSION,
        "dataset_adapter": ADAPTER_NAME,
        "camera_name": str(args.camera_name),
        "video_name": str(video_name),
        "episode_fps": float(episode_fps),
        "keyframes_per_sec_min": int(args.keyframes_per_sec_min),
        "keyframes_per_sec_max": int(args.keyframes_per_sec_max),
        "base_seed": int(args.keyframe_seed),
        "load_stride": int(args.fps),
        "max_num_frames": int(args.max_num_frames),
        "window_start": int(start_index),
        "window_stop": int(stop_index),
        "selected_frame_count": int(source_frame_indices.size),
    }
    spec_hash = _schedule_spec_hash(schedule_spec)
    schedule_path = (
        out_dir
        / video_name
        / _QUERY_FRAME_SHARED_DIRNAME
        / f"query_frame_schedule_v{_QUERY_FRAME_SCHEDULE_VERSION}_{spec_hash}.json"
    )
    if schedule_path.is_file():
        return schedule_path

    derived_seed = _derive_window_schedule_seed(
        base_seed=int(args.keyframe_seed),
        video_name=video_name,
        spec_hash=spec_hash,
    )
    (
        candidate_source_frame_indices,
        short_tail_source_indices,
        query_frame_source_indices,
    ) = _sample_window_query_source_indices(
        source_frame_indices=source_frame_indices,
        episode_fps=episode_fps,
        keyframes_per_sec_min=int(args.keyframes_per_sec_min),
        keyframes_per_sec_max=int(args.keyframes_per_sec_max),
        seed=derived_seed,
    )
    _atomic_write_json(
        schedule_path,
        {
            **schedule_spec,
            "derived_seed": int(derived_seed),
            "candidate_source_frame_indices": candidate_source_frame_indices.tolist(),
            "dropped_short_tail_source_indices": short_tail_source_indices.tolist(),
            "query_frame_source_indices": query_frame_source_indices.tolist(),
        },
    )
    logger.info(
        f"{video_name}: prepared per-window query-frame schedule "
        f"({len(query_frame_source_indices)} frames, fps={episode_fps:.3f}, "
        f"kps={args.keyframes_per_sec_min}~{args.keyframes_per_sec_max})"
    )
    return schedule_path


def _find_episode_dirs(
    dataset_root: Path,
    *,
    episode_glob: str,
    episode_limit: int | None,
) -> list[Path]:
    annotation_paths = sorted(dataset_root.glob(f"{episode_glob}/annotation.hdf5"))
    if not annotation_paths:
        annotation_paths = sorted(dataset_root.rglob("annotation.hdf5"))
    episode_dirs = [path.parent.resolve() for path in annotation_paths]
    if episode_limit is not None and episode_limit > 0:
        episode_dirs = episode_dirs[: int(episode_limit)]
    return episode_dirs


def _resolve_frame_count(handle: h5py.File) -> int:
    return int(
        min(
            handle["video/frame_number"].shape[0],
            handle["depth/depth"].shape[0],
            handle["slam/trans_xyz"].shape[0],
            handle["slam/quat_wxyz"].shape[0],
        )
    )


def _build_window_ranges(
    *,
    frame_count: int,
    start_index: int,
    stop_index: int | None,
    window_size: int,
    window_step: int | None,
) -> list[tuple[int, int]]:
    resolved_start = max(0, int(start_index))
    resolved_stop = frame_count if stop_index is None else min(int(stop_index), int(frame_count))
    if resolved_start >= resolved_stop:
        raise ValueError(
            f"Invalid Xperience window range [{resolved_start}, {resolved_stop}) for frame_count={frame_count}"
        )
    if window_size <= 0:
        return [(resolved_start, resolved_stop)]
    step = int(window_step if window_step is not None else window_size)
    ranges: list[tuple[int, int]] = []
    current = resolved_start
    while current < resolved_stop:
        window_stop = min(current + int(window_size), resolved_stop)
        ranges.append((current, window_stop))
        current += step
    return ranges


def _build_video_name(episode_dir: Path, camera_name: str, start_index: int, stop_index: int) -> str:
    return f"{episode_dir.parent.name}__{episode_dir.name}__{camera_name}__{start_index:06d}_{stop_index:06d}"


def _load_scene_bundle(
    episode_dir: Path,
    *,
    dataset_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
    start_index: int,
    stop_index: int,
    video_name: str,
) -> dict[str, object]:
    paths = open_xperience_episode(episode_dir, camera_name=args.camera_name)
    with h5py.File(paths.annotation_path, "r") as handle:
        relative_indices = build_candidate_source_frame_indices(
            stop_index - start_index,
            stride=int(args.fps),
            max_num_frames=args.max_num_frames,
        )
        if relative_indices.size == 0:
            raise ValueError(f"No frames selected from {episode_dir} in range [{start_index}, {stop_index})")
        source_frame_indices = (relative_indices + int(start_index)).astype(np.int32, copy=False)
        episode_fps = _resolve_episode_fps(handle, float(args.fallback_episode_fps))
        query_frame_schedule_path = _prepare_query_frame_schedule(
            out_dir=out_dir,
            video_name=video_name,
            source_frame_indices=source_frame_indices,
            episode_fps=episode_fps,
            args=args,
            start_index=start_index,
            stop_index=stop_index,
        )
        first_frame = load_video_frame(paths.video_path, int(source_frame_indices[0]))
        depth_frames = np.asarray(handle["depth/depth"][source_frame_indices], dtype=np.float32)
        depth_conf = np.asarray(handle["depth/confidence"][source_frame_indices], dtype=np.float32)
        target_hw = tuple(int(value) for value in depth_frames.shape[1:3])
        source_hw = tuple(int(value) for value in first_frame.shape[:2])
        rgb_frames = load_video_frames(paths.video_path, source_frame_indices, target_hw=target_hw)
        intrinsics_single = build_stereo_left_intrinsics(handle, source_hw=source_hw, target_hw=target_hw)
        intrinsics = np.repeat(intrinsics_single[None, :, :], source_frame_indices.shape[0], axis=0)
        extrinsics = build_stereo_left_extrinsics(handle, source_frame_indices=source_frame_indices)
        language_text = read_caption_main_task(handle)

    source_descriptor = build_xperience_source_descriptor(
        dataset_root=dataset_root,
        episode_dir=episode_dir,
        camera_name=args.camera_name,
        window_start=start_index,
        window_stop=stop_index,
        source_hw=source_hw,
        target_hw=target_hw,
    )
    source_descriptor["episode_fps"] = float(episode_fps)
    source_descriptor["selected_frame_count"] = int(source_frame_indices.shape[0])
    source_descriptor["language_text"] = str(language_text)
    return {
        "video_ten": torch.from_numpy(rgb_frames).permute(0, 3, 1, 2).float() / 255.0,
        "depth_npy": depth_frames.astype(np.float32, copy=False),
        "depth_conf": depth_conf.astype(np.float32, copy=False),
        "intrs_npy": intrinsics.astype(np.float32, copy=False),
        "extrs_npy": extrinsics.astype(np.float32, copy=False),
        "original_filenames": [f"{int(frame_idx):06d}" for frame_idx in source_frame_indices.tolist()],
        "source_frame_indices": source_frame_indices.astype(np.int32, copy=False),
        "camera_name": str(args.camera_name),
        "scene_id": _build_video_name(episode_dir, args.camera_name, start_index, stop_index),
        "language_text": str(language_text),
        "source_descriptor": source_descriptor,
        "query_frame_schedule_path": str(query_frame_schedule_path),
        "video_path": None,
        "depth_path": None,
        "stride": int(args.fps),
    }


def _write_language_file(output_root: Path, video_name: str, language_text: str) -> None:
    if not language_text:
        return
    output_dir = output_root / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lang.txt").write_text(str(language_text).strip() + "\n", encoding="utf-8")


def _reindex_window_tasks(tasks: list[WindowTask]) -> list[WindowTask]:
    total_tasks = len(tasks)
    return [
        WindowTask(
            task_index=task_index,
            total_tasks=total_tasks,
            episode_dir=task.episode_dir,
            video_name=task.video_name,
            window_start=task.window_start,
            window_stop=task.window_stop,
            output_dir=task.output_dir,
        )
        for task_index, task in enumerate(tasks, start=1)
    ]


def _build_window_tasks(
    *,
    episode_dirs: list[Path],
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[WindowTask], int]:
    pending_tasks: list[WindowTask] = []
    skipped = 0

    for episode_dir in episode_dirs:
        with h5py.File(episode_dir / "annotation.hdf5", "r") as handle:
            frame_count = _resolve_frame_count(handle)
        windows = _build_window_ranges(
            frame_count=frame_count,
            start_index=args.start_index,
            stop_index=args.stop_index,
            window_size=args.window_size,
            window_step=args.window_step,
        )
        for window_start, window_stop in windows:
            video_name = _build_video_name(episode_dir, args.camera_name, window_start, window_stop)
            output_path = out_dir / video_name
            if args.skip_existing and output_path.exists() and is_traceforge_output_complete(output_path):
                skipped += 1
                logger.info(f"Skipping {video_name} because output already exists and is complete")
                continue
            pending_tasks.append(
                WindowTask(
                    task_index=0,
                    total_tasks=0,
                    episode_dir=episode_dir,
                    video_name=video_name,
                    window_start=window_start,
                    window_stop=window_stop,
                    output_dir=output_path,
                )
            )
    return _reindex_window_tasks(pending_tasks), skipped


def _safe_per_query_seconds(total_seconds: float | None, query_frame_count: int | None) -> float | None:
    if total_seconds is None or query_frame_count is None or query_frame_count <= 0:
        return None
    return float(total_seconds / float(query_frame_count))


def _build_window_task_metric_record(
    *,
    task: WindowTask,
    gpu_id: int | None,
    args: argparse.Namespace,
    worker_slot: WorkerSlot | None,
    query_frame_schedule_path: str | None,
    selected_frame_count: int | None,
    query_frame_count: int | None,
    process_seconds: float | None,
    save_seconds: float | None,
    started_at_unix: float,
    finished_at_unix: float,
    status: str,
    retryable_cuda_error: bool,
    error_message: str | None,
    original_frame_height: int | None = None,
    original_frame_width: int | None = None,
    processing_frame_height: int | None = None,
    processing_frame_width: int | None = None,
) -> dict[str, Any]:
    total_seconds = None
    if process_seconds is not None and save_seconds is not None:
        total_seconds = float(process_seconds + save_seconds)
    return {
        "task_index": int(task.task_index),
        "total_tasks": int(task.total_tasks),
        "episode_uuid": task.episode_dir.parent.name,
        "episode_name": task.episode_dir.name,
        "camera_name": str(args.camera_name),
        "video_name": task.video_name,
        "window_start": int(task.window_start),
        "window_stop": int(task.window_stop),
        "gpu_id": gpu_id,
        "worker_label": worker_slot.label if worker_slot is not None else None,
        "worker_index": worker_slot.worker_index if worker_slot is not None else None,
        "gpu_slot_index": worker_slot.gpu_slot_index if worker_slot is not None else None,
        "gpu_slot_count": worker_slot.gpu_slot_count if worker_slot is not None else None,
        "device": getattr(args, "device", None),
        "num_iters": int(args.num_iters),
        "depth_filter_workers": int(args.depth_filter_workers),
        "traj_filter_profile": getattr(args, "traj_filter_profile", None),
        "fps": int(args.fps),
        "max_num_frames": int(args.max_num_frames),
        "future_len": int(args.future_len),
        "grid_size": int(args.grid_size),
        "grid_width": getattr(args, "grid_width", None),
        "grid_height": getattr(args, "grid_height", None),
        "query_grid_hw": list(getattr(args, "query_grid_hw", ()) or []),
        "processing_resize_hw": list(getattr(args, "processing_resize_hw", ()) or []),
        "selected_frame_count": selected_frame_count,
        "query_frame_schedule_path": query_frame_schedule_path,
        "query_frame_count": query_frame_count,
        "process_seconds": process_seconds,
        "save_seconds": save_seconds,
        "total_seconds": total_seconds,
        "process_seconds_per_query": _safe_per_query_seconds(process_seconds, query_frame_count),
        "save_seconds_per_query": _safe_per_query_seconds(save_seconds, query_frame_count),
        "total_seconds_per_query": _safe_per_query_seconds(total_seconds, query_frame_count),
        "status": status,
        "retryable_cuda_error": bool(retryable_cuda_error),
        "error_message": error_message,
        "started_at_unix": float(started_at_unix),
        "finished_at_unix": float(finished_at_unix),
        "output_dir": str(task.output_dir.resolve()),
    }


def _build_window_task_profile_record(
    *,
    task: WindowTask,
    gpu_id: int | None,
    args: argparse.Namespace,
    worker_slot: WorkerSlot | None,
    query_frame_schedule_path: str | None,
    query_frame_count: int | None,
    process_seconds: float | None,
    save_seconds: float | None,
    started_at_unix: float,
    finished_at_unix: float,
    status: str,
    retryable_cuda_error: bool,
    error_message: str | None,
    profile_stats: dict[str, float] | None,
    save_profile_stats: dict[str, float] | None,
    per_query_save_seconds: dict[int, float] | None,
    scene_finalize_overhead_seconds: float | None,
    original_frame_height: int | None = None,
    original_frame_width: int | None = None,
    processing_frame_height: int | None = None,
    processing_frame_width: int | None = None,
) -> dict[str, Any]:
    normalized_per_query_save_seconds = {
        str(int(query_frame_idx)): float(seconds)
        for query_frame_idx, seconds in (per_query_save_seconds or {}).items()
    }
    return {
        "task_index": int(task.task_index),
        "total_tasks": int(task.total_tasks),
        "episode_uuid": task.episode_dir.parent.name,
        "episode_name": task.episode_dir.name,
        "camera_name": str(args.camera_name),
        "video_name": task.video_name,
        "window_start": int(task.window_start),
        "window_stop": int(task.window_stop),
        "gpu_id": gpu_id,
        "worker_label": worker_slot.label if worker_slot is not None else None,
        "worker_index": worker_slot.worker_index if worker_slot is not None else None,
        "gpu_slot_index": worker_slot.gpu_slot_index if worker_slot is not None else None,
        "gpu_slot_count": worker_slot.gpu_slot_count if worker_slot is not None else None,
        "device": getattr(args, "device", None),
        "num_iters": int(args.num_iters),
        "depth_filter_workers": int(args.depth_filter_workers),
        "traj_filter_profile": getattr(args, "traj_filter_profile", None),
        "query_frame_schedule_path": query_frame_schedule_path,
        "query_frame_count": query_frame_count,
        "process_seconds": process_seconds,
        "save_seconds": save_seconds,
        "status": status,
        "retryable_cuda_error": bool(retryable_cuda_error),
        "error_message": error_message,
        "started_at_unix": float(started_at_unix),
        "finished_at_unix": float(finished_at_unix),
        "profile_stats": dict(profile_stats or {}),
        "save_profile_stats": dict(save_profile_stats or {}),
        "per_query_save_seconds": normalized_per_query_save_seconds,
        "scene_finalize_overhead_seconds": (
            float(scene_finalize_overhead_seconds)
            if scene_finalize_overhead_seconds is not None
            else None
        ),
        "original_frame_height": (
            int(original_frame_height) if original_frame_height is not None else None
        ),
        "original_frame_width": (
            int(original_frame_width) if original_frame_width is not None else None
        ),
        "processing_frame_height": (
            int(processing_frame_height) if processing_frame_height is not None else None
        ),
        "processing_frame_width": (
            int(processing_frame_width) if processing_frame_width is not None else None
        ),
        "output_dir": str(task.output_dir.resolve()),
    }


def _build_batch_run_summary(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    out_dir: Path,
    telemetry_out_dir: Path,
    episode_count: int,
    window_task_count: int,
    window_success_count: int,
    window_fail_count: int,
    window_skip_existing_count: int,
    wall_clock_seconds: float,
    gpu_ids: list[int],
    telemetry_gpu_ids: list[int],
    gpu_info: list[dict[str, Any]],
    worker_slot_count: int,
    host_name: str | None,
) -> dict[str, Any]:
    return {
        "dataset_adapter": ADAPTER_NAME,
        "dataset_root": str(dataset_root.resolve()),
        "out_dir": str(out_dir.resolve()),
        "telemetry_out_dir": str(telemetry_out_dir.resolve()),
        "camera_name": str(args.camera_name),
        "episode_glob": str(args.episode_glob),
        "episode_limit": getattr(args, "episode_limit", None),
        "episode_count": int(episode_count),
        "window_task_count": int(window_task_count),
        "window_success_count": int(window_success_count),
        "window_fail_count": int(window_fail_count),
        "window_skip_existing_count": int(window_skip_existing_count),
        "wall_clock_seconds": float(wall_clock_seconds),
        "gpu_ids": [int(gpu_id) for gpu_id in gpu_ids],
        "telemetry_gpu_ids": [int(gpu_id) for gpu_id in telemetry_gpu_ids],
        "gpu_info": list(gpu_info or []),
        "host_name": host_name,
        "workers_per_gpu": int(args.workers_per_gpu),
        "worker_slot_count": int(worker_slot_count),
        "hardware_telemetry_interval_sec": float(args.hardware_telemetry_interval_sec),
        "collect_profile_stats": bool(args.collect_profile_stats),
        "num_iters": int(args.num_iters),
        "depth_filter_workers": int(args.depth_filter_workers),
        "fps": int(args.fps),
        "frame_drop_rate": int(args.frame_drop_rate),
        "keyframe_seed": int(args.keyframe_seed),
        "keyframes_per_sec_min": int(args.keyframes_per_sec_min),
        "keyframes_per_sec_max": int(args.keyframes_per_sec_max),
        "fallback_episode_fps": float(args.fallback_episode_fps),
        "max_num_frames": int(args.max_num_frames),
        "future_len": int(args.future_len),
        "grid_size": int(args.grid_size),
        "grid_width": getattr(args, "grid_width", None),
        "grid_height": getattr(args, "grid_height", None),
        "query_grid_hw": list(getattr(args, "query_grid_hw", ()) or []),
        "window_size": int(args.window_size),
        "window_step": int(args.window_step) if args.window_step is not None else None,
        "start_index": int(args.start_index),
        "stop_index": int(args.stop_index) if args.stop_index is not None else None,
        "scene_storage_mode": str(args.scene_storage_mode),
        "output_layout": str(args.output_layout),
        "filter_level": str(args.filter_level),
        "traj_filter_profile": str(args.traj_filter_profile),
        "support_grid_ratio": float(args.support_grid_ratio),
    }


def _run_window_task(
    *,
    task: WindowTask,
    dataset_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
    model_3dtracker,
    gpu_id: int | None = None,
    worker_slot: WorkerSlot | None = None,
    telemetry_writer: BatchTelemetryWriter | None = None,
) -> tuple[bool, bool]:
    window_args = copy.copy(args)
    started_at_unix = time.time()
    process_seconds: float | None = None
    save_seconds: float | None = None
    selected_frame_count: int | None = None
    query_frame_count: int | None = None
    status = "failed"
    retryable_cuda_error = False
    error_message: str | None = None
    scene_bundle: dict[str, object] | None = None
    result: dict[str, Any] | None = None
    save_artifacts: dict[str, Any] | None = None
    query_frame_schedule_path: str | None = None

    logger.info(
        f"{task.video_name}: run "
        f"(device={window_args.device}, window=[{task.window_start}, {task.window_stop}))"
    )

    try:
        safe_empty_cuda_cache(f"{task.video_name}: pre-run cleanup")
        scene_bundle = _load_scene_bundle(
            task.episode_dir,
            dataset_root=dataset_root,
            out_dir=out_dir,
            args=window_args,
            start_index=task.window_start,
            stop_index=task.window_stop,
            video_name=task.video_name,
        )
        query_frame_schedule_path = str(scene_bundle.get("query_frame_schedule_path") or "")
        window_args.query_frame_schedule_path = query_frame_schedule_path or None
        selected_frame_indices = scene_bundle.get("source_frame_indices")
        if selected_frame_indices is not None:
            selected_frame_count = int(len(selected_frame_indices))

        process_start = time.perf_counter()
        result = infer.process_single_scene_bundle(
            scene_bundle,
            window_args,
            model_3dtracker,
            video_name=task.video_name,
            output_dir=str(out_dir),
        )
        process_seconds = time.perf_counter() - process_start
        query_frame_results = result.get("query_frame_results") or {}
        query_frame_count = int(len(query_frame_results))

        save_start = time.perf_counter()
        save_artifacts = infer.save_structured_data(
            video_name=task.video_name,
            output_dir=str(out_dir),
            video_tensor=result["video_tensor"],
            depths=result["depths"],
            coords=result["coords"],
            visibs=result["visibs"],
            intrinsics=result["intrinsics"],
            extrinsics=result["extrinsics"],
            query_points_per_frame=result["query_points_per_frame"],
            original_filenames=result["original_filenames"],
            query_frame_results=result.get("query_frame_results"),
            future_len=window_args.future_len,
            grid_size=window_args.grid_size,
            filter_args=window_args,
            full_video_tensor=result["full_video_tensor"],
            full_depths=result["full_depths"],
            full_intrinsics=result["full_intrinsics"],
            full_extrinsics=result["full_extrinsics"],
            depth_conf=result["depth_conf"],
            video_source_path=None,
            depth_source_path=None,
            source_descriptor=scene_bundle["source_descriptor"],
            source_frame_indices=result["source_frame_indices"],
            query_frame_metadata=result.get("query_frame_metadata"),
            original_frame_height=result.get("original_frame_height"),
            original_frame_width=result.get("original_frame_width"),
            processing_frame_height=result.get("processing_frame_height"),
            processing_frame_width=result.get("processing_frame_width"),
        )
        save_seconds = time.perf_counter() - save_start
        _write_language_file(out_dir, task.video_name, str(scene_bundle.get("language_text") or ""))
        status = "success"
        logger.info(f"Finished Xperience window {task.video_name}")
        return True, False
    except Exception as exc:
        if is_retryable_cuda_error(exc):
            retryable_cuda_error = True
            error_message = str(exc)
            logger.exception(f"{task.video_name} hit retryable CUDA failure: {exc}")
            return False, True
        error_message = str(exc)
        logger.exception(f"{task.video_name} failed: {exc}")
        return False, False
    finally:
        finished_at_unix = time.time()
        if telemetry_writer is not None:
            telemetry_writer.record_task(
                _build_window_task_metric_record(
                    task=task,
                    gpu_id=gpu_id,
                    args=window_args,
                    worker_slot=worker_slot,
                    query_frame_schedule_path=(query_frame_schedule_path or None),
                    selected_frame_count=selected_frame_count,
                    query_frame_count=query_frame_count,
                    process_seconds=process_seconds,
                    save_seconds=save_seconds,
                    started_at_unix=started_at_unix,
                    finished_at_unix=finished_at_unix,
                    status=status,
                    retryable_cuda_error=retryable_cuda_error,
                    error_message=error_message,
                    original_frame_height=(
                        result.get("original_frame_height")
                        if isinstance(result, dict)
                        else None
                    ),
                    original_frame_width=(
                        result.get("original_frame_width")
                        if isinstance(result, dict)
                        else None
                    ),
                    processing_frame_height=(
                        result.get("processing_frame_height")
                        if isinstance(result, dict)
                        else None
                    ),
                    processing_frame_width=(
                        result.get("processing_frame_width")
                        if isinstance(result, dict)
                        else None
                    ),
                )
            )
            if bool(getattr(window_args, "collect_profile_stats", False)):
                telemetry_writer.record_task_profile(
                    _build_window_task_profile_record(
                        task=task,
                        gpu_id=gpu_id,
                        args=window_args,
                        worker_slot=worker_slot,
                        query_frame_schedule_path=(query_frame_schedule_path or None),
                        query_frame_count=query_frame_count,
                        process_seconds=process_seconds,
                        save_seconds=save_seconds,
                        started_at_unix=started_at_unix,
                        finished_at_unix=finished_at_unix,
                        status=status,
                        retryable_cuda_error=retryable_cuda_error,
                        error_message=error_message,
                        profile_stats=(
                            result.get("profile_stats")
                            if isinstance(result, dict)
                            else None
                        ),
                        save_profile_stats=(
                            save_artifacts.get("save_profile_stats")
                            if isinstance(save_artifacts, dict)
                            else None
                        ),
                        per_query_save_seconds=(
                            save_artifacts.get("per_query_save_seconds")
                            if isinstance(save_artifacts, dict)
                            else None
                        ),
                        scene_finalize_overhead_seconds=(
                            save_artifacts.get("scene_finalize_overhead_seconds")
                            if isinstance(save_artifacts, dict)
                            else None
                        ),
                        original_frame_height=(
                            result.get("original_frame_height")
                            if isinstance(result, dict)
                            else None
                        ),
                        original_frame_width=(
                            result.get("original_frame_width")
                            if isinstance(result, dict)
                            else None
                        ),
                        processing_frame_height=(
                            result.get("processing_frame_height")
                            if isinstance(result, dict)
                            else None
                        ),
                        processing_frame_width=(
                            result.get("processing_frame_width")
                            if isinstance(result, dict)
                            else None
                        ),
                    )
                )
        del save_artifacts
        del result
        del scene_bundle
        safe_empty_cuda_cache(f"{task.video_name}: run cleanup")


def _process_window_tasks_on_gpu(
    *,
    worker_slot: WorkerSlot,
    task_queue,
    dataset_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
    stop_event,
    remaining_tasks=None,
    telemetry_writer: BatchTelemetryWriter | None = None,
) -> tuple[int, int, float]:
    worker_args = copy.deepcopy(args)
    worker_args.device = worker_slot.device
    worker_label = worker_slot.label

    worker_start = time.time()
    total_success = 0
    total_fail = 0
    model_3dtracker = None
    try:
        while not stop_event.is_set():
            if model_3dtracker is None:
                if not wait_for_gpu_recovery(
                    gpu_id=worker_slot.gpu_id,
                    args=worker_args,
                    stop_event=stop_event,
                ):
                    break
                logger.info(f"[{worker_label}] start dynamic worker on {worker_args.device}")
                try:
                    model_3dtracker = load_model(worker_args.checkpoint).to(worker_args.device)
                    warm_up_cuda_linalg(worker_args.device)
                except Exception as exc:
                    model_3dtracker = unload_tracker_model(model_3dtracker)
                    if is_retryable_cuda_error(exc):
                        logger.exception(
                            f"[{worker_label}] worker startup failed with retryable CUDA error: {exc}"
                        )
                        stop_event.wait(max(worker_args.gpu_recovery_poll_sec, 1.0))
                        continue
                    raise

            try:
                task = task_queue.get(timeout=min(max(worker_args.gpu_recovery_poll_sec, 1.0), 5.0))
            except queue.Empty:
                continue

            if task is None:
                task_queue.task_done()
                break

            try:
                logger.info(
                    f"[{worker_label}] "
                    f"[{task.task_index}/{task.total_tasks}] {task.video_name}"
                )
                ok, retire_worker = _run_window_task(
                    task=task,
                    dataset_root=dataset_root,
                    out_dir=out_dir,
                    args=worker_args,
                    model_3dtracker=model_3dtracker,
                    gpu_id=worker_slot.gpu_id,
                    worker_slot=worker_slot,
                    telemetry_writer=telemetry_writer,
                )
                if ok:
                    total_success += 1
                    mark_task_completed(remaining_tasks)
                elif retire_worker:
                    task_queue.put(task)
                    model_3dtracker = unload_tracker_model(model_3dtracker)
                    logger.warning(
                        f"[{worker_label}] re-queued {task.video_name} after retryable CUDA failure; "
                        "waiting for GPU recovery."
                    )
                else:
                    total_fail += 1
                    mark_task_completed(remaining_tasks)
            finally:
                task_queue.task_done()
    finally:
        model_3dtracker = unload_tracker_model(model_3dtracker)

    elapsed = time.time() - worker_start
    logger.info(
        f"[{worker_label}] dynamic worker done in {elapsed/60:.1f} min "
        f"(window_success={total_success}, window_fail={total_fail})"
    )
    return total_success, total_fail, elapsed


def _process_window_tasks_on_gpu_entrypoint(
    *,
    worker_slot: WorkerSlot,
    task_queue,
    dataset_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
    stop_event,
    remaining_tasks,
    result_queue,
    telemetry_writer: BatchTelemetryWriter | None = None,
) -> None:
    try:
        success_count, fail_count, elapsed = _process_window_tasks_on_gpu(
            worker_slot=worker_slot,
            task_queue=task_queue,
            dataset_root=dataset_root,
            out_dir=out_dir,
            args=args,
            stop_event=stop_event,
            remaining_tasks=remaining_tasks,
            telemetry_writer=telemetry_writer,
        )
        result_queue.put(
            WorkerProcessResult(
                worker_label=worker_slot.label,
                success_count=success_count,
                fail_count=fail_count,
                elapsed=elapsed,
                error=None,
            )
        )
    except Exception as exc:
        logger.exception(f"[{worker_slot.label}] dynamic worker failed: {exc}")
        result_queue.put(
            WorkerProcessResult(
                worker_label=worker_slot.label,
                success_count=0,
                fail_count=0,
                elapsed=0.0,
                error=str(exc),
            )
        )


def run(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Xperience dataset_root does not exist: {dataset_root}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    telemetry_out_dir = (
        Path(args.telemetry_out_dir).expanduser().resolve()
        if args.telemetry_out_dir
        else out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if telemetry_out_dir != out_dir:
        telemetry_out_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = _find_episode_dirs(
        dataset_root,
        episode_glob=args.episode_glob,
        episode_limit=args.episode_limit,
    )
    if not episode_dirs:
        raise SystemExit(f"No Xperience episodes found under {dataset_root} with pattern {args.episode_glob!r}")

    tasks, skipped = _build_window_tasks(
        episode_dirs=episode_dirs,
        out_dir=out_dir,
        args=args,
    )
    gpu_ids = parse_gpu_ids(args.gpu_id)
    worker_slots = build_worker_slots(gpu_ids, workers_per_gpu=args.workers_per_gpu)
    probe_gpu_ids = list(dict.fromkeys(gpu_ids))
    telemetry_gpu_ids = resolve_telemetry_gpu_ids(
        gpu_ids=gpu_ids,
        device=args.device,
    )
    gpu_memory: dict[int, GpuMemoryInfo | None] = {}
    skipped_gpu_ids: list[int] = []
    telemetry_writer: BatchTelemetryWriter | None = None
    hardware_sampler: HardwareTelemetrySampler | None = None
    host_name = socket.gethostname()
    gpu_info = collect_gpu_static_info(telemetry_gpu_ids)

    if probe_gpu_ids:
        _available_gpu_ids, gpu_memory, skipped_gpu_ids = filter_gpu_ids_by_free_memory(
            probe_gpu_ids,
            min_free_gpu_mem_gb=args.min_free_gpu_mem_gb,
        )
        if args.min_free_gpu_mem_gb > 0 and skipped_gpu_ids and len(skipped_gpu_ids) == len(probe_gpu_ids):
            logger.warning(
                "No GPUs currently pass the free-memory filter; dynamic workers will wait for recovery."
            )

    logger.info("=" * 80)
    logger.info("Native Xperience batch inference")
    logger.info(f"dataset_root={dataset_root}")
    logger.info(f"out_dir={out_dir}")
    logger.info(f"telemetry_out_dir={telemetry_out_dir}")
    logger.info(
        f"episodes={len(episode_dirs)}, pending_windows={len(tasks)}, skip_existing_windows={skipped}"
    )
    logger.info(
        f"device={args.device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    if gpu_ids:
        logger.info(f"gpu_ids={gpu_ids}")
        logger.info(
            f"workers_per_gpu={args.workers_per_gpu}, total_dynamic_workers={len(worker_slots)}"
        )
        for gpu_id in probe_gpu_ids:
            mem_info = gpu_memory.get(gpu_id)
            if mem_info is None:
                logger.warning(f"[GPU {gpu_id}] free-memory probe unavailable; keeping GPU enabled.")
                continue
            logger.info(
                f"[GPU {gpu_id}] free_mem={mem_info.free_gb:.1f} GiB / {mem_info.total_gb:.1f} GiB"
            )
        for gpu_id in skipped_gpu_ids:
            mem_info = gpu_memory[gpu_id]
            assert mem_info is not None
            logger.warning(
                f"[GPU {gpu_id}] currently below the free-memory threshold and will wait for recovery: "
                f"{mem_info.free_gb:.1f} GiB < {args.min_free_gpu_mem_gb:.1f} GiB"
            )
    logger.info(
        f"camera={args.camera_name}, keyframes_per_sec={args.keyframes_per_sec_min}~{args.keyframes_per_sec_max}, "
        f"future_len={args.future_len}, grid_size={args.grid_size}, "
        f"grid_hw={getattr(args, 'query_grid_hw', None)}, load_stride={args.fps}, "
        f"window_size={args.window_size}, window_step={args.window_step}, "
        f"depth_filter_workers={args.depth_filter_workers}"
    )
    logger.info(
        f"collect_profile_stats={args.collect_profile_stats}, "
        f"hardware_telemetry_interval_sec={args.hardware_telemetry_interval_sec}"
    )
    if gpu_info:
        logger.info(f"telemetry_gpu_ids={telemetry_gpu_ids}")
        for gpu_record in gpu_info:
            logger.info(
                f"[GPU {gpu_record['gpu_id']}] model={gpu_record.get('name')} "
                f"total_mem_gib={gpu_record.get('memory_total_gib')}"
            )
    logger.info("=" * 80)

    if not tasks:
        logger.info("No pending Xperience windows after filtering.")
        return 0

    window_success = 0
    window_fail = 0
    batch_start = time.time()

    try:
        if gpu_ids:
            mp_ctx = mp.get_context("spawn")
            telemetry_writer = BatchTelemetryWriter(
                telemetry_out_dir,
                enable_profile_records=args.collect_profile_stats,
                enable_hardware_records=args.hardware_telemetry_interval_sec > 0.0,
                lock=mp_ctx.Lock(),
            )
            if args.hardware_telemetry_interval_sec > 0.0:
                hardware_sampler = HardwareTelemetrySampler(
                    telemetry_writer=telemetry_writer,
                    interval_sec=args.hardware_telemetry_interval_sec,
                    host_name=host_name,
                    gpu_ids=telemetry_gpu_ids,
                )
                hardware_sampler.start()

            task_queue = mp_ctx.JoinableQueue()
            for task in tasks:
                task_queue.put(task)
            stop_event = mp_ctx.Event()
            remaining_tasks = mp_ctx.Value("i", len(tasks))
            result_queue = mp_ctx.Queue()
            worker_processes: list[tuple[WorkerSlot, mp.Process]] = []

            for worker_slot in worker_slots:
                process = mp_ctx.Process(
                    target=_process_window_tasks_on_gpu_entrypoint,
                    kwargs={
                        "worker_slot": worker_slot,
                        "task_queue": task_queue,
                        "dataset_root": dataset_root,
                        "out_dir": out_dir,
                        "args": args,
                        "stop_event": stop_event,
                        "remaining_tasks": remaining_tasks,
                        "result_queue": result_queue,
                        "telemetry_writer": telemetry_writer,
                    },
                    name=f"xperience-{worker_slot.gpu_id}-{worker_slot.gpu_slot_index}",
                )
                process.start()
                worker_processes.append((worker_slot, process))

            try:
                while remaining_tasks.value > 0:
                    dead_workers = [
                        (worker_slot, process)
                        for worker_slot, process in worker_processes
                        if not process.is_alive()
                    ]
                    if dead_workers:
                        logger.error("A dynamic GPU worker exited before all Xperience window tasks completed.")
                        break
                    time.sleep(min(max(args.gpu_recovery_poll_sec, 1.0), 30.0))
            finally:
                stop_event.set()
                for _ in worker_slots:
                    task_queue.put(None)
                for worker_slot, process in worker_processes:
                    process.join(timeout=max(args.gpu_recovery_poll_sec, 1.0) + 10.0)
                    if process.is_alive():
                        logger.warning(
                            f"[{worker_slot.label}] worker did not exit promptly; terminating."
                        )
                        process.terminate()
                        process.join(timeout=5.0)

            worker_results: dict[str, WorkerProcessResult] = {}
            while True:
                try:
                    worker_result = result_queue.get_nowait()
                except queue.Empty:
                    break
                worker_results[worker_result.worker_label] = worker_result

            for worker_slot, process in worker_processes:
                worker_result = worker_results.get(worker_slot.label)
                if worker_result is not None:
                    window_success += worker_result.success_count
                    window_fail += worker_result.fail_count
                    if worker_result.error is not None:
                        logger.error(
                            f"[{worker_result.worker_label}] dynamic worker reported error: "
                            f"{worker_result.error}"
                        )
                    continue
                if process.exitcode not in (0, None):
                    logger.error(
                        f"[{worker_slot.label}] dynamic worker exited with code {process.exitcode}"
                    )

            remaining_task_count = remaining_tasks.value
            if remaining_task_count > 0:
                window_fail += remaining_task_count
                logger.error(
                    f"Dynamic scheduler left {remaining_task_count} Xperience window tasks unprocessed."
                )
        else:
            telemetry_writer = BatchTelemetryWriter(
                telemetry_out_dir,
                enable_profile_records=args.collect_profile_stats,
                enable_hardware_records=args.hardware_telemetry_interval_sec > 0.0,
            )
            if args.hardware_telemetry_interval_sec > 0.0:
                hardware_sampler = HardwareTelemetrySampler(
                    telemetry_writer=telemetry_writer,
                    interval_sec=args.hardware_telemetry_interval_sec,
                    host_name=host_name,
                    gpu_ids=telemetry_gpu_ids,
                )
                hardware_sampler.start()

            logger.info(f"Loading 3D tracker once on {args.device}")
            model_3dtracker = load_model(args.checkpoint).to(args.device)
            warm_up_cuda_linalg(args.device)
            try:
                for task in tasks:
                    logger.info(f"[{task.task_index}/{task.total_tasks}] {task.video_name}")
                    ok, retryable_cuda_failure = _run_window_task(
                        task=task,
                        dataset_root=dataset_root,
                        out_dir=out_dir,
                        args=args,
                        model_3dtracker=model_3dtracker,
                        telemetry_writer=telemetry_writer,
                    )
                    if ok:
                        window_success += 1
                    else:
                        window_fail += 1
                        if retryable_cuda_failure:
                            model_3dtracker = unload_tracker_model(model_3dtracker)
                            logger.warning(
                                "Reloading tracker after retryable CUDA failure in single-device mode."
                            )
                            model_3dtracker = load_model(args.checkpoint).to(args.device)
                            warm_up_cuda_linalg(args.device)
            finally:
                unload_tracker_model(model_3dtracker)
    finally:
        if hardware_sampler is not None:
            hardware_sampler.stop()

    wall_clock_seconds = time.time() - batch_start
    if telemetry_writer is not None:
        telemetry_writer.write_summary(
            _build_batch_run_summary(
                args=args,
                dataset_root=dataset_root,
                out_dir=out_dir,
                telemetry_out_dir=telemetry_out_dir,
                episode_count=len(episode_dirs),
                window_task_count=len(tasks),
                window_success_count=window_success,
                window_fail_count=window_fail,
                window_skip_existing_count=skipped,
                wall_clock_seconds=wall_clock_seconds,
                gpu_ids=gpu_ids,
                telemetry_gpu_ids=telemetry_gpu_ids,
                gpu_info=gpu_info,
                worker_slot_count=len(worker_slots),
                host_name=host_name,
            )
        )

    logger.info("=" * 80)
    if gpu_ids:
        logger.info(
            f"Done dynamic gpu mode. window_success={window_success}, "
            f"window_fail={window_fail}, skipped={skipped}"
        )
    else:
        logger.info(
            f"Done. window_success={window_success}, "
            f"window_fail={window_fail}, skipped={skipped}"
        )
    logger.info(f"wall_clock_seconds={wall_clock_seconds:.3f}")
    logger.info("=" * 80)
    return 1 if window_fail else 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
