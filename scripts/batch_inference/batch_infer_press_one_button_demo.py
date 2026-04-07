#!/usr/bin/env python3
"""
Press-one-button demo 数据集批量推理脚本。

目标数据结构：
    base_path/
        episode_00000/
            lang.txt
            trajectory_valid.h5
            rgb/
                varied_camera_1/*.png
                varied_camera_2/*.png
                varied_camera_3/*.png
            depth/
                varied_camera_1/*.npy
                varied_camera_2/*.npy
                varied_camera_3/*.npy

脚本设计参考：
1. external-only TraceForge 推理与保存逻辑；
2. episode/camera 粒度的多 GPU 调度。

推荐多卡使用方式：
- 使用 `--gpu_id 0,1,...` 启动动态调度；
- 每张卡对应一个常驻 worker，只加载一次 3D tracker；
- worker 从共享任务队列中按 `episode/camera` 粒度领取下一个任务，直到队列清空。

默认输出方式：
- 就地写回到每个 episode 下的 `trajectory/<camera_name>/...`；
- 如需兼容旧流程，仍可通过 `--out_dir` 指定外部输出根目录。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any
from pathlib import Path

import h5py
import numpy as np
import torch
from loguru import logger

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import infer
from utils.keyframe_schedule_utils import (
    build_candidate_source_frame_indices,
    filter_query_local_indices_by_remaining_frames,
    sample_query_source_indices_per_second,
)
from utils.traceforge_artifact_utils import is_traceforge_output_complete


DEFAULT_CAMERAS = [
    "varied_camera_1",
    "varied_camera_2",
]

_CUDA_LINALG_WARMUP_LOCK = threading.Lock()
_CUDA_LINALG_WARMED_DEVICES: set[str] = set()
_QUERY_FRAME_SCHEDULE_VERSION = 3
_QUERY_FRAME_SHARED_DIRNAME = "_shared"
_BATCH_RUN_SUMMARY_BASENAME = "_batch_run_summary.json"
_CAMERA_TASK_METRICS_BASENAME = "_camera_task_metrics.jsonl"
_CAMERA_TASK_PROFILES_BASENAME = "_camera_task_profiles.jsonl"
_HARDWARE_TELEMETRY_BASENAME = "_hardware_telemetry.jsonl"
_DEFAULT_DEPTH_FILTER_WORKERS = 8


@dataclass(frozen=True)
class CameraTask:
    task_index: int
    total_tasks: int
    episode_dir: Path
    out_episode_dir: Path
    camera_name: str
    query_frame_schedule_path: Path | None = None


@dataclass(frozen=True)
class GpuMemoryInfo:
    free_gb: float
    total_gb: float


@dataclass(frozen=True)
class WorkerSlot:
    worker_index: int
    total_workers: int
    gpu_id: int
    gpu_slot_index: int
    gpu_slot_count: int

    @property
    def device(self) -> str:
        return f"cuda:{self.gpu_id}"

    @property
    def label(self) -> str:
        if self.gpu_slot_count <= 1:
            return f"GPU {self.gpu_id}"
        return f"GPU {self.gpu_id} slot {self.gpu_slot_index}/{self.gpu_slot_count}"


@dataclass(frozen=True)
class WorkerProcessResult:
    worker_label: str
    success_count: int
    fail_count: int
    elapsed: float
    error: str | None = None


@dataclass(frozen=True)
class CpuIoSnapshot:
    captured_at_unix: float
    cpu_total_ticks: int
    cpu_iowait_ticks: int
    disk_read_bytes: int
    disk_write_bytes: int


class BatchTelemetryWriter:
    def __init__(
        self,
        out_root: Path,
        *,
        enable_profile_records: bool = False,
        enable_hardware_records: bool = False,
        lock=None,
    ):
        self.out_root = out_root.resolve()
        self.summary_path = self.out_root / _BATCH_RUN_SUMMARY_BASENAME
        self.task_metrics_path = self.out_root / _CAMERA_TASK_METRICS_BASENAME
        self.task_profiles_path = self.out_root / _CAMERA_TASK_PROFILES_BASENAME
        self.hardware_telemetry_path = self.out_root / _HARDWARE_TELEMETRY_BASENAME
        self.enable_profile_records = bool(enable_profile_records)
        self.enable_hardware_records = bool(enable_hardware_records)
        self._lock = lock
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.task_metrics_path.write_text("", encoding="utf-8")
        if self.enable_profile_records:
            self.task_profiles_path.write_text("", encoding="utf-8")
        if self.enable_hardware_records:
            self.hardware_telemetry_path.write_text("", encoding="utf-8")

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if self._lock is None:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
            return
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")

    def record_task(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(self.task_metrics_path, payload)

    def record_task_profile(self, payload: dict[str, Any]) -> None:
        if not self.enable_profile_records:
            return
        self._append_jsonl(self.task_profiles_path, payload)

    def record_hardware_sample(self, payload: dict[str, Any]) -> None:
        if not self.enable_hardware_records:
            return
        self._append_jsonl(self.hardware_telemetry_path, payload)

    def write_summary(self, payload: dict[str, Any]) -> None:
        if self._lock is None:
            _atomic_write_json(self.summary_path, payload)
            return
        with self._lock:
            _atomic_write_json(self.summary_path, payload)


def parse_camera_names(camera_names: str) -> list[str]:
    values = [item.strip() for item in camera_names.split(",") if item.strip()]
    if not values:
        raise ValueError("camera_names must contain at least one camera name")
    return values


def resolve_schedule_camera_names(
    camera_names: list[str],
    shared_schedule_camera_names: str | list[str] | None,
) -> list[str]:
    if shared_schedule_camera_names is None:
        return list(camera_names)
    if isinstance(shared_schedule_camera_names, str):
        return parse_camera_names(shared_schedule_camera_names)
    values = [str(item).strip() for item in shared_schedule_camera_names if str(item).strip()]
    if not values:
        raise ValueError("shared_schedule_camera_names must contain at least one camera name")
    return values


def parse_camera_int_overrides(
    raw: str | None,
    *,
    option_name: str,
) -> dict[str, int]:
    if raw is None:
        return {}

    overrides: dict[str, int] = {}
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        camera_name, separator, value_str = item.partition(":")
        camera_name = camera_name.strip()
        value_str = value_str.strip()
        if separator != ":" or not camera_name or not value_str:
            raise ValueError(
                f"{option_name} expects comma-separated camera:value pairs, got {item!r}"
            )
        value = int(value_str)
        if value <= 0:
            raise ValueError(f"{option_name} values must be >= 1, got {item!r}")
        if camera_name in overrides:
            raise ValueError(f"{option_name} contains duplicate camera {camera_name!r}")
        overrides[camera_name] = value
    return overrides


def resolve_camera_num_iters(
    *,
    base_num_iters: int,
    camera_name: str,
    overrides: dict[str, int] | None,
) -> int:
    if not overrides:
        return int(base_num_iters)
    return int(overrides.get(camera_name, base_num_iters))


def resolve_traj_filter_profile(camera_name: str, requested_profile: str) -> str:
    camera_name = camera_name.lower()
    is_wrist_like = (
        camera_name.endswith("camera_3")
        or "wrist" in camera_name
        or "hand" in camera_name
    )
    if requested_profile == "auto":
        return "external"
    if requested_profile in {
        "wrist_pick_place",
        "wrist_pick_place_no_heatmap",
    }:
        if is_wrist_like:
            return requested_profile
        return "external"
    return requested_profile


def _safe_per_query_seconds(total_seconds: float | None, query_frame_count: int | None) -> float | None:
    if total_seconds is None or query_frame_count is None or query_frame_count <= 0:
        return None
    return float(total_seconds / float(query_frame_count))


def build_camera_task_metric_record(
    *,
    task: CameraTask,
    gpu_id: int | None,
    args: argparse.Namespace,
    worker_label: str | None,
    worker_index: int | None,
    gpu_slot_index: int | None,
    gpu_slot_count: int | None,
    query_frame_count: int | None,
    process_seconds: float | None,
    save_seconds: float | None,
    started_at_unix: float,
    finished_at_unix: float,
    status: str,
    retryable_cuda_error: bool,
    error_message: str | None,
) -> dict[str, Any]:
    total_seconds = None
    if process_seconds is not None and save_seconds is not None:
        total_seconds = float(process_seconds + save_seconds)
    return {
        "task_index": int(task.task_index),
        "total_tasks": int(task.total_tasks),
        "episode_name": task.episode_dir.name,
        "camera_name": task.camera_name,
        "gpu_id": gpu_id,
        "worker_label": worker_label,
        "worker_index": worker_index,
        "gpu_slot_index": gpu_slot_index,
        "gpu_slot_count": gpu_slot_count,
        "device": getattr(args, "device", None),
        "num_iters": int(args.num_iters),
        "camera_num_iters_overrides": dict(getattr(args, "camera_num_iters_overrides", {})),
        "depth_filter_workers": int(getattr(args, "depth_filter_workers", _DEFAULT_DEPTH_FILTER_WORKERS)),
        "traj_filter_profile": getattr(args, "traj_filter_profile", None),
        "shared_schedule_camera_names": list(
            getattr(args, "shared_schedule_camera_names", getattr(args, "camera_names", []))
        ),
        "query_frame_schedule_path": (
            str(task.query_frame_schedule_path.resolve())
            if task.query_frame_schedule_path is not None
            else None
        ),
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
        "output_camera_dir": str((task.out_episode_dir / task.camera_name).resolve()),
    }


def build_camera_task_profile_record(
    *,
    task: CameraTask,
    gpu_id: int | None,
    args: argparse.Namespace,
    worker_label: str | None,
    worker_index: int | None,
    gpu_slot_index: int | None,
    gpu_slot_count: int | None,
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
) -> dict[str, Any]:
    normalized_per_query_save_seconds = {
        str(int(query_frame_idx)): float(seconds)
        for query_frame_idx, seconds in (per_query_save_seconds or {}).items()
    }
    return {
        "task_index": int(task.task_index),
        "total_tasks": int(task.total_tasks),
        "episode_name": task.episode_dir.name,
        "camera_name": task.camera_name,
        "gpu_id": gpu_id,
        "worker_label": worker_label,
        "worker_index": worker_index,
        "gpu_slot_index": gpu_slot_index,
        "gpu_slot_count": gpu_slot_count,
        "device": getattr(args, "device", None),
        "num_iters": int(args.num_iters),
        "camera_num_iters_overrides": dict(getattr(args, "camera_num_iters_overrides", {})),
        "depth_filter_workers": int(getattr(args, "depth_filter_workers", _DEFAULT_DEPTH_FILTER_WORKERS)),
        "traj_filter_profile": getattr(args, "traj_filter_profile", None),
        "shared_schedule_camera_names": list(
            getattr(args, "shared_schedule_camera_names", getattr(args, "camera_names", []))
        ),
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
        "query_frame_schedule_path": (
            str(task.query_frame_schedule_path.resolve())
            if task.query_frame_schedule_path is not None
            else None
        ),
        "output_camera_dir": str((task.out_episode_dir / task.camera_name).resolve()),
    }


def build_batch_run_summary(
    *,
    args: argparse.Namespace,
    base_path: Path,
    out_dir: Path | None,
    gpu_ids: list[int],
    telemetry_gpu_ids: list[int],
    host_name: str | None,
    gpu_info: list[dict[str, Any]] | None,
    worker_slot_count: int,
    episode_count: int,
    camera_task_count: int,
    total_camera_success: int,
    total_camera_fail: int,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    return {
        "base_path": str(base_path.resolve()),
        "out_dir": str(out_dir.resolve()) if out_dir is not None else None,
        "camera_names": list(args.camera_names),
        "shared_schedule_camera_names": list(
            getattr(args, "shared_schedule_camera_names", getattr(args, "camera_names", []))
        ),
        "gpu_ids": [int(gpu_id) for gpu_id in gpu_ids],
        "telemetry_gpu_ids": [int(gpu_id) for gpu_id in telemetry_gpu_ids],
        "host_name": host_name,
        "gpu_info": list(gpu_info or []),
        "workers_per_gpu": int(getattr(args, "workers_per_gpu", 1)),
        "worker_slot_count": int(worker_slot_count),
        "episode_count": int(episode_count),
        "camera_task_count": int(camera_task_count),
        "camera_success_count": int(total_camera_success),
        "camera_fail_count": int(total_camera_fail),
        "wall_clock_seconds": float(wall_clock_seconds),
        "collect_profile_stats": bool(getattr(args, "collect_profile_stats", False)),
        "hardware_telemetry_interval_sec": float(
            getattr(args, "hardware_telemetry_interval_sec", 0.0)
        ),
        "num_iters": int(args.num_iters),
        "camera_num_iters_overrides": dict(getattr(args, "camera_num_iters_overrides", {})),
        "depth_filter_workers": int(getattr(args, "depth_filter_workers", _DEFAULT_DEPTH_FILTER_WORKERS)),
        "keyframe_seed": int(args.keyframe_seed),
        "keyframes_per_sec_min": int(args.keyframes_per_sec_min),
        "keyframes_per_sec_max": int(args.keyframes_per_sec_max),
        "fps": int(args.fps),
        "max_num_frames": int(args.max_num_frames),
        "future_len": int(args.future_len),
        "grid_size": int(args.grid_size),
        "support_grid_ratio": float(args.support_grid_ratio),
        "filter_level": args.filter_level,
        "traj_filter_profile": args.traj_filter_profile,
        "external_geom_name": args.external_geom_name,
        "external_extr_mode": args.external_extr_mode,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference for press_one_button_demo_v1"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Dataset root, e.g. /data1/yaoxuran/press_one_button_demo_v1",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional TraceForge output root. If omitted, write in-place under each episode.",
    )
    parser.add_argument(
        "--trajectory_dirname",
        type=str,
        default="trajectory",
        help="Directory name used for in-place output under each episode",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
        help="Comma-separated physical GPU IDs, e.g. 1,3,5,6. IDs may be sparse; pass the currently usable cards.",
    )
    parser.add_argument(
        "--min_free_gpu_mem_gb",
        type=float,
        default=0.0,
        help=(
            "In --gpu_id mode, wait to load the model on a GPU until its free memory "
            "reaches this threshold. Useful on shared machines."
        ),
    )
    parser.add_argument(
        "--gpu_recovery_poll_sec",
        type=float,
        default=30.0,
        help=(
            "Polling interval in seconds for re-checking GPUs that are temporarily "
            "unavailable in dynamic multi-GPU mode."
        ),
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help=(
            "Number of resident workers to launch per listed physical GPU. "
            "Use values >1 when sharing lightly loaded GPUs."
        ),
    )
    parser.add_argument(
        "--collect_profile_stats",
        action="store_true",
        help="Persist infer.py process/save timing breakdowns into batch telemetry JSONL.",
    )
    parser.add_argument(
        "--hardware_telemetry_interval_sec",
        type=float,
        default=0.0,
        help=(
            "If >0 and --out_dir is set, periodically record GPU/CPU/IO telemetry "
            "into _hardware_telemetry.jsonl."
        ),
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        default=",".join(DEFAULT_CAMERAS),
        help="Comma-separated camera names to process",
    )
    parser.add_argument(
        "--shared_schedule_camera_names",
        type=str,
        default=None,
        help=(
            "Optional comma-separated camera names used only for building the shared "
            "query-frame schedule. Use this to keep query frames aligned across split "
            "batch submissions."
        ),
    )
    parser.add_argument(
        "--episode_name",
        type=str,
        default=None,
        help="Only process a single episode, e.g. episode_00000",
    )
    parser.add_argument(
        "--episode_names_file",
        type=str,
        default=None,
        help="Optional text file containing one episode name per line to restrict the batch.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit total episodes; useful for testing",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a camera if its TraceForge output is already complete",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print discovered work without loading models",
    )
    parser.add_argument(
        "--copy_lang",
        action="store_true",
        help="Copy episode lang.txt into output episode directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/tapip3d_final.pth",
    )
    parser.add_argument(
        "--depth_pose_method",
        type=str,
        default="external",
        choices=infer.video_depth_pose_dict.keys(),
        help="Maintained mode: external, using trajectory_valid.h5 per episode",
    )
    parser.add_argument(
        "--external_geom_name",
        type=str,
        default="trajectory_valid.h5",
        help="Per-episode geometry filename",
    )
    parser.add_argument(
        "--external_extr_mode",
        type=str,
        default="w2c",
        choices=["w2c", "c2w"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Single-GPU execution device. In --gpu_id mode each worker binds its own CUDA device automatically.",
    )
    parser.add_argument("--num_iters", type=int, default=5)
    parser.add_argument(
        "--camera_num_iters",
        type=str,
        default=None,
        help=(
            "Optional comma-separated camera:num_iters overrides, for example "
            "varied_camera_1:4,varied_camera_2:4."
        ),
    )
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max_num_frames", type=int, default=512)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument(
        "--output_layout",
        type=str,
        default="v2",
        choices=["v2", "legacy"],
        help="Artifact layout to write for each camera output.",
    )
    parser.add_argument(
        "--scene_storage_mode",
        type=str,
        default="source_ref",
        choices=["source_ref", "cache"],
        help=(
            "Storage backend for v2 artifacts. source_ref is the default and stores source RGB/depth/geometry "
            "references in scene_meta.json; cache writes local scene.h5 and scene_rgb.mp4."
        ),
    )
    parser.add_argument(
        "--save_visibility",
        action="store_true",
        default=False,
        help="Store per-query visibility arrays in sample NPZ files.",
    )
    parser.add_argument(
        "--keyframes_per_sec_min",
        type=int,
        default=2,
        help="Minimum number of query frames sampled per second for each episode.",
    )
    parser.add_argument(
        "--keyframes_per_sec_max",
        type=int,
        default=3,
        help="Maximum number of query frames sampled per second for each episode.",
    )
    parser.add_argument(
        "--keyframe_seed",
        type=int,
        default=0,
        help="Base random seed for deterministic per-episode keyframe schedules.",
    )
    parser.add_argument(
        "--fallback_episode_fps",
        type=float,
        default=0.0,
        help="Fallback FPS used only when trajectory_valid.h5 is missing root attr 'fps'. <=0 disables fallback.",
    )
    parser.add_argument(
        "--future_len",
        type=int,
        default=32,
        help="Tracking window per query frame",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=80,
        help="Grid size per query frame; 80 means 6400 points",
    )
    parser.add_argument(
        "--query_prefilter_mode",
        type=str,
        default="off",
        choices=["off", "profile_aware_static_v1"],
        help="Optional static query prefilter applied before tracking.",
    )
    parser.add_argument(
        "--query_prefilter_wrist_rank_keep_ratio",
        type=float,
        default=0.30,
        help="For wrist_manipulator* query prefiltering, keep the nearest query-depth ranks up to this ratio.",
    )
    parser.add_argument(
        "--support_grid_ratio",
        type=float,
        default=0.8,
        help="Support-point grid ratio relative to grid_size. 0 disables support points.",
    )
    parser.add_argument(
        "--depth_filter_workers",
        type=int,
        default=_DEFAULT_DEPTH_FILTER_WORKERS,
        help="Thread count used by infer.py when precomputing filtered depth segments.",
    )
    parser.add_argument(
        "--filter_level",
        type=str,
        default="standard",
        choices=["none", "basic", "standard", "strict"],
        help="Trajectory filtering level for sample traj_valid_mask",
    )
    parser.add_argument(
        "--traj_filter_profile",
        type=str,
        default="external",
        choices=[
            "auto",
            "external",
            "external_manipulator",
            "external_manipulator_v2",
            "wrist",
            "wrist_pick_place",
            "wrist_pick_place_no_heatmap",
            "wrist_manipulator_top95",
            "wrist_manipulator",
        ],
        help=(
            "Trajectory filtering profile. The maintained external-only default is external for all cameras; "
            "auto is retained as a compatibility alias and currently resolves to external. "
            "external_manipulator, external_manipulator_v2, wrist_pick_place, "
            "wrist_pick_place_no_heatmap, wrist_manipulator_top95, "
            "and wrist_manipulator "
            "must be requested explicitly."
        ),
    )
    parser.add_argument(
        "--traj_filter_ablation_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "wrist_seed_top95",
            "wrist_no_query_edge",
            "wrist_no_manipulator_depth",
            "wrist_no_manipulator_motion",
            "wrist_no_manipulator_cluster",
        ],
        help="Optional save-time wrist filter ablation for analysis only.",
    )
    parser.add_argument(
        "--min_valid_frames",
        type=int,
        default=None,
        help="Minimum valid frames per trajectory (overrides filter_level default)",
    )
    parser.add_argument(
        "--visibility_threshold",
        type=float,
        default=None,
        help="Minimum visibility ratio (overrides filter_level default)",
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.01,
        help="Minimum depth value in meters",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=10.0,
        help="Maximum depth value in meters",
    )
    parser.add_argument(
        "--boundary_margin",
        type=int,
        default=None,
        help="Projection boundary margin in pixels (overrides filter_level default)",
    )
    parser.add_argument(
        "--depth_change_threshold",
        type=float,
        default=None,
        help="Depth change std threshold in meters (overrides filter_level default)",
    )
    args = parser.parse_args()
    args.camera_names = parse_camera_names(args.camera_names)
    args.shared_schedule_camera_names = resolve_schedule_camera_names(
        args.camera_names,
        args.shared_schedule_camera_names,
    )
    args.camera_num_iters_overrides = parse_camera_int_overrides(
        args.camera_num_iters,
        option_name="--camera_num_iters",
    )

    if args.fps <= 0:
        raise ValueError("--fps must be >= 1 for shared per-second keyframe sampling.")
    if args.keyframes_per_sec_min <= 0 or args.keyframes_per_sec_max <= 0:
        raise ValueError("--keyframes_per_sec_min/max must both be >= 1")
    if args.keyframes_per_sec_min > args.keyframes_per_sec_max:
        raise ValueError("--keyframes_per_sec_min must be <= --keyframes_per_sec_max")
    if args.query_prefilter_wrist_rank_keep_ratio < 0.0 or args.query_prefilter_wrist_rank_keep_ratio > 1.0:
        raise ValueError("--query_prefilter_wrist_rank_keep_ratio must be within [0, 1]")
    if args.support_grid_ratio < 0.0:
        raise ValueError("--support_grid_ratio must be >= 0")
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers_per_gpu must be >= 1")
    if args.depth_filter_workers <= 0:
        raise ValueError("--depth_filter_workers must be >= 1")
    if args.hardware_telemetry_interval_sec < 0.0:
        raise ValueError("--hardware_telemetry_interval_sec must be >= 0")
    unknown_override_cameras = sorted(
        camera_name
        for camera_name in args.camera_num_iters_overrides
        if camera_name not in args.camera_names
    )
    if unknown_override_cameras:
        raise ValueError(
            "--camera_num_iters contains cameras outside --camera_names: "
            + ",".join(unknown_override_cameras)
        )

    return args


def parse_gpu_ids(gpu_id: str | None) -> list[int]:
    if gpu_id is None:
        return []
    values = [item.strip() for item in gpu_id.split(",") if item.strip()]
    if not values:
        return []
    return [int(item) for item in values]


def build_worker_slots(gpu_ids: list[int], *, workers_per_gpu: int) -> list[WorkerSlot]:
    if not gpu_ids:
        return []

    gpu_slot_counts: Counter[int] = Counter()
    for gpu_id in gpu_ids:
        gpu_slot_counts[gpu_id] += workers_per_gpu

    gpu_slot_progress = {gpu_id: 0 for gpu_id in gpu_slot_counts}
    total_workers = sum(gpu_slot_counts.values())

    worker_slots: list[WorkerSlot] = []
    worker_index = 1
    for gpu_id in gpu_ids:
        for _ in range(workers_per_gpu):
            gpu_slot_progress[gpu_id] += 1
            worker_slots.append(
                WorkerSlot(
                    worker_index=worker_index,
                    total_workers=total_workers,
                    gpu_id=gpu_id,
                    gpu_slot_index=gpu_slot_progress[gpu_id],
                    gpu_slot_count=gpu_slot_counts[gpu_id],
                )
            )
            worker_index += 1
    return worker_slots


def get_gpu_memory_info(gpu_id: int) -> GpuMemoryInfo | None:
    query_cmd = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            query_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        line = result.stdout.strip().splitlines()[0]
        free_mib_str, total_mib_str = [part.strip() for part in line.split(",", maxsplit=1)]
        mib = 1024.0
        return GpuMemoryInfo(
            free_gb=float(free_mib_str) / mib,
            total_gb=float(total_mib_str) / mib,
        )
    except Exception:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
            gib = float(1024 ** 3)
            return GpuMemoryInfo(
                free_gb=free_bytes / gib,
                total_gb=total_bytes / gib,
            )
        except Exception:
            return None


def _parse_optional_float(raw: str) -> float | None:
    value = raw.strip()
    if not value or value in {"N/A", "[N/A]", "[Not Supported]"}:
        return None
    return float(value)


def resolve_telemetry_gpu_ids(
    *,
    gpu_ids: list[int],
    device: str | None,
) -> list[int]:
    if gpu_ids:
        return list(dict.fromkeys(int(gpu_id) for gpu_id in gpu_ids))
    if device is None or not str(device).startswith("cuda"):
        return []
    if str(device) == "cuda":
        return [0]
    try:
        return [int(str(device).split(":", maxsplit=1)[1])]
    except Exception:
        return [0]


def get_gpu_static_info(gpu_id: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"gpu_id": int(gpu_id)}
    query_cmd = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            query_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        line = result.stdout.strip().splitlines()[0]
        gpu_id_str, name, total_mib_str = [part.strip() for part in line.split(",", maxsplit=2)]
        payload.update(
            {
                "gpu_id": int(gpu_id_str),
                "name": name,
                "memory_total_gib": float(total_mib_str) / 1024.0,
                "probe_source": "nvidia-smi",
            }
        )
        return payload
    except Exception:
        payload["probe_source"] = "fallback"
        try:
            payload["name"] = torch.cuda.get_device_name(gpu_id)
            _free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
            payload["memory_total_gib"] = float(total_bytes) / float(1024 ** 3)
        except Exception as exc:
            payload["probe_error"] = str(exc)
        return payload


def collect_gpu_static_info(gpu_ids: list[int]) -> list[dict[str, Any]]:
    return [get_gpu_static_info(gpu_id) for gpu_id in gpu_ids]


def collect_gpu_runtime_samples(gpu_ids: list[int]) -> tuple[list[dict[str, Any]], str | None]:
    if not gpu_ids:
        return [], None

    query_cmd = [
        "nvidia-smi",
        f"--id={','.join(str(gpu_id) for gpu_id in gpu_ids)}",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            query_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        return [], str(exc)

    samples: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=6)]
        if len(parts) != 7:
            continue
        gpu_id_str, name, util_gpu_str, util_mem_str, mem_used_str, mem_total_str, power_draw_str = parts
        samples.append(
            {
                "gpu_id": int(gpu_id_str),
                "name": name,
                "utilization_gpu_pct": _parse_optional_float(util_gpu_str),
                "utilization_memory_pct": _parse_optional_float(util_mem_str),
                "memory_used_mib": _parse_optional_float(mem_used_str),
                "memory_total_mib": _parse_optional_float(mem_total_str),
                "power_draw_watts": _parse_optional_float(power_draw_str),
            }
        )
    return samples, None


def _list_sampled_block_devices() -> set[str]:
    sys_block = Path("/sys/block")
    if not sys_block.is_dir():
        return set()
    ignored_prefixes = ("loop", "ram", "dm-", "md", "sr", "fd")
    device_names = {
        path.name
        for path in sys_block.iterdir()
        if path.is_dir() and not path.name.startswith(ignored_prefixes)
    }
    return device_names


def _read_cpu_total_and_iowait_ticks() -> tuple[int, int] | None:
    proc_stat = Path("/proc/stat")
    if not proc_stat.is_file():
        return None
    for line in proc_stat.read_text(encoding="utf-8").splitlines():
        if not line.startswith("cpu "):
            continue
        fields = [int(value) for value in line.split()[1:]]
        if len(fields) < 5:
            return None
        return int(sum(fields)), int(fields[4])
    return None


def _read_disk_io_bytes(block_devices: set[str]) -> tuple[int, int] | None:
    proc_diskstats = Path("/proc/diskstats")
    if not proc_diskstats.is_file():
        return None
    read_bytes = 0
    write_bytes = 0
    for line in proc_diskstats.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 10:
            continue
        device_name = parts[2]
        if device_name not in block_devices:
            continue
        read_bytes += int(parts[5]) * 512
        write_bytes += int(parts[9]) * 512
    return read_bytes, write_bytes


def capture_cpu_io_snapshot(block_devices: set[str]) -> CpuIoSnapshot | None:
    cpu_ticks = _read_cpu_total_and_iowait_ticks()
    disk_bytes = _read_disk_io_bytes(block_devices)
    if cpu_ticks is None or disk_bytes is None:
        return None
    return CpuIoSnapshot(
        captured_at_unix=time.time(),
        cpu_total_ticks=cpu_ticks[0],
        cpu_iowait_ticks=cpu_ticks[1],
        disk_read_bytes=disk_bytes[0],
        disk_write_bytes=disk_bytes[1],
    )


def build_cpu_io_metrics(
    previous_snapshot: CpuIoSnapshot | None,
    current_snapshot: CpuIoSnapshot | None,
) -> dict[str, Any] | None:
    if previous_snapshot is None or current_snapshot is None:
        return None

    elapsed_seconds = float(
        max(
            current_snapshot.captured_at_unix - previous_snapshot.captured_at_unix,
            1e-6,
        )
    )
    delta_cpu_total = max(
        current_snapshot.cpu_total_ticks - previous_snapshot.cpu_total_ticks,
        0,
    )
    delta_cpu_iowait = max(
        current_snapshot.cpu_iowait_ticks - previous_snapshot.cpu_iowait_ticks,
        0,
    )
    delta_read_bytes = max(
        current_snapshot.disk_read_bytes - previous_snapshot.disk_read_bytes,
        0,
    )
    delta_write_bytes = max(
        current_snapshot.disk_write_bytes - previous_snapshot.disk_write_bytes,
        0,
    )
    return {
        "sample_window_seconds": elapsed_seconds,
        "cpu_iowait_pct": (
            float(100.0 * float(delta_cpu_iowait) / float(delta_cpu_total))
            if delta_cpu_total > 0
            else None
        ),
        "disk_read_bytes_per_sec": float(delta_read_bytes / elapsed_seconds),
        "disk_write_bytes_per_sec": float(delta_write_bytes / elapsed_seconds),
    }


class HardwareTelemetrySampler:
    def __init__(
        self,
        *,
        telemetry_writer: BatchTelemetryWriter,
        interval_sec: float,
        host_name: str | None,
        gpu_ids: list[int],
    ) -> None:
        self._telemetry_writer = telemetry_writer
        self._interval_sec = float(interval_sec)
        self._host_name = host_name
        self._gpu_ids = list(gpu_ids)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample_index = 0
        self._block_devices = _list_sampled_block_devices()
        self._previous_cpu_io_snapshot = capture_cpu_io_snapshot(self._block_devices)

    def start(self) -> None:
        if self._interval_sec <= 0.0:
            return
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="traceforge-hardware-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=max(self._interval_sec, 1.0) + 2.0)
        self._thread = None

    def _run(self) -> None:
        self._record_sample(include_cpu_io_metrics=False)
        while not self._stop_event.wait(self._interval_sec):
            self._record_sample(include_cpu_io_metrics=True)

    def _record_sample(self, *, include_cpu_io_metrics: bool) -> None:
        current_snapshot = capture_cpu_io_snapshot(self._block_devices)
        cpu_io_metrics = None
        if include_cpu_io_metrics:
            cpu_io_metrics = build_cpu_io_metrics(
                self._previous_cpu_io_snapshot,
                current_snapshot,
            )
        gpu_samples, gpu_sample_error = collect_gpu_runtime_samples(self._gpu_ids)
        self._telemetry_writer.record_hardware_sample(
            {
                "sample_index": int(self._sample_index),
                "captured_at_unix": float(time.time()),
                "host_name": self._host_name,
                "gpu_samples": gpu_samples,
                "gpu_sample_error": gpu_sample_error,
                "cpu_io_metrics": cpu_io_metrics,
            }
        )
        self._sample_index += 1
        self._previous_cpu_io_snapshot = current_snapshot


def filter_gpu_ids_by_free_memory(
    gpu_ids: list[int],
    *,
    min_free_gpu_mem_gb: float,
) -> tuple[list[int], dict[int, GpuMemoryInfo | None], list[int]]:
    gpu_memory: dict[int, GpuMemoryInfo | None] = {}
    available_gpu_ids: list[int] = []
    skipped_gpu_ids: list[int] = []

    for gpu_id in gpu_ids:
        mem_info = get_gpu_memory_info(gpu_id)
        gpu_memory[gpu_id] = mem_info
        if mem_info is None:
            available_gpu_ids.append(gpu_id)
            continue
        if min_free_gpu_mem_gb > 0 and mem_info.free_gb < min_free_gpu_mem_gb:
            skipped_gpu_ids.append(gpu_id)
            continue
        available_gpu_ids.append(gpu_id)

    return available_gpu_ids, gpu_memory, skipped_gpu_ids


def is_retryable_cuda_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True

    message = str(exc).lower()
    retryable_markers = (
        "cuda out of memory",
        "outofmemoryerror",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
        "cuda error: all cuda-capable devices are busy or unavailable",
        "cuda-capable device(s) is/are busy or unavailable",
        "device busy",
        "device unavailable",
        "illegal memory access",
        "lazy wrapper should be called at most once",
    )
    return any(marker in message for marker in retryable_markers)


def wait_for_gpu_recovery(
    *,
    gpu_id: int,
    args: argparse.Namespace,
    stop_event,
) -> bool:
    threshold = args.min_free_gpu_mem_gb
    poll_sec = max(args.gpu_recovery_poll_sec, 1.0)

    if threshold <= 0:
        return not stop_event.is_set()

    wait_logged = False
    last_log_time = 0.0
    while not stop_event.is_set():
        mem_info = get_gpu_memory_info(gpu_id)
        if mem_info is None:
            if not wait_logged:
                logger.warning(
                    f"[GPU {gpu_id}] free-memory probe unavailable during recovery; "
                    "attempting startup without filtering."
                )
            return True

        if mem_info.free_gb >= threshold:
            if wait_logged:
                logger.info(
                    f"[GPU {gpu_id}] recovered: free_mem={mem_info.free_gb:.1f} GiB "
                    f">= {threshold:.1f} GiB"
                )
            return True

        now = time.time()
        if not wait_logged or (now - last_log_time) >= max(poll_sec * 4, 120.0):
            logger.warning(
                f"[GPU {gpu_id}] waiting for recovery: "
                f"free_mem={mem_info.free_gb:.1f} GiB < {threshold:.1f} GiB"
            )
            wait_logged = True
            last_log_time = now
        stop_event.wait(poll_sec)

    return False


def safe_empty_cuda_cache(context: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning(f"{context}: torch.cuda.empty_cache() failed: {exc}")


def unload_tracker_model(model_3dtracker):
    if model_3dtracker is not None:
        del model_3dtracker
    safe_empty_cuda_cache("unload_tracker_model")
    return None


def mark_task_completed(remaining_tasks) -> None:
    if remaining_tasks is None:
        return
    with remaining_tasks.get_lock():
        remaining_tasks.value -= 1


def warm_up_cuda_linalg(device: str) -> None:
    if not device.startswith("cuda"):
        return
    if device in _CUDA_LINALG_WARMED_DEVICES:
        return

    with _CUDA_LINALG_WARMUP_LOCK:
        if device in _CUDA_LINALG_WARMED_DEVICES:
            return

        logger.info(
            f"[{device}] warming CUDA linalg to avoid threaded lazy-load races"
        )
        eye = torch.eye(4, device=device, dtype=torch.float32)
        inv_eye = torch.linalg.inv(eye)
        torch.cuda.synchronize(device)
        del eye
        del inv_eye
        _CUDA_LINALG_WARMED_DEVICES.add(device)


def resolve_output_root(args: argparse.Namespace) -> Path | None:
    if args.out_dir is None:
        return None
    out_dir = args.out_dir.strip()
    if not out_dir:
        return None
    return Path(out_dir).resolve()


def resolve_episode_output_dir(
    episode_dir: Path,
    *,
    args: argparse.Namespace,
    out_root: Path | None,
) -> Path:
    if out_root is not None:
        return out_root / episode_dir.name
    return episode_dir / args.trajectory_dirname


def describe_output_target(args: argparse.Namespace, out_root: Path | None) -> str:
    if out_root is not None:
        return str(out_root)
    return f"<episode>/{args.trajectory_dirname}"


def _has_files(dir_path: Path, suffixes: tuple[str, ...]) -> bool:
    return dir_path.is_dir() and any(
        path.is_file() and path.suffix.lower() in suffixes for path in dir_path.iterdir()
    )


def find_valid_episodes(base_path: Path, camera_names: list[str], geom_name: str) -> list[Path]:
    episodes: list[Path] = []
    for episode_dir in sorted(base_path.iterdir()):
        if not episode_dir.is_dir():
            continue
        if not (episode_dir.name.startswith("episode_") or episode_dir.name.isdigit()):
            continue

        geom_path = episode_dir / geom_name
        if not geom_path.is_file():
            continue

        has_any_camera = False
        for camera_name in camera_names:
            rgb_dir = episode_dir / "rgb" / camera_name
            depth_dir = episode_dir / "depth" / camera_name
            if _has_files(rgb_dir, (".png", ".jpg", ".jpeg")) and _has_files(depth_dir, (".npy", ".png")):
                has_any_camera = True
                break

        if has_any_camera:
            episodes.append(episode_dir)
    return episodes


def camera_output_complete(out_episode_dir: Path, camera_name: str) -> bool:
    camera_dir = out_episode_dir / camera_name
    return is_traceforge_output_complete(camera_dir)


def copy_episode_lang(episode_dir: Path, out_episode_dir: Path) -> None:
    lang_path = episode_dir / "lang.txt"
    if not lang_path.is_file():
        return
    out_episode_dir.mkdir(parents=True, exist_ok=True)
    target = out_episode_dir / "lang.txt"
    target.write_text(lang_path.read_text(encoding="utf-8"), encoding="utf-8")


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
    tmp_path = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def ensure_episode_query_frame_schedule(
    *,
    episode_dir: Path,
    out_episode_dir: Path,
    args: argparse.Namespace,
) -> Path:
    geom_path = episode_dir / args.external_geom_name
    episode_fps = _read_episode_fps(geom_path, args.fallback_episode_fps)
    schedule_camera_names = list(
        getattr(args, "shared_schedule_camera_names", getattr(args, "camera_names", []))
    )

    camera_raw_frame_counts: dict[str, int] = {}
    for camera_name in schedule_camera_names:
        rgb_dir = episode_dir / "rgb" / camera_name
        depth_dir = episode_dir / "depth" / camera_name
        if not _has_files(rgb_dir, (".png", ".jpg", ".jpeg")):
            continue
        if not _has_files(depth_dir, (".npy", ".png")):
            continue
        rgb_count = _count_files(rgb_dir, (".png", ".jpg", ".jpeg"))
        depth_count = _count_files(depth_dir, (".npy", ".png"))
        geom_count = _read_geom_frame_count(geom_path, camera_name)
        camera_raw_frame_counts[camera_name] = min(rgb_count, depth_count, geom_count)

    if not camera_raw_frame_counts:
        raise ValueError(f"No schedulable cameras found under {episode_dir}")

    common_raw_frame_count = min(camera_raw_frame_counts.values())
    candidate_source_frame_indices = build_candidate_source_frame_indices(
        common_raw_frame_count,
        stride=int(args.fps),
        max_num_frames=args.max_num_frames,
    )
    if candidate_source_frame_indices.size == 0:
        raise ValueError(
            f"{episode_dir.name}: no candidate frames remain after stride/max_num_frames filtering"
        )
    candidate_local_indices = np.arange(candidate_source_frame_indices.size, dtype=np.int32)
    candidate_local_indices, short_tail_local_indices = (
        filter_query_local_indices_by_remaining_frames(
            candidate_local_indices,
            video_length=int(candidate_source_frame_indices.size),
        )
    )
    short_tail_source_indices = candidate_source_frame_indices[short_tail_local_indices]
    candidate_source_frame_indices = candidate_source_frame_indices[candidate_local_indices]
    if short_tail_source_indices.size > 0:
        logger.info(
            f"{episode_dir.name}: dropped {short_tail_source_indices.size} candidate query frames "
            "because <= 8 frames remain to the end (inclusive)"
        )
    if candidate_source_frame_indices.size == 0:
        raise ValueError(
            f"{episode_dir.name}: no candidate frames remain after "
            "stride/max_num_frames/tail filtering"
        )

    schedule_spec = {
        "version": _QUERY_FRAME_SCHEDULE_VERSION,
        "external_geom_name": args.external_geom_name,
        "camera_names": schedule_camera_names,
        "episode_fps": float(episode_fps),
        "keyframes_per_sec_min": int(args.keyframes_per_sec_min),
        "keyframes_per_sec_max": int(args.keyframes_per_sec_max),
        "base_seed": int(args.keyframe_seed),
        "load_stride": int(args.fps),
        "max_num_frames": int(args.max_num_frames),
        "common_raw_frame_count": int(common_raw_frame_count),
    }
    spec_hash = _schedule_spec_hash(schedule_spec)
    schedule_dir = out_episode_dir / _QUERY_FRAME_SHARED_DIRNAME
    schedule_path = schedule_dir / f"query_frame_schedule_v{_QUERY_FRAME_SCHEDULE_VERSION}_{spec_hash}.json"
    if schedule_path.is_file():
        return schedule_path

    derived_seed = _derive_episode_schedule_seed(
        base_seed=int(args.keyframe_seed),
        episode_name=episode_dir.name,
        spec_hash=spec_hash,
    )
    query_frame_source_indices = sample_query_source_indices_per_second(
        candidate_source_frame_indices,
        episode_fps=episode_fps,
        keyframes_per_sec_min=int(args.keyframes_per_sec_min),
        keyframes_per_sec_max=int(args.keyframes_per_sec_max),
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
            "dropped_short_tail_source_indices": short_tail_source_indices.tolist(),
            "query_frame_source_indices": query_frame_source_indices.tolist(),
        },
    )
    logger.info(
        f"{episode_dir.name}: prepared shared query-frame schedule "
        f"({len(query_frame_source_indices)} frames, fps={episode_fps:.3f}, "
        f"kps={args.keyframes_per_sec_min}~{args.keyframes_per_sec_max})"
    )
    return schedule_path


def build_camera_tasks(
    episodes: list[Path],
    *,
    args: argparse.Namespace,
    out_dir: Path | None,
) -> list[CameraTask]:
    pending: list[tuple[Path, Path, str, Path | None]] = []

    for episode_dir in episodes:
        out_episode_dir = resolve_episode_output_dir(
            episode_dir,
            args=args,
            out_root=out_dir,
        )
        schedule_path: Path | None = None
        for camera_name in args.camera_names:
            rgb_dir = episode_dir / "rgb" / camera_name
            depth_dir = episode_dir / "depth" / camera_name
            if not _has_files(rgb_dir, (".png", ".jpg", ".jpeg")):
                logger.warning(f"{episode_dir.name}/{camera_name}: skip, RGB missing")
                continue
            if not _has_files(depth_dir, (".npy", ".png")):
                logger.warning(f"{episode_dir.name}/{camera_name}: skip, depth missing")
                continue
            if args.skip_existing and camera_output_complete(out_episode_dir, camera_name):
                logger.info(f"{episode_dir.name}/{camera_name}: skip_existing")
                continue
            if schedule_path is None:
                schedule_path = ensure_episode_query_frame_schedule(
                    episode_dir=episode_dir,
                    out_episode_dir=out_episode_dir,
                    args=args,
                )
            pending.append((episode_dir, out_episode_dir, camera_name, schedule_path))

    total_tasks = len(pending)
    tasks: list[CameraTask] = []
    for task_index, (episode_dir, out_episode_dir, camera_name, schedule_path) in enumerate(pending, start=1):
        tasks.append(
            CameraTask(
                task_index=task_index,
                total_tasks=total_tasks,
                episode_dir=episode_dir,
                out_episode_dir=out_episode_dir,
                camera_name=camera_name,
                query_frame_schedule_path=schedule_path,
            )
        )
    return tasks


def build_camera_args(
    base_args: argparse.Namespace,
    episode_dir: Path,
    camera_name: str,
    *,
    query_frame_schedule_path: Path | None,
) -> argparse.Namespace:
    camera_args = copy.deepcopy(base_args)
    camera_args.mask_dir = None
    camera_args.camera_name = camera_name
    camera_args.traj_filter_profile = resolve_traj_filter_profile(
        camera_name,
        base_args.traj_filter_profile,
    )
    camera_args.num_iters = resolve_camera_num_iters(
        base_num_iters=int(base_args.num_iters),
        camera_name=camera_name,
        overrides=getattr(base_args, "camera_num_iters_overrides", None),
    )
    camera_args.external_geom_npz = str(episode_dir / base_args.external_geom_name)
    camera_args.query_frame_schedule_path = (
        str(query_frame_schedule_path) if query_frame_schedule_path is not None else None
    )
    return camera_args


def save_result(
    *,
    episode_dir: Path,
    out_episode_dir: Path,
    camera_name: str,
    result: dict,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return infer.save_structured_data(
        video_name=camera_name,
        output_dir=str(out_episode_dir),
        video_tensor=result["video_tensor"],
        depths=result["depths"],
        coords=result["coords"],
        visibs=result["visibs"],
        intrinsics=result["intrinsics"],
        extrinsics=result["extrinsics"],
        query_points_per_frame=result["query_points_per_frame"],
        original_filenames=result["original_filenames"],
        query_frame_results=result.get("query_frame_results"),
        future_len=args.future_len,
        grid_size=args.grid_size,
        filter_args=args,
        full_video_tensor=result["full_video_tensor"],
        full_depths=result["full_depths"],
        full_intrinsics=result["full_intrinsics"],
        full_extrinsics=result["full_extrinsics"],
        depth_conf=result["depth_conf"],
        video_source_path=str(episode_dir / "rgb" / camera_name),
        depth_source_path=str(episode_dir / "depth" / camera_name),
        source_frame_indices=result["source_frame_indices"],
        query_frame_metadata=result.get("query_frame_metadata"),
    )


def run_camera_task(
    *,
    task: CameraTask,
    args: argparse.Namespace,
    model_3dtracker,
    gpu_id: int | None = None,
    worker_slot: WorkerSlot | None = None,
    telemetry_writer: BatchTelemetryWriter | None = None,
) -> tuple[bool, bool]:
    if args.copy_lang and not (task.out_episode_dir / "lang.txt").is_file():
        copy_episode_lang(task.episode_dir, task.out_episode_dir)

    camera_args = build_camera_args(
        args,
        task.episode_dir,
        task.camera_name,
        query_frame_schedule_path=task.query_frame_schedule_path,
    )
    logger.info(
        f"{task.episode_dir.name}/{task.camera_name}: run "
        f"(device={camera_args.device}, depth_pose_method={camera_args.depth_pose_method})"
    )

    started_at_unix = time.time()
    process_seconds: float | None = None
    save_seconds: float | None = None
    query_frame_count: int | None = None
    status = "failed"
    retryable_cuda_error = False
    error_message: str | None = None
    result: dict[str, Any] | None = None
    save_artifacts: dict[str, Any] | None = None
    try:
        model_depth_pose = infer.video_depth_pose_dict[camera_args.depth_pose_method](camera_args)
        process_start = time.perf_counter()
        result = infer.process_single_video(
            str(task.episode_dir / "rgb" / task.camera_name),
            str(task.episode_dir / "depth" / task.camera_name),
            camera_args,
            model_3dtracker,
            model_depth_pose,
        )
        process_seconds = time.perf_counter() - process_start
        query_frame_results = result.get("query_frame_results") or {}
        query_frame_count = int(len(query_frame_results))
        save_start = time.perf_counter()
        save_artifacts = save_result(
            episode_dir=task.episode_dir,
            out_episode_dir=task.out_episode_dir,
            camera_name=task.camera_name,
            result=result,
            args=camera_args,
        )
        save_seconds = time.perf_counter() - save_start
        status = "success"
        return True, False
    except Exception as exc:
        if is_retryable_cuda_error(exc):
            retryable_cuda_error = True
            error_message = str(exc)
            logger.exception(
                f"{task.episode_dir.name}/{task.camera_name} hit retryable CUDA failure: {exc}"
            )
            return False, True
        error_message = str(exc)
        logger.exception(f"{task.episode_dir.name}/{task.camera_name} failed: {exc}")
        return False, False
    finally:
        finished_at_unix = time.time()
        if telemetry_writer is not None:
            telemetry_writer.record_task(
                build_camera_task_metric_record(
                    task=task,
                    gpu_id=gpu_id,
                    args=camera_args,
                    worker_label=worker_slot.label if worker_slot is not None else None,
                    worker_index=worker_slot.worker_index if worker_slot is not None else None,
                    gpu_slot_index=worker_slot.gpu_slot_index if worker_slot is not None else None,
                    gpu_slot_count=worker_slot.gpu_slot_count if worker_slot is not None else None,
                    query_frame_count=query_frame_count,
                    process_seconds=process_seconds,
                    save_seconds=save_seconds,
                    started_at_unix=started_at_unix,
                    finished_at_unix=finished_at_unix,
                    status=status,
                    retryable_cuda_error=retryable_cuda_error,
                    error_message=error_message,
                )
            )
            if bool(getattr(camera_args, "collect_profile_stats", False)):
                telemetry_writer.record_task_profile(
                    build_camera_task_profile_record(
                        task=task,
                        gpu_id=gpu_id,
                        args=camera_args,
                        worker_label=worker_slot.label if worker_slot is not None else None,
                        worker_index=worker_slot.worker_index if worker_slot is not None else None,
                        gpu_slot_index=worker_slot.gpu_slot_index if worker_slot is not None else None,
                        gpu_slot_count=worker_slot.gpu_slot_count if worker_slot is not None else None,
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
                    )
                )
        if "model_depth_pose" in locals():
            del model_depth_pose
        safe_empty_cuda_cache(
            f"{task.episode_dir.name}/{task.camera_name}: run_camera_task cleanup"
        )


def run_episode(
    *,
    episode_dir: Path,
    out_episode_dir: Path,
    args: argparse.Namespace,
    model_3dtracker,
    telemetry_writer: BatchTelemetryWriter | None = None,
) -> tuple[int, int]:
    success_count = 0
    fail_count = 0

    pending_cameras: list[str] = []
    for camera_name in args.camera_names:
        rgb_dir = episode_dir / "rgb" / camera_name
        depth_dir = episode_dir / "depth" / camera_name
        if not _has_files(rgb_dir, (".png", ".jpg", ".jpeg")):
            logger.warning(f"{episode_dir.name}/{camera_name}: skip, RGB missing")
            continue
        if not _has_files(depth_dir, (".npy", ".png")):
            logger.warning(f"{episode_dir.name}/{camera_name}: skip, depth missing")
            continue
        if args.skip_existing and camera_output_complete(out_episode_dir, camera_name):
            logger.info(f"{episode_dir.name}/{camera_name}: skip_existing")
            continue
        pending_cameras.append(camera_name)

    schedule_path: Path | None = None
    if pending_cameras:
        schedule_path = ensure_episode_query_frame_schedule(
            episode_dir=episode_dir,
            out_episode_dir=out_episode_dir,
            args=args,
        )

    tasks = [
        CameraTask(
            task_index=idx,
            total_tasks=len(pending_cameras),
            episode_dir=episode_dir,
            out_episode_dir=out_episode_dir,
            camera_name=camera_name,
            query_frame_schedule_path=schedule_path,
        )
        for idx, camera_name in enumerate(pending_cameras, start=1)
    ]
    if args.copy_lang and not tasks:
        copy_episode_lang(episode_dir, out_episode_dir)

    for task in tasks:
        ok, _retire_worker = run_camera_task(
            task=task,
            args=args,
            model_3dtracker=model_3dtracker,
            telemetry_writer=telemetry_writer,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def process_camera_tasks_on_gpu(
    *,
    worker_slot: WorkerSlot,
    task_queue,
    args: argparse.Namespace,
    stop_event,
    remaining_tasks=None,
    telemetry_writer: BatchTelemetryWriter | None = None,
) -> tuple[int, int, float]:
    worker_args = copy.deepcopy(args)
    worker_args.device = worker_slot.device
    worker_label = worker_slot.label

    worker_start = time.time()
    total_camera_success = 0
    total_camera_fail = 0
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
                    model_3dtracker = infer.load_model(worker_args.checkpoint).to(worker_args.device)
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
                    f"[{task.task_index}/{task.total_tasks}] {task.episode_dir.name}/{task.camera_name}"
                )
                ok, retire_worker = run_camera_task(
                    task=task,
                    args=worker_args,
                    model_3dtracker=model_3dtracker,
                    gpu_id=worker_slot.gpu_id,
                    worker_slot=worker_slot,
                    telemetry_writer=telemetry_writer,
                )
                if ok:
                    total_camera_success += 1
                    mark_task_completed(remaining_tasks)
                elif retire_worker:
                    task_queue.put(task)
                    model_3dtracker = unload_tracker_model(model_3dtracker)
                    logger.warning(
                        f"[{worker_label}] re-queued {task.episode_dir.name}/{task.camera_name} "
                        "after retryable CUDA failure; waiting for GPU recovery."
                    )
                else:
                    total_camera_fail += 1
                    mark_task_completed(remaining_tasks)
            finally:
                task_queue.task_done()
    finally:
        model_3dtracker = unload_tracker_model(model_3dtracker)

    elapsed = time.time() - worker_start
    logger.info(
        f"[{worker_label}] dynamic worker done in {elapsed/60:.1f} min "
        f"(camera_success={total_camera_success}, camera_fail={total_camera_fail})"
    )
    return total_camera_success, total_camera_fail, elapsed


def process_camera_tasks_on_gpu_entrypoint(
    *,
    worker_slot: WorkerSlot,
    task_queue,
    args: argparse.Namespace,
    stop_event,
    remaining_tasks,
    result_queue,
    telemetry_writer: BatchTelemetryWriter | None = None,
) -> None:
    try:
        success_count, fail_count, elapsed = process_camera_tasks_on_gpu(
            worker_slot=worker_slot,
            task_queue=task_queue,
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


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_path).resolve()
    out_dir = resolve_output_root(args)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
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
        available_gpu_ids, gpu_memory, skipped_gpu_ids = filter_gpu_ids_by_free_memory(
            probe_gpu_ids,
            min_free_gpu_mem_gb=args.min_free_gpu_mem_gb,
        )
        if args.min_free_gpu_mem_gb > 0 and not available_gpu_ids:
            logger.warning(
                "No GPUs currently pass the free-memory filter; "
                "dynamic workers will wait for recovery."
            )

    episodes = find_valid_episodes(base_path, args.camera_names, args.external_geom_name)
    if not episodes:
        logger.error(f"No valid episodes found under {base_path}")
        return

    if args.episode_name is not None and args.episode_names_file is not None:
        raise ValueError("--episode_name and --episode_names_file are mutually exclusive")

    if args.episode_name is not None:
        episodes = [episode for episode in episodes if episode.name == args.episode_name]
        if not episodes:
            logger.error(f"Episode not found: {args.episode_name}")
            return
    elif args.episode_names_file is not None:
        episode_names_path = Path(args.episode_names_file).expanduser()
        requested_names = [
            line.strip()
            for line in episode_names_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not requested_names:
            logger.error(f"No episode names found in {episode_names_path}")
            return
        requested_name_set = set(requested_names)
        episodes = [episode for episode in episodes if episode.name in requested_name_set]
        if not episodes:
            logger.error(f"No requested episodes found under {base_path} from {episode_names_path}")
            return
        missing_names = [name for name in requested_names if name not in {episode.name for episode in episodes}]
        if missing_names:
            logger.warning(
                f"Ignoring {len(missing_names)} unknown episode names from {episode_names_path}: "
                f"{missing_names[:5]}{'...' if len(missing_names) > 5 else ''}"
            )
    elif args.max_episodes is not None and args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    episodes_for_run = episodes

    logger.info("=" * 80)
    logger.info("Press-one-button demo batch inference")
    logger.info(f"base_path={base_path}")
    logger.info(f"out_dir={describe_output_target(args, out_dir)}")
    logger.info(f"cameras={args.camera_names}")
    logger.info(
        f"episodes={len(episodes_for_run)}"
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
        f"keyframes_per_sec={args.keyframes_per_sec_min}~{args.keyframes_per_sec_max}, "
        f"future_len={args.future_len}, grid_size={args.grid_size}, "
        f"query_prefilter={args.query_prefilter_mode}, support_grid_ratio={args.support_grid_ratio}, "
        f"load_stride={args.fps}, depth_filter_workers={args.depth_filter_workers}"
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

    dynamic_tasks: list[CameraTask] | None = None
    if gpu_ids:
        dynamic_tasks = build_camera_tasks(
            episodes_for_run,
            args=args,
            out_dir=out_dir,
        )
        logger.info(f"dynamic camera tasks={len(dynamic_tasks)}")

    if args.dry_run:
        if dynamic_tasks is not None:
            for task in dynamic_tasks:
                logger.info(
                    f"[dry_run {task.task_index:03d}/{task.total_tasks:03d}] "
                    f"{task.episode_dir.name}/{task.camera_name} -> {task.out_episode_dir}"
                )
        else:
            for idx, episode in enumerate(episodes_for_run, start=1):
                logger.info(f"[dry_run {idx:03d}/{len(episodes_for_run):03d}] {episode}")
        return

    total_camera_success = 0
    total_camera_fail = 0
    batch_start = time.time()
    camera_task_count = 0

    try:
        if gpu_ids:
            assert dynamic_tasks is not None
            camera_task_count = len(dynamic_tasks)
            if not dynamic_tasks:
                logger.info("No pending camera tasks after filtering.")
                logger.info("=" * 80)
                return
            if args.copy_lang:
                for episode_dir in episodes_for_run:
                    copy_episode_lang(
                        episode_dir,
                        resolve_episode_output_dir(
                            episode_dir,
                            args=args,
                            out_root=out_dir,
                        ),
                    )

            mp_ctx = mp.get_context("spawn")
            if out_dir is not None:
                telemetry_writer = BatchTelemetryWriter(
                    out_dir,
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
            for task in dynamic_tasks:
                task_queue.put(task)
            stop_event = mp_ctx.Event()
            remaining_tasks = mp_ctx.Value("i", len(dynamic_tasks))
            result_queue = mp_ctx.Queue()
            worker_processes: list[tuple[WorkerSlot, mp.Process]] = []

            for worker_slot in worker_slots:
                process = mp_ctx.Process(
                    target=process_camera_tasks_on_gpu_entrypoint,
                    kwargs={
                        "worker_slot": worker_slot,
                        "task_queue": task_queue,
                        "args": args,
                        "stop_event": stop_event,
                        "remaining_tasks": remaining_tasks,
                        "result_queue": result_queue,
                        "telemetry_writer": telemetry_writer,
                    },
                    name=f"traceforge-{worker_slot.gpu_id}-{worker_slot.gpu_slot_index}",
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
                        logger.error("A dynamic GPU worker exited before all tasks completed.")
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
                    total_camera_success += worker_result.success_count
                    total_camera_fail += worker_result.fail_count
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
                total_camera_fail += remaining_task_count
                logger.error(
                    f"Dynamic scheduler left {remaining_task_count} camera tasks unprocessed."
                )
        else:
            if out_dir is not None:
                telemetry_writer = BatchTelemetryWriter(
                    out_dir,
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
            model_3dtracker = infer.load_model(args.checkpoint).to(args.device)
            try:
                for idx, episode_dir in enumerate(episodes_for_run, start=1):
                    logger.info(f"[{idx}/{len(episodes_for_run)}] episode={episode_dir.name}")
                    out_episode_dir = resolve_episode_output_dir(
                        episode_dir,
                        args=args,
                        out_root=out_dir,
                    )
                    success_count, fail_count = run_episode(
                        episode_dir=episode_dir,
                        out_episode_dir=out_episode_dir,
                        args=args,
                        model_3dtracker=model_3dtracker,
                        telemetry_writer=telemetry_writer,
                    )
                    total_camera_success += success_count
                    total_camera_fail += fail_count
                camera_task_count = total_camera_success + total_camera_fail
            finally:
                del model_3dtracker
                safe_empty_cuda_cache("single_gpu_main cleanup")
    finally:
        if hardware_sampler is not None:
            hardware_sampler.stop()

    wall_clock_seconds = time.time() - batch_start
    if telemetry_writer is not None:
        telemetry_writer.write_summary(
            build_batch_run_summary(
                args=args,
                base_path=base_path,
                out_dir=out_dir,
                gpu_ids=gpu_ids,
                telemetry_gpu_ids=telemetry_gpu_ids,
                host_name=host_name,
                gpu_info=gpu_info,
                worker_slot_count=len(worker_slots),
                episode_count=len(episodes_for_run),
                camera_task_count=camera_task_count,
                total_camera_success=total_camera_success,
                total_camera_fail=total_camera_fail,
                wall_clock_seconds=wall_clock_seconds,
            )
        )

    logger.info("=" * 80)
    if gpu_ids:
        logger.info(
            f"Done dynamic gpu mode. camera_success={total_camera_success}, "
            f"camera_fail={total_camera_fail}"
        )
    else:
        logger.info(
            f"Done. camera_success={total_camera_success}, camera_fail={total_camera_fail}"
        )
    logger.info(f"wall_clock_seconds={wall_clock_seconds:.3f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
