#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
import batch_infer_press_one_button_demo as batch_infer
from utils.traj_filter_utils import compute_accessed_high_volatility_mask, resolve_traj_filter_config


_SCAN_MANIFEST_BASENAME = "empty_sample_scan_manifest.json"
_RESULTS_JSONL_BASENAME = "repair_results.jsonl"
_SUMMARY_JSON_BASENAME = "repair_summary.json"


@dataclass(frozen=True)
class EmptySampleRecord:
    episode_name: str
    episode_dir: str
    out_episode_dir: str
    camera_name: str
    sample_path: str
    query_frame_local: int
    query_frame_source: int
    segment_len: int
    future_len: int | None
    track_count: int
    before_valid_count: int


@dataclass(frozen=True)
class RepairTask:
    task_index: int
    total_tasks: int
    episode_name: str
    episode_dir: str
    out_episode_dir: str
    camera_name: str
    query_frames_local: tuple[int, ...]
    query_frames_source: tuple[int, ...]
    sample_paths: tuple[str, ...]
    schedule_path: str


def _timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def _load_scene_meta(camera_output_dir: Path) -> dict[str, Any]:
    meta_path = camera_output_dir / "scene_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing scene_meta.json: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"scene_meta.json must contain a JSON object: {meta_path}")
    return payload


def _resolve_runtime_root(base_path: Path, requested: str | None, dirname: str, run_tag: str) -> Path:
    if requested is not None and requested.strip():
        return Path(requested).expanduser().resolve()
    return (base_path / dirname / run_tag).resolve()


def _load_requested_episode_names(args) -> list[str] | None:
    if args.episode_name is not None and args.episode_names_file is not None:
        raise ValueError("--episode_name and --episode_names_file are mutually exclusive")
    if args.episode_name is not None:
        return [str(args.episode_name)]
    if args.episode_names_file is None:
        return None

    episode_names_path = Path(args.episode_names_file).expanduser()
    requested_names = [
        line.strip()
        for line in episode_names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not requested_names:
        raise ValueError(f"No episode names found in {episode_names_path}")
    return requested_names


def resolve_target_episodes(base_path: Path, args) -> list[Path]:
    episodes = batch_infer.find_valid_episodes(base_path, args.camera_names, args.external_geom_name)
    if not episodes:
        return []

    requested_names = _load_requested_episode_names(args)
    if requested_names is not None:
        requested_name_set = set(requested_names)
        episodes = [episode for episode in episodes if episode.name in requested_name_set]
        if not episodes:
            return []
        missing_names = [name for name in requested_names if name not in {episode.name for episode in episodes}]
        if missing_names:
            logger.warning(
                f"Ignoring {len(missing_names)} unknown episode names: "
                f"{missing_names[:5]}{'...' if len(missing_names) > 5 else ''}"
            )
    elif args.max_episodes is not None and args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    return episodes


def scan_empty_sample_records(
    *,
    base_path: Path,
    args,
    out_root: Path | None,
    episodes: list[Path] | None = None,
) -> list[EmptySampleRecord]:
    if episodes is None:
        episodes = resolve_target_episodes(base_path, args)

    records: list[EmptySampleRecord] = []
    for episode_dir in episodes:
        out_episode_dir = batch_infer.resolve_episode_output_dir(
            episode_dir,
            args=args,
            out_root=out_root,
        )
        for camera_name in args.camera_names:
            camera_output_dir = out_episode_dir / camera_name
            samples_dir = camera_output_dir / "samples"
            if not samples_dir.is_dir():
                continue

            scene_meta = _load_scene_meta(camera_output_dir)
            source_frame_indices = scene_meta.get("source_frame_indices") or []
            future_len_raw = scene_meta.get("future_len")
            future_len = None if future_len_raw is None else int(future_len_raw)

            for sample_path in sorted(samples_dir.glob("*.npz")):
                with np.load(sample_path, allow_pickle=False) as data:
                    if "traj_valid_mask" not in data:
                        continue
                    traj_valid_mask = np.asarray(data["traj_valid_mask"]).astype(bool, copy=False)
                    before_valid_count = int(np.count_nonzero(traj_valid_mask))
                    if before_valid_count > 0:
                        continue

                    query_frame_local = int(np.asarray(data["query_frame_index"]).reshape(-1)[0])
                    if query_frame_local < len(source_frame_indices):
                        query_frame_source = int(source_frame_indices[query_frame_local])
                    else:
                        query_frame_source = query_frame_local
                    segment_len = int(np.asarray(data["segment_frame_indices"]).reshape(-1).shape[0])
                    records.append(
                        EmptySampleRecord(
                            episode_name=episode_dir.name,
                            episode_dir=str(episode_dir.resolve()),
                            out_episode_dir=str(out_episode_dir.resolve()),
                            camera_name=camera_name,
                            sample_path=str(sample_path.resolve()),
                            query_frame_local=query_frame_local,
                            query_frame_source=query_frame_source,
                            segment_len=segment_len,
                            future_len=future_len,
                            track_count=int(traj_valid_mask.size),
                            before_valid_count=before_valid_count,
                        )
                    )

    records.sort(key=lambda item: (item.episode_name, item.camera_name, item.query_frame_local))
    return records


def build_backup_path(sample_path: Path, *, mirror_root: Path, backup_root: Path) -> Path:
    sample_path = sample_path.resolve()
    mirror_root = mirror_root.resolve()
    try:
        relative_path = sample_path.relative_to(mirror_root)
    except ValueError as exc:
        raise ValueError(f"{sample_path} is not under mirror_root={mirror_root}") from exc
    return (backup_root / relative_path).resolve()


def _build_schedule_payload(task_records: list[EmptySampleRecord]) -> dict[str, Any]:
    first_record = task_records[0]
    return {
        "version": "empty_sample_repair_v1",
        "repair_mode": "empty_sample_only",
        "episode_name": first_record.episode_name,
        "camera_name": first_record.camera_name,
        "query_frame_local_indices": [int(record.query_frame_local) for record in task_records],
        "query_frame_source_indices": [int(record.query_frame_source) for record in task_records],
        "sample_paths": [record.sample_path for record in task_records],
    }


def build_repair_tasks(
    records: list[EmptySampleRecord],
    *,
    report_root: Path,
) -> list[RepairTask]:
    grouped: dict[tuple[str, str], list[EmptySampleRecord]] = {}
    for record in records:
        grouped.setdefault((record.episode_name, record.camera_name), []).append(record)

    schedule_dir = report_root / "schedules"
    schedule_dir.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[str, str, list[EmptySampleRecord], Path]] = []
    for (episode_name, camera_name), task_records in sorted(grouped.items()):
        task_records = sorted(task_records, key=lambda item: item.query_frame_local)
        payload = _build_schedule_payload(task_records)
        payload_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:12]
        schedule_path = schedule_dir / f"{episode_name}__{camera_name}__{payload_hash}.json"
        _write_json(schedule_path, payload)
        pending.append((episode_name, camera_name, task_records, schedule_path))

    tasks: list[RepairTask] = []
    total_tasks = len(pending)
    for task_index, (_episode_name, _camera_name, task_records, schedule_path) in enumerate(pending, start=1):
        first_record = task_records[0]
        tasks.append(
            RepairTask(
                task_index=task_index,
                total_tasks=total_tasks,
                episode_name=first_record.episode_name,
                episode_dir=first_record.episode_dir,
                out_episode_dir=first_record.out_episode_dir,
                camera_name=first_record.camera_name,
                query_frames_local=tuple(int(record.query_frame_local) for record in task_records),
                query_frames_source=tuple(int(record.query_frame_source) for record in task_records),
                sample_paths=tuple(record.sample_path for record in task_records),
                schedule_path=str(schedule_path.resolve()),
            )
        )
    return tasks


def summarize_empty_sample_records(records: list[EmptySampleRecord]) -> dict[str, Any]:
    by_camera: dict[str, int] = {}
    by_episode_camera: dict[str, int] = {}
    unique_episode_query_pairs: set[tuple[str, int]] = set()
    for record in records:
        by_camera[record.camera_name] = by_camera.get(record.camera_name, 0) + 1
        episode_camera = f"{record.episode_name}/{record.camera_name}"
        by_episode_camera[episode_camera] = by_episode_camera.get(episode_camera, 0) + 1
        unique_episode_query_pairs.add((record.episode_name, record.query_frame_local))

    return {
        "empty_sample_count": len(records),
        "episode_count": len({record.episode_name for record in records}),
        "episode_camera_group_count": len(by_episode_camera),
        "unique_episode_query_pair_count": len(unique_episode_query_pairs),
        "by_camera": dict(sorted(by_camera.items())),
    }


def write_scan_manifest(
    *,
    records: list[EmptySampleRecord],
    tasks: list[RepairTask],
    base_path: Path,
    out_root: Path | None,
    report_root: Path,
    backup_root: Path,
    manifest_path: Path,
) -> None:
    payload = {
        "base_path": str(base_path.resolve()),
        "out_root": None if out_root is None else str(out_root.resolve()),
        "report_root": str(report_root.resolve()),
        "backup_root": str(backup_root.resolve()),
        "summary": summarize_empty_sample_records(records),
        "records": [asdict(record) for record in records],
        "tasks": [
            {
                "task_index": task.task_index,
                "total_tasks": task.total_tasks,
                "episode_name": task.episode_name,
                "camera_name": task.camera_name,
                "query_frames_local": list(task.query_frames_local),
                "query_frames_source": list(task.query_frames_source),
                "schedule_path": task.schedule_path,
            }
            for task in tasks
        ],
    }
    _write_json(manifest_path, payload)


def _atomic_save_npz(target_path: Path, sample_data: dict[str, np.ndarray]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target_path.parent,
        prefix=f".{target_path.stem}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        np.savez(temp_path, **sample_data)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _ensure_sample_backed_up(sample_path: Path, backup_path: Path) -> str:
    if backup_path.exists():
        if sample_path.exists():
            raise FileExistsError(
                f"Backup already exists while original sample still exists: {backup_path}"
            )
        return "existing_backup"

    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample to back up: {sample_path}")

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(sample_path), str(backup_path))
    return "moved"


def _restore_sample_from_backup(sample_path: Path, backup_path: Path) -> None:
    if backup_path.is_file() and not sample_path.exists():
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_path, sample_path)


def _remove_sample_after_backup(sample_path: Path, backup_path: Path) -> str:
    backup_status = _ensure_sample_backed_up(sample_path, backup_path)
    if sample_path.exists():
        sample_path.unlink()
    return backup_status


def _compute_subset_high_volatility_mask(
    *,
    camera_args,
    prepared_bundles: dict[int, dict[str, object]],
    full_depths: np.ndarray,
) -> np.ndarray | None:
    filter_config = resolve_traj_filter_config(camera_args)
    use_temporal_depth_consistency = bool(filter_config["enabled"] and filter_config["use_temporal_depth_consistency"])
    use_depth_volatility_guidance = bool(
        use_temporal_depth_consistency and filter_config["use_depth_volatility_guidance"]
    )
    if not use_depth_volatility_guidance:
        return None

    full_depths_np = np.asarray(full_depths, dtype=np.float32)
    accessed_pixel_mask = infer._build_accessed_pixel_mask(
        prepared_bundles,
        image_height=int(full_depths_np.shape[1]),
        image_width=int(full_depths_np.shape[2]),
    )
    high_volatility_mask, _ = compute_accessed_high_volatility_mask(
        full_depths_np,
        accessed_pixel_mask=accessed_pixel_mask,
        min_depth=float(filter_config["min_depth"]),
        max_depth=float(filter_config["max_depth"]),
        low_percentile=float(filter_config["volatility_low_percentile"]),
        high_percentile=float(filter_config["volatility_high_percentile"]),
        mask_percentile=float(filter_config["volatility_mask_percentile"]),
    )
    return high_volatility_mask


def run_repair_task(
    *,
    task: RepairTask,
    args,
    model_3dtracker,
    backup_root: Path,
    mirror_root: Path,
    gpu_id: int | None = None,
) -> tuple[bool, bool, dict[str, Any]]:
    episode_dir = Path(task.episode_dir)
    out_episode_dir = Path(task.out_episode_dir)
    schedule_path = Path(task.schedule_path)
    camera_args = batch_infer.build_camera_args(
        args,
        episode_dir,
        task.camera_name,
        query_frame_schedule_path=schedule_path,
    )
    task_record: dict[str, Any] = {
        "task_index": int(task.task_index),
        "total_tasks": int(task.total_tasks),
        "episode_name": task.episode_name,
        "camera_name": task.camera_name,
        "gpu_id": gpu_id,
        "query_frames_local": [int(value) for value in task.query_frames_local],
        "query_frames_source": [int(value) for value in task.query_frames_source],
        "sample_paths": list(task.sample_paths),
        "schedule_path": str(schedule_path),
        "sample_results": [],
        "status": "failed",
        "error_message": None,
        "process_seconds": None,
        "save_seconds": None,
        "query_frame_count": int(len(task.query_frames_local)),
        "fixed_count": 0,
        "still_empty_count": 0,
        "removed_short_tail_count": 0,
        "failed_sample_count": 0,
    }

    started_at_unix = time.time()
    try:
        model_depth_pose = infer.video_depth_pose_dict[camera_args.depth_pose_method](camera_args)
        process_start = time.perf_counter()
        result = infer.process_single_video(
            str(episode_dir / "rgb" / task.camera_name),
            str(episode_dir / "depth" / task.camera_name),
            camera_args,
            model_3dtracker,
            model_depth_pose,
        )
        task_record["process_seconds"] = float(time.perf_counter() - process_start)

        query_frame_results = result.get("query_frame_results") or {}
        skipped_query_frames = {
            int(value)
            for value in (
                result.get("skipped_query_frame_indices_local")
                or result.get("query_frame_metadata", {}).get("skipped_short_tail_query_frame_indices_local")
                or []
            )
        }
        missing_query_frames = sorted(
            set(task.query_frames_local) - set(query_frame_results) - skipped_query_frames
        )
        if missing_query_frames:
            raise ValueError(
                f"Missing rerun results for query frames {missing_query_frames} "
                f"under {task.episode_name}/{task.camera_name}"
            )

        prepared_bundles: dict[int, dict[str, object]] = {}
        high_volatility_mask = None
        if query_frame_results:
            filter_config = resolve_traj_filter_config(camera_args)
            prepared_bundles = infer._prepare_query_frame_sample_bundles(
                query_frame_results=query_frame_results,
                grid_size=int(camera_args.grid_size),
                filter_args=camera_args,
                filter_config=filter_config,
                full_depths=np.asarray(result["full_depths"], dtype=np.float32),
                include_query_frame_image=False,
            )
            high_volatility_mask = _compute_subset_high_volatility_mask(
                camera_args=camera_args,
                prepared_bundles=prepared_bundles,
                full_depths=np.asarray(result["full_depths"], dtype=np.float32),
            )

        save_start = time.perf_counter()
        sample_results: list[dict[str, Any]] = []
        for query_frame_local, query_frame_source, sample_path_str in zip(
            task.query_frames_local,
            task.query_frames_source,
            task.sample_paths,
        ):
            sample_path = Path(sample_path_str)
            backup_path = build_backup_path(sample_path, mirror_root=mirror_root, backup_root=backup_root)
            sample_result = {
                "episode_name": task.episode_name,
                "camera_name": task.camera_name,
                "query_frame_local": int(query_frame_local),
                "query_frame_source": int(query_frame_source),
                "sample_path": str(sample_path),
                "backup_path": str(backup_path),
                "before_valid_count": 0,
                "after_valid_count": None,
                "status": "failed",
                "error_message": None,
                "skip_reason": None,
                "backup_status": None,
            }
            try:
                if int(query_frame_local) in skipped_query_frames:
                    sample_result["backup_status"] = _remove_sample_after_backup(sample_path, backup_path)
                    sample_result["after_valid_count"] = 0
                    sample_result["skip_reason"] = "short_tail_segment_len<=8"
                    sample_result["status"] = "removed_short_tail"
                    sample_results.append(sample_result)
                    continue

                built_sample = infer.build_v2_sample_data(
                    prepared_bundle=prepared_bundles[int(query_frame_local)],
                    filter_args=camera_args,
                    high_volatility_mask=high_volatility_mask,
                )
                sample_data = built_sample["sample_data"]
                sample_result["after_valid_count"] = int(
                    np.count_nonzero(np.asarray(sample_data["traj_valid_mask"], dtype=bool))
                )
                _ensure_sample_backed_up(sample_path, backup_path)
                _atomic_save_npz(sample_path, sample_data)
                sample_result["status"] = (
                    "fixed" if int(sample_result["after_valid_count"]) > 0 else "still_empty"
                )
            except Exception as exc:
                sample_result["error_message"] = str(exc)
                _restore_sample_from_backup(sample_path, backup_path)
            sample_results.append(sample_result)

        task_record["save_seconds"] = float(time.perf_counter() - save_start)
        task_record["sample_results"] = sample_results
        task_record["fixed_count"] = int(
            sum(1 for item in sample_results if item["status"] == "fixed")
        )
        task_record["still_empty_count"] = int(
            sum(1 for item in sample_results if item["status"] == "still_empty")
        )
        task_record["removed_short_tail_count"] = int(
            sum(1 for item in sample_results if item["status"] == "removed_short_tail")
        )
        task_record["failed_sample_count"] = int(
            sum(1 for item in sample_results if item["status"] == "failed")
        )
        task_record["status"] = (
            "success" if task_record["failed_sample_count"] == 0 else "partial"
        )
        return True, False, task_record
    except Exception as exc:
        task_record["error_message"] = str(exc)
        if batch_infer.is_retryable_cuda_error(exc):
            logger.exception(
                f"{task.episode_name}/{task.camera_name} hit retryable CUDA failure: {exc}"
            )
            return False, True, task_record
        logger.exception(f"{task.episode_name}/{task.camera_name} failed: {exc}")
        return False, False, task_record
    finally:
        task_record["started_at_unix"] = float(started_at_unix)
        task_record["finished_at_unix"] = float(time.time())
        if "model_depth_pose" in locals():
            del model_depth_pose
        batch_infer.safe_empty_cuda_cache(
            f"{task.episode_name}/{task.camera_name}: repair task cleanup"
        )


def process_repair_tasks_on_gpu(
    *,
    worker_slot: batch_infer.WorkerSlot,
    task_queue,
    result_queue,
    args,
    stop_event,
    remaining_tasks,
    backup_root: Path,
    mirror_root: Path,
) -> None:
    worker_args = copy.deepcopy(args)
    worker_args.device = worker_slot.device
    worker_label = worker_slot.label
    model_3dtracker = None
    try:
        while not stop_event.is_set():
            if model_3dtracker is None:
                if not batch_infer.wait_for_gpu_recovery(
                    gpu_id=worker_slot.gpu_id,
                    args=worker_args,
                    stop_event=stop_event,
                ):
                    break
                logger.info(f"[{worker_label}] start repair worker on {worker_args.device}")
                try:
                    model_3dtracker = infer.load_model(worker_args.checkpoint).to(worker_args.device)
                    batch_infer.warm_up_cuda_linalg(worker_args.device)
                except Exception as exc:
                    model_3dtracker = batch_infer.unload_tracker_model(model_3dtracker)
                    if batch_infer.is_retryable_cuda_error(exc):
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
                    f"[{worker_label}] [{task.task_index}/{task.total_tasks}] "
                    f"{task.episode_name}/{task.camera_name}"
                )
                ok, retire_worker, task_record = run_repair_task(
                    task=task,
                    args=worker_args,
                    model_3dtracker=model_3dtracker,
                    backup_root=backup_root,
                    mirror_root=mirror_root,
                    gpu_id=worker_slot.gpu_id,
                )
                if ok:
                    batch_infer.mark_task_completed(remaining_tasks)
                    result_queue.put(task_record)
                elif retire_worker:
                    task_queue.put(task)
                    model_3dtracker = batch_infer.unload_tracker_model(model_3dtracker)
                    logger.warning(
                        f"[{worker_label}] re-queued {task.episode_name}/{task.camera_name} "
                        "after retryable CUDA failure; waiting for GPU recovery."
                    )
                else:
                    batch_infer.mark_task_completed(remaining_tasks)
                    result_queue.put(task_record)
            finally:
                task_queue.task_done()
    finally:
        model_3dtracker = batch_infer.unload_tracker_model(model_3dtracker)


def process_repair_tasks_on_gpu_entrypoint(
    *,
    worker_slot: batch_infer.WorkerSlot,
    task_queue,
    result_queue,
    args,
    stop_event,
    remaining_tasks,
    backup_root: str,
    mirror_root: str,
) -> None:
    try:
        process_repair_tasks_on_gpu(
            worker_slot=worker_slot,
            task_queue=task_queue,
            result_queue=result_queue,
            args=args,
            stop_event=stop_event,
            remaining_tasks=remaining_tasks,
            backup_root=Path(backup_root),
            mirror_root=Path(mirror_root),
        )
    except Exception as exc:
        logger.exception(f"[{worker_slot.label}] repair worker failed: {exc}")
        result_queue.put(
            {
                "kind": "worker_error",
                "worker_label": worker_slot.label,
                "error_message": str(exc),
            }
        )


def run_repair_tasks_single_process(
    *,
    tasks: list[RepairTask],
    args,
    backup_root: Path,
    mirror_root: Path,
) -> list[dict[str, Any]]:
    device = getattr(args, "device", "cuda")
    logger.info(f"[single-process] loading tracker on {device}")
    model_3dtracker = infer.load_model(args.checkpoint).to(device)
    batch_infer.warm_up_cuda_linalg(device)
    task_records: list[dict[str, Any]] = []
    try:
        for task in tasks:
            logger.info(
                f"[single-process] [{task.task_index}/{task.total_tasks}] "
                f"{task.episode_name}/{task.camera_name}"
            )
            ok, _retire_worker, task_record = run_repair_task(
                task=task,
                args=args,
                model_3dtracker=model_3dtracker,
                backup_root=backup_root,
                mirror_root=mirror_root,
            )
            if not ok and task_record.get("status") == "failed" and task_record.get("error_message") is not None:
                task_records.append(task_record)
                continue
            task_records.append(task_record)
    finally:
        model_3dtracker = batch_infer.unload_tracker_model(model_3dtracker)
    return task_records


def run_repair_tasks_multi_gpu(
    *,
    tasks: list[RepairTask],
    args,
    backup_root: Path,
    mirror_root: Path,
) -> list[dict[str, Any]]:
    gpu_ids = batch_infer.parse_gpu_ids(args.gpu_id)
    worker_slots = batch_infer.build_worker_slots(
        gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
    )
    if not worker_slots:
        return run_repair_tasks_single_process(
            tasks=tasks,
            args=args,
            backup_root=backup_root,
            mirror_root=mirror_root,
        )

    mp_ctx = mp.get_context("spawn")
    task_queue = mp_ctx.JoinableQueue()
    result_queue = mp_ctx.Queue()
    stop_event = mp_ctx.Event()
    remaining_tasks = mp_ctx.Value("i", len(tasks))

    for task in tasks:
        task_queue.put(task)

    worker_processes: list[tuple[batch_infer.WorkerSlot, mp.Process]] = []
    for worker_slot in worker_slots:
        process = mp_ctx.Process(
            target=process_repair_tasks_on_gpu_entrypoint,
            kwargs={
                "worker_slot": worker_slot,
                "task_queue": task_queue,
                "result_queue": result_queue,
                "args": args,
                "stop_event": stop_event,
                "remaining_tasks": remaining_tasks,
                "backup_root": str(backup_root),
                "mirror_root": str(mirror_root),
            },
            name=f"repair-empty-samples-{worker_slot.gpu_id}-{worker_slot.gpu_slot_index}",
        )
        process.start()
        worker_processes.append((worker_slot, process))

    task_records: list[dict[str, Any]] = []
    try:
        while True:
            try:
                result = result_queue.get(timeout=5.0)
                if isinstance(result, dict) and result.get("kind") == "worker_error":
                    stop_event.set()
                    raise RuntimeError(
                        f"{result.get('worker_label')} failed: {result.get('error_message')}"
                    )
                task_records.append(result)
            except queue.Empty:
                pass

            if remaining_tasks.value <= 0:
                break

            dead_workers = [
                (worker_slot, process)
                for worker_slot, process in worker_processes
                if not process.is_alive() and process.exitcode not in (None, 0)
            ]
            if dead_workers:
                stop_event.set()
                first_dead = dead_workers[0]
                raise RuntimeError(
                    f"{first_dead[0].label} exited with code {first_dead[1].exitcode} "
                    f"before all repair tasks completed"
                )

        task_queue.join()
    finally:
        for _ in worker_slots:
            task_queue.put(None)
        for worker_slot, process in worker_processes:
            process.join(timeout=10.0)
            if process.is_alive():
                logger.warning(f"[{worker_slot.label}] repair worker did not exit promptly; terminating.")
                process.terminate()
                process.join(timeout=5.0)
        while True:
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(result, dict) and result.get("kind") != "worker_error":
                task_records.append(result)

    task_records.sort(key=lambda item: (item["episode_name"], item["camera_name"]))
    return task_records


def summarize_repair_results(task_records: list[dict[str, Any]]) -> dict[str, Any]:
    sample_records = [
        sample_result
        for task_record in task_records
        for sample_result in task_record.get("sample_results", [])
    ]
    return {
        "task_count": len(task_records),
        "sample_count": len(sample_records),
        "fixed_count": int(sum(1 for item in sample_records if item["status"] == "fixed")),
        "still_empty_count": int(sum(1 for item in sample_records if item["status"] == "still_empty")),
        "removed_short_tail_count": int(
            sum(1 for item in sample_records if item["status"] == "removed_short_tail")
        ),
        "failed_sample_count": int(sum(1 for item in sample_records if item["status"] == "failed")),
        "failed_task_count": int(sum(1 for item in task_records if item.get("status") == "failed")),
        "partial_task_count": int(sum(1 for item in task_records if item.get("status") == "partial")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair empty TraceForge samples for press_one_button_demo outputs."
    )
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--trajectory_dirname", type=str, default="trajectory")
    parser.add_argument("--backup_root", type=str, default=None)
    parser.add_argument("--report_root", type=str, default=None)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--gpu_id", type=str, default=None)
    parser.add_argument("--min_free_gpu_mem_gb", type=float, default=0.0)
    parser.add_argument("--gpu_recovery_poll_sec", type=float, default=30.0)
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument(
        "--camera_names",
        type=str,
        default=None,
        help=(
            "Comma-separated camera names to repair. Required. Only these camera outputs "
            "are scanned and repaired."
        ),
    )
    parser.add_argument("--episode_name", type=str, default=None)
    parser.add_argument("--episode_names_file", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/tapip3d_final.pth")
    parser.add_argument(
        "--depth_pose_method",
        type=str,
        default="external",
        choices=infer.video_depth_pose_dict.keys(),
    )
    parser.add_argument("--external_geom_name", type=str, default="trajectory_valid.h5")
    parser.add_argument("--external_extr_mode", type=str, default="w2c", choices=["w2c", "c2w"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max_num_frames", type=int, default=512)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--output_layout", type=str, default="v2", choices=["v2", "legacy"])
    parser.add_argument(
        "--scene_storage_mode",
        type=str,
        default="source_ref",
        choices=["source_ref", "cache"],
    )
    parser.add_argument("--save_visibility", action="store_true", default=False)
    parser.add_argument("--future_len", type=int, default=32)
    parser.add_argument("--grid_size", type=int, default=80)
    parser.add_argument("--query_prefilter_mode", type=str, default="off", choices=["off", "profile_aware_static_v1"])
    parser.add_argument("--query_prefilter_wrist_rank_keep_ratio", type=float, default=0.30)
    parser.add_argument("--support_grid_ratio", type=float, default=0.8)
    parser.add_argument("--filter_level", type=str, default="standard", choices=["none", "basic", "standard", "strict"])
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
    )
    parser.add_argument("--min_valid_frames", type=int, default=None)
    parser.add_argument("--visibility_threshold", type=float, default=None)
    parser.add_argument("--min_depth", type=float, default=0.01)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--boundary_margin", type=int, default=None)
    parser.add_argument("--depth_change_threshold", type=float, default=None)
    args = parser.parse_args()
    args.camera_names = batch_infer.parse_camera_names(
        args.camera_names,
        option_name="--camera_names",
    )
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers_per_gpu must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_path).expanduser().resolve()
    out_root = batch_infer.resolve_output_root(args)
    run_tag = _timestamp_tag()
    backup_root = _resolve_runtime_root(
        base_path,
        args.backup_root,
        "_empty_sample_repair_backups",
        run_tag,
    )
    report_root = _resolve_runtime_root(
        base_path,
        args.report_root,
        "_empty_sample_repair_reports",
        run_tag,
    )
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path is not None and args.manifest_path.strip()
        else report_root / _SCAN_MANIFEST_BASENAME
    )
    results_jsonl_path = report_root / _RESULTS_JSONL_BASENAME
    summary_json_path = report_root / _SUMMARY_JSON_BASENAME
    mirror_root = out_root if out_root is not None else base_path

    episodes = resolve_target_episodes(base_path, args)
    if not episodes:
        logger.error(f"No valid episodes found under {base_path}")
        return

    records = scan_empty_sample_records(
        base_path=base_path,
        args=args,
        out_root=out_root,
        episodes=episodes,
    )
    tasks = build_repair_tasks(records, report_root=report_root)
    write_scan_manifest(
        records=records,
        tasks=tasks,
        base_path=base_path,
        out_root=out_root,
        report_root=report_root,
        backup_root=backup_root,
        manifest_path=manifest_path,
    )

    summary = summarize_empty_sample_records(records)
    logger.info("=" * 80)
    logger.info("Empty sample repair")
    logger.info(f"base_path={base_path}")
    logger.info(f"out_root={out_root if out_root is not None else '<episode>/trajectory'}")
    logger.info(f"episodes={summary['episode_count']}")
    logger.info(f"empty_samples={summary['empty_sample_count']}")
    logger.info(f"episode_camera_groups={summary['episode_camera_group_count']}")
    logger.info(f"report_root={report_root}")
    logger.info(f"backup_root={backup_root}")
    logger.info(f"manifest_path={manifest_path}")
    logger.info(f"device={args.device}, gpu_id={args.gpu_id}")
    logger.info("=" * 80)

    if args.dry_run:
        return

    report_root.mkdir(parents=True, exist_ok=True)
    results_jsonl_path.write_text("", encoding="utf-8")

    if args.gpu_id:
        task_records = run_repair_tasks_multi_gpu(
            tasks=tasks,
            args=args,
            backup_root=backup_root,
            mirror_root=mirror_root,
        )
    else:
        task_records = run_repair_tasks_single_process(
            tasks=tasks,
            args=args,
            backup_root=backup_root,
            mirror_root=mirror_root,
        )

    for task_record in task_records:
        _append_jsonl(results_jsonl_path, task_record)

    repair_summary = summarize_repair_results(task_records)
    repair_summary.update(
        {
            "base_path": str(base_path),
            "out_root": None if out_root is None else str(out_root),
            "report_root": str(report_root),
            "backup_root": str(backup_root),
            "manifest_path": str(manifest_path),
        }
    )
    _write_json(summary_json_path, repair_summary)
    logger.info(
        "Repair finished: "
        f"fixed={repair_summary['fixed_count']} "
        f"still_empty={repair_summary['still_empty_count']} "
        f"removed_short_tail={repair_summary['removed_short_tail_count']} "
        f"failed={repair_summary['failed_sample_count']}"
    )


if __name__ == "__main__":
    main()
