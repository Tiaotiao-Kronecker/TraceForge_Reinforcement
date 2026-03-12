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
1. `batch_infer.py` 的批量发现与批量执行思路；
2. `infer_bridge_v2.py` 的“单条样本多相机推理并保存标准 TraceForge 输出”逻辑。

推荐多卡使用方式：
- 默认使用 `--gpu_id 0,1,...` 启动动态调度；
- 每张卡对应一个常驻 worker，只加载一次 3D tracker；
- worker 从共享任务队列中按 `episode/camera` 粒度领取下一个任务，直到队列清空；
- 如需兼容旧的平均分片方式，可切回 `--gpu_schedule_mode static`。

默认输出方式：
- 就地写回到每个 episode 下的 `trajectory/<camera_name>/...`；
- 如需兼容旧流程，仍可通过 `--out_dir` 指定外部输出根目录。
"""

from __future__ import annotations

import argparse
import copy
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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


DEFAULT_CAMERAS = [
    "varied_camera_1",
    "varied_camera_2",
    "varied_camera_3",
]


@dataclass(frozen=True)
class CameraTask:
    task_index: int
    total_tasks: int
    episode_dir: Path
    out_episode_dir: Path
    camera_name: str


def parse_camera_names(camera_names: str) -> list[str]:
    values = [item.strip() for item in camera_names.split(",") if item.strip()]
    if not values:
        raise ValueError("camera_names must contain at least one camera name")
    return values


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
        help="Comma-separated GPU IDs, e.g. 0,1,2,3. If set, run multi-GPU in one command.",
    )
    parser.add_argument(
        "--gpu_schedule_mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "static"],
        help="In --gpu_id mode, choose dynamic camera-task scheduling or legacy static sharding.",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        default=",".join(DEFAULT_CAMERAS),
        help="Comma-separated camera names to process",
    )
    parser.add_argument(
        "--episode_name",
        type=str,
        default=None,
        help="Only process a single episode, e.g. episode_00000",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit total episodes before sharding; useful for testing",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total shard count for multi-process launch",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Current shard index in [0, num_shards)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a camera if images/samples/main npz already exist",
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
        help="Recommended: external, using trajectory_valid.h5 per episode",
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
        help="Use cuda:0 when launching each shard with CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--num_iters", type=int, default=6)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max_num_frames", type=int, default=384)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument(
        "--use_all_trajectories",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--frame_drop_rate",
        type=int,
        default=15,
        help="Query every N frames; 15 reproduces the 0/15/30/45 demo pattern",
    )
    parser.add_argument(
        "--future_len",
        type=int,
        default=32,
        help="Tracking window per query frame",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=80,
        help="Grid size per query frame; 80 means 6400 points",
    )
    args = parser.parse_args()
    args.camera_names = parse_camera_names(args.camera_names)

    if args.num_shards <= 0:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    return args


def parse_gpu_ids(gpu_id: str | None) -> list[int]:
    if gpu_id is None:
        return []
    values = [item.strip() for item in gpu_id.split(",") if item.strip()]
    if not values:
        return []
    return [int(item) for item in values]


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


def resolve_launcher_log_dir(
    *,
    base_path: Path,
    args: argparse.Namespace,
    out_root: Path | None,
) -> Path:
    if out_root is not None:
        return out_root / "_logs"
    return base_path / f"_{args.trajectory_dirname}_batch_logs"


def describe_output_target(args: argparse.Namespace, out_root: Path | None) -> str:
    if out_root is not None:
        return str(out_root)
    return f"<episode>/{args.trajectory_dirname}"


def build_worker_cmd(
    *,
    script_path: str,
    args: argparse.Namespace,
    worker_count: int,
    worker_index: int,
) -> list[str]:
    cmd = [
        sys.executable,
        script_path,
        "--base_path", args.base_path,
        "--camera_names", ",".join(args.camera_names),
        "--num_shards", str(worker_count),
        "--shard_index", str(worker_index),
        "--checkpoint", args.checkpoint,
        "--depth_pose_method", args.depth_pose_method,
        "--external_geom_name", args.external_geom_name,
        "--external_extr_mode", args.external_extr_mode,
        "--device", "cuda:0",
        "--num_iters", str(args.num_iters),
        "--fps", str(args.fps),
        "--max_num_frames", str(args.max_num_frames),
        "--horizon", str(args.horizon),
        "--frame_drop_rate", str(args.frame_drop_rate),
        "--future_len", str(args.future_len),
        "--max_frames_per_video", str(args.max_frames_per_video),
        "--grid_size", str(args.grid_size),
        "--trajectory_dirname", args.trajectory_dirname,
    ]

    if args.out_dir is not None and args.out_dir.strip():
        cmd.extend(["--out_dir", args.out_dir])
    if args.episode_name is not None:
        cmd.extend(["--episode_name", args.episode_name])
    if args.max_episodes is not None:
        cmd.extend(["--max_episodes", str(args.max_episodes)])
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.copy_lang:
        cmd.append("--copy_lang")
    if args.save_video:
        cmd.append("--save_video")
    if args.dry_run:
        cmd.append("--dry_run")

    return cmd


def launch_worker_process(
    *,
    gpu_id: int,
    worker_index: int,
    worker_count: int,
    args: argparse.Namespace,
    log_dir: Path,
) -> tuple[bool, float, str | None]:
    script_path = os.path.abspath(__file__)
    cmd = build_worker_cmd(
        script_path=script_path,
        args=args,
        worker_count=worker_count,
        worker_index=worker_index,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPYCACHEPREFIX"] = env.get("PYTHONPYCACHEPREFIX", "/tmp/traceforge_pycache")
    env["PYTHONPATH"] = (
        _PROJECT_ROOT + os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH")
        else _PROJECT_ROOT
    )

    log_path = log_dir / f"worker_gpu{gpu_id}.log"
    start_time = time.time()
    logger.info(
        f"[GPU {gpu_id}] launch worker {worker_index}/{worker_count}: "
        f"log={log_path}"
    )

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=log_file,
            env=env,
            check=False,
            text=True,
        )

    elapsed = time.time() - start_time
    if proc.returncode == 0:
        logger.info(f"[GPU {gpu_id}] worker finished in {elapsed/60:.1f} min")
        return True, elapsed, None

    err = f"worker exit code {proc.returncode}"
    logger.error(f"[GPU {gpu_id}] {err}; see {log_path}")
    return False, elapsed, err


def _has_files(dir_path: Path, suffixes: tuple[str, ...]) -> bool:
    return dir_path.is_dir() and any(
        path.is_file() and path.suffix.lower() in suffixes for path in dir_path.iterdir()
    )


def find_valid_episodes(base_path: Path, camera_names: list[str], geom_name: str) -> list[Path]:
    episodes: list[Path] = []
    for episode_dir in sorted(base_path.iterdir()):
        if not episode_dir.is_dir():
            continue
        if not episode_dir.name.startswith("episode_"):
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


def shard_episodes(episodes: list[Path], num_shards: int, shard_index: int) -> list[Path]:
    return episodes[shard_index::num_shards]


def camera_output_complete(out_episode_dir: Path, camera_name: str) -> bool:
    camera_dir = out_episode_dir / camera_name
    images_dir = camera_dir / "images"
    samples_dir = camera_dir / "samples"
    main_npz = camera_dir / f"{camera_name}.npz"
    has_images = images_dir.is_dir() and any(images_dir.iterdir())
    has_samples = samples_dir.is_dir() and any(samples_dir.iterdir())
    return has_images and has_samples and main_npz.is_file()


def copy_episode_lang(episode_dir: Path, out_episode_dir: Path) -> None:
    lang_path = episode_dir / "lang.txt"
    if not lang_path.is_file():
        return
    out_episode_dir.mkdir(parents=True, exist_ok=True)
    target = out_episode_dir / "lang.txt"
    target.write_text(lang_path.read_text(encoding="utf-8"), encoding="utf-8")


def build_camera_tasks(
    episodes: list[Path],
    *,
    args: argparse.Namespace,
    out_dir: Path | None,
) -> list[CameraTask]:
    pending: list[tuple[Path, Path, str]] = []

    for episode_dir in episodes:
        out_episode_dir = resolve_episode_output_dir(
            episode_dir,
            args=args,
            out_root=out_dir,
        )
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
            pending.append((episode_dir, out_episode_dir, camera_name))

    total_tasks = len(pending)
    tasks: list[CameraTask] = []
    for task_index, (episode_dir, out_episode_dir, camera_name) in enumerate(pending, start=1):
        tasks.append(
            CameraTask(
                task_index=task_index,
                total_tasks=total_tasks,
                episode_dir=episode_dir,
                out_episode_dir=out_episode_dir,
                camera_name=camera_name,
            )
        )
    return tasks


def build_camera_args(
    base_args: argparse.Namespace,
    episode_dir: Path,
    camera_name: str,
) -> argparse.Namespace:
    camera_args = copy.deepcopy(base_args)
    camera_args.mask_dir = None
    camera_args.camera_name = camera_name
    camera_args.external_geom_npz = str(episode_dir / base_args.external_geom_name)
    return camera_args


def save_result(
    *,
    out_episode_dir: Path,
    camera_name: str,
    result: dict,
    args: argparse.Namespace,
    save_lock: threading.Lock | None = None,
) -> None:
    if save_lock is not None:
        save_lock.acquire()
    try:
        infer.args = args
        infer.save_structured_data(
            video_name=camera_name,
            output_dir=str(out_episode_dir),
            video_tensor=result["video_tensor"],
            depths=result["depths"],
            coords=result["coords"],
            visibs=result["visibs"],
            intrinsics=result["intrinsics"],
            extrinsics=result["extrinsics"],
            query_points_per_frame=result["query_points_per_frame"],
            horizon=args.horizon,
            original_filenames=result["original_filenames"],
            use_all_trajectories=args.use_all_trajectories,
            query_frame_results=result.get("query_frame_results"),
            future_len=args.future_len,
            grid_size=args.grid_size,
        )
    finally:
        if save_lock is not None:
            save_lock.release()

    camera_dir = out_episode_dir / camera_name
    main_npz_path = camera_dir / f"{camera_name}.npz"
    data_npz = {
        "coords": result["coords"].cpu().numpy(),
        "extrinsics": result["full_extrinsics"].cpu().numpy(),
        "intrinsics": result["full_intrinsics"].cpu().numpy(),
        "height": result["video_tensor"].shape[-2],
        "width": result["video_tensor"].shape[-1],
        "depths": result["depths"].cpu().numpy().astype(np.float16),
        "unc_metric": result["depth_conf"].astype(np.float16),
        "visibs": result["visibs"][..., None].cpu().numpy(),
    }
    if args.save_video:
        data_npz["video"] = result["video_tensor"].cpu().numpy()
    np.savez(main_npz_path, **data_npz)
    logger.info(f"Saved {main_npz_path}")


def run_camera_task(
    *,
    task: CameraTask,
    args: argparse.Namespace,
    model_3dtracker,
    save_lock: threading.Lock | None = None,
) -> bool:
    if args.copy_lang and not (task.out_episode_dir / "lang.txt").is_file():
        copy_episode_lang(task.episode_dir, task.out_episode_dir)

    camera_args = build_camera_args(args, task.episode_dir, task.camera_name)
    logger.info(
        f"{task.episode_dir.name}/{task.camera_name}: run "
        f"(device={camera_args.device}, depth_pose_method={camera_args.depth_pose_method})"
    )

    try:
        model_depth_pose = infer.video_depth_pose_dict[camera_args.depth_pose_method](camera_args)
        result = infer.process_single_video(
            str(task.episode_dir / "rgb" / task.camera_name),
            str(task.episode_dir / "depth" / task.camera_name),
            camera_args,
            model_3dtracker,
            model_depth_pose,
        )
        save_result(
            out_episode_dir=task.out_episode_dir,
            camera_name=task.camera_name,
            result=result,
            args=camera_args,
            save_lock=save_lock,
        )
        return True
    except Exception as exc:
        logger.exception(f"{task.episode_dir.name}/{task.camera_name} failed: {exc}")
        return False
    finally:
        if "model_depth_pose" in locals():
            del model_depth_pose
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_episode(
    *,
    episode_dir: Path,
    out_episode_dir: Path,
    args: argparse.Namespace,
    model_3dtracker,
    save_lock: threading.Lock | None = None,
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

    tasks = [
        CameraTask(
            task_index=idx,
            total_tasks=len(pending_cameras),
            episode_dir=episode_dir,
            out_episode_dir=out_episode_dir,
            camera_name=camera_name,
        )
        for idx, camera_name in enumerate(pending_cameras, start=1)
    ]
    if args.copy_lang and not tasks:
        copy_episode_lang(episode_dir, out_episode_dir)

    for task in tasks:
        ok = run_camera_task(
            task=task,
            args=args,
            model_3dtracker=model_3dtracker,
            save_lock=save_lock,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def process_camera_tasks_on_gpu(
    *,
    gpu_id: int,
    task_queue: queue.Queue[CameraTask],
    args: argparse.Namespace,
    save_lock: threading.Lock,
) -> tuple[int, int, float]:
    worker_args = copy.deepcopy(args)
    worker_args.device = f"cuda:{gpu_id}"

    logger.info(f"[GPU {gpu_id}] start dynamic worker on {worker_args.device}")
    worker_start = time.time()
    model_3dtracker = infer.load_model(worker_args.checkpoint).to(worker_args.device)

    total_camera_success = 0
    total_camera_fail = 0
    try:
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                break

            logger.info(
                f"[GPU {gpu_id}] "
                f"[{task.task_index}/{task.total_tasks}] {task.episode_dir.name}/{task.camera_name}"
            )
            ok = run_camera_task(
                task=task,
                args=worker_args,
                model_3dtracker=model_3dtracker,
                save_lock=save_lock,
            )
            if ok:
                total_camera_success += 1
            else:
                total_camera_fail += 1
    finally:
        del model_3dtracker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - worker_start
    logger.info(
        f"[GPU {gpu_id}] dynamic worker done in {elapsed/60:.1f} min "
        f"(camera_success={total_camera_success}, camera_fail={total_camera_fail})"
    )
    return total_camera_success, total_camera_fail, elapsed


def process_episodes_on_gpu(
    *,
    gpu_id: int,
    episodes: list[Path],
    args: argparse.Namespace,
    out_dir: Path | None,
    save_lock: threading.Lock,
) -> tuple[int, int, float]:
    worker_args = copy.deepcopy(args)
    worker_args.device = f"cuda:{gpu_id}"

    logger.info(
        f"[GPU {gpu_id}] start worker with {len(episodes)} episodes on {worker_args.device}"
    )
    worker_start = time.time()
    model_3dtracker = infer.load_model(worker_args.checkpoint).to(worker_args.device)

    total_camera_success = 0
    total_camera_fail = 0
    try:
        for idx, episode_dir in enumerate(episodes, start=1):
            logger.info(
                f"[GPU {gpu_id}] [{idx}/{len(episodes)}] episode={episode_dir.name}"
            )
            out_episode_dir = resolve_episode_output_dir(
                episode_dir,
                args=worker_args,
                out_root=out_dir,
            )
            success_count, fail_count = run_episode(
                episode_dir=episode_dir,
                out_episode_dir=out_episode_dir,
                args=worker_args,
                model_3dtracker=model_3dtracker,
                save_lock=save_lock,
            )
            total_camera_success += success_count
            total_camera_fail += fail_count
    finally:
        del model_3dtracker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - worker_start
    logger.info(
        f"[GPU {gpu_id}] done in {elapsed/60:.1f} min "
        f"(camera_success={total_camera_success}, camera_fail={total_camera_fail})"
    )
    return total_camera_success, total_camera_fail, elapsed


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_path).resolve()
    out_dir = resolve_output_root(args)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = parse_gpu_ids(args.gpu_id)

    episodes = find_valid_episodes(base_path, args.camera_names, args.external_geom_name)
    if not episodes:
        logger.error(f"No valid episodes found under {base_path}")
        return

    if args.episode_name is not None:
        episodes = [episode for episode in episodes if episode.name == args.episode_name]
        if not episodes:
            logger.error(f"Episode not found: {args.episode_name}")
            return
    elif args.max_episodes is not None and args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    total_before_shard = len(episodes)
    if gpu_ids:
        episodes_for_run = episodes
    else:
        episodes_for_run = shard_episodes(episodes, args.num_shards, args.shard_index)

    logger.info("=" * 80)
    logger.info("Press-one-button demo batch inference")
    logger.info(f"base_path={base_path}")
    logger.info(f"out_dir={describe_output_target(args, out_dir)}")
    logger.info(f"cameras={args.camera_names}")
    logger.info(
        f"episodes(before shard)={total_before_shard}, "
        f"episodes(this shard)={len(episodes_for_run)}, "
        f"shard={args.shard_index}/{args.num_shards}"
    )
    logger.info(
        f"device={args.device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    if gpu_ids:
        logger.info(f"gpu_ids={gpu_ids}")
        logger.info(f"gpu_schedule_mode={args.gpu_schedule_mode}")
    logger.info(
        f"frame_drop_rate={args.frame_drop_rate}, future_len={args.future_len}, grid_size={args.grid_size}"
    )
    logger.info("=" * 80)

    dynamic_tasks: list[CameraTask] | None = None
    if gpu_ids and args.gpu_schedule_mode == "dynamic":
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

    if gpu_ids:
        if args.num_shards != 1 or args.shard_index != 0:
            logger.warning(
                "gpu_id mode ignores the incoming shard settings and manages sharding internally."
            )
        worker_count = len(gpu_ids)
        if args.gpu_schedule_mode == "static":
            log_dir = resolve_launcher_log_dir(
                base_path=base_path,
                args=args,
                out_root=out_dir,
            )
            log_dir.mkdir(parents=True, exist_ok=True)

            episodes_per_worker = [len(episodes_for_run[i::worker_count]) for i in range(worker_count)]
            for worker_index, gpu in enumerate(gpu_ids):
                logger.info(
                    f"[GPU {gpu}] assigned {episodes_per_worker[worker_index]} episodes "
                    f"(worker shard {worker_index}/{worker_count})"
                )

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        launch_worker_process,
                        gpu_id=gpu,
                        worker_index=worker_index,
                        worker_count=worker_count,
                        args=args,
                        log_dir=log_dir,
                    ): gpu
                    for worker_index, gpu in enumerate(gpu_ids)
                }
                for future in as_completed(future_map):
                    gpu = future_map[future]
                    try:
                        ok, _elapsed, _err = future.result()
                        if ok:
                            total_camera_success += 1
                        else:
                            total_camera_fail += 1
                    except Exception as exc:
                        total_camera_fail += 1
                        logger.exception(f"[GPU {gpu}] launcher thread failed: {exc}")
        else:
            assert dynamic_tasks is not None
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

            save_lock = threading.Lock()
            task_queue: queue.Queue[CameraTask] = queue.Queue()
            for task in dynamic_tasks:
                task_queue.put(task)

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        process_camera_tasks_on_gpu,
                        gpu_id=gpu,
                        task_queue=task_queue,
                        args=args,
                        save_lock=save_lock,
                    ): gpu
                    for gpu in gpu_ids
                }
                for future in as_completed(future_map):
                    gpu = future_map[future]
                    try:
                        success_count, fail_count, _elapsed = future.result()
                        total_camera_success += success_count
                        total_camera_fail += fail_count
                    except Exception as exc:
                        logger.exception(f"[GPU {gpu}] dynamic worker failed: {exc}")
            remaining_tasks = task_queue.qsize()
            if remaining_tasks > 0:
                total_camera_fail += remaining_tasks
                logger.error(f"Dynamic scheduler left {remaining_tasks} camera tasks unprocessed.")
    else:
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
                )
                total_camera_success += success_count
                total_camera_fail += fail_count
        finally:
            del model_3dtracker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("=" * 80)
    if gpu_ids:
        if args.gpu_schedule_mode == "static":
            logger.info(
                f"Done launcher mode. workers_ok={total_camera_success}, "
                f"workers_failed={total_camera_fail}"
            )
        else:
            logger.info(
                f"Done dynamic gpu mode. camera_success={total_camera_success}, "
                f"camera_fail={total_camera_fail}"
            )
    else:
        logger.info(
            f"Done. shard={args.shard_index}/{args.num_shards}, "
            f"camera_success={total_camera_success}, camera_fail={total_camera_fail}"
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
