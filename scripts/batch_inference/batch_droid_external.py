#!/usr/bin/env python3
"""
DROID 数据集批量推理（外部几何直通版）：
使用外部深度 + 外部内外参，跳过 VGGT，并支持多 GPU 均匀负载。

用法:
    python scripts/batch_inference/batch_droid_external.py \
        --base_path /home/zoyo/projects/droid_preprocess_pipeline/droid_raw \
        --gpu_id 0,1,2,3,4,5,6,7 \
        --frame_drop_rate 5 \
        --grid_size 80
"""

import os
import sys
import argparse
import subprocess
import time
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

CAMERAS = ["hand_camera", "varied_camera_1", "varied_camera_2"]
RGB_EXTS = ("*.jpg", "*.jpeg", "*.png")
DEPTH_EXTS = ("*.npy", "*.png", "*.jpg", "*.jpeg")


def has_files(dir_path: Path, patterns) -> bool:
    """检查目录下是否存在匹配文件。"""
    if not dir_path.is_dir():
        return False
    for pattern in patterns:
        if any(dir_path.glob(pattern)):
            return True
    return False


def resolve_geom_path(dataset_path: Path, camera_name: str, geom_source: str) -> Path | None:
    npz_path = dataset_path / f"external_geom_{camera_name}_left.npz"
    h5_path = dataset_path / "trajectory_valid.h5"

    if geom_source == "npz":
        return npz_path if npz_path.exists() else None
    if geom_source == "h5":
        return h5_path if h5_path.exists() else None
    if npz_path.exists():
        return npz_path
    if h5_path.exists():
        return h5_path
    return None


def resolve_depth_path(dataset_path: Path, camera_name: str) -> Path:
    depth_nested = dataset_path / "depth" / camera_name / "depth"
    depth_flat = dataset_path / "depth" / camera_name
    if has_files(depth_nested, DEPTH_EXTS):
        return depth_nested
    if has_files(depth_flat, DEPTH_EXTS):
        return depth_flat
    return depth_nested


def find_droid_datasets(base_path, geom_source):
    """扫描 DROID 数据集目录，返回有效数据集列表。"""
    base = Path(base_path).resolve()
    if not base.is_dir():
        return []

    datasets = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue

        rgb_root = d / "rgb_stereo_valid"
        depth_root = d / "depth"
        if not rgb_root.is_dir() or not depth_root.is_dir():
            continue

        has_camera = False
        for cam in CAMERAS:
            video_path = rgb_root / cam
            depth_path = resolve_depth_path(d, cam)
            geom_path = resolve_geom_path(d, cam, geom_source)
            has_rgb = has_files(video_path, RGB_EXTS) or has_files(video_path / "left", RGB_EXTS)
            has_depth = has_files(depth_path, DEPTH_EXTS)
            if has_rgb and has_depth and geom_path is not None:
                has_camera = True
                break

        if has_camera:
            datasets.append(d)

    return datasets


def is_camera_output_complete(dataset_path: Path, camera_name: str) -> bool:
    """检查单相机输出是否完整。"""
    camera_dir = dataset_path / "trajectory" / camera_name
    main_npz = camera_dir / f"{camera_name}.npz"
    images_dir = camera_dir / "images"
    samples_dir = camera_dir / "samples"
    depth_dir = camera_dir / "depth"

    if not main_npz.exists() or not images_dir.is_dir() or not samples_dir.is_dir() or not depth_dir.is_dir():
        return False

    n_images = len(list(images_dir.glob("*.png")))
    n_samples = len(list(samples_dir.glob("*.npz")))
    n_depth_raw = len(list(depth_dir.glob("*_raw.npz")))
    return n_images > 0 and n_images == n_samples == n_depth_raw


def process_single_camera(dataset_path, camera_name, args, gpu_id, task_index, total_tasks, print_lock):
    """处理单个数据集的单个相机。"""
    dataset_name = dataset_path.name
    task_name = f"{dataset_name}/{camera_name}"

    with print_lock:
        print(f"[{task_index}/{total_tasks}] 开始 {task_name} (GPU {gpu_id})")

    video_path = dataset_path / "rgb_stereo_valid" / camera_name
    depth_path = resolve_depth_path(dataset_path, camera_name)
    geom_path = resolve_geom_path(dataset_path, camera_name, args.geom_source)
    trajectory_root = dataset_path / "trajectory"
    camera_out_dir = trajectory_root / camera_name
    logs_dir = trajectory_root / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    task_log = logs_dir / f"{camera_name}_external.log"

    has_rgb = has_files(video_path, RGB_EXTS) or has_files(video_path / "left", RGB_EXTS)
    has_depth = has_files(depth_path, DEPTH_EXTS)
    if (not has_rgb) or (not has_depth) or geom_path is None:
        with print_lock:
            print(f"⚠️  [{task_index}/{total_tasks}] {task_name} 跳过（缺少 RGB/深度/外部几何）")
        return (True, task_name, 0, None)

    if camera_out_dir.exists():
        shutil.rmtree(camera_out_dir)
    camera_out_dir.parent.mkdir(parents=True, exist_ok=True)

    python_bin = "/usr/local/miniconda3/envs/traceforge/bin/python"
    if not os.path.exists(python_bin):
        python_bin = sys.executable

    infer_script = os.path.join(_project_root, "scripts/batch_inference/infer.py")
    cmd = [
        python_bin, infer_script,
        "--depth_pose_method", "external",
        "--external_geom_npz", str(geom_path),
        "--camera_name", camera_name,
        "--depth_path", str(depth_path),
        "--video_path", str(video_path),
        "--video_name", camera_name,
        "--out_dir", str(trajectory_root),
        "--device", f"cuda:{gpu_id}",
        "--frame_drop_rate", str(args.frame_drop_rate),
        "--grid_size", str(args.grid_size),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = _project_root
    env["CUDA_HOME"] = "/usr/local/cuda-12.8"
    env["PATH"] = "/usr/local/cuda-12.8/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + env.get("LD_LIBRARY_PATH", "")

    task_start = time.time()
    timeout_s = None if args.task_timeout <= 0 else args.task_timeout
    try:
        with open(task_log, "w", encoding="utf-8") as f_log:
            subprocess.run(
                cmd,
                check=True,
                stdout=f_log,
                stderr=f_log,
                env=env,
                timeout=timeout_s,
            )
        elapsed = time.time() - task_start
        with print_lock:
            print(f"✅ [{task_index}/{total_tasks}] {task_name} 完成 ({elapsed/60:.1f}min, GPU{gpu_id})")
        return (True, task_name, elapsed, None)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - task_start
        with print_lock:
            print(f"❌ [{task_index}/{total_tasks}] {task_name} 超时 ({elapsed/60:.1f}min)")
            print(f"   timeout={args.task_timeout}s")
            if task_log.exists():
                lines = task_log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
        return (False, task_name, elapsed, f"timeout after {args.task_timeout}s")
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - task_start
        with print_lock:
            print(f"❌ [{task_index}/{total_tasks}] {task_name} 失败 ({elapsed/60:.1f}min)")
            if task_log.exists():
                lines = task_log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
        return (False, task_name, elapsed, str(e))
    except Exception as e:
        elapsed = time.time() - task_start
        with print_lock:
            print(f"❌ [{task_index}/{total_tasks}] {task_name} 异常: {e}")
        return (False, task_name, elapsed, str(e))


def process_tasks_on_gpu(gpu_id, gpu_tasks, args, total_tasks, print_lock):
    """每个 GPU 一个工作线程，串行处理分配到该 GPU 的任务。"""
    results = []
    for task_index, dataset, camera in gpu_tasks:
        result = process_single_camera(
            dataset, camera, args, gpu_id, task_index, total_tasks, print_lock
        )
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="DROID 批量推理（external 几何直通）")
    parser.add_argument("--base_path", type=str, required=True, help="DROID 数据集根目录")
    parser.add_argument("--gpu_id", type=str, default="0,1,2,3,4,5,6,7", help="GPU IDs，如 0,1,2,3")
    parser.add_argument("--frame_drop_rate", type=int, default=5, help="帧间隔")
    parser.add_argument("--grid_size", type=int, default=80, help="网格大小")
    parser.add_argument("--max_datasets", type=int, default=None, help="最多处理数据集数（测试用）")
    parser.add_argument("--max_workers", type=int, default=None, help="并行 worker 数，默认等于 GPU 数")
    parser.add_argument("--task_timeout", type=int, default=0, help="单任务超时秒数；<=0 表示不设超时")
    parser.add_argument("--only_incomplete", action="store_true", default=False, help="仅处理输出不完整的相机任务")
    parser.add_argument(
        "--geom_source",
        type=str,
        default="auto",
        choices=["auto", "npz", "h5"],
        help="外部几何来源：auto(优先 external_geom_*.npz, 回退 trajectory_valid.h5)",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path).resolve()
    if not base_path.is_dir():
        print(f"错误：数据集目录不存在: {base_path}")
        return

    datasets = find_droid_datasets(base_path, args.geom_source)
    if not datasets:
        print(f"未找到有效的 DROID 数据集: {base_path}")
        return

    if args.max_datasets is not None and args.max_datasets > 0:
        datasets = datasets[:args.max_datasets]
        print(f"限制处理前 {len(datasets)} 个数据集")

    print(f"找到 {len(datasets)} 个数据集")

    tasks = []
    for dataset in datasets:
        for camera in CAMERAS:
            if args.only_incomplete and is_camera_output_complete(dataset, camera):
                continue
            tasks.append((dataset, camera))

    print(f"总任务数: {len(tasks)} ({len(datasets)} 数据集 × {len(CAMERAS)} 相机)")
    if not tasks:
        print("没有需要处理的任务（可能都已完整）")
        return

    gpu_ids = [int(x.strip()) for x in args.gpu_id.split(",")]
    if not gpu_ids:
        print("错误：未提供有效 GPU ID")
        return
    max_workers = args.max_workers if args.max_workers is not None else len(gpu_ids)
    max_workers = max(1, min(max_workers, len(gpu_ids)))
    worker_gpus = gpu_ids[:max_workers]
    print(f"GPU: {worker_gpus}, 并行数: {max_workers} (每GPU单进程)\n")

    print_lock = threading.Lock()
    success_count = 0
    fail_count = 0
    task_times = []
    start_time = time.time()

    tasks_by_gpu = {gpu: [] for gpu in worker_gpus}
    for i, (dataset, camera) in enumerate(tasks, 1):
        gpu = worker_gpus[(i - 1) % len(worker_gpus)]
        tasks_by_gpu[gpu].append((i, dataset, camera))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_tasks_on_gpu, gpu, gpu_tasks, args, len(tasks), print_lock
            ): gpu
            for gpu, gpu_tasks in tasks_by_gpu.items()
            if gpu_tasks
        }
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                gpu_results = future.result()
                for ok, _name, elapsed, _err in gpu_results:
                    task_times.append(elapsed)
                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1
            except Exception as e:
                fail_count += len(tasks_by_gpu[gpu])
                with print_lock:
                    print(f"❌ GPU {gpu} worker 异常: {e}")

    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"完成: 成功 {success_count}/{len(tasks)}, 失败 {fail_count}")
    if task_times:
        avg = sum(task_times) / len(task_times)
        print(f"总耗时: {total_elapsed/3600:.1f} h, 平均每任务: {avg/60:.1f} min")


if __name__ == "__main__":
    main()
