#!/usr/bin/env python3
"""
DROID数据集批量推理脚本：处理多个数据集，每个数据集3个相机，使用多GPU并行

用法:
    python scripts/batch_inference/batch_droid.py \
        --base_path /home/zoyo/projects/droid_preprocess_pipeline/droid_raw \
        --gpu_id 0,1,2,3,4,5,6,7 \
        --frame_drop_rate 5 \
        --grid_size 80

    测试（只跑前2个数据集）:
    python scripts/batch_inference/batch_droid.py \
        --base_path /home/zoyo/projects/droid_preprocess_pipeline/droid_raw \
        --gpu_id 0,1 \
        --max_datasets 2 \
        --frame_drop_rate 300

说明:
    该脚本默认走 TraceForge 的纯 VGGT 流程（仅 RGB），不再传入外部深度或外参。
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

from utils.traceforge_artifact_utils import is_traceforge_output_complete

CAMERAS = ["hand_camera", "varied_camera_1", "varied_camera_2"]
FRAME_EXTS = ("*.jpg", "*.jpeg", "*.png")


def has_frame_files(dir_path: Path) -> bool:
    """检查目录下是否有图像帧文件。"""
    if not dir_path.is_dir():
        return False
    for pattern in FRAME_EXTS:
        if any(dir_path.glob(pattern)):
            return True
    return False


def find_droid_datasets(base_path):
    """扫描DROID数据集目录，返回有效数据集列表"""
    base = Path(base_path).resolve()
    if not base.is_dir():
        return []

    datasets = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue

        # 纯VGGT流程只要求RGB目录结构存在
        rgb_dir = d / "rgb_stereo_valid"
        if not rgb_dir.is_dir():
            continue

        # 检查是否至少有一个相机的数据
        has_camera = False
        for cam in CAMERAS:
            cam_root = rgb_dir / cam
            if has_frame_files(cam_root) or has_frame_files(cam_root / "left"):
                has_camera = True
                break

        if has_camera:
            datasets.append(d)

    return datasets


def is_camera_output_complete(dataset_path: Path, camera_name: str) -> bool:
    """检查单相机输出是否完整。"""
    camera_dir = dataset_path / "trajectory" / camera_name
    return is_traceforge_output_complete(camera_dir)


def process_single_camera(dataset_path, camera_name, args, gpu_id, task_index, total_tasks, print_lock):
    """处理单个数据集的单个相机"""
    dataset_name = dataset_path.name
    task_name = f"{dataset_name}/{camera_name}"

    with print_lock:
        print(f"[{task_index}/{total_tasks}] 开始 {task_name} (GPU {gpu_id})")

    # 构建路径
    # 传相机目录本身；infer.py 内部会在必要时自动回退到 left 子目录读取帧
    video_path = dataset_path / "rgb_stereo_valid" / camera_name
    trajectory_root = dataset_path / "trajectory"
    camera_out_dir = trajectory_root / camera_name
    logs_dir = trajectory_root / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    task_log = logs_dir / f"{camera_name}.log"

    # 检查路径是否存在
    if (not video_path.is_dir()) or (not has_frame_files(video_path) and not has_frame_files(video_path / "left")):
        with print_lock:
            print(f"⚠️  [{task_index}/{total_tasks}] {task_name} 跳过（路径不存在）")
        return (True, task_name, 0, None)  # 标记为成功但跳过

    # 覆盖旧结果：每个相机目录单独清理
    if camera_out_dir.exists():
        shutil.rmtree(camera_out_dir)
    camera_out_dir.parent.mkdir(parents=True, exist_ok=True)

    # Python路径
    python_bin = "/usr/local/miniconda3/envs/traceforge/bin/python"
    if not os.path.exists(python_bin):
        python_bin = sys.executable

    # 构建命令
    infer_script = os.path.join(_project_root, "scripts/batch_inference/infer.py")
    cmd = [
        python_bin, infer_script,
        "--video_path", str(video_path),
        "--video_name", camera_name,
        "--out_dir", str(trajectory_root),
        "--device", f"cuda:{gpu_id}",
        "--frame_drop_rate", str(args.frame_drop_rate),
        "--grid_size", str(args.grid_size),
        "--output_layout", args.output_layout,
    ]
    if args.save_visibility:
        cmd.append("--save_visibility")

    # 环境变量
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
    parser = argparse.ArgumentParser(description="DROID数据集批量推理（多数据集×3相机）")
    parser.add_argument("--base_path", type=str, required=True, help="DROID数据集根目录")
    parser.add_argument("--gpu_id", type=str, default="0,1,2,3,4,5,6,7", help="GPU IDs，如 0,1,2,3")
    parser.add_argument("--frame_drop_rate", type=int, default=5, help="帧间隔")
    parser.add_argument("--grid_size", type=int, default=80, help="网格大小")
    parser.add_argument("--max_datasets", type=int, default=None, help="最多处理数据集数（测试用）")
    parser.add_argument("--max_workers", type=int, default=None, help="并行 worker 数，默认等于 GPU 数")
    parser.add_argument("--task_timeout", type=int, default=0, help="单任务超时秒数；<=0 表示不设超时")
    parser.add_argument("--only_incomplete", action="store_true", default=False, help="仅处理输出不完整的相机任务")
    parser.add_argument(
        "--output_layout",
        type=str,
        default="v2",
        choices=["v2", "legacy"],
        help="Artifact layout passed through to infer.py.",
    )
    parser.add_argument(
        "--save_visibility",
        action="store_true",
        default=False,
        help="Store per-query visibility arrays in sample NPZ files.",
    )
    args = parser.parse_args()

    base_path = Path(args.base_path).resolve()
    if not base_path.is_dir():
        print(f"错误：数据集目录不存在: {base_path}")
        return

    # 扫描数据集
    datasets = find_droid_datasets(base_path)
    if not datasets:
        print(f"未找到有效的DROID数据集: {base_path}")
        return

    if args.max_datasets is not None and args.max_datasets > 0:
        datasets = datasets[:args.max_datasets]
        print(f"限制处理前 {len(datasets)} 个数据集")

    print(f"找到 {len(datasets)} 个数据集")

    # 创建任务列表：(dataset_path, camera_name)
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

    # GPU配置
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

    # 将任务按 worker GPU 轮转分配；每个 GPU worker 串行执行，避免单卡并发过载
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
