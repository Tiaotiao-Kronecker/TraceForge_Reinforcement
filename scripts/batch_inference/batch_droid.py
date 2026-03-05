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
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

CAMERAS = ["hand_camera", "varied_camera_1", "varied_camera_2"]


def find_droid_datasets(base_path):
    """扫描DROID数据集目录，返回有效数据集列表"""
    base = Path(base_path).resolve()
    if not base.is_dir():
        return []

    datasets = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue

        # 检查是否有必需的目录结构
        h5_file = d / "trajectory_valid.h5"
        rgb_dir = d / "rgb_stereo_valid"
        depth_dir = d / "depth"

        if not (h5_file.exists() and rgb_dir.is_dir() and depth_dir.is_dir()):
            continue

        # 检查是否至少有一个相机的数据
        has_camera = False
        for cam in CAMERAS:
            rgb_cam = rgb_dir / cam / "left"
            depth_cam = depth_dir / cam / "depth"
            if rgb_cam.is_dir() and depth_cam.is_dir():
                has_camera = True
                break

        if has_camera:
            datasets.append(d)

    return datasets


def process_single_camera(dataset_path, camera_name, args, gpu_id, task_index, total_tasks, print_lock):
    """处理单个数据集的单个相机"""
    dataset_name = dataset_path.name
    task_name = f"{dataset_name}/{camera_name}"

    with print_lock:
        print(f"[{task_index}/{total_tasks}] 开始 {task_name} (GPU {gpu_id})")

    # 构建路径
    h5_file = dataset_path / "trajectory_valid.h5"
    video_path = dataset_path / "rgb_stereo_valid" / camera_name / "left"
    depth_path = dataset_path / "depth" / camera_name / "depth"
    out_dir = dataset_path / "trajectory"

    # 检查路径是否存在
    if not video_path.is_dir() or not depth_path.is_dir():
        with print_lock:
            print(f"⚠️  [{task_index}/{total_tasks}] {task_name} 跳过（路径不存在）")
        return (True, task_name, 0, None)  # 标记为成功但跳过

    # Python路径
    python_bin = "/usr/local/miniconda3/envs/traceforge/bin/python"
    if not os.path.exists(python_bin):
        python_bin = sys.executable

    # 构建命令
    infer_script = os.path.join(_project_root, "scripts/batch_inference/infer.py")
    cmd = [
        python_bin, infer_script,
        "--depth_pose_method", "external",
        "--external_geom_npz", str(h5_file),
        "--camera_name", camera_name,
        "--depth_path", str(depth_path),
        "--video_path", str(video_path),
        "--out_dir", str(out_dir),
        "--device", f"cuda:{gpu_id}",
        "--frame_drop_rate", str(args.frame_drop_rate),
        "--grid_size", str(args.grid_size),
    ]

    # 环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = _project_root
    env["CUDA_HOME"] = "/usr/local/cuda-12.8"
    env["PATH"] = "/usr/local/cuda-12.8/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + env.get("LD_LIBRARY_PATH", "")

    task_start = time.time()
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, env=env, timeout=7200
        )
        elapsed = time.time() - task_start
        with print_lock:
            print(f"✅ [{task_index}/{total_tasks}] {task_name} 完成 ({elapsed/60:.1f}min, GPU{gpu_id})")
        return (True, task_name, elapsed, None)
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - task_start
        with print_lock:
            print(f"❌ [{task_index}/{total_tasks}] {task_name} 失败 ({elapsed/60:.1f}min)")
            if e.stderr:
                for line in e.stderr.strip().split("\n")[-10:]:
                    if line.strip():
                        print(f"   {line}")
        return (False, task_name, elapsed, str(e))
    except Exception as e:
        elapsed = time.time() - task_start
        with print_lock:
            print(f"❌ [{task_index}/{total_tasks}] {task_name} 异常: {e}")
        return (False, task_name, elapsed, str(e))


def main():
    parser = argparse.ArgumentParser(description="DROID数据集批量推理（多数据集×3相机）")
    parser.add_argument("--base_path", type=str, required=True, help="DROID数据集根目录")
    parser.add_argument("--gpu_id", type=str, default="0,1,2,3,4,5,6,7", help="GPU IDs，如 0,1,2,3")
    parser.add_argument("--frame_drop_rate", type=int, default=5, help="帧间隔")
    parser.add_argument("--grid_size", type=int, default=80, help="网格大小")
    parser.add_argument("--max_datasets", type=int, default=None, help="最多处理数据集数（测试用）")
    parser.add_argument("--max_workers", type=int, default=None, help="并行数，默认等于GPU数")
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
            tasks.append((dataset, camera))

    print(f"总任务数: {len(tasks)} ({len(datasets)} 数据集 × {len(CAMERAS)} 相机)")

    # GPU配置
    gpu_ids = [int(x.strip()) for x in args.gpu_id.split(",")]
    max_workers = args.max_workers if args.max_workers is not None else len(gpu_ids)
    print(f"GPU: {gpu_ids}, 并行数: {max_workers}\n")

    print_lock = threading.Lock()
    success_count = 0
    fail_count = 0
    task_times = []
    start_time = time.time()

    # 并行执行
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, (dataset, camera) in enumerate(tasks, 1):
                gpu_id = gpu_ids[(i - 1) % len(gpu_ids)]
                future = executor.submit(
                    process_single_camera,
                    dataset, camera, args, gpu_id, i, len(tasks), print_lock
                )
                futures[future] = (i, f"{dataset.name}/{camera}")

            for future in as_completed(futures):
                try:
                    ok, name, elapsed, err = future.result()
                    task_times.append(elapsed)
                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    with print_lock:
                        print(f"❌ 任务异常: {e}")
    else:
        # 串行执行
        for i, (dataset, camera) in enumerate(tasks, 1):
            gpu_id = gpu_ids[0]
            ok, name, elapsed, err = process_single_camera(
                dataset, camera, args, gpu_id, i, len(tasks), print_lock
            )
            task_times.append(elapsed)
            if ok:
                success_count += 1
            else:
                fail_count += 1

    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"完成: 成功 {success_count}/{len(tasks)}, 失败 {fail_count}")
    if task_times:
        avg = sum(task_times) / len(task_times)
        print(f"总耗时: {total_elapsed/3600:.1f} h, 平均每任务: {avg/60:.1f} min")


if __name__ == "__main__":
    main()
