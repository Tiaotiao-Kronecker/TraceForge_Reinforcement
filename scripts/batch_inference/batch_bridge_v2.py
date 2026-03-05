#!/usr/bin/env python3
"""
BridgeV2 批量推理：对 base_path 下所有轨迹调用 infer_bridge_v2（三相机 + lang/obs_dict/policy_out）。

与 batch_infer.py 的区别：
- 每条轨迹有 3 个相机，输出到 out_dir/{traj_id}/images0、images1、images2
- 会复制 lang.txt、obs_dict.pkl、policy_out.pkl 到每条轨迹输出目录
- 使用修正后的帧序（按数字排序），与旧单相机批处理结果可对比

容量预估（基于实测，且考虑 frame_drop_rate）：
- 参考：单相机 frame_drop_rate=1、grid_size=20 约 142MB/条（如 07985）
- 三相机、frame_drop_rate=1：约 426MB/条（grid_size=20）或 884MB/条（grid_size=80）
- 帧相关部分随 frame_drop_rate 近似按 1/fdr 缩放；批处理常用 fdr=5 → 约 1/5 体量
- 10500 条、grid_size=20：fdr=1 约 4.47 TB，fdr=5 约 0.9 TB
- 10500 条、grid_size=80：fdr=1 约 9.3 TB，fdr=5 约 1.9 TB

用法:
    python scripts/batch_inference/batch_bridge_v2.py \
        --base_path /data1/dataset/dataset/opt/dataset_temp/bridge_depth \
        --out_dir /data1/wangchen/projects/TraceForge/output_bridge_v2_full \
        --skip_existing \
        --frame_drop_rate 5 \
        --gpu_id 0,1,2,3

    测试（只跑前 5 条）:
    python scripts/batch_inference/batch_bridge_v2.py \
        --base_path /data1/dataset/dataset/opt/dataset_temp/bridge_depth \
        --out_dir ./output_bridge_v2_test \
        --max_trajs 5 \
        --gpu_id 0
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

# 参考：单相机单条轨迹（frame_drop_rate=1，grid_size=20）约 142MB
SINGLE_CAM_TRAJ_MB_FDR1 = 142
# 三相机单条轨迹（frame_drop_rate=1）≈ 3×单相机；grid_size=80 时约 884MB（实测 00000）
THREE_CAM_TRAJ_MB_FDR1 = SINGLE_CAM_TRAJ_MB_FDR1 * 3  # ~426
THREE_CAM_TRAJ_MB_GRID80_FDR1 = 884
# 每轨迹固定开销（元数据等）约 1MB，其余随帧数近似按 1/frame_drop_rate 缩放
TRAJ_OVERHEAD_MB = 1
CAMERAS_PER_TRAJ = 3


def find_bridge_v2_trajs(base_path):
    """与 infer_bridge_v2 一致：扫描满足 BridgeV2 结构的轨迹目录。"""
    base = Path(base_path).resolve()
    if not base.is_dir():
        return []
    trajs = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        for cam in range(3):
            im_dir = d / f"images{cam}"
            dep_dir = d / f"depth_images{cam}"
            if im_dir.is_dir() and dep_dir.is_dir():
                has_im = any(
                    f.suffix.lower() in (".jpg", ".jpeg", ".png")
                    for f in im_dir.iterdir() if f.is_file()
                )
                has_dep = any(
                    f.suffix.lower() in (".png", ".jpg", ".jpeg")
                    for f in dep_dir.iterdir() if f.is_file()
                )
                if has_im and has_dep:
                    trajs.append(d)
                    break
    return trajs


def estimate_disk_mb(num_trajs, grid_size=20, frame_drop_rate=1):
    """
    预估总占用（MB）。帧相关部分按 1/frame_drop_rate 缩放（fdr=5 约为 fdr=1 的 1/5）。
    """
    fdr = max(1, int(frame_drop_rate))
    if grid_size is not None and grid_size > 20:
        # 可变部分 ≈ (884 - 1) / fdr，固定 ≈ 1
        per_traj = TRAJ_OVERHEAD_MB + (THREE_CAM_TRAJ_MB_GRID80_FDR1 - TRAJ_OVERHEAD_MB) / fdr
    else:
        per_traj = TRAJ_OVERHEAD_MB + (THREE_CAM_TRAJ_MB_FDR1 - TRAJ_OVERHEAD_MB) / fdr
    return num_trajs * per_traj


def get_fs_avail_mb(path):
    """获取 path 所在文件系统可用空间（MB）。"""
    path = Path(path).resolve()
    path = path if path.is_dir() else path.parent
    if not path.exists():
        path = path.parent
    try:
        stat = os.statvfs(path)
        return (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
    except Exception:
        return None


def process_single_traj(traj_id, args, bridge_v2_script, gpu_ids, traj_index, total_trajs, print_lock):
    """对单条轨迹执行 infer_bridge_v2.py（三相机 + 元数据）。"""
    traj_out_dir = os.path.join(args.out_dir, traj_id)
    if gpu_ids and isinstance(gpu_ids[0], int):
        device = f"cuda:{gpu_ids[(traj_index - 1) % len(gpu_ids)]}"
    else:
        device = gpu_ids[0] if gpu_ids else "cuda"

    with print_lock:
        print(f"[{traj_index}/{total_trajs}] 开始 {traj_id} (device={device})")

    conda_env_python = "/home/wangchen/.conda/envs/traceforge/bin/python"
    if not os.path.exists(conda_env_python):
        conda_env_python = "/usr/local/miniconda3/envs/traceforge/bin/python"
    if not os.path.exists(conda_env_python):
        conda_env_python = sys.executable

    cmd = [
        conda_env_python, bridge_v2_script,
        "--base_path", args.base_path,
        "--out_dir", args.out_dir,
        "--traj_id", traj_id,
        "--device", device,
        "--frame_drop_rate", str(args.frame_drop_rate),
        "--max_cameras", str(getattr(args, "max_cameras", 3)),
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")
    if getattr(args, "grid_size", None) is not None:
        cmd.extend(["--grid_size", str(args.grid_size)])
    if getattr(args, "depth_only", False):
        cmd.append("--depth_only")

    env = os.environ.copy()
    env["PYTHONPATH"] = _project_root + (os.pathsep + env.get("PYTHONPATH", "")) if env.get("PYTHONPATH") else _project_root
    if os.path.exists(conda_env_python):
        env["PATH"] = os.path.dirname(conda_env_python) + os.pathsep + env.get("PATH", "")

    traj_start = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, timeout=3600)
        elapsed = time.time() - traj_start
        with print_lock:
            print(f"✅ [{traj_index}/{total_trajs}] {traj_id} 完成 ({elapsed:.1f}s, {device})")
        return (True, traj_id, elapsed, None)
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - traj_start
        with print_lock:
            print(f"❌ [{traj_index}/{total_trajs}] {traj_id} 失败 ({elapsed:.1f}s): 返回码 {e.returncode}")
            if e.stderr:
                for line in e.stderr.strip().split("\n")[-15:]:
                    if line.strip():
                        print(f"   {line}")
        return (False, traj_id, elapsed, str(e))
    except Exception as e:
        elapsed = time.time() - traj_start
        with print_lock:
            print(f"❌ [{traj_index}/{total_trajs}] {traj_id} 异常: {e}")
        return (False, traj_id, elapsed, str(e))


def main():
    parser = argparse.ArgumentParser(description="BridgeV2 批量推理（三相机 + 元数据）")
    parser.add_argument("--base_path", type=str, required=True, help="BridgeV2 数据集根目录")
    parser.add_argument("--out_dir", type=str, default="./output_bridge_v2_batch", help="输出根目录")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已有完整输出的轨迹")
    parser.add_argument("--frame_drop_rate", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=None, help="默认 20，可设为 80 等")
    parser.add_argument("--max_cameras", type=int, default=3)
    parser.add_argument("--max_trajs", type=int, default=None, help="最多处理条数（测试用）")
    parser.add_argument("--start_after", type=str, default=None, help="从该轨迹 ID 之后开始（如 08669 表示从 08670 起，用于续跑）")
    parser.add_argument("--gpu_id", type=str, default=None, help="如 0,1,2,3")
    parser.add_argument("--max_workers", type=int, default=None, help="并行数，默认等于 GPU 数")
    parser.add_argument("--no_parallel", action="store_true", help="串行执行")
    parser.add_argument("--depth_only", action="store_true", help="仅输出深度相关结果（RGB/深度/位姿），不生成 keypoint 轨迹及 samples/*.npz")
    parser.add_argument("--scene_group_size", type=int, default=None, help="按轨迹排序后，每 scene_group_size 条视为一个场景")
    parser.add_argument("--scene_skip_groups", type=int, default=0, help="跳过前多少个场景组（与 scene_group_size 联用）")
    args = parser.parse_args()

    base_path = Path(args.base_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs = find_bridge_v2_trajs(base_path)
    if not trajs:
        print(f"未找到 BridgeV2 轨迹: {base_path}")
        return
    # 按名称排序保证分组稳定
    trajs = sorted(trajs, key=lambda p: p.name)
    traj_ids = [d.name for d in trajs]
    total_count = len(traj_ids)
    if getattr(args, "start_after", None):
        traj_ids = [t for t in traj_ids if t > args.start_after]
        print(f"续跑：从 {args.start_after} 之后开始，共 {len(traj_ids)} 条待处理（总 {total_count} 条）")

    # 按“每 N 条为一组场景，只取每组第一个”的方式筛选轨迹，
    # 例如 scene_group_size=5, scene_skip_groups=1 → 从第 2 组开始，每组只取第 1 条。
    if getattr(args, "scene_group_size", None) is not None and args.scene_group_size > 0:
        g = args.scene_group_size
        skip_g = max(0, int(args.scene_skip_groups))
        selected = []
        for idx, tid in enumerate(traj_ids):
            group_idx = idx // g
            if group_idx < skip_g:
                continue
            if idx % g == 0:
                selected.append(tid)
        print(f"按 scene_group_size={g}, scene_skip_groups={skip_g} 过滤后，将处理 {len(selected)} 条轨迹（原始 {len(traj_ids)} 条）")
        traj_ids = selected
    if args.max_trajs is not None and args.max_trajs > 0:
        traj_ids = traj_ids[: args.max_trajs]
        print(f"限制处理前 {len(traj_ids)} 条")
    if not traj_ids:
        print("没有待处理轨迹")
        return
    print(f"将处理 {len(traj_ids)} 条轨迹")

    # 容量预估与磁盘检查（考虑 frame_drop_rate）
    fdr = getattr(args, "frame_drop_rate", 5)
    est_mb = estimate_disk_mb(len(traj_ids), getattr(args, "grid_size", None), fdr)
    est_gb = est_mb / 1024
    avail_mb = get_fs_avail_mb(out_dir)
    print(f"\n📊 容量预估 (frame_drop_rate={fdr}, {len(traj_ids)} 条): 约 {est_gb:.2f} GB ({est_mb:.0f} MB)")
    if avail_mb is not None:
        avail_gb = avail_mb / 1024
        print(f"   输出目录所在盘可用: {avail_gb:.2f} GB ({avail_mb:.0f} MB)")
        if avail_mb < est_mb * 1.1:
            print(f"   ⚠️  可用空间可能不足，建议预留 ≥ {est_gb * 1.15:.1f} GB 或更换输出目录")
        else:
            print(f"   ✓ 当前空间充足")
    else:
        print("   (无法读取磁盘可用空间)")

    # GPU
    try:
        import torch
        if args.gpu_id:
            gpu_ids = [int(x.strip()) for x in args.gpu_id.split(",")]
            gpu_ids = [g for g in gpu_ids if 0 <= g < torch.cuda.device_count()]
        else:
            gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else ["cuda"]
    except Exception:
        gpu_ids = [0] if args.gpu_id is None else [int(x.strip()) for x in args.gpu_id.split(",")]

    max_workers = 1
    if not args.no_parallel and gpu_ids and isinstance(gpu_ids[0], int):
        max_workers = args.max_workers if args.max_workers is not None else len(gpu_ids)
    print(f"🎮 GPU: {gpu_ids}, 并行数: {max_workers}\n")

    bridge_v2_script = os.path.join(os.path.dirname(__file__), "infer_bridge_v2.py")
    print_lock = threading.Lock()
    success_count = 0
    fail_count = 0
    traj_times = []
    start_time = time.time()

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            f2i = {
                executor.submit(
                    process_single_traj,
                    tid, args, bridge_v2_script, gpu_ids, i + 1, len(traj_ids), print_lock
                ): (i + 1, tid)
                for i, tid in enumerate(traj_ids)
            }
            for future in as_completed(f2i):
                try:
                    ok, name, elapsed, err = future.result()
                    traj_times.append(elapsed)
                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    with print_lock:
                        print(f"❌ 任务异常: {e}")
    else:
        for i, tid in enumerate(traj_ids, 1):
            ok, name, elapsed, err = process_single_traj(
                tid, args, bridge_v2_script, gpu_ids, i, len(traj_ids), print_lock
            )
            traj_times.append(elapsed)
            if ok:
                success_count += 1
            else:
                fail_count += 1

    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"完成: 成功 {success_count}/{len(traj_ids)}, 失败 {fail_count}")
    print(f"输出: {out_dir}")
    if traj_times:
        avg = sum(traj_times) / len(traj_times)
        print(f"总耗时: {total_elapsed/60:.1f} min, 平均每条: {avg:.1f} s")
        if total_count > len(traj_ids) and success_count > 0:
            est_full = avg * total_count
            print(f"若跑满 {total_count} 条（当前并行 {max_workers}）: 约 {est_full/3600:.1f} h")


if __name__ == "__main__":
    main()
