#!/usr/bin/env python3
"""
快速检查 output_bridge_depth_grid80 推理结果，找出生成失败的样本

失败定义：
1. 无输出目录（未处理）
2. 空目录
3. 缺少 images0/images 或 images0/samples 或两者为空
4. images 或 samples 目录无有效文件

用法:
    python check_failed_inference.py --out_dir /home/wangchen/projects/TraceForge/output_bridge_depth_grid80
    python check_failed_inference.py --out_dir ./output_bridge_depth_grid80 --dataset_path /usr/data/dataset/opt/dataset_temp/bridge_depth  # 对比数据集找出未处理的
"""

import os
import argparse
from pathlib import Path
from typing import Tuple


def is_success(traj_dir: Path) -> Tuple[bool, str]:
    """
    检查单个 traj 输出是否成功。
    返回 (是否成功, 失败原因)
    """
    images_dir = traj_dir / "images0" / "images"
    samples_dir = traj_dir / "images0" / "samples"

    if not images_dir.exists():
        return False, "缺少 images0/images"
    if not samples_dir.exists():
        return False, "缺少 images0/samples"

    # 检查是否有 PNG
    images = list(images_dir.glob("*.png"))
    if not images:
        return False, "images 目录为空"

    # 检查是否有 NPZ
    samples = list(samples_dir.glob("*.npz"))
    if not samples:
        return False, "samples 目录为空"

    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str,
                        default="/home/wangchen/projects/TraceForge/output_bridge_depth_grid80",
                        help="推理输出目录")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="数据集路径，若提供则对比找出未处理的样本")
    parser.add_argument("--save", type=str, default=None,
                        help="保存失败样本 ID 到文件")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        print(f"❌ 输出目录不存在: {out_dir}")
        return

    # 获取所有输出 traj
    traj_dirs = sorted([d for d in out_dir.iterdir() if d.is_dir()])
    n_total = len(traj_dirs)

    failed = []
    for d in traj_dirs:
        ok, reason = is_success(d)
        if not ok:
            failed.append((d.name, reason))

    n_failed = len(failed)
    n_ok = n_total - n_failed

    print("=" * 60)
    print("推理结果快速检查")
    print("=" * 60)
    print(f"输出目录: {out_dir}")
    print(f"输出 traj 总数: {n_total}")
    print(f"✅ 成功: {n_ok}")
    print(f"❌ 失败: {n_failed}")
    if n_total > 0:
        print(f"成功率: {n_ok/n_total*100:.2f}%")
    print()

    if failed:
        print("失败样本 (前 30 个):")
        for tid, reason in failed[:30]:
            print(f"  {tid}: {reason}")
        if len(failed) > 30:
            print(f"  ... 共 {len(failed)} 个失败")

    # 对比数据集找出未处理的
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        if dataset_path.exists():
            expected = set(d.name for d in dataset_path.iterdir()
                          if d.is_dir() and (d / "images0").exists() and (d / "depth_images0").exists())
            output_ids = set(d.name for d in traj_dirs)
            not_processed = sorted(expected - output_ids)
            print(f"\n未处理样本 (数据集有但无输出): {len(not_processed)} 个")
            if not_processed:
                print(f"  示例: {not_processed[:15]}")
                if len(not_processed) > 15:
                    print(f"  ... 共 {len(not_processed)} 个")
            failed_ids = [tid for tid, _ in failed]
            all_failed = failed_ids + not_processed
        else:
            all_failed = [tid for tid, _ in failed]
    else:
        all_failed = [tid for tid, _ in failed]

    if args.save and all_failed:
        with open(args.save, "w") as f:
            f.write("\n".join(sorted(set(all_failed))))
        print(f"\n已保存失败/未处理样本 ID 到: {args.save} (共 {len(all_failed)} 个)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
