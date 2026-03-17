#!/usr/bin/env python3
"""
TraceForge 推理数据集采样脚本

从大量 TraceForge 推理结果中采样指定数量的 case，支持随机采样与均匀采样。
输出采样结果到文件，并可选择复制或创建软链接到目标目录。

当前脚本按 `legacy` case 目录结构工作，要求每个 case 下存在
`images0/images` 与 `images0/samples`。它不会解析 press-one-button demo
当前默认的 `v2` scene cache 布局。

用法:
    python sample_traceforge_dataset.py \
        --data_dir /data2/dataset/output_bridge_v2_full_grid80 \
        --n_sample 1000 \
        --output_list sampled_1000.txt

    # 复制到新目录
    python sample_traceforge_dataset.py \
        --data_dir /data2/dataset/output_bridge_v2_full_grid80 \
        --n_sample 1000 \
        --output_dir /data2/dataset/output_bridge_v2_sampled_1000 \
        --mode copy

    # 仅有效样本（通过完整性检查）
    python sample_traceforge_dataset.py \
        --data_dir /data2/dataset/output_bridge_v2_full_grid80 \
        --n_sample 1000 \
        --valid_only
"""

import argparse
import random
import shutil
from pathlib import Path


def is_valid_case(case_dir: Path) -> bool:
    """检查 legacy case 是否有效（含 images0/images 与 samples）"""
    images_dir = case_dir / "images0" / "images"
    samples_dir = case_dir / "images0" / "samples"
    if not images_dir.exists() or not samples_dir.exists():
        return False
    if not list(images_dir.glob("*.png")) or not list(samples_dir.glob("*.npz")):
        return False
    return True


def get_all_cases(data_dir: Path, valid_only: bool = False) -> list[str]:
    """获取所有 case ID 列表"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_dirs = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if valid_only:
        all_dirs = [d for d in all_dirs if is_valid_case(data_dir / d)]
    return all_dirs


def sample_random(cases: list[str], n: int, seed: int = None) -> list[str]:
    """随机采样"""
    if seed is not None:
        random.seed(seed)
    n = min(n, len(cases))
    return sorted(random.sample(cases, n))


def sample_uniform(cases: list[str], n: int) -> list[str]:
    """均匀采样（按索引等间隔选取）"""
    n = min(n, len(cases))
    if n >= len(cases):
        return cases
    indices = [int(i * (len(cases) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    return [cases[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description="TraceForge 数据集采样")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data2/dataset/output_bridge_v2_full_grid80",
        help="TraceForge legacy 推理输出根目录",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=1000,
        help="采样数量（默认 1000）",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "uniform"],
        default="random",
        help="采样方式：random 随机，uniform 均匀",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（仅 random 有效，默认 42）",
    )
    parser.add_argument(
        "--valid_only",
        action="store_true",
        help="仅从有效 legacy case 中采样（通过 images0/images 与 samples 检查）",
    )
    parser.add_argument(
        "--output_list",
        type=str,
        default=None,
        help="保存采样 ID 列表到文件（每行一个 ID）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="目标目录，配合 --mode 复制或链接采样结果",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["copy", "symlink"],
        default="copy",
        help="输出模式：copy 复制，symlink 软链接（需 --output_dir）",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cases = get_all_cases(data_dir, valid_only=args.valid_only)
    n_total = len(cases)

    if n_total == 0:
        print("❌ 未找到有效 case")
        return

    if args.method == "random":
        sampled = sample_random(cases, args.n_sample, seed=args.seed)
    else:
        sampled = sample_uniform(cases, args.n_sample)

    n_sampled = len(sampled)
    print(f"总 case 数: {n_total}")
    print(f"采样数量: {n_sampled}")
    print(f"采样方式: {args.method}" + (f" (seed={args.seed})" if args.method == "random" else ""))

    if args.output_list:
        out_path = Path(args.output_list)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(sampled) + "\n")
        print(f"✅ 采样 ID 已保存: {out_path}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for cid in sampled:
            src = data_dir / cid
            dst = out_dir / cid
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                if args.mode == "copy":
                    shutil.copytree(src, dst)
                else:
                    dst.symlink_to(src.resolve())
        print(f"✅ 已{args.mode} {n_sampled} 个 case 到 {out_dir}")


if __name__ == "__main__":
    main()
