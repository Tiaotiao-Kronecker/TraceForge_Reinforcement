#!/usr/bin/env python3
"""
TraceForge / press-one-button 数据集采样脚本。

支持两类布局：
- legacy TraceForge 推理输出
- press-one-button / sim episode 原始目录

可输出采样列表、manifest，并可复制或创建软链接到目标目录。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


SUPPORTED_LAYOUTS = ("legacy", "press_one_button")


def is_valid_legacy_case(case_dir: Path) -> bool:
    """检查 legacy case 是否有效（含 images0/images 与 samples）"""
    images_dir = case_dir / "images0" / "images"
    samples_dir = case_dir / "images0" / "samples"
    if not images_dir.exists() or not samples_dir.exists():
        return False
    if not list(images_dir.glob("*.png")) or not list(samples_dir.glob("*.npz")):
        return False
    return True


def is_valid_press_one_button_episode(case_dir: Path) -> bool:
    """检查 press-one-button episode 是否有效。"""
    if not case_dir.is_dir() or not case_dir.name.startswith("episode_"):
        return False
    if not (case_dir / "trajectory_valid.h5").is_file():
        return False
    rgb_root = case_dir / "rgb"
    depth_root = case_dir / "depth"
    if not rgb_root.is_dir() or not depth_root.is_dir():
        return False

    for rgb_dir in sorted(path for path in rgb_root.iterdir() if path.is_dir()):
        depth_dir = depth_root / rgb_dir.name
        has_rgb = any(path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in rgb_dir.iterdir())
        has_depth = depth_dir.is_dir() and any(
            path.is_file() and path.suffix.lower() in {".npy", ".png"} for path in depth_dir.iterdir()
        )
        if has_rgb and has_depth:
            return True
    return False


def is_valid_case(case_dir: Path, *, layout: str) -> bool:
    if layout == "legacy":
        return is_valid_legacy_case(case_dir)
    if layout == "press_one_button":
        return is_valid_press_one_button_episode(case_dir)
    raise ValueError(f"Unsupported layout: {layout}")


def get_all_cases(
    data_dir: Path,
    *,
    valid_only: bool = False,
    layout: str = "legacy",
) -> list[str]:
    """获取所有 case ID 列表。"""
    data_dir = Path(data_dir)
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(f"layout must be one of {SUPPORTED_LAYOUTS}, got: {layout}")
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    all_dirs = sorted(
        d.name
        for d in data_dir.iterdir()
        if d.is_dir() and (layout != "press_one_button" or d.name.startswith("episode_"))
    )
    if valid_only:
        all_dirs = [d for d in all_dirs if is_valid_case(data_dir / d, layout=layout)]
    return all_dirs


def sample_random(cases: list[str], n: int, seed: int | None = None) -> list[str]:
    """随机采样。"""
    if seed is not None:
        random.seed(seed)
    n = min(n, len(cases))
    return sorted(random.sample(cases, n))


def sample_uniform(cases: list[str], n: int) -> list[str]:
    """均匀采样（按索引等间隔选取）。"""
    n = min(n, len(cases))
    if n >= len(cases):
        return cases
    indices = [int(i * (len(cases) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    return [cases[i] for i in indices]


def parse_exclude_dir_names(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_sample_manifest(
    *,
    data_dir: Path,
    sampled: list[str],
    layout: str,
    method: str,
    n_requested: int,
    seed: int | None,
    exclude_dir_names: list[str],
    output_dir: Path | None,
) -> dict[str, Any]:
    resolved_output_dir = output_dir.resolve() if output_dir is not None else None
    return {
        "dataset_root": str(resolved_output_dir or data_dir.resolve()),
        "source_dataset_root": str(data_dir.resolve()),
        "sampled_dataset_root": str(resolved_output_dir) if resolved_output_dir is not None else None,
        "layout": layout,
        "sample_method": method,
        "seed": seed,
        "n_requested": int(n_requested),
        "n_sampled": int(len(sampled)),
        "excluded_dir_names": list(exclude_dir_names),
        "episodes": list(sampled),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _copy_case_tree(src: Path, dst: Path, *, exclude_dir_names: list[str]) -> None:
    ignore = shutil.ignore_patterns(*exclude_dir_names) if exclude_dir_names else None
    shutil.copytree(src, dst, ignore=ignore)


def _symlink_case_tree(src: Path, dst: Path, *, exclude_dir_names: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=False)
    excluded = set(exclude_dir_names)
    for child in sorted(src.iterdir()):
        if child.name in excluded:
            continue
        (dst / child.name).symlink_to(child.resolve())


def materialize_sampled_cases(
    *,
    data_dir: Path,
    sampled: list[str],
    output_dir: Path,
    mode: str,
    exclude_dir_names: list[str],
    overwrite: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cid in sampled:
        src = data_dir / cid
        dst = output_dir / cid
        if not src.exists():
            raise FileNotFoundError(f"采样 case 不存在: {src}")
        if dst.exists():
            if not overwrite:
                raise FileExistsError(f"目标 case 已存在: {dst}")
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        if mode == "copy":
            _copy_case_tree(src, dst, exclude_dir_names=exclude_dir_names)
        elif mode == "symlink":
            _symlink_case_tree(src, dst, exclude_dir_names=exclude_dir_names)
        else:
            raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TraceForge / press-one-button 数据集采样")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data2/dataset/output_bridge_v2_full_grid80",
        help="输入数据集根目录",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=SUPPORTED_LAYOUTS,
        default="legacy",
        help="输入数据布局类型",
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
        help="仅从有效 case 中采样",
    )
    parser.add_argument(
        "--output_list",
        type=str,
        default=None,
        help="保存采样 ID 列表到文件（每行一个 ID）",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default=None,
        help="保存采样 manifest JSON",
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
    parser.add_argument(
        "--exclude_dir_names",
        type=str,
        default="",
        help="复制/链接时排除的目录名，逗号分隔，例如 trajectory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖 output_dir 下已存在的目标 case",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    exclude_dir_names = parse_exclude_dir_names(args.exclude_dir_names)
    cases = get_all_cases(data_dir, valid_only=args.valid_only, layout=args.layout)
    n_total = len(cases)

    if n_total == 0:
        print("❌ 未找到有效 case")
        return

    if args.method == "random":
        sampled = sample_random(cases, args.n_sample, seed=args.seed)
        seed_value: int | None = int(args.seed)
    else:
        sampled = sample_uniform(cases, args.n_sample)
        seed_value = None

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    manifest = build_sample_manifest(
        data_dir=data_dir,
        sampled=sampled,
        layout=args.layout,
        method=args.method,
        n_requested=args.n_sample,
        seed=seed_value,
        exclude_dir_names=exclude_dir_names,
        output_dir=output_dir,
    )

    print(f"总 case 数: {n_total}")
    print(f"采样数量: {len(sampled)}")
    print(f"布局类型: {args.layout}")
    print(f"采样方式: {args.method}" + (f" (seed={args.seed})" if args.method == "random" else ""))
    if exclude_dir_names:
        print(f"排除目录: {','.join(exclude_dir_names)}")

    if args.output_list:
        out_path = Path(args.output_list)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(sampled) + "\n", encoding="utf-8")
        print(f"✅ 采样 ID 已保存: {out_path}")

    if args.output_dir:
        assert output_dir is not None
        materialize_sampled_cases(
            data_dir=data_dir,
            sampled=sampled,
            output_dir=output_dir,
            mode=args.mode,
            exclude_dir_names=exclude_dir_names,
            overwrite=bool(args.overwrite),
        )
        print(f"✅ 已{args.mode} {len(sampled)} 个 case 到 {output_dir}")

    if args.output_manifest:
        manifest_path = Path(args.output_manifest)
        _write_json(manifest_path, manifest)
        print(f"✅ 采样 manifest 已保存: {manifest_path}")


if __name__ == "__main__":
    main()
