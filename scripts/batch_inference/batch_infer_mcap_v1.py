#!/usr/bin/env python3
"""
MCAP v1 数据集批量推理入口。

目标数据结构：
    base_path/
        00000/
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

实现策略：
- 复用维护中的 press_one_button_demo 批处理主逻辑；
- 仅替换 episode 发现规则，兼容 `00000` 这类纯数字目录名；
- 仍然使用 external-only TraceForge 推理、共享 query-frame schedule、
  以及现有的多 GPU / shared-GPU worker 调度。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import batch_infer_press_one_button_demo as press_one_button_demo


def _has_files(dir_path: Path, suffixes: tuple[str, ...]) -> bool:
    return dir_path.is_dir() and any(
        path.is_file() and path.suffix.lower() in suffixes for path in dir_path.iterdir()
    )


def _is_supported_episode_dir_name(name: str) -> bool:
    return name.isdigit() or name.startswith("episode_")


def find_valid_episodes(
    base_path: Path,
    camera_names: list[str],
    geom_name: str,
) -> list[Path]:
    episodes: list[Path] = []
    for episode_dir in sorted(base_path.iterdir()):
        if not episode_dir.is_dir():
            continue
        if not _is_supported_episode_dir_name(episode_dir.name):
            continue

        geom_path = episode_dir / geom_name
        if not geom_path.is_file():
            continue

        has_any_camera = False
        for camera_name in camera_names:
            rgb_dir = episode_dir / "rgb" / camera_name
            depth_dir = episode_dir / "depth" / camera_name
            if _has_files(rgb_dir, (".png", ".jpg", ".jpeg")) and _has_files(
                depth_dir, (".npy", ".png")
            ):
                has_any_camera = True
                break

        if has_any_camera:
            episodes.append(episode_dir)

    return episodes


def main() -> None:
    press_one_button_demo.find_valid_episodes = find_valid_episodes
    press_one_button_demo.main()


if __name__ == "__main__":
    main()
