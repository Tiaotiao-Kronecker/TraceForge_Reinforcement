#!/usr/bin/env python3
"""
Visualization helpers for the Xperience-10M sample dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.data_analysis.xperience_sample_utils import FrameSample, XperienceSampleDataset


RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
RESAMPLE_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def add_label(image: Image.Image, label: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = _font()
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    width = right - left
    height = bottom - top
    draw.rectangle((8, 8, 16 + width, 16 + height), fill=(0, 0, 0))
    draw.text((12, 12), label, fill=(255, 255, 255), font=font)
    return image


def array_to_panel(array: np.ndarray, size: tuple[int, int], label: str) -> Image.Image:
    image = Image.fromarray(array).convert("RGB")
    image = ImageOps.contain(image, size, RESAMPLE_LANCZOS)
    panel = Image.new("RGB", size, color=(28, 32, 38))
    offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
    panel.paste(image, offset)
    return add_label(panel, label)


def colorize_depth(depth: np.ndarray, size: tuple[int, int] = (320, 320)) -> Image.Image:
    valid = depth > 0
    normalized = np.zeros_like(depth, dtype=np.uint8)
    if np.any(valid):
        values = depth[valid]
        low = float(np.min(values))
        high = float(np.max(values))
        scaled = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
        normalized = (scaled * 255).astype(np.uint8)
    image = Image.fromarray(normalized, mode="L").resize(size, RESAMPLE_NEAREST)
    return ImageOps.colorize(image, black="#081d58", white="#ffd166")


def colorize_confidence(confidence: np.ndarray, size: tuple[int, int] = (320, 320)) -> Image.Image:
    values = confidence.astype(np.float32)
    values -= values.min()
    if values.max() > 0:
        values /= values.max()
    image = Image.fromarray((values * 255).astype(np.uint8), mode="L").resize(size, RESAMPLE_NEAREST)
    return ImageOps.colorize(image, black="#081c15", white="#95d5b2")


def project_points(points: np.ndarray, canvas_size: tuple[int, int], percentiles: tuple[float, float]) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    low, high = np.percentile(points, percentiles, axis=0)
    clipped = np.clip(points, low, high)
    span = np.maximum(high - low, 1e-6)
    normalized = (clipped - low) / span
    width, height = canvas_size
    projected = np.empty_like(normalized)
    projected[:, 0] = normalized[:, 0] * (width - 1)
    projected[:, 1] = (1.0 - normalized[:, 1]) * (height - 1)
    return projected.astype(np.int32)


def render_slam_panel(dataset: XperienceSampleDataset, current_index: int, size: tuple[int, int] = (620, 280)) -> Image.Image:
    point_cloud = np.asarray(dataset.slam_point_cloud)[:, [0, 2]]
    trajectory = np.asarray(dataset.slam_translations)[:, [0, 2]]
    current_point = trajectory[current_index : current_index + 1]

    points_xy = project_points(point_cloud, (size[0] - 24, size[1] - 24), (1, 99))
    path_xy = project_points(trajectory, (size[0] - 24, size[1] - 24), (1, 99))
    current_xy = project_points(current_point, (size[0] - 24, size[1] - 24), (1, 99))

    panel = Image.new("RGB", size, color=(18, 20, 24))
    draw = ImageDraw.Draw(panel)
    margin = 12

    for x, y in points_xy:
        draw.point((int(x + margin), int(y + margin)), fill=(84, 92, 110))

    if len(path_xy) > 1:
        draw.line([(int(x + margin), int(y + margin)) for x, y in path_xy], fill=(244, 180, 0), width=3)

    if len(current_xy) == 1:
        x, y = current_xy[0]
        draw.ellipse((int(x + margin) - 6, int(y + margin) - 6, int(x + margin) + 6, int(y + margin) + 6), fill=(231, 76, 60))

    return add_label(panel, "SLAM x/z trajectory")


def format_metrics(sample: FrameSample) -> str:
    lines = [
        f"frame {sample.index}  t={sample.relative_time_sec:.2f}s",
        f"translation xyz: {np.round(sample.slam_translation, 4).tolist()}",
    ]
    if sample.active_segment is not None:
        lines.append(f"sub task: {sample.active_segment.sub_task}")
        if sample.active_segment.action_labels:
            lines.append("actions: " + ", ".join(sample.active_segment.action_labels))
    if sample.depth is not None:
        valid = sample.depth[sample.depth > 0]
        if valid.size:
            lines.append(f"depth range: {float(np.min(valid)):.2f}m .. {float(np.max(valid)):.2f}m")
    if sample.contacts is not None:
        lines.append(f"body contact mean: {float(np.mean(sample.contacts)):.3f}")
    if sample.imu is not None:
        accel_norm = np.linalg.norm(sample.imu["accel_xyz"], axis=1)
        gyro_norm = np.linalg.norm(sample.imu["gyro_xyz"], axis=1)
        lines.append(f"imu accel norm mean: {float(np.mean(accel_norm)):.3f}")
        lines.append(f"imu gyro norm mean: {float(np.mean(gyro_norm)):.3f}")
    return "\n".join(lines)


def render_frame_dashboard(
    dataset: XperienceSampleDataset,
    sample: FrameSample,
    *,
    main_camera: str = "stereo_left",
    secondary_cameras: Sequence[str] = ("stereo_right", "fisheye_cam1"),
) -> Image.Image:
    canvas = Image.new("RGB", (1600, 980), color=(14, 16, 20))
    draw = ImageDraw.Draw(canvas)
    font = _font()

    title = dataset.caption_config.get("config", {}).get("Main Task", "Xperience sample")
    draw.text((28, 20), title, fill=(255, 255, 255), font=font)
    draw.text((28, 44), f"dataset frame {sample.index} / {len(dataset) - 1}", fill=(184, 192, 208), font=font)

    main_panel = array_to_panel(sample.video_frames[main_camera], (920, 690), main_camera)
    canvas.paste(main_panel, (24, 84))

    if sample.depth is not None:
        canvas.paste(add_label(colorize_depth(sample.depth), "depth"), (970, 84))
    if sample.depth_confidence is not None:
        canvas.paste(add_label(colorize_confidence(sample.depth_confidence), "depth confidence"), (1250, 84))

    for slot, stream in enumerate(secondary_cameras):
        frame = sample.video_frames.get(stream)
        if frame is None:
            continue
        panel = array_to_panel(frame, (285, 220), stream)
        canvas.paste(panel, (970 + slot * 295, 426))

    slam_panel = render_slam_panel(dataset, sample.index)
    canvas.paste(slam_panel, (970, 666))

    draw.multiline_text((24, 802), format_metrics(sample), fill=(226, 232, 240), font=font, spacing=6)
    return canvas


def build_storyboard(
    dataset: XperienceSampleDataset,
    indices: Sequence[int],
    *,
    main_camera: str = "stereo_left",
    size: tuple[int, int] = (360, 240),
) -> Image.Image:
    columns = 3
    padding = 18
    title_height = 52
    rows = int(np.ceil(len(indices) / columns))
    canvas = Image.new(
        "RGB",
        (columns * size[0] + (columns + 1) * padding, rows * (size[1] + title_height) + (rows + 1) * padding),
        color=(14, 16, 20),
    )
    draw = ImageDraw.Draw(canvas)
    font = _font()

    for position, index in enumerate(indices):
        sample = dataset.get_frame(
            index,
            video_streams=(main_camera,),
            load_video=True,
            load_depth=False,
            load_mocap=False,
            load_imu=False,
        )
        row = position // columns
        column = position % columns
        x = padding + column * (size[0] + padding)
        y = padding + row * (size[1] + title_height + padding)
        panel = array_to_panel(sample.video_frames[main_camera], size, main_camera)
        canvas.paste(panel, (x, y + title_height))
        subtitle = sample.active_segment.sub_task if sample.active_segment else "no segment"
        draw.text((x, y), f"frame {index}  t={sample.relative_time_sec:.1f}s", fill=(255, 255, 255), font=font)
        draw.text((x, y + 18), subtitle[:52], fill=(184, 192, 208), font=font)

    return canvas


def save_gif(frames: Sequence[Image.Image], output_path: Path, duration_ms: int = 180) -> None:
    prepared = [frame.convert("P", palette=Image.Palette.ADAPTIVE) for frame in frames]
    prepared[0].save(
        output_path,
        save_all=True,
        append_images=prepared[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
