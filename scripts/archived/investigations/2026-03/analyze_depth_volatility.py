#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.traceforge_artifact_utils import SceneReader, list_sample_query_frames, normalize_sample_data
from utils.traj_filter_utils import compute_depth_volatility_map, compute_high_volatility_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw depth volatility for one TraceForge camera directory.")
    parser.add_argument("--camera_dir", type=str, required=True, help="Path to one camera output directory.")
    parser.add_argument("--query_frame", type=int, default=None, help="Optional sample query frame for track overlay.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory.")
    parser.add_argument(
        "--volatility_percentile",
        type=float,
        default=99.0,
        help="Percentile used to threshold the volatility heatmap.",
    )
    parser.add_argument("--min_depth", type=float, default=0.01, help="Minimum valid raw depth in meters.")
    parser.add_argument("--max_depth", type=float, default=10.0, help="Maximum valid raw depth in meters.")
    return parser.parse_args()


def summarize_array(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def build_region_summary(volatility_map: np.ndarray) -> dict[str, dict[str, float]]:
    height, width = volatility_map.shape
    top_h = max(1, int(round(height * 0.2)))
    bottom_h = max(1, int(round(height * 0.2)))
    center_y0 = max(0, height // 2 - top_h // 2)
    center_y1 = min(height, center_y0 + top_h)
    center_x0 = max(0, width // 2 - max(1, int(round(width * 0.2))) // 2)
    center_x1 = min(width, center_x0 + max(1, int(round(width * 0.2))))

    summary = {
        "global": summarize_array(volatility_map),
        "top20": summarize_array(volatility_map[:top_h]),
        "center20": summarize_array(volatility_map[center_y0:center_y1, center_x0:center_x1]),
        "bottom20": summarize_array(volatility_map[height - bottom_h :]),
    }

    block_h = max(1, height // 3)
    block_w = max(1, width // 3)
    blocks: dict[str, dict[str, float]] = {}
    row_names = ("top", "center", "bottom")
    col_names = ("left", "center", "right")
    for row_idx, row_name in enumerate(row_names):
        y0 = row_idx * block_h
        y1 = height if row_idx == 2 else min(height, (row_idx + 1) * block_h)
        for col_idx, col_name in enumerate(col_names):
            x0 = col_idx * block_w
            x1 = width if col_idx == 2 else min(width, (col_idx + 1) * block_w)
            blocks[f"{row_name}_{col_name}"] = summarize_array(volatility_map[y0:y1, x0:x1])
    summary["blocks_3x3"] = blocks
    return summary


def save_heatmap(volatility_map: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(volatility_map, cmap="inferno")
    ax.set_title("Raw Depth Volatility (p95 - p05)")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="meters")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_overlay(rgb: np.ndarray, high_volatility_mask: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(rgb)
    overlay = np.zeros((*high_volatility_mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = high_volatility_mask.astype(np.float32) * 0.35
    ax.imshow(overlay)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_tracks_overlay(
    rgb: np.ndarray,
    high_volatility_mask: np.ndarray,
    *,
    sample: dict[str, Any] | None,
    output_path: Path,
    query_frame: int | None,
) -> dict[str, Any] | None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(rgb)

    overlay = np.zeros((*high_volatility_mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = high_volatility_mask.astype(np.float32) * 0.28
    ax.imshow(overlay)

    sample_summary: dict[str, Any] | None = None
    if sample is not None:
        keypoints = np.asarray(sample["keypoints"], dtype=np.float32)
        traj_valid_mask = np.asarray(sample["traj_valid_mask"]).astype(bool, copy=False)
        high_hits = sample.get("traj_high_volatility_hit")
        if high_hits is None:
            high_hits = np.zeros_like(traj_valid_mask)
        else:
            high_hits = np.asarray(high_hits).astype(bool, copy=False)
        reason_bits = sample.get("traj_mask_reason_bits")
        if reason_bits is not None:
            reason_bits = np.asarray(reason_bits, dtype=np.uint8)

        kept = traj_valid_mask
        filtered = ~traj_valid_mask
        filtered_high = filtered & high_hits
        filtered_other = filtered & (~high_hits)

        if np.any(kept):
            ax.scatter(keypoints[kept, 0], keypoints[kept, 1], s=6, c="#7CFC00", alpha=0.55, linewidths=0)
        if np.any(filtered_other):
            ax.scatter(
                keypoints[filtered_other, 0],
                keypoints[filtered_other, 1],
                s=7,
                c="#FFD166",
                alpha=0.75,
                linewidths=0,
            )
        if np.any(filtered_high):
            ax.scatter(
                keypoints[filtered_high, 0],
                keypoints[filtered_high, 1],
                s=9,
                c="#EF476F",
                alpha=0.9,
                linewidths=0,
            )

        sample_summary = {
            "query_frame": int(query_frame) if query_frame is not None else None,
            "num_tracks": int(keypoints.shape[0]),
            "kept_tracks": int(kept.sum()),
            "filtered_tracks": int(filtered.sum()),
            "high_volatility_hit_tracks": int(high_hits.sum()),
        }
        if reason_bits is not None:
            unique_bits, counts = np.unique(reason_bits, return_counts=True)
            sample_summary["reason_bit_histogram"] = {
                str(int(bit_value)): int(count)
                for bit_value, count in zip(unique_bits, counts)
            }
    else:
        ax.text(
            0.02,
            0.98,
            "No sample found for track overlay",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 6},
        )

    title = "Depth Volatility + Track Overlay"
    if query_frame is not None:
        title += f" (query_frame={query_frame})"
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return sample_summary


def load_all_depths(reader: SceneReader, frame_count: int) -> np.ndarray:
    return np.stack([reader.get_depth_frame(frame_idx) for frame_idx in range(frame_count)], axis=0).astype(np.float32)


def resolve_track_query_frame(camera_dir: Path, requested_query_frame: int | None) -> int | None:
    if requested_query_frame is not None:
        return int(requested_query_frame)
    frames = list_sample_query_frames(camera_dir)
    if not frames:
        return None
    return int(frames[0])


def load_sample_if_available(camera_dir: Path, query_frame: int | None) -> dict[str, Any] | None:
    if query_frame is None:
        return None
    sample_path = camera_dir / "samples" / f"{camera_dir.name}_{query_frame}.npz"
    if not sample_path.is_file():
        return None
    return normalize_sample_data(sample_path)


def main() -> None:
    args = parse_args()
    camera_dir = Path(args.camera_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else camera_dir / "_depth_volatility_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    with SceneReader(camera_dir) as reader:
        intrinsics, _ = reader.get_camera_arrays()
        frame_count = int(intrinsics.shape[0])
        depth_video = load_all_depths(reader, frame_count)
        query_frame = resolve_track_query_frame(camera_dir, args.query_frame)
        frame_for_overlay = query_frame if query_frame is not None else 0
        rgb_frame = reader.get_rgb_frame(int(frame_for_overlay))

    volatility_map = compute_depth_volatility_map(
        depth_video,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    high_volatility_mask, threshold = compute_high_volatility_mask(
        volatility_map,
        percentile=args.volatility_percentile,
    )

    save_heatmap(volatility_map, output_dir / "depth_volatility_heatmap.png")
    save_overlay(
        rgb_frame,
        high_volatility_mask,
        output_dir / "depth_volatility_overlay.png",
        title=f"High Volatility Overlay (p{args.volatility_percentile:.1f}, frame={frame_for_overlay})",
    )

    sample = load_sample_if_available(camera_dir, query_frame)
    sample_summary = save_tracks_overlay(
        rgb_frame,
        high_volatility_mask,
        sample=sample,
        output_path=output_dir / "depth_volatility_tracks_overlay.png",
        query_frame=query_frame,
    )

    summary = {
        "camera_dir": str(camera_dir),
        "frame_count": int(frame_count),
        "image_height": int(volatility_map.shape[0]),
        "image_width": int(volatility_map.shape[1]),
        "volatility_percentile": float(args.volatility_percentile),
        "high_volatility_threshold": float(threshold) if np.isfinite(threshold) else None,
        "high_volatility_pixel_count": int(high_volatility_mask.sum()),
        "high_volatility_pixel_ratio": float(high_volatility_mask.mean()),
        "region_summary": build_region_summary(volatility_map),
        "track_overlay_query_frame": query_frame,
        "sample_summary": sample_summary,
    }
    (output_dir / "depth_volatility_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
