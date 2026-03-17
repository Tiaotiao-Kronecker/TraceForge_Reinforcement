#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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

from utils.traceforge_artifact_utils import SceneReader, normalize_sample_data, traj_uvz_to_world


DEFAULT_TAGS = ("none_ref", "basic", "standard", "strict")
DEFAULT_PRIMARY_TAG = "standard"
DEFAULT_PRIMARY_EPISODE = "episode_00000_blue"
DEFAULT_PRIMARY_CAMERA = "varied_camera_1"
DEFAULT_PRIMARY_QUERY_FRAME = 15


def parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def md5_digest(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def arrays_equal_or_both_none(lhs: np.ndarray | None, rhs: np.ndarray | None) -> bool:
    if lhs is None and rhs is None:
        return True
    if lhs is None or rhs is None:
        return False
    return np.array_equal(lhs, rhs)


def sample_cloud(points: np.ndarray, colors: np.ndarray, max_points: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices], colors[indices]


def build_axis_limits(point_sets: list[np.ndarray]) -> tuple[np.ndarray, float]:
    non_empty = [points for points in point_sets if points.size > 0]
    if not non_empty:
        return np.zeros(3, dtype=np.float32), 1.0
    stacked = np.concatenate(non_empty, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    return center.astype(np.float32), max(radius, 1e-3)


def apply_axis_limits(ax: plt.Axes, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def maybe_visibility(sample: dict[str, Any], track_mask: np.ndarray) -> np.ndarray | None:
    visibility = sample.get("visibility")
    if visibility is None:
        return None
    vis = np.asarray(visibility)
    if vis.ndim == 3 and vis.shape[-1] == 1:
        vis = vis.squeeze(-1)
    traj = sample["traj_uvz"]
    if vis.shape == (traj.shape[1], traj.shape[0]):
        vis = vis.T
    if vis.shape != (traj.shape[0], traj.shape[1]):
        return None
    return vis[track_mask].astype(np.float32, copy=False)


def unproject_depth_to_world(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    z = depth.astype(np.float32, copy=False)
    x_cam = (xs - cx) * z / max(fx, 1e-8)
    y_cam = (ys - cy) * z / max(fy, 1e-8)
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1)
    pts_cam_h = np.concatenate(
        [pts_cam, np.ones((*pts_cam.shape[:2], 1), dtype=np.float32)],
        axis=-1,
    )
    c2w = np.linalg.inv(w2c).astype(np.float32)
    pts_world = (c2w @ pts_cam_h.reshape(-1, 4).T).T.reshape(height, width, 4)
    return pts_world[..., :3].astype(np.float32)


def build_pointcloud_variants(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    *,
    downsample: int,
    depth_min: float,
    depth_max: float,
) -> dict[str, Any]:
    pts_world = unproject_depth_to_world(depth, intrinsics, w2c)
    points = pts_world[::downsample, ::downsample].reshape(-1, 3)
    colors = rgb[::downsample, ::downsample].reshape(-1, 3).astype(np.float32) / 255.0
    depth_ds = depth[::downsample, ::downsample].reshape(-1)

    base_mask = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(colors).all(axis=1)
        & np.isfinite(depth_ds)
    )
    worldz_mask = base_mask & (points[:, 2] > depth_min) & (points[:, 2] < depth_max)
    camera_depth_mask = base_mask & (depth_ds > depth_min) & (depth_ds < depth_max)
    worldz_only = worldz_mask & ~camera_depth_mask
    camera_only = camera_depth_mask & ~worldz_mask

    finite_depth = depth[np.isfinite(depth)]
    max_depth_value = float(finite_depth.max()) if finite_depth.size else math.nan
    gradient_y, gradient_x = np.gradient(depth.astype(np.float32))
    gradient_mag = np.sqrt(np.nan_to_num(gradient_x) ** 2 + np.nan_to_num(gradient_y) ** 2)
    finite_gradient = gradient_mag[np.isfinite(gradient_mag)]
    gradient_threshold = float(np.quantile(finite_gradient, 0.995)) if finite_gradient.size else math.nan

    return {
        "points": points,
        "colors": colors,
        "depth_ds": depth_ds,
        "worldz_mask": worldz_mask,
        "camera_depth_mask": camera_depth_mask,
        "worldz_only_mask": worldz_only,
        "camera_only_mask": camera_only,
        "stats": {
            "downsampled_total_points": int(base_mask.size),
            "base_valid_points": int(base_mask.sum()),
            "worldz_kept_points": int(worldz_mask.sum()),
            "camera_depth_kept_points": int(camera_depth_mask.sum()),
            "worldz_only_points": int(worldz_only.sum()),
            "camera_depth_only_points": int(camera_only.sum()),
            "mismatch_ratio": float(np.mean(worldz_mask != camera_depth_mask)),
            "depth_min": float(depth_min),
            "depth_max": float(depth_max),
            "max_depth_value": max_depth_value,
            "gradient_threshold_p995": gradient_threshold,
        },
        "overlays": {
            "low_depth": np.isfinite(depth) & (depth <= depth_min),
            "plateau_depth": np.isfinite(depth) & (depth >= max_depth_value - 1e-4),
            "high_gradient": np.isfinite(gradient_mag) & (gradient_mag >= gradient_threshold),
        },
    }


def choose_track_indices(metrics_rows: list[dict[str, float]], max_tracks: int) -> np.ndarray:
    if not metrics_rows:
        return np.empty(0, dtype=np.int32)
    order = sorted(
        range(len(metrics_rows)),
        key=lambda idx: (
            metrics_rows[idx]["uv_span"],
            metrics_rows[idx]["depth_span"],
            metrics_rows[idx]["valid_frame_count"],
        ),
        reverse=True,
    )
    return np.asarray(order[: min(max_tracks, len(order))], dtype=np.int32)


def compute_track_metrics(
    traj_uvz: np.ndarray,
    traj_world: np.ndarray,
    visibility: np.ndarray | None,
    camera_center: np.ndarray,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    finite = np.isfinite(traj_world).all(axis=-1)
    for track_idx in range(traj_world.shape[0]):
        valid = finite[track_idx]
        valid_count = int(valid.sum())
        if valid_count < 2:
            continue
        pts = traj_world[track_idx, valid]
        uvz = traj_uvz[track_idx, valid]
        vectors = pts - camera_center[None]
        radii = np.linalg.norm(vectors, axis=1)
        unit_vectors = vectors / np.maximum(radii[:, None], 1e-8)
        mean_direction = unit_vectors.mean(axis=0)
        mean_direction /= max(float(np.linalg.norm(mean_direction)), 1e-8)
        angles_deg = np.degrees(np.arccos(np.clip(unit_vectors @ mean_direction, -1.0, 1.0)))
        visibility_ratio = float(visibility[track_idx, valid].mean()) if visibility is not None else math.nan
        rows.append(
            {
                "track_index": float(track_idx),
                "valid_frame_count": float(valid_count),
                "query_depth": float(uvz[0, 2]),
                "depth_span": float(uvz[:, 2].max() - uvz[:, 2].min()),
                "uv_span": float(np.linalg.norm(uvz[:, :2].max(axis=0) - uvz[:, :2].min(axis=0))),
                "radial_span": float(radii.max() - radii.min()),
                "ray_angle_median_deg": float(np.median(angles_deg)),
                "ray_angle_p90_deg": float(np.percentile(angles_deg, 90)),
                "visibility_ratio": visibility_ratio,
            }
        )
    return rows


def summarize_metrics(rows: list[dict[str, float]], threshold_deg: float, radial_span_threshold: float) -> dict[str, Any]:
    if not rows:
        return {
            "track_count": 0,
            "suspicious_track_count": 0,
            "suspicious_ratio": math.nan,
        }

    def values(key: str) -> np.ndarray:
        out = np.asarray([row[key] for row in rows], dtype=np.float32)
        return out[np.isfinite(out)]

    def stats(values_array: np.ndarray) -> dict[str, float]:
        if values_array.size == 0:
            return {}
        return {
            "min": float(values_array.min()),
            "p01": float(np.quantile(values_array, 0.01)),
            "p05": float(np.quantile(values_array, 0.05)),
            "p50": float(np.quantile(values_array, 0.50)),
            "p95": float(np.quantile(values_array, 0.95)),
            "p99": float(np.quantile(values_array, 0.99)),
            "max": float(values_array.max()),
        }

    suspicious = [
        row
        for row in rows
        if row["ray_angle_p90_deg"] < threshold_deg and row["radial_span"] > radial_span_threshold
    ]
    return {
        "track_count": int(len(rows)),
        "suspicious_track_count": int(len(suspicious)),
        "suspicious_ratio": float(len(suspicious) / max(len(rows), 1)),
        "thresholds": {
            "ray_angle_p90_deg_lt": float(threshold_deg),
            "radial_span_gt": float(radial_span_threshold),
        },
        "ray_angle_p90_deg": stats(values("ray_angle_p90_deg")),
        "radial_span": stats(values("radial_span")),
        "depth_span": stats(values("depth_span")),
        "uv_span": stats(values("uv_span")),
        "query_depth": stats(values("query_depth")),
        "visibility_ratio": stats(values("visibility_ratio")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_dense_comparison(
    output_path: Path,
    *,
    current_points: np.ndarray,
    current_colors: np.ndarray,
    alt_points: np.ndarray,
    alt_colors: np.ndarray,
    current_only_points: np.ndarray,
    alt_only_points: np.ndarray,
    max_points: int,
) -> None:
    current_plot, current_colors_plot = sample_cloud(current_points, current_colors, max_points=max_points, seed=0)
    alt_plot, alt_colors_plot = sample_cloud(alt_points, alt_colors, max_points=max_points, seed=1)
    current_only_plot, _ = sample_cloud(
        current_only_points,
        np.zeros((len(current_only_points), 3), dtype=np.float32),
        max_points=max_points,
        seed=2,
    )
    alt_only_plot, _ = sample_cloud(
        alt_only_points,
        np.zeros((len(alt_only_points), 3), dtype=np.float32),
        max_points=max_points,
        seed=3,
    )

    center, radius = build_axis_limits([current_plot, alt_plot, current_only_plot, alt_only_plot])

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]

    axes[0].scatter(
        current_plot[:, 0], current_plot[:, 1], current_plot[:, 2],
        c=current_colors_plot, s=0.4, alpha=0.8, linewidths=0.0,
    )
    axes[0].set_title("Current filter (world z)")

    axes[1].scatter(
        alt_plot[:, 0], alt_plot[:, 1], alt_plot[:, 2],
        c=alt_colors_plot, s=0.4, alpha=0.8, linewidths=0.0,
    )
    axes[1].set_title("Alternative filter (camera depth)")

    if current_only_plot.size > 0:
        axes[2].scatter(
            current_only_plot[:, 0], current_only_plot[:, 1], current_only_plot[:, 2],
            c="tab:red", s=0.8, alpha=0.8, linewidths=0.0, label="world-z only",
        )
    if alt_only_plot.size > 0:
        axes[2].scatter(
            alt_only_plot[:, 0], alt_only_plot[:, 1], alt_only_plot[:, 2],
            c="tab:blue", s=0.8, alpha=0.8, linewidths=0.0, label="camera-depth only",
        )
    axes[2].set_title("Mismatch points")
    axes[2].legend(loc="upper right")

    for ax in axes:
        apply_axis_limits(ax, center, radius)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=22, azim=-58)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def overlay_mask_on_rgb(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def plot_depth_overlays(
    output_path: Path,
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    overlays: dict[str, np.ndarray],
) -> None:
    low_depth_rgb = overlay_mask_on_rgb(rgb, overlays["low_depth"], (255, 0, 0))
    plateau_rgb = overlay_mask_on_rgb(rgb, overlays["plateau_depth"], (0, 128, 255))
    gradient_rgb = overlay_mask_on_rgb(rgb, overlays["high_gradient"], (255, 192, 0))

    finite_depth = depth[np.isfinite(depth)]
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4),
    ]
    axes[0].imshow(low_depth_rgb)
    axes[0].set_title("RGB + low-depth mask (red)")
    axes[1].imshow(plateau_rgb)
    axes[1].set_title("RGB + plateau-depth mask (blue)")
    axes[2].imshow(gradient_rgb)
    axes[2].set_title("RGB + high-gradient mask (orange)")
    if finite_depth.size > 0:
        axes[3].hist(finite_depth.ravel(), bins=80, color="tab:gray")
    axes[3].set_title("Depth histogram")
    axes[3].set_xlabel("depth")
    axes[3].set_ylabel("count")
    for ax in axes[:3]:
        ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_keypoint_diagnostics(
    output_path: Path,
    *,
    rgb: np.ndarray,
    traj_uvz: np.ndarray,
    traj_world: np.ndarray,
    metric_rows: list[dict[str, float]],
    selected_indices: np.ndarray,
    threshold_deg: float,
    radial_span_threshold: float,
) -> None:
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    ax_img = fig.add_subplot(2, 2, 1)
    ax_3d = fig.add_subplot(2, 2, 2, projection="3d")
    ax_scatter = fig.add_subplot(2, 2, 3)
    ax_text = fig.add_subplot(2, 2, 4)

    ax_img.imshow(rgb)
    cmap = plt.cm.turbo(np.linspace(0.0, 1.0, max(len(selected_indices), 1), endpoint=True))
    selected_points: list[np.ndarray] = []
    for order_idx, track_idx in enumerate(selected_indices):
        traj_2d = traj_uvz[track_idx, :, :2]
        valid = np.isfinite(traj_2d).all(axis=1)
        if np.count_nonzero(valid) < 2:
            continue
        pts = traj_2d[valid]
        color = cmap[order_idx]
        ax_img.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.0, alpha=0.8)
        ax_img.scatter(pts[0, 0], pts[0, 1], color=color, s=10, linewidths=0.0)

        traj_3d = traj_world[track_idx]
        valid_world = np.isfinite(traj_3d).all(axis=1)
        pts_3d = traj_3d[valid_world]
        if len(pts_3d) < 2:
            continue
        selected_points.append(pts_3d)
        ax_3d.plot(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], color=color, linewidth=1.2, alpha=0.8)
        ax_3d.scatter(pts_3d[0, 0], pts_3d[0, 1], pts_3d[0, 2], color=color, s=8, depthshade=False)

    all_angles = np.asarray([row["ray_angle_p90_deg"] for row in metric_rows], dtype=np.float32)
    all_radial = np.asarray([row["radial_span"] for row in metric_rows], dtype=np.float32)
    ax_scatter.scatter(all_angles, all_radial, s=5, alpha=0.35, color="tab:purple")
    ax_scatter.axvline(threshold_deg, color="tab:red", linestyle="--", linewidth=1.0)
    ax_scatter.axhline(radial_span_threshold, color="tab:red", linestyle="--", linewidth=1.0)
    ax_scatter.set_xlabel("ray angle p90 (deg)")
    ax_scatter.set_ylabel("radial span (m)")
    ax_scatter.set_title("Per-track rayness")

    suspicious_count = int(
        np.count_nonzero((all_angles < threshold_deg) & (all_radial > radial_span_threshold))
    )
    ax_text.axis("off")
    text_lines = [
        f"tracks after traj_valid_mask: {len(metric_rows)}",
        f"suspicious tracks: {suspicious_count}",
        f"threshold: angle p90 < {threshold_deg:.1f} deg",
        f"threshold: radial span > {radial_span_threshold:.2f} m",
    ]
    if len(metric_rows) > 0:
        query_depth = np.asarray([row["query_depth"] for row in metric_rows], dtype=np.float32)
        uv_span = np.asarray([row["uv_span"] for row in metric_rows], dtype=np.float32)
        radial_span = np.asarray([row["radial_span"] for row in metric_rows], dtype=np.float32)
        text_lines.extend(
            [
                "",
                f"query depth p50/p95: {np.quantile(query_depth, 0.5):.3f} / {np.quantile(query_depth, 0.95):.3f}",
                f"uv span p50/p95: {np.quantile(uv_span, 0.5):.3f} / {np.quantile(uv_span, 0.95):.3f}",
                f"radial span p50/p95: {np.quantile(radial_span, 0.5):.5f} / {np.quantile(radial_span, 0.95):.5f}",
            ]
        )
    ax_text.text(0.0, 1.0, "\n".join(text_lines), va="top", ha="left", family="monospace")

    ax_img.set_title("Selected 2D tracks")
    ax_img.set_axis_off()
    if selected_points:
        center, radius = build_axis_limits(selected_points)
        apply_axis_limits(ax_3d, center, radius)
    ax_3d.set_title("Selected 3D tracks")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.view_init(elev=22, azim=-58)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_joint_overlay(
    output_path: Path,
    *,
    cloud_points: np.ndarray,
    cloud_colors: np.ndarray,
    traj_world: np.ndarray,
    selected_indices: np.ndarray,
    max_points: int,
) -> None:
    cloud_plot, cloud_colors_plot = sample_cloud(cloud_points, cloud_colors, max_points=max_points, seed=0)
    selected_tracks = [traj_world[idx][np.isfinite(traj_world[idx]).all(axis=1)] for idx in selected_indices]
    center, radius = build_axis_limits([cloud_plot, *selected_tracks])

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(
        cloud_plot[:, 0],
        cloud_plot[:, 1],
        cloud_plot[:, 2],
        c=cloud_colors_plot,
        s=0.4,
        alpha=0.65,
        linewidths=0.0,
    )
    colors = plt.cm.turbo(np.linspace(0.0, 1.0, max(len(selected_indices), 1), endpoint=True))
    for order_idx, track_idx in enumerate(selected_indices):
        pts = traj_world[track_idx][np.isfinite(traj_world[track_idx]).all(axis=1)]
        if len(pts) < 2:
            continue
        color = colors[order_idx]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.1, alpha=0.85)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color=color, s=8, depthshade=False)

    apply_axis_limits(ax, center, radius)
    ax.set_title("Joint overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-58)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def common_inference_relpaths(benchmark_root: Path, tags: list[str], pattern: str) -> list[Path]:
    path_sets = []
    for tag in tags:
        root = benchmark_root / tag / "inference"
        path_sets.append({path.relative_to(root) for path in root.glob(pattern)})
    common = set.intersection(*path_sets) if path_sets else set()
    return sorted(common)


def audit_consistency(benchmark_root: Path, tags: list[str]) -> dict[str, Any]:
    scene_relpaths = common_inference_relpaths(benchmark_root, tags, "*/*/scene.h5")
    scene_identical = 0
    scene_examples: list[dict[str, Any]] = []
    for relpath in scene_relpaths:
        digests = {tag: md5_digest(benchmark_root / tag / "inference" / relpath) for tag in tags}
        if len(set(digests.values())) == 1:
            scene_identical += 1
        elif len(scene_examples) < 5:
            scene_examples.append({"relpath": relpath.as_posix(), "digests": digests})

    sample_relpaths = common_inference_relpaths(benchmark_root, tags, "*/*/samples/*.npz")
    sample_rows = []
    traj_uvz_identical = 0
    visibility_identical = 0
    for relpath in sample_relpaths:
        payloads: dict[str, dict[str, Any]] = {}
        for tag in tags:
            payloads[tag] = normalize_sample_data(benchmark_root / tag / "inference" / relpath)
        reference = payloads[tags[0]]
        traj_match = all(np.array_equal(payloads[tag]["traj_uvz"], reference["traj_uvz"]) for tag in tags[1:])
        visibility_match = all(
            arrays_equal_or_both_none(payloads[tag].get("visibility"), reference.get("visibility"))
            for tag in tags[1:]
        )
        mask_changed = any(
            not np.array_equal(payloads[tag]["traj_valid_mask"], reference["traj_valid_mask"])
            for tag in tags[1:]
        )
        if traj_match:
            traj_uvz_identical += 1
        if visibility_match:
            visibility_identical += 1
        sample_rows.append(
            {
                "sample_path": relpath.as_posix(),
                "traj_uvz_identical_across_tags": traj_match,
                "visibility_identical_across_tags": visibility_match,
                "traj_valid_mask_differs": mask_changed,
            }
        )

    return {
        "tags": tags,
        "scene_h5": {
            "common_count": int(len(scene_relpaths)),
            "identical_count": int(scene_identical),
            "all_identical": bool(scene_identical == len(scene_relpaths)),
            "non_identical_examples": scene_examples,
        },
        "sample_npz": {
            "common_count": int(len(sample_relpaths)),
            "traj_uvz_identical_count": int(traj_uvz_identical),
            "visibility_identical_count": int(visibility_identical),
            "rows": sample_rows,
        },
    }


def load_primary_case(
    benchmark_root: Path,
    *,
    tag: str,
    episode: str,
    camera: str,
    query_frame: int,
) -> dict[str, Any]:
    camera_dir = benchmark_root / tag / "inference" / episode / camera
    sample_path = camera_dir / "samples" / f"{camera}_{query_frame}.npz"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Primary sample not found: {sample_path}")
    sample = normalize_sample_data(sample_path)
    track_mask = sample["traj_valid_mask"].astype(bool, copy=False)
    traj_uvz = sample["traj_uvz"].astype(np.float32)[track_mask]
    visibility = maybe_visibility(sample, track_mask)
    with SceneReader(camera_dir) as scene_reader:
        intrinsics_all, extrinsics_all = scene_reader.get_camera_arrays()
        depth = scene_reader.get_depth_frame(query_frame).astype(np.float32)
        rgb = scene_reader.get_rgb_frame(query_frame).astype(np.uint8)
    query_intrinsics = intrinsics_all[query_frame].astype(np.float32)
    query_w2c = extrinsics_all[query_frame].astype(np.float32)
    traj_world = traj_uvz_to_world(traj_uvz, query_intrinsics, query_w2c)
    camera_center = np.linalg.inv(query_w2c)[:3, 3].astype(np.float32)
    return {
        "camera_dir": camera_dir,
        "sample_path": sample_path,
        "traj_uvz": traj_uvz,
        "traj_world": traj_world,
        "visibility": visibility,
        "depth": depth,
        "rgb": rgb,
        "intrinsics": query_intrinsics,
        "w2c": query_w2c,
        "camera_center": camera_center,
    }


def build_dense_scan_rows(
    benchmark_root: Path,
    *,
    tag: str,
    downsample: int,
    depth_min: float,
    depth_max: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_root = benchmark_root / tag / "inference"
    for sample_path in sorted(sample_root.glob("*/*/samples/*.npz")):
        sample = normalize_sample_data(sample_path)
        query_frame = int(sample["query_frame_index"])
        camera_dir = sample_path.parents[1]
        with SceneReader(camera_dir) as scene_reader:
            intrinsics_all, extrinsics_all = scene_reader.get_camera_arrays()
            depth = scene_reader.get_depth_frame(query_frame).astype(np.float32)
            rgb = scene_reader.get_rgb_frame(query_frame).astype(np.uint8)
        dense_bundle = build_pointcloud_variants(
            depth,
            rgb,
            intrinsics_all[query_frame].astype(np.float32),
            extrinsics_all[query_frame].astype(np.float32),
            downsample=downsample,
            depth_min=depth_min,
            depth_max=depth_max,
        )
        rows.append(
            {
                "sample_path": sample_path.relative_to(sample_root).as_posix(),
                **dense_bundle["stats"],
            }
        )
    rows.sort(key=lambda row: row["camera_depth_only_points"], reverse=True)
    return rows


def build_keypoint_scan_rows(
    benchmark_root: Path,
    *,
    tag: str,
    threshold_deg: float,
    radial_span_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_root = benchmark_root / tag / "inference"
    for sample_path in sorted(sample_root.glob("*/*/samples/*.npz")):
        sample = normalize_sample_data(sample_path)
        track_mask = sample["traj_valid_mask"].astype(bool, copy=False)
        traj_uvz = sample["traj_uvz"].astype(np.float32)[track_mask]
        if len(traj_uvz) == 0:
            rows.append(
                {
                    "sample_path": sample_path.relative_to(sample_root).as_posix(),
                    "track_count": 0,
                    "suspicious_track_count": 0,
                    "suspicious_ratio": math.nan,
                    "query_depth_p50": math.nan,
                    "radial_span_p95": math.nan,
                    "ray_angle_p90_p95": math.nan,
                }
            )
            continue
        visibility = maybe_visibility(sample, track_mask)
        query_frame = int(sample["query_frame_index"])
        camera_dir = sample_path.parents[1]
        with SceneReader(camera_dir) as scene_reader:
            intrinsics_all, extrinsics_all = scene_reader.get_camera_arrays()
        query_intrinsics = intrinsics_all[query_frame].astype(np.float32)
        query_w2c = extrinsics_all[query_frame].astype(np.float32)
        traj_world = traj_uvz_to_world(traj_uvz, query_intrinsics, query_w2c)
        camera_center = np.linalg.inv(query_w2c)[:3, 3].astype(np.float32)
        metric_rows = compute_track_metrics(traj_uvz, traj_world, visibility, camera_center)
        summary = summarize_metrics(metric_rows, threshold_deg, radial_span_threshold)
        rows.append(
            {
                "sample_path": sample_path.relative_to(sample_root).as_posix(),
                "track_count": summary["track_count"],
                "suspicious_track_count": summary["suspicious_track_count"],
                "suspicious_ratio": summary["suspicious_ratio"],
                "query_depth_p50": summary.get("query_depth", {}).get("p50", math.nan),
                "radial_span_p95": summary.get("radial_span", {}).get("p95", math.nan),
                "ray_angle_p90_p95": summary.get("ray_angle_p90_deg", {}).get("p95", math.nan),
            }
        )
    rows.sort(
        key=lambda row: (
            np.nan_to_num(row["suspicious_ratio"], nan=-1.0),
            row["suspicious_track_count"],
            np.nan_to_num(row["radial_span_p95"], nan=-1.0),
        ),
        reverse=True,
    )
    return rows


def write_summary_markdown(
    output_path: Path,
    *,
    benchmark_root: Path,
    primary_case: dict[str, Any],
    consistency: dict[str, Any],
    dense_stats: dict[str, Any],
    keypoint_summary: dict[str, Any],
    dense_scan_rows: list[dict[str, Any]],
    keypoint_scan_rows: list[dict[str, Any]],
) -> None:
    top_dense = dense_scan_rows[:5]
    top_keypoint = keypoint_scan_rows[:5]
    lines = [
        "# Ray Artifact Investigation Summary",
        "",
        f"- benchmark_root: `{benchmark_root}`",
        f"- primary_sample: `{primary_case['sample_path']}`",
        "",
        "## Consistency",
        "",
        f"- `scene.h5` identical across tags: `{consistency['scene_h5']['all_identical']}` "
        f"({consistency['scene_h5']['identical_count']}/{consistency['scene_h5']['common_count']})",
        f"- `traj_uvz` identical across tags: "
        f"`{consistency['sample_npz']['traj_uvz_identical_count']}/{consistency['sample_npz']['common_count']}`",
        f"- `visibility` identical across tags: "
        f"`{consistency['sample_npz']['visibility_identical_count']}/{consistency['sample_npz']['common_count']}`",
        "",
        "## Primary Dense Case",
        "",
        f"- current world-z kept points: `{dense_stats['worldz_kept_points']}`",
        f"- camera-depth kept points: `{dense_stats['camera_depth_kept_points']}`",
        f"- current-only points: `{dense_stats['worldz_only_points']}`",
        f"- camera-depth-only points: `{dense_stats['camera_depth_only_points']}`",
        f"- mismatch ratio: `{dense_stats['mismatch_ratio']:.4f}`",
        "",
        "## Primary Keypoint Case",
        "",
        f"- track_count: `{keypoint_summary['track_count']}`",
        f"- suspicious_track_count: `{keypoint_summary['suspicious_track_count']}`",
        f"- suspicious_ratio: `{keypoint_summary['suspicious_ratio']:.6f}`",
        "",
        "## Top Dense Mismatch Cases",
        "",
    ]
    for row in top_dense:
        lines.append(
            f"- `{row['sample_path']}`: current-only={row['worldz_only_points']}, "
            f"camera-depth-only={row['camera_depth_only_points']}, mismatch={row['mismatch_ratio']:.4f}"
        )
    lines.extend(["", "## Top Keypoint Cases", ""])
    for row in top_keypoint:
        ratio = row["suspicious_ratio"]
        ratio_str = "nan" if not np.isfinite(ratio) else f"{ratio:.6f}"
        lines.append(
            f"- `{row['sample_path']}`: suspicious={row['suspicious_track_count']}/{row['track_count']}, "
            f"ratio={ratio_str}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 3D ray-like artifacts from existing TraceForge outputs.")
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("data_tmp/traj_filter_benchmark/2026-03-15"),
        help="Benchmark root containing per-tag inference outputs.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=",".join(DEFAULT_TAGS),
        help="Comma-separated tag list used for consistency checks.",
    )
    parser.add_argument(
        "--primary_tag",
        type=str,
        default=DEFAULT_PRIMARY_TAG,
        help="Tag used for the primary case diagnostic figures.",
    )
    parser.add_argument(
        "--primary_episode",
        type=str,
        default=DEFAULT_PRIMARY_EPISODE,
        help="Primary episode name.",
    )
    parser.add_argument(
        "--primary_camera",
        type=str,
        default=DEFAULT_PRIMARY_CAMERA,
        help="Primary camera name.",
    )
    parser.add_argument(
        "--primary_query_frame",
        type=int,
        default=DEFAULT_PRIMARY_QUERY_FRAME,
        help="Primary query frame index.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("data_tmp/ray_artifact_investigation/2026-03-16"),
        help="Output directory for diagnostic artifacts.",
    )
    parser.add_argument(
        "--depth_min",
        type=float,
        default=0.01,
        help="Depth minimum used for point-cloud filtering.",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=10.0,
        help="Depth maximum used for point-cloud filtering.",
    )
    parser.add_argument(
        "--primary_downsample",
        type=int,
        default=4,
        help="Point-cloud downsample for the primary diagnostic case.",
    )
    parser.add_argument(
        "--scan_downsample",
        type=int,
        default=8,
        help="Point-cloud downsample for the benchmark-wide dense scan.",
    )
    parser.add_argument(
        "--max_plot_cloud_points",
        type=int,
        default=20000,
        help="Maximum cloud points rendered in each diagnostic figure.",
    )
    parser.add_argument(
        "--max_plot_tracks",
        type=int,
        default=128,
        help="Maximum track count rendered in keypoint and joint figures.",
    )
    parser.add_argument(
        "--ray_angle_threshold_deg",
        type=float,
        default=2.0,
        help="Ray-angle p90 threshold used to flag ray-like trajectories.",
    )
    parser.add_argument(
        "--radial_span_threshold",
        type=float,
        default=0.2,
        help="Minimum radial span (meters) used to flag ray-like trajectories.",
    )
    args = parser.parse_args()

    tags = parse_csv_items(args.tags)
    benchmark_root = args.benchmark_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    consistency = audit_consistency(benchmark_root, tags)
    primary_case = load_primary_case(
        benchmark_root,
        tag=args.primary_tag,
        episode=args.primary_episode,
        camera=args.primary_camera,
        query_frame=args.primary_query_frame,
    )

    dense_bundle = build_pointcloud_variants(
        primary_case["depth"],
        primary_case["rgb"],
        primary_case["intrinsics"],
        primary_case["w2c"],
        downsample=args.primary_downsample,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )

    keypoint_rows = compute_track_metrics(
        primary_case["traj_uvz"],
        primary_case["traj_world"],
        primary_case["visibility"],
        primary_case["camera_center"],
    )
    keypoint_summary = summarize_metrics(
        keypoint_rows,
        args.ray_angle_threshold_deg,
        args.radial_span_threshold,
    )
    selected_track_indices = choose_track_indices(keypoint_rows, args.max_plot_tracks)

    current_points = dense_bundle["points"][dense_bundle["worldz_mask"]]
    current_colors = dense_bundle["colors"][dense_bundle["worldz_mask"]]
    alt_points = dense_bundle["points"][dense_bundle["camera_depth_mask"]]
    alt_colors = dense_bundle["colors"][dense_bundle["camera_depth_mask"]]
    current_only_points = dense_bundle["points"][dense_bundle["worldz_only_mask"]]
    alt_only_points = dense_bundle["points"][dense_bundle["camera_only_mask"]]

    dense_plot_path = output_root / "dense_only" / "primary_case_pointcloud_filter_comparison.png"
    plot_dense_comparison(
        dense_plot_path,
        current_points=current_points,
        current_colors=current_colors,
        alt_points=alt_points,
        alt_colors=alt_colors,
        current_only_points=current_only_points,
        alt_only_points=alt_only_points,
        max_points=args.max_plot_cloud_points,
    )

    depth_overlay_path = output_root / "dense_only" / "primary_case_depth_mask_overlay.png"
    plot_depth_overlays(
        depth_overlay_path,
        rgb=primary_case["rgb"],
        depth=primary_case["depth"],
        overlays=dense_bundle["overlays"],
    )

    keypoint_plot_path = output_root / "keypoint_only" / "primary_case_keypoint_diagnostics.png"
    plot_keypoint_diagnostics(
        keypoint_plot_path,
        rgb=primary_case["rgb"],
        traj_uvz=primary_case["traj_uvz"],
        traj_world=primary_case["traj_world"],
        metric_rows=keypoint_rows,
        selected_indices=selected_track_indices,
        threshold_deg=args.ray_angle_threshold_deg,
        radial_span_threshold=args.radial_span_threshold,
    )

    joint_worldz_path = output_root / "joint_overlays" / "primary_case_joint_overlay_worldz.png"
    plot_joint_overlay(
        joint_worldz_path,
        cloud_points=current_points,
        cloud_colors=current_colors,
        traj_world=primary_case["traj_world"],
        selected_indices=selected_track_indices,
        max_points=args.max_plot_cloud_points,
    )

    joint_camdepth_path = output_root / "joint_overlays" / "primary_case_joint_overlay_camera_depth.png"
    plot_joint_overlay(
        joint_camdepth_path,
        cloud_points=alt_points,
        cloud_colors=alt_colors,
        traj_world=primary_case["traj_world"],
        selected_indices=selected_track_indices,
        max_points=args.max_plot_cloud_points,
    )

    dense_scan_rows = build_dense_scan_rows(
        benchmark_root,
        tag=args.primary_tag,
        downsample=args.scan_downsample,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
    keypoint_scan_rows = build_keypoint_scan_rows(
        benchmark_root,
        tag=args.primary_tag,
        threshold_deg=args.ray_angle_threshold_deg,
        radial_span_threshold=args.radial_span_threshold,
    )

    dense_scan_csv = output_root / "dense_filter_scan.csv"
    keypoint_scan_csv = output_root / "keypoint_scan.csv"
    write_csv(dense_scan_csv, dense_scan_rows)
    write_csv(keypoint_scan_csv, keypoint_scan_rows)

    summary = {
        "benchmark_root": str(benchmark_root),
        "primary_case": {
            "tag": args.primary_tag,
            "episode": args.primary_episode,
            "camera": args.primary_camera,
            "query_frame": args.primary_query_frame,
            "sample_path": str(primary_case["sample_path"]),
            "dense_stats": dense_bundle["stats"],
            "keypoint_summary": keypoint_summary,
            "figure_paths": {
                "dense_comparison": str(dense_plot_path),
                "depth_overlays": str(depth_overlay_path),
                "keypoint_diagnostics": str(keypoint_plot_path),
                "joint_overlay_worldz": str(joint_worldz_path),
                "joint_overlay_camera_depth": str(joint_camdepth_path),
            },
        },
        "consistency": consistency,
        "dense_filter_scan": {
            "csv_path": str(dense_scan_csv),
            "top_rows": dense_scan_rows[:10],
        },
        "keypoint_scan": {
            "csv_path": str(keypoint_scan_csv),
            "top_rows": keypoint_scan_rows[:10],
        },
        "provisional_classification": {
            "label": "B",
            "reason": (
                "Dense-pointcloud artifacts show a strong, repeatable discrepancy between the current "
                "world-z filter and a camera-depth filter, while the saved traj_valid_mask-filtered "
                "keypoint trajectories in the benchmark do not reproduce strong ray-like geometry "
                "under the same thresholds."
            ),
        },
    }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    markdown_path = output_root / "summary.md"
    write_summary_markdown(
        markdown_path,
        benchmark_root=benchmark_root,
        primary_case=summary["primary_case"],
        consistency=consistency,
        dense_stats=dense_bundle["stats"],
        keypoint_summary=keypoint_summary,
        dense_scan_rows=dense_scan_rows,
        keypoint_scan_rows=keypoint_scan_rows,
    )

    print(f"summary={summary_path}")
    print(f"dense_scan_csv={dense_scan_csv}")
    print(f"keypoint_scan_csv={keypoint_scan_csv}")
    print(f"dense_figure={dense_plot_path}")
    print(f"keypoint_figure={keypoint_plot_path}")


if __name__ == "__main__":
    main()
