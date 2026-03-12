#!/usr/bin/env python3
"""Export first-frame point clouds from DROID inference outputs as PLY.

This utility reuses the point-cloud generation logic from
`/data1/zoyo/projects/droid-preprocess-pipeline/scripts/step06_generate_pointclouds.py`
for the default `w2c -> c2w` path, and adds an explicit `c2w` path for
datasets/results whose stored extrinsics are already camera-to-world.

It is tailored to the six inference result folders used in the current
TraceForge DROID extrinsics sanity check:
  - outputs_droid_hand_c2w
  - outputs_droid_hand_w2c
  - outputs_droid_varied1_c2w
  - outputs_droid_varied1_w2c
  - outputs_droid_varied2_c2w
  - outputs_droid_varied2_w2c

Outputs:
  <output_root>/individual/<group>/<camera>_frame00000.ply
  <output_root>/combined/<group>_frame00000_combined.ply
  <output_root>/summary.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


DEFAULT_INPUTS = {
    "c2w": {
        "hand_camera": Path("/home/wangchen/projects/TraceForge/outputs_droid_hand_c2w/hand_camera"),
        "varied_camera_1": Path("/home/wangchen/projects/TraceForge/outputs_droid_varied1_c2w/varied_camera_1"),
        "varied_camera_2": Path("/home/wangchen/projects/TraceForge/outputs_droid_varied2_c2w/varied_camera_2"),
    },
    "w2c": {
        "hand_camera": Path("/home/wangchen/projects/TraceForge/outputs_droid_hand_w2c/hand_camera"),
        "varied_camera_1": Path("/home/wangchen/projects/TraceForge/outputs_droid_varied1_w2c/varied_camera_1"),
        "varied_camera_2": Path("/home/wangchen/projects/TraceForge/outputs_droid_varied2_w2c/varied_camera_2"),
    },
}

DEFAULT_REFERENCE_H5 = Path(
    "/data1/zoyo/projects/droid_preprocess_pipeline/droid_raw/"
    "AUTOLab+5d05c5aa+2023-09-02-10h-41m-09s/trajectory_valid.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export first-frame PLYs from DROID inference outputs."
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data1/wangchen/projects/TraceForge/outputs_droid_firstframe_ply_check"),
        help="Directory for exported PLYs and summary.json.",
    )
    parser.add_argument(
        "--step06_script",
        type=Path,
        default=Path("/data1/zoyo/projects/droid-preprocess-pipeline/scripts/step06_generate_pointclouds.py"),
        help="Path to the reference DROID pointcloud script.",
    )
    parser.add_argument(
        "--reference_h5",
        type=Path,
        default=DEFAULT_REFERENCE_H5,
        help="Optional raw dataset trajectory_valid.h5 for extrinsic comparison.",
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=0,
        help="Frame index to export.",
    )
    parser.add_argument(
        "--depth_min",
        type=float,
        default=0.1,
        help="Min valid depth in meters.",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=2.0,
        help="Max valid depth in meters.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.003,
        help="Voxel size for downsampling.",
    )
    return parser.parse_args()


def load_step06_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("step06_generate_pointclouds", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load step06 script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_camera_bundle(camera_dir: Path, frame_index: int) -> dict[str, Any]:
    camera_name = camera_dir.name
    main_npz = camera_dir / f"{camera_name}.npz"
    rgb_path = camera_dir / "images" / f"{camera_name}_{frame_index}.png"
    depth_path = camera_dir / "depth" / f"{camera_name}_{frame_index}_raw.npz"

    if not main_npz.is_file():
        raise FileNotFoundError(main_npz)
    if not rgb_path.is_file():
        raise FileNotFoundError(rgb_path)
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)

    with np.load(main_npz) as data:
        intrinsics = data["intrinsics"][frame_index].astype(np.float32)
        extrinsics = data["extrinsics"][frame_index].astype(np.float32)

    with np.load(depth_path) as data:
        depth = data["depth"].astype(np.float32)

    rgb = np.array(Image.open(rgb_path).convert("RGB"))

    return {
        "camera_name": camera_name,
        "main_npz": main_npz,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "depth": depth,
        "rgb": rgb,
    }


def create_pointcloud_from_c2w(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    c2w: np.ndarray,
    depth_min: float,
    depth_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate world-frame xyz/rgb assuming extrinsics are stored as c2w."""
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    valid = np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)
    if not np.any(valid):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    u_valid = u[valid].astype(np.float64)
    v_valid = v[valid].astype(np.float64)
    z_valid = depth[valid].astype(np.float64)

    x_cam = (u_valid - cx) * z_valid / fx
    y_cam = (v_valid - cy) * z_valid / fy
    pts_cam = np.stack([x_cam, y_cam, z_valid], axis=-1)

    pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
    colors = rgb[valid].astype(np.float32) / 255.0
    return pts_world.astype(np.float32), colors.astype(np.float32)


def save_ply_binary(output_path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors_u8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    vertex = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors_u8[:, 0]
    vertex["green"] = colors_u8[:, 1]
    vertex["blue"] = colors_u8[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with output_path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertex.tofile(f)


def approx_symmetric_chamfer(a: np.ndarray, b: np.ndarray, max_points: int = 3000) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")

    rng = np.random.default_rng(0)
    if len(a) > max_points:
        a = a[rng.choice(len(a), max_points, replace=False)]
    if len(b) > max_points:
        b = b[rng.choice(len(b), max_points, replace=False)]

    def mean_nn(x: np.ndarray, y: np.ndarray, chunk: int = 256) -> float:
        dists = []
        for start in range(0, len(x), chunk):
            xx = x[start : start + chunk]
            d2 = ((xx[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
            dists.append(np.sqrt(d2.min(axis=1)))
        return float(np.concatenate(dists).mean())

    return 0.5 * (mean_nn(a, b) + mean_nn(b, a))


def load_reference_extrinsics(reference_h5: Path) -> dict[str, dict[str, np.ndarray]] | None:
    if not reference_h5.is_file():
        return None

    cameras = ["hand_camera", "varied_camera_1", "varied_camera_2"]
    out: dict[str, dict[str, np.ndarray]] = {}
    with h5py.File(reference_h5, "r") as f:
        for camera in cameras:
            out[camera] = {
                "extrinsics": f[f"observation/camera/extrinsics/{camera}_left"][0].astype(np.float32),
                "intrinsics": f[f"observation/camera/intrinsics/{camera}_left"][0].astype(np.float32),
            }
    return out


def main() -> None:
    args = parse_args()
    step06 = load_step06_module(args.step06_script)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "frame_index": args.frame_index,
        "depth_min": args.depth_min,
        "depth_max": args.depth_max,
        "voxel_size": args.voxel_size,
        "step06_script": str(args.step06_script),
        "reference_h5": str(args.reference_h5),
        "groups": {},
        "dataset_match": {},
        "cross_mode_consistency": {},
    }

    loaded: dict[str, dict[str, dict[str, Any]]] = {"c2w": {}, "w2c": {}}
    for group, camera_dirs in DEFAULT_INPUTS.items():
        group_points = []
        group_colors = []
        summary["groups"][group] = {}
        for camera_name, camera_dir in camera_dirs.items():
            bundle = load_camera_bundle(camera_dir, args.frame_index)
            if group == "w2c":
                points, colors = step06.create_pointcloud_arrays(
                    rgb=bundle["rgb"],
                    depth=bundle["depth"],
                    fx=float(bundle["intrinsics"][0, 0]),
                    fy=float(bundle["intrinsics"][1, 1]),
                    cx=float(bundle["intrinsics"][0, 2]),
                    cy=float(bundle["intrinsics"][1, 2]),
                    extrinsic=bundle["extrinsics"],
                    depth_min=args.depth_min,
                    depth_max=args.depth_max,
                )
            else:
                points, colors = create_pointcloud_from_c2w(
                    rgb=bundle["rgb"],
                    depth=bundle["depth"],
                    intrinsics=bundle["intrinsics"],
                    c2w=bundle["extrinsics"],
                    depth_min=args.depth_min,
                    depth_max=args.depth_max,
                )

            points, colors = step06.voxel_downsample(points, colors, args.voxel_size)
            loaded[group][camera_name] = {
                "bundle": bundle,
                "points": points,
                "colors": colors,
            }
            group_points.append(points)
            group_colors.append(colors)

            indiv_path = (
                args.output_root
                / "individual"
                / group
                / f"{camera_name}_frame{args.frame_index:05d}.ply"
            )
            save_ply_binary(indiv_path, points, colors)

            summary["groups"][group][camera_name] = {
                "input_dir": str(camera_dir),
                "stored_extrinsics_mode": group,
                "points": int(len(points)),
                "ply_path": str(indiv_path),
                "main_npz": str(bundle["main_npz"]),
                "image_path": str(bundle["rgb_path"]),
                "depth_path": str(bundle["depth_path"]),
            }

        merged_points = np.concatenate(group_points, axis=0)
        merged_colors = np.concatenate(group_colors, axis=0)
        combined_path = args.output_root / "combined" / f"{group}_frame{args.frame_index:05d}_combined.ply"
        save_ply_binary(combined_path, merged_points, merged_colors)
        summary["groups"][group]["combined"] = {
            "points": int(len(merged_points)),
            "ply_path": str(combined_path),
        }

    reference = load_reference_extrinsics(args.reference_h5)
    if reference is not None:
        for group in ["c2w", "w2c"]:
            summary["dataset_match"][group] = {}
            for camera_name in ["hand_camera", "varied_camera_1", "varied_camera_2"]:
                bundle = loaded[group][camera_name]["bundle"]
                summary["dataset_match"][group][camera_name] = {
                    "extrinsics_l2_diff": float(
                        np.linalg.norm(bundle["extrinsics"] - reference[camera_name]["extrinsics"])
                    ),
                    "intrinsics_l2_diff": float(
                        np.linalg.norm(bundle["intrinsics"] - reference[camera_name]["intrinsics"])
                    ),
                }

    for camera_name in ["hand_camera", "varied_camera_1", "varied_camera_2"]:
        c2w_extr = loaded["c2w"][camera_name]["bundle"]["extrinsics"]
        w2c_extr = loaded["w2c"][camera_name]["bundle"]["extrinsics"]
        summary["cross_mode_consistency"][camera_name] = {
            "extrinsics_inverse_l2_diff": float(np.linalg.norm(np.linalg.inv(c2w_extr) - w2c_extr)),
            "pointcloud_chamfer_after_correct_interpretation": approx_symmetric_chamfer(
                loaded["c2w"][camera_name]["points"],
                loaded["w2c"][camera_name]["points"],
            ),
        }

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
