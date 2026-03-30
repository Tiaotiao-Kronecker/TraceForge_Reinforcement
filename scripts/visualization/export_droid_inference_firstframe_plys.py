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

The input folders are expected to be current TraceForge `v2` camera output
directories with `scene_meta.json` source references, not legacy main-NPZ
artifacts.

Outputs:
  <output_root>/individual/<group>/<camera>_frame00000.ply
  <output_root>/combined/<group>_frame00000_combined.ply
  <output_root>/summary.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.traceforge_artifact_utils import SceneReader, load_scene_meta


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


def _resolve_meta_path(episode_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return episode_dir / path


def _stringify_path(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _load_raw_camera_arrays(*, geom_path: Path, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    if geom_path.suffix.lower() == ".h5":
        intr_key_with_suffix = f"observation/camera/intrinsics/{camera_name}_left"
        extr_key_with_suffix = f"observation/camera/extrinsics/{camera_name}_left"
        intr_key_no_suffix = f"observation/camera/intrinsics/{camera_name}"
        extr_key_no_suffix = f"observation/camera/extrinsics/{camera_name}"
        with h5py.File(geom_path, "r") as f:
            if intr_key_with_suffix in f and extr_key_with_suffix in f:
                intrinsics = f[intr_key_with_suffix][:].astype(np.float32)
                extrinsics = f[extr_key_with_suffix][:].astype(np.float32)
            elif intr_key_no_suffix in f and extr_key_no_suffix in f:
                intrinsics = f[intr_key_no_suffix][:].astype(np.float32)
                extrinsics = f[extr_key_no_suffix][:].astype(np.float32)
            else:
                available = list(f["observation/camera/intrinsics"].keys()) if "observation/camera/intrinsics" in f else []
                raise KeyError(
                    f"H5 geometry must contain either '{intr_key_with_suffix}' or '{intr_key_no_suffix}'. "
                    f"Available cameras: {available}"
                )
        return intrinsics, extrinsics

    data = np.load(geom_path)
    try:
        if "intrinsics" not in data or "extrinsics" not in data:
            raise KeyError(f"NPZ geometry must contain 'intrinsics' and 'extrinsics': {geom_path}")
        intrinsics = data["intrinsics"].astype(np.float32)
        extrinsics = data["extrinsics"].astype(np.float32)
    finally:
        data.close()
    return intrinsics, extrinsics


def load_camera_bundle(camera_dir: Path, frame_index: int) -> dict[str, Any]:
    camera_dir = camera_dir.resolve()
    if not camera_dir.is_dir():
        raise FileNotFoundError(f"Episode directory not found: {camera_dir}")
    camera_name = camera_dir.name
    scene_meta = load_scene_meta(camera_dir)
    if scene_meta is None:
        raise FileNotFoundError(f"Missing scene_meta.json under {camera_dir}")

    source_geom_path = _resolve_meta_path(camera_dir, scene_meta.get("source_geom_path"))
    if source_geom_path is None or not source_geom_path.is_file():
        raise FileNotFoundError(f"Missing source geometry file for {camera_dir}: {source_geom_path}")

    stored_extrinsics_mode = str(
        scene_meta.get("source_extrinsics_mode") or scene_meta.get("extrinsics_mode") or ""
    ).strip().lower()
    if stored_extrinsics_mode not in {"w2c", "c2w"}:
        raise ValueError(
            f"Unsupported stored extrinsics mode for {camera_dir}: {stored_extrinsics_mode!r}"
        )

    source_frame_indices = np.asarray(
        scene_meta.get("source_frame_indices", []),
        dtype=np.int32,
    ).reshape(-1)
    source_frame_index = int(frame_index) if len(source_frame_indices) == 0 else int(source_frame_indices[frame_index])

    intrinsics_all, extrinsics_all = _load_raw_camera_arrays(
        geom_path=source_geom_path,
        camera_name=str(scene_meta.get("source_camera_name") or camera_name),
    )
    if source_frame_index < 0 or source_frame_index >= len(intrinsics_all) or source_frame_index >= len(extrinsics_all):
        raise IndexError(
            f"source frame {source_frame_index} exceeds geometry length for {camera_dir}: "
            f"intrinsics={len(intrinsics_all)}, extrinsics={len(extrinsics_all)}"
        )

    with SceneReader(camera_dir) as scene_reader:
        rgb = scene_reader.get_rgb_frame(frame_index)
        depth = scene_reader.get_depth_frame(frame_index)

    return {
        "episode_dir": camera_dir,
        "camera_name": camera_name,
        "scene_meta_path": camera_dir / "scene_meta.json",
        "source_geom_path": source_geom_path,
        "source_rgb_path": _resolve_meta_path(camera_dir, scene_meta.get("source_rgb_path")),
        "source_depth_path": _resolve_meta_path(camera_dir, scene_meta.get("source_depth_path")),
        "source_frame_index": source_frame_index,
        "stored_extrinsics_mode": stored_extrinsics_mode,
        "intrinsics": intrinsics_all[source_frame_index].astype(np.float32),
        "extrinsics": extrinsics_all[source_frame_index].astype(np.float32),
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
            if bundle["stored_extrinsics_mode"] != group:
                raise ValueError(
                    f"Input group '{group}' does not match stored extrinsics mode "
                    f"'{bundle['stored_extrinsics_mode']}' for {camera_dir}"
                )
            if bundle["stored_extrinsics_mode"] == "w2c":
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
                "input_dir": str(bundle["episode_dir"]),
                "stored_extrinsics_mode": bundle["stored_extrinsics_mode"],
                "points": int(len(points)),
                "ply_path": str(indiv_path),
                "scene_meta_path": str(bundle["scene_meta_path"]),
                "source_geom_path": str(bundle["source_geom_path"]),
                "source_rgb_path": _stringify_path(bundle["source_rgb_path"]),
                "source_depth_path": _stringify_path(bundle["source_depth_path"]),
                "source_frame_index": int(bundle["source_frame_index"]),
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
