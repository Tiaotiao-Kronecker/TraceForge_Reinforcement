#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


CURRENT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORBSLAM3_ROOT = CURRENT_REPO_ROOT / "third_party" / "orbslam3_src"
DEFAULT_ORBSLAM3_RUNTIME_LIB_DIR = CURRENT_REPO_ROOT / ".conda_envs" / "orbslam3" / "lib"
DEFAULT_CAMERA_FPS = 30.0
DEFAULT_DEPTH_MAP_FACTOR = 1000.0
DEFAULT_STEREO_BASELINE_M = 0.07732
DEFAULT_STEREO_TH_DEPTH = 40.0
DEFAULT_ORB_N_FEATURES = 1000
DEFAULT_ORB_SCALE_FACTOR = 1.2
DEFAULT_ORB_N_LEVELS = 8
DEFAULT_ORB_INI_TH_FAST = 20
DEFAULT_ORB_MIN_TH_FAST = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and run an ORB-SLAM3 RGB-D mature baseline for one TraceForge case, "
            "then convert the resulting TUM trajectory into TraceForge geom npz."
        )
    )
    parser.add_argument(
        "--case_dir",
        type=Path,
        required=True,
        help="Case directory, e.g. .../stereo_left_start_00435_officialprep",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="stereo_left",
        help="Camera name under rgb/<camera_name> and depth/<camera_name>.",
    )
    parser.add_argument(
        "--orbslam3_root",
        type=Path,
        default=DEFAULT_ORBSLAM3_ROOT,
        help="ORB-SLAM3 source root that contains Vocabulary/ and Examples/RGB-D/.",
    )
    parser.add_argument(
        "--orb_binary",
        type=Path,
        default=None,
        help="Optional explicit path to rgbd_tum. Defaults to <orbslam3_root>/Examples/RGB-D/rgbd_tum.",
    )
    parser.add_argument(
        "--orb_vocab",
        type=Path,
        default=None,
        help="Optional explicit path to ORBvoc.txt. Defaults to <orbslam3_root>/Vocabulary/ORBvoc.txt.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "ORB working directory. Defaults to "
            "<case_dir>/_analysis_mature_baseline/orbslam3_rgbd."
        ),
    )
    parser.add_argument(
        "--settings_path",
        type=Path,
        default=None,
        help="Optional explicit ORB-SLAM3 settings YAML path.",
    )
    parser.add_argument(
        "--source_geom_npz",
        type=Path,
        default=None,
        help="Source TraceForge geometry npz. Defaults to geom/geom_<camera_name>_official_w2c.npz.",
    )
    parser.add_argument(
        "--dest_geom_npz",
        type=Path,
        default=None,
        help="Destination ORB-based geometry npz. Defaults to geom/geom_<camera_name>_orbslam3_rgbd_w2c.npz.",
    )
    parser.add_argument(
        "--camera_fps",
        type=float,
        default=DEFAULT_CAMERA_FPS,
        help="Fallback Camera.fps written into the ORB settings YAML when no trajectory_valid.h5 attr is present.",
    )
    parser.add_argument(
        "--depth_map_factor",
        type=float,
        default=DEFAULT_DEPTH_MAP_FACTOR,
        help="Meters-to-uint16 scale used to export depth PNGs and written into RGBD.DepthMapFactor.",
    )
    parser.add_argument(
        "--stereo_baseline_m",
        type=float,
        default=DEFAULT_STEREO_BASELINE_M,
        help="Stereo.b value written into the ORB settings YAML for RGB-D mode.",
    )
    parser.add_argument(
        "--stereo_th_depth",
        type=float,
        default=DEFAULT_STEREO_TH_DEPTH,
        help="Stereo.ThDepth value written into the ORB settings YAML for RGB-D mode.",
    )
    parser.add_argument(
        "--max_depth_m",
        type=float,
        default=65.0,
        help="Clamp exported uint16 depth PNG values to this metric depth before scaling.",
    )
    parser.add_argument(
        "--missing_pose_strategy",
        choices=("error", "hold_prev"),
        default="error",
        help="How to handle frames missing from CameraTrajectory.txt.",
    )
    parser.add_argument(
        "--skip_run",
        action="store_true",
        help="Prepare assets and convert an existing CameraTrajectory.txt without invoking ORB-SLAM3.",
    )
    parser.add_argument(
        "--skip_existing_depth",
        action="store_true",
        help="Reuse existing converted depth PNGs when the target file already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing settings, association, and converted geometry artifacts.",
    )
    return parser.parse_args()


def _sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return int(stem), stem
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if digits:
        return int(digits[-1]), stem
    return 0, stem


def _collect_files(directory: Path) -> list[Path]:
    files = [path for path in directory.iterdir() if path.is_file()]
    files.sort(key=_sort_key)
    return files


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    case_dir = args.case_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (case_dir / "_analysis_mature_baseline" / "orbslam3_rgbd").resolve()
    )
    source_geom_npz = (
        args.source_geom_npz.resolve()
        if args.source_geom_npz is not None
        else (case_dir / "geom" / f"geom_{args.camera_name}_official_w2c.npz").resolve()
    )
    dest_geom_npz = (
        args.dest_geom_npz.resolve()
        if args.dest_geom_npz is not None
        else (case_dir / "geom" / f"geom_{args.camera_name}_orbslam3_rgbd_w2c.npz").resolve()
    )
    settings_path = (
        args.settings_path.resolve()
        if args.settings_path is not None
        else (output_dir / f"{args.camera_name}_rgbd.yaml").resolve()
    )
    orbslam3_root = args.orbslam3_root.resolve()
    orb_binary = (
        args.orb_binary.resolve()
        if args.orb_binary is not None
        else (orbslam3_root / "Examples" / "RGB-D" / "rgbd_tum").resolve()
    )
    orb_vocab = (
        args.orb_vocab.resolve()
        if args.orb_vocab is not None
        else (orbslam3_root / "Vocabulary" / "ORBvoc.txt").resolve()
    )
    return {
        "case_dir": case_dir,
        "output_dir": output_dir,
        "source_geom_npz": source_geom_npz,
        "dest_geom_npz": dest_geom_npz,
        "settings_path": settings_path,
        "orbslam3_root": orbslam3_root,
        "orb_binary": orb_binary,
        "orb_vocab": orb_vocab,
    }


def _resolve_case_inputs(case_dir: Path, camera_name: str) -> tuple[Path, Path]:
    rgb_dir = case_dir / "rgb" / camera_name
    depth_dir = case_dir / "depth" / camera_name
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing RGB directory: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing depth directory: {depth_dir}")
    return rgb_dir, depth_dir


def _infer_case_fps(case_dir: Path, fallback_fps: float) -> float:
    traj_h5 = case_dir / "trajectory_valid.h5"
    if traj_h5.is_file() and h5py is not None:
        with h5py.File(traj_h5, "r") as f:
            raw_fps = f.attrs.get("fps")
        if raw_fps is not None:
            value = float(raw_fps)
            if value > 0:
                return value
    return float(fallback_fps)


def _load_rgb_size(rgb_path: Path) -> tuple[int, int]:
    with Image.open(rgb_path) as image:
        width, height = image.size
    return width, height


def _load_intrinsics(source_geom_npz: Path) -> np.ndarray:
    data = np.load(source_geom_npz)
    try:
        if "intrinsics" not in data:
            raise KeyError(f"Missing 'intrinsics' in {source_geom_npz}")
        intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
    finally:
        data.close()
    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")
    return intrinsics


def _write_settings_yaml(
    settings_path: Path,
    *,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    camera_fps: float,
    depth_map_factor: float,
    stereo_baseline_m: float,
    stereo_th_depth: float,
    overwrite: bool,
) -> None:
    if settings_path.exists() and not overwrite:
        return
    k = intrinsics[0]
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    settings_text = f"""%YAML:1.0

File.version: "1.0"

Camera.type: "PinHole"

Camera1.fx: {fx:.9f}
Camera1.fy: {fy:.9f}
Camera1.cx: {cx:.9f}
Camera1.cy: {cy:.9f}

Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera1.k3: 0.0

Camera.width: {int(width)}
Camera.height: {int(height)}
Camera.fps: {int(round(camera_fps))}
Camera.RGB: 1

Stereo.ThDepth: {float(stereo_th_depth):.6f}
Stereo.b: {float(stereo_baseline_m):.6f}

RGBD.DepthMapFactor: {float(depth_map_factor):.6f}

ORBextractor.nFeatures: {int(DEFAULT_ORB_N_FEATURES)}
ORBextractor.scaleFactor: {float(DEFAULT_ORB_SCALE_FACTOR):.6f}
ORBextractor.nLevels: {int(DEFAULT_ORB_N_LEVELS)}
ORBextractor.iniThFAST: {int(DEFAULT_ORB_INI_TH_FAST)}
ORBextractor.minThFAST: {int(DEFAULT_ORB_MIN_TH_FAST)}

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(settings_text, encoding="utf-8")


def _convert_depth_to_uint16_png(
    src_path: Path,
    dst_path: Path,
    *,
    depth_map_factor: float,
    max_depth_m: float,
    skip_existing: bool,
) -> None:
    if dst_path.exists() and skip_existing:
        return
    if src_path.suffix.lower() != ".npy":
        raise ValueError(
            f"Expected float32 .npy depth inputs for ORB conversion, got: {src_path}"
        )
    depth_m = np.asarray(np.load(src_path), dtype=np.float32)
    depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    depth_m = np.clip(depth_m, 0.0, float(max_depth_m))
    depth_u16 = np.rint(depth_m * float(depth_map_factor)).astype(np.uint16)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_u16).save(dst_path)


def _build_orb_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    if DEFAULT_ORBSLAM3_RUNTIME_LIB_DIR.is_dir():
        existing = env.get("LD_LIBRARY_PATH", "").strip()
        orb_lib = str(DEFAULT_ORBSLAM3_RUNTIME_LIB_DIR)
        env["LD_LIBRARY_PATH"] = f"{orb_lib}:{existing}" if existing else orb_lib
    return env


def _write_association_file(
    assoc_path: Path,
    *,
    case_dir: Path,
    rgb_paths: list[Path],
    depth_png_paths: list[Path],
    overwrite: bool,
) -> None:
    if assoc_path.exists() and not overwrite:
        return
    lines: list[str] = []
    for index, (rgb_path, depth_path) in enumerate(zip(rgb_paths, depth_png_paths)):
        timestamp = f"{float(index):.6f}"
        rgb_rel = rgb_path.relative_to(case_dir).as_posix()
        depth_rel = depth_path.relative_to(case_dir).as_posix()
        lines.append(f"{timestamp} {rgb_rel} {timestamp} {depth_rel}")
    assoc_path.parent.mkdir(parents=True, exist_ok=True)
    assoc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0:
        raise ValueError("Encountered zero-norm quaternion in CameraTrajectory.txt")
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )


def _fill_missing_poses_hold_prev(extrinsics: list[np.ndarray | None]) -> tuple[np.ndarray, list[int]]:
    missing_indices = [index for index, value in enumerate(extrinsics) if value is None]
    if not missing_indices:
        return np.stack([value for value in extrinsics if value is not None], axis=0), []
    first_valid = next((index for index, value in enumerate(extrinsics) if value is not None), None)
    if first_valid is None:
        raise RuntimeError("ORB-SLAM3 did not produce any valid poses.")
    for index in range(first_valid):
        extrinsics[index] = extrinsics[first_valid].copy()
    for index in range(first_valid + 1, len(extrinsics)):
        if extrinsics[index] is None:
            extrinsics[index] = extrinsics[index - 1].copy()
    return np.stack([value for value in extrinsics if value is not None], axis=0), missing_indices


def _convert_tum_to_geom(
    trajectory_path: Path,
    *,
    source_geom_npz: Path,
    dest_geom_npz: Path,
    missing_pose_strategy: str,
) -> dict[str, Any]:
    intrinsics = _load_intrinsics(source_geom_npz)
    extrinsics_by_frame: list[np.ndarray | None] = [None] * int(intrinsics.shape[0])
    tracked_frame_indices: list[int] = []
    for raw_line in trajectory_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        timestamp = float(parts[0])
        frame_index = int(round(timestamp))
        if frame_index < 0 or frame_index >= len(extrinsics_by_frame):
            raise RuntimeError(
                f"Trajectory timestamp {timestamp} mapped to out-of-range frame {frame_index} "
                f"for {len(extrinsics_by_frame)} frames."
            )
        _, tx, ty, tz, qx, qy, qz, qw = (float(value) for value in parts[:8])
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = _quat_xyzw_to_rot(qx, qy, qz, qw)
        c2w[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        extrinsics_by_frame[frame_index] = np.linalg.inv(c2w).astype(np.float32)
        tracked_frame_indices.append(frame_index)

    missing_indices = [index for index, value in enumerate(extrinsics_by_frame) if value is None]
    if missing_indices:
        if missing_pose_strategy == "error":
            raise RuntimeError(
                "ORB-SLAM3 trajectory is missing frames. "
                f"Missing frame indices: {missing_indices}"
            )
        if missing_pose_strategy != "hold_prev":
            raise ValueError(f"Unsupported missing pose strategy: {missing_pose_strategy}")
        extrinsics, missing_indices = _fill_missing_poses_hold_prev(extrinsics_by_frame)
    else:
        extrinsics = np.stack([value for value in extrinsics_by_frame if value is not None], axis=0)

    dest_geom_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest_geom_npz, intrinsics=intrinsics.astype(np.float32), extrinsics=extrinsics.astype(np.float32))
    return {
        "source_geom_npz": str(source_geom_npz),
        "dest_geom_npz": str(dest_geom_npz),
        "frame_count": int(intrinsics.shape[0]),
        "tracked_frame_count": int(len(tracked_frame_indices)),
        "tracked_frame_indices": tracked_frame_indices,
        "missing_frame_indices": missing_indices,
        "missing_pose_strategy": missing_pose_strategy,
    }


def _write_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    resolved = _resolve_paths(args)
    case_dir = resolved["case_dir"]
    output_dir = resolved["output_dir"]
    source_geom_npz = resolved["source_geom_npz"]
    dest_geom_npz = resolved["dest_geom_npz"]
    settings_path = resolved["settings_path"]
    orb_binary = resolved["orb_binary"]
    orb_vocab = resolved["orb_vocab"]

    rgb_dir, depth_dir = _resolve_case_inputs(case_dir, args.camera_name)
    rgb_paths = _collect_files(rgb_dir)
    depth_paths = _collect_files(depth_dir)
    if not rgb_paths:
        raise FileNotFoundError(f"No RGB files found under {rgb_dir}")
    if len(rgb_paths) != len(depth_paths):
        raise RuntimeError(
            f"RGB/depth file count mismatch for {args.camera_name}: "
            f"{len(rgb_paths)} vs {len(depth_paths)}"
        )

    if not source_geom_npz.is_file():
        raise FileNotFoundError(f"Missing source geometry npz: {source_geom_npz}")
    if not orb_binary.is_file():
        raise FileNotFoundError(f"Missing ORB-SLAM3 binary: {orb_binary}")
    if not orb_vocab.is_file():
        raise FileNotFoundError(f"Missing ORB vocabulary file: {orb_vocab}")

    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = _load_rgb_size(rgb_paths[0])
    intrinsics = _load_intrinsics(source_geom_npz)
    camera_fps = _infer_case_fps(case_dir, args.camera_fps)

    _write_settings_yaml(
        settings_path,
        intrinsics=intrinsics,
        width=width,
        height=height,
        camera_fps=camera_fps,
        depth_map_factor=args.depth_map_factor,
        stereo_baseline_m=args.stereo_baseline_m,
        stereo_th_depth=args.stereo_th_depth,
        overwrite=args.overwrite,
    )

    depth_png_dir = output_dir / "depth_png" / args.camera_name
    depth_png_paths: list[Path] = []
    for depth_path in depth_paths:
        dst_path = depth_png_dir / f"{depth_path.stem}.png"
        _convert_depth_to_uint16_png(
            depth_path,
            dst_path,
            depth_map_factor=args.depth_map_factor,
            max_depth_m=args.max_depth_m,
            skip_existing=args.skip_existing_depth and not args.overwrite,
        )
        depth_png_paths.append(dst_path)

    assoc_path = output_dir / "associate.txt"
    _write_association_file(
        assoc_path,
        case_dir=case_dir,
        rgb_paths=rgb_paths,
        depth_png_paths=depth_png_paths,
        overwrite=args.overwrite,
    )

    prepare_summary = {
        "case_dir": str(case_dir),
        "camera_name": args.camera_name,
        "rgb_dir": str(rgb_dir),
        "depth_dir": str(depth_dir),
        "depth_png_dir": str(depth_png_dir),
        "settings_path": str(settings_path),
        "assoc_path": str(assoc_path),
        "source_geom_npz": str(source_geom_npz),
        "dest_geom_npz": str(dest_geom_npz),
        "orb_binary": str(orb_binary),
        "orb_vocab": str(orb_vocab),
        "frame_count": len(rgb_paths),
        "width": width,
        "height": height,
        "camera_fps": camera_fps,
        "depth_map_factor": float(args.depth_map_factor),
        "stereo_baseline_m": float(args.stereo_baseline_m),
        "stereo_th_depth": float(args.stereo_th_depth),
    }
    _write_summary(output_dir / "prepare_summary.json", prepare_summary)

    trajectory_path = output_dir / "CameraTrajectory.txt"
    if not args.skip_run:
        command = [
            str(orb_binary),
            str(orb_vocab),
            str(settings_path),
            str(case_dir),
            str(assoc_path),
        ]
        subprocess.run(command, cwd=output_dir, env=_build_orb_runtime_env(), check=True)
        _write_summary(
            output_dir / "run_summary.json",
            {
                "command": command,
                "cwd": str(output_dir),
                "orb_runtime_lib_dir": (
                    str(DEFAULT_ORBSLAM3_RUNTIME_LIB_DIR)
                    if DEFAULT_ORBSLAM3_RUNTIME_LIB_DIR.is_dir()
                    else None
                ),
                "camera_trajectory_path": str(trajectory_path),
                "keyframe_trajectory_path": str(output_dir / "KeyFrameTrajectory.txt"),
            },
        )

    if not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Missing CameraTrajectory.txt at {trajectory_path}. "
            "If ORB-SLAM3 was skipped, ensure the trajectory already exists."
        )

    conversion_summary = _convert_tum_to_geom(
        trajectory_path,
        source_geom_npz=source_geom_npz,
        dest_geom_npz=dest_geom_npz,
        missing_pose_strategy=args.missing_pose_strategy,
    )
    _write_summary(output_dir / "conversion_summary.json", conversion_summary)

    result = {
        "prepare_summary": str(output_dir / "prepare_summary.json"),
        "run_summary": str(output_dir / "run_summary.json"),
        "conversion_summary": str(output_dir / "conversion_summary.json"),
        "camera_trajectory_path": str(trajectory_path),
        "dest_geom_npz": str(dest_geom_npz),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
