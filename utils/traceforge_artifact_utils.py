from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from utils.extrinsics_utils import normalize_extrinsics_to_w2c


V2_LAYOUT = "v2"
LEGACY_LAYOUT = "legacy"
SCENE_H5_NAME = "scene.h5"
SCENE_META_NAME = "scene_meta.json"
SCENE_RGB_NAME = "scene_rgb.mp4"
SCENE_STORAGE_CACHE = "cache"
SCENE_STORAGE_SOURCE_REF = "source_ref"
DEFAULT_SCENE_STORAGE_MODE = SCENE_STORAGE_SOURCE_REF

_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg")
_RGB_FRAME_EXTS = (".jpg", ".jpeg", ".png")
_DEPTH_FRAME_EXTS = (".npy", ".png", ".jpg", ".jpeg")


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def detect_output_layout(episode_dir: str | Path) -> str:
    episode_dir = _to_path(episode_dir)
    samples_dir = episode_dir / "samples"
    if samples_dir.is_dir() and (
        (episode_dir / SCENE_META_NAME).is_file()
        or (episode_dir / SCENE_H5_NAME).is_file()
    ):
        return V2_LAYOUT
    return LEGACY_LAYOUT


def get_scene_storage_mode(meta: dict[str, Any] | None) -> str:
    if not meta:
        return SCENE_STORAGE_CACHE
    return str(meta.get("scene_storage_mode", SCENE_STORAGE_CACHE))


def _resolve_meta_path(base_dir: str | Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = _to_path(raw_path)
    if path.is_absolute():
        return path
    return _to_path(base_dir) / path


def _frame_sort_key(path: str | Path) -> tuple[int, str]:
    path = str(path)
    stem = Path(path).stem
    if stem.isdigit():
        return int(stem), path
    digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
    if digits:
        return int(digits[-1]), path
    return 0, path


def _collect_and_sort_frame_files(root: str | Path, extensions: tuple[str, ...]) -> list[Path]:
    root = _to_path(root)
    files: list[Path] = []
    for ext in extensions:
        files.extend(sorted(root.glob(f"*{ext}")))
    if not files:
        left_dir = root / "left"
        if left_dir.is_dir():
            for ext in extensions:
                files.extend(sorted(left_dir.glob(f"*{ext}")))
    files.sort(key=_frame_sort_key)
    return files


def _load_rgb_image(path: str | Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_depth_image(path: str | Path) -> np.ndarray:
    path = _to_path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    image = Image.open(path).convert("I;16")
    return np.array(image).astype(np.float32) / 1000.0


def _load_external_geom_arrays(
    *,
    geom_path: str | Path,
    camera_name: str,
    extr_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    geom_path = _to_path(geom_path)
    if not geom_path.exists():
        raise FileNotFoundError(f"Missing source geometry file: {geom_path}")

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
    else:
        data = np.load(geom_path)
        try:
            if "intrinsics" not in data or "extrinsics" not in data:
                raise KeyError(
                    f"NPZ geometry must contain 'intrinsics' and 'extrinsics': {geom_path}"
                )
            intrinsics = data["intrinsics"].astype(np.float32)
            extrinsics = data["extrinsics"].astype(np.float32)
        finally:
            data.close()

    if intrinsics.shape[0] != extrinsics.shape[0]:
        raise ValueError(
            f"Geometry intrinsics/extrinsics length mismatch: {intrinsics.shape[0]} vs {extrinsics.shape[0]}"
        )
    extrinsics_w2c = normalize_extrinsics_to_w2c(
        extrinsics,
        extr_mode=extr_mode,
        context="SceneReader source geometry",
    )
    return intrinsics, extrinsics_w2c


def is_traceforge_output_complete(output_dir: str | Path) -> bool:
    output_dir = _to_path(output_dir)
    samples_dir = output_dir / "samples"
    has_samples = samples_dir.is_dir() and any(samples_dir.glob("*.npz"))
    if not has_samples:
        return False

    if detect_output_layout(output_dir) == V2_LAYOUT:
        meta = load_scene_meta(output_dir)
        if meta is None:
            return False
        storage_mode = get_scene_storage_mode(meta)
        if storage_mode == SCENE_STORAGE_CACHE:
            scene_h5_path = _resolve_meta_path(output_dir, meta.get("scene_h5_path", SCENE_H5_NAME))
            rgb_cache_path = _resolve_meta_path(output_dir, meta.get("rgb_cache_path", SCENE_RGB_NAME))
            return bool(
                scene_h5_path is not None
                and scene_h5_path.is_file()
                and rgb_cache_path is not None
                and rgb_cache_path.is_file()
            )
        if storage_mode == SCENE_STORAGE_SOURCE_REF:
            source_rgb_path = _resolve_meta_path(output_dir, meta.get("source_rgb_path"))
            source_depth_path = _resolve_meta_path(output_dir, meta.get("source_depth_path"))
            source_geom_path = _resolve_meta_path(output_dir, meta.get("source_geom_path"))
            return bool(
                source_rgb_path is not None
                and source_rgb_path.exists()
                and source_depth_path is not None
                and source_depth_path.exists()
                and source_geom_path is not None
                and source_geom_path.exists()
            )
        return False

    images_dir = output_dir / "images"
    depth_dir = output_dir / "depth"
    main_npz = output_dir / f"{output_dir.name}.npz"
    return (
        main_npz.is_file()
        and images_dir.is_dir()
        and depth_dir.is_dir()
        and any(images_dir.iterdir())
    )


def path_kind(path: str | None) -> str | None:
    if not path:
        return None
    path_obj = Path(path)
    if path_obj.is_dir():
        return "directory"
    if path_obj.is_file():
        return "file"
    return "missing"


def ensure_uint8_video(video_tensor: np.ndarray) -> np.ndarray:
    video = np.asarray(video_tensor)
    if video.ndim != 4:
        raise ValueError(f"Expected video with shape (T,H,W,3), got {video.shape}")
    if video.dtype == np.uint8:
        return video
    if np.issubdtype(video.dtype, np.floating):
        return np.clip(np.round(video * 255.0), 0, 255).astype(np.uint8)
    return np.clip(video, 0, 255).astype(np.uint8)


def write_scene_h5(
    scene_h5_path: str | Path,
    *,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
) -> None:
    scene_h5_path = _to_path(scene_h5_path)
    scene_h5_path.parent.mkdir(parents=True, exist_ok=True)
    depths = np.asarray(depths, dtype=np.float16)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics_w2c = np.asarray(extrinsics_w2c, dtype=np.float32)
    _, height, width = depths.shape

    with h5py.File(scene_h5_path, "w") as f:
        camera_group = f.create_group("camera")
        dense_group = f.create_group("dense")
        camera_group.create_dataset("intrinsics", data=intrinsics)
        camera_group.create_dataset("extrinsics_w2c", data=extrinsics_w2c)
        dense_group.create_dataset(
            "depth",
            data=depths,
            chunks=(1, height, width),
            compression="gzip",
            shuffle=True,
        )


def write_scene_rgb_mp4(
    rgb_path: str | Path,
    *,
    video_frames: np.ndarray,
    fps: int = 10,
) -> None:
    rgb_path = _to_path(rgb_path)
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    frames = ensure_uint8_video(video_frames)
    writer = imageio.get_writer(
        str(rgb_path),
        fps=max(int(fps), 1),
        codec="libx264",
        quality=7,
        macro_block_size=1,
        pixelformat="yuv420p",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def write_scene_meta(meta_path: str | Path, meta: dict[str, Any]) -> None:
    meta_path = _to_path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def load_scene_meta(episode_dir: str | Path) -> dict[str, Any] | None:
    meta_path = _to_path(episode_dir) / SCENE_META_NAME
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text())


def infer_sample_valid_steps_from_traj(traj_uvz: np.ndarray) -> np.ndarray:
    traj_uvz = np.asarray(traj_uvz)
    if traj_uvz.ndim != 3:
        raise ValueError(f"Expected traj with shape (N,T,3), got {traj_uvz.shape}")
    return ~np.all(np.isinf(traj_uvz), axis=(0, 2))


def is_prefix_valid_steps(valid_steps: np.ndarray) -> bool:
    valid_steps = np.asarray(valid_steps, dtype=bool)
    if valid_steps.ndim != 1:
        return False
    valid_len = int(valid_steps.sum())
    if valid_len == 0:
        return True
    return bool(valid_steps[:valid_len].all() and (~valid_steps[valid_len:]).all())


def normalize_sample_data(sample_path: str | Path) -> dict[str, Any]:
    sample_path = _to_path(sample_path)
    data = np.load(sample_path)
    try:
        if "traj_uvz" in data:
            traj_uvz = data["traj_uvz"].astype(np.float32)
            keypoints = data["keypoints"].astype(np.float32)
            query_frame_index = int(np.asarray(data["query_frame_index"]).reshape(-1)[0])
            segment_frame_indices = data["segment_frame_indices"].astype(np.int32)
            num_tracks = int(traj_uvz.shape[0])
            traj_valid_mask = (
                np.asarray(data["traj_valid_mask"]).astype(bool, copy=False)
                if "traj_valid_mask" in data
                else np.ones(traj_uvz.shape[0], dtype=bool)
            )
            dense_query_count = (
                int(np.asarray(data["dense_query_count"]).reshape(-1)[0])
                if "dense_query_count" in data
                else int(traj_uvz.shape[0])
            )
            tracked_query_count = (
                int(np.asarray(data["tracked_query_count"]).reshape(-1)[0])
                if "tracked_query_count" in data
                else int(np.isfinite(traj_uvz).any(axis=(1, 2)).sum())
            )
            support_grid_size = (
                int(np.asarray(data["support_grid_size"]).reshape(-1)[0])
                if "support_grid_size" in data
                else None
            )
            visibility = data["visibility"].astype(np.float16) if "visibility" in data else None
            traj_depth_consistency_ratio = (
                data["traj_depth_consistency_ratio"].astype(np.float16)
                if "traj_depth_consistency_ratio" in data
                else None
            )
            traj_stable_depth_consistency_ratio = (
                data["traj_stable_depth_consistency_ratio"].astype(np.float16)
                if "traj_stable_depth_consistency_ratio" in data
                else None
            )
            traj_high_volatility_hit = (
                np.asarray(data["traj_high_volatility_hit"]).astype(bool, copy=False)
                if "traj_high_volatility_hit" in data
                else None
            )
            traj_volatility_exposure_ratio = (
                data["traj_volatility_exposure_ratio"].astype(np.float16)
                if "traj_volatility_exposure_ratio" in data
                else None
            )
            traj_compare_frame_count = (
                data["traj_compare_frame_count"].astype(np.uint16)
                if "traj_compare_frame_count" in data
                else None
            )
            traj_stable_compare_frame_count = (
                data["traj_stable_compare_frame_count"].astype(np.uint16)
                if "traj_stable_compare_frame_count" in data
                else None
            )
            traj_mask_reason_bits = (
                data["traj_mask_reason_bits"].astype(np.uint8)
                if "traj_mask_reason_bits" in data
                else None
            )
            traj_supervision_mask = (
                np.asarray(data["traj_supervision_mask"]).astype(bool, copy=False)
                if "traj_supervision_mask" in data
                else None
            )
            traj_supervision_prefix_len = (
                data["traj_supervision_prefix_len"].astype(np.uint16)
                if "traj_supervision_prefix_len" in data
                else None
            )
            traj_supervision_count = (
                data["traj_supervision_count"].astype(np.uint16)
                if "traj_supervision_count" in data
                else None
            )
            traj_wrist_seed_mask = (
                np.asarray(data["traj_wrist_seed_mask"]).astype(bool, copy=False)
                if "traj_wrist_seed_mask" in data
                else np.zeros(num_tracks, dtype=bool)
            )
            traj_query_depth_rank = (
                data["traj_query_depth_rank"].astype(np.float16)
                if "traj_query_depth_rank" in data
                else np.full(num_tracks, np.nan, dtype=np.float16)
            )
            traj_query_depth_edge_mask = (
                np.asarray(data["traj_query_depth_edge_mask"]).astype(bool, copy=False)
                if "traj_query_depth_edge_mask" in data
                else np.zeros(num_tracks, dtype=bool)
            )
            traj_query_depth_patch_valid_ratio = (
                data["traj_query_depth_patch_valid_ratio"].astype(np.float16)
                if "traj_query_depth_patch_valid_ratio" in data
                else np.full(num_tracks, np.nan, dtype=np.float16)
            )
            traj_query_depth_patch_std = (
                data["traj_query_depth_patch_std"].astype(np.float16)
                if "traj_query_depth_patch_std" in data
                else np.full(num_tracks, np.nan, dtype=np.float16)
            )
            traj_query_depth_edge_risk_mask = (
                np.asarray(data["traj_query_depth_edge_risk_mask"]).astype(bool, copy=False)
                if "traj_query_depth_edge_risk_mask" in data
                else np.zeros(num_tracks, dtype=bool)
            )
            traj_motion_extent = (
                data["traj_motion_extent"].astype(np.float16)
                if "traj_motion_extent" in data
                else np.full(num_tracks, np.nan, dtype=np.float16)
            )
            traj_motion_step_median = (
                data["traj_motion_step_median"].astype(np.float16)
                if "traj_motion_step_median" in data
                else np.full(num_tracks, np.nan, dtype=np.float16)
            )
            traj_manipulator_candidate_mask = (
                np.asarray(data["traj_manipulator_candidate_mask"]).astype(bool, copy=False)
                if "traj_manipulator_candidate_mask" in data
                else np.zeros(num_tracks, dtype=bool)
            )
            traj_manipulator_cluster_id = (
                data["traj_manipulator_cluster_id"].astype(np.int16)
                if "traj_manipulator_cluster_id" in data
                else np.full(num_tracks, -1, dtype=np.int16)
            )
            traj_manipulator_component_size = (
                data["traj_manipulator_component_size"].astype(np.uint16)
                if "traj_manipulator_component_size" in data
                else np.zeros(num_tracks, dtype=np.uint16)
            )
            traj_manipulator_cluster_fallback_used = (
                bool(np.asarray(data["traj_manipulator_cluster_fallback_used"]).reshape(-1)[0])
                if "traj_manipulator_cluster_fallback_used" in data
                else False
            )
            return {
                "layout": V2_LAYOUT,
                "traj_uvz": traj_uvz,
                "traj_2d": traj_uvz[..., :2].astype(np.float32, copy=False),
                "keypoints": keypoints,
                "query_frame_index": query_frame_index,
                "segment_frame_indices": segment_frame_indices,
                "dense_query_count": dense_query_count,
                "tracked_query_count": tracked_query_count,
                "support_grid_size": support_grid_size,
                "traj_valid_mask": traj_valid_mask,
                "visibility": visibility,
                "traj_depth_consistency_ratio": traj_depth_consistency_ratio,
                "traj_stable_depth_consistency_ratio": traj_stable_depth_consistency_ratio,
                "traj_high_volatility_hit": traj_high_volatility_hit,
                "traj_volatility_exposure_ratio": traj_volatility_exposure_ratio,
                "traj_compare_frame_count": traj_compare_frame_count,
                "traj_stable_compare_frame_count": traj_stable_compare_frame_count,
                "traj_mask_reason_bits": traj_mask_reason_bits,
                "traj_supervision_mask": traj_supervision_mask,
                "traj_supervision_prefix_len": traj_supervision_prefix_len,
                "traj_supervision_count": traj_supervision_count,
                "traj_wrist_seed_mask": traj_wrist_seed_mask,
                "traj_query_depth_rank": traj_query_depth_rank,
                "traj_query_depth_edge_mask": traj_query_depth_edge_mask,
                "traj_query_depth_patch_valid_ratio": traj_query_depth_patch_valid_ratio,
                "traj_query_depth_patch_std": traj_query_depth_patch_std,
                "traj_query_depth_edge_risk_mask": traj_query_depth_edge_risk_mask,
                "traj_motion_extent": traj_motion_extent,
                "traj_motion_step_median": traj_motion_step_median,
                "traj_manipulator_candidate_mask": traj_manipulator_candidate_mask,
                "traj_manipulator_cluster_id": traj_manipulator_cluster_id,
                "traj_manipulator_component_size": traj_manipulator_component_size,
                "traj_manipulator_cluster_fallback_used": traj_manipulator_cluster_fallback_used,
                "frame_aligned": True,
            }

        traj_uvz = data["traj"].astype(np.float32)
        traj_2d = (
            data["traj_2d"].astype(np.float32)
            if "traj_2d" in data
            else traj_uvz[..., :2].astype(np.float32, copy=False)
        )
        keypoints = data["keypoints"].astype(np.float32)
        query_frame_index = int(np.asarray(data["frame_index"]).reshape(-1)[0])
        num_tracks = int(traj_uvz.shape[0])
        if "traj_valid_mask" in data:
            traj_valid_mask = np.asarray(data["traj_valid_mask"]).astype(bool, copy=False)
        else:
            traj_valid_mask = np.ones(traj_uvz.shape[0], dtype=bool)
        dense_query_count = (
            int(np.asarray(data["dense_query_count"]).reshape(-1)[0])
            if "dense_query_count" in data
            else int(traj_uvz.shape[0])
        )
        tracked_query_count = (
            int(np.asarray(data["tracked_query_count"]).reshape(-1)[0])
            if "tracked_query_count" in data
            else int(np.isfinite(traj_uvz).any(axis=(1, 2)).sum())
        )
        support_grid_size = (
            int(np.asarray(data["support_grid_size"]).reshape(-1)[0])
            if "support_grid_size" in data
            else None
        )

        if "valid_steps" in data:
            valid_steps = np.asarray(data["valid_steps"]).astype(bool, copy=False)
        else:
            valid_steps = infer_sample_valid_steps_from_traj(traj_uvz)

        frame_aligned = is_prefix_valid_steps(valid_steps)
        if frame_aligned:
            valid_len = int(valid_steps.sum())
        else:
            valid_len = int(traj_uvz.shape[1])
        segment_frame_indices = query_frame_index + np.arange(valid_len, dtype=np.int32)
        return {
            "layout": LEGACY_LAYOUT,
            "traj_uvz": traj_uvz,
            "traj_2d": traj_2d,
            "keypoints": keypoints,
            "query_frame_index": query_frame_index,
            "segment_frame_indices": segment_frame_indices,
            "dense_query_count": dense_query_count,
            "tracked_query_count": tracked_query_count,
            "support_grid_size": support_grid_size,
            "traj_valid_mask": traj_valid_mask,
            "visibility": None,
            "traj_depth_consistency_ratio": (
                data["traj_depth_consistency_ratio"].astype(np.float16)
                if "traj_depth_consistency_ratio" in data
                else None
            ),
            "traj_stable_depth_consistency_ratio": (
                data["traj_stable_depth_consistency_ratio"].astype(np.float16)
                if "traj_stable_depth_consistency_ratio" in data
                else None
            ),
            "traj_high_volatility_hit": (
                np.asarray(data["traj_high_volatility_hit"]).astype(bool, copy=False)
                if "traj_high_volatility_hit" in data
                else None
            ),
            "traj_volatility_exposure_ratio": (
                data["traj_volatility_exposure_ratio"].astype(np.float16)
                if "traj_volatility_exposure_ratio" in data
                else None
            ),
            "traj_compare_frame_count": (
                data["traj_compare_frame_count"].astype(np.uint16)
                if "traj_compare_frame_count" in data
                else None
            ),
            "traj_stable_compare_frame_count": (
                data["traj_stable_compare_frame_count"].astype(np.uint16)
                if "traj_stable_compare_frame_count" in data
                else None
            ),
            "traj_mask_reason_bits": (
                data["traj_mask_reason_bits"].astype(np.uint8)
                if "traj_mask_reason_bits" in data
                else None
            ),
            "traj_supervision_mask": (
                np.asarray(data["traj_supervision_mask"]).astype(bool, copy=False)
                if "traj_supervision_mask" in data
                else None
            ),
            "traj_supervision_prefix_len": (
                data["traj_supervision_prefix_len"].astype(np.uint16)
                if "traj_supervision_prefix_len" in data
                else None
            ),
            "traj_supervision_count": (
                data["traj_supervision_count"].astype(np.uint16)
                if "traj_supervision_count" in data
                else None
            ),
            "traj_wrist_seed_mask": (
                np.asarray(data["traj_wrist_seed_mask"]).astype(bool, copy=False)
                if "traj_wrist_seed_mask" in data
                else np.zeros(num_tracks, dtype=bool)
            ),
            "traj_query_depth_rank": (
                data["traj_query_depth_rank"].astype(np.float16)
                if "traj_query_depth_rank" in data
                else np.full(num_tracks, np.nan, dtype=np.float16)
            ),
            "traj_query_depth_edge_mask": (
                np.asarray(data["traj_query_depth_edge_mask"]).astype(bool, copy=False)
                if "traj_query_depth_edge_mask" in data
                else np.zeros(num_tracks, dtype=bool)
            ),
            "traj_query_depth_patch_valid_ratio": (
                data["traj_query_depth_patch_valid_ratio"].astype(np.float16)
                if "traj_query_depth_patch_valid_ratio" in data
                else np.full(num_tracks, np.nan, dtype=np.float16)
            ),
            "traj_query_depth_patch_std": (
                data["traj_query_depth_patch_std"].astype(np.float16)
                if "traj_query_depth_patch_std" in data
                else np.full(num_tracks, np.nan, dtype=np.float16)
            ),
            "traj_query_depth_edge_risk_mask": (
                np.asarray(data["traj_query_depth_edge_risk_mask"]).astype(bool, copy=False)
                if "traj_query_depth_edge_risk_mask" in data
                else np.zeros(num_tracks, dtype=bool)
            ),
            "traj_motion_extent": (
                data["traj_motion_extent"].astype(np.float16)
                if "traj_motion_extent" in data
                else np.full(num_tracks, np.nan, dtype=np.float16)
            ),
            "traj_motion_step_median": (
                data["traj_motion_step_median"].astype(np.float16)
                if "traj_motion_step_median" in data
                else np.full(num_tracks, np.nan, dtype=np.float16)
            ),
            "traj_manipulator_candidate_mask": (
                np.asarray(data["traj_manipulator_candidate_mask"]).astype(bool, copy=False)
                if "traj_manipulator_candidate_mask" in data
                else np.zeros(num_tracks, dtype=bool)
            ),
            "traj_manipulator_cluster_id": (
                data["traj_manipulator_cluster_id"].astype(np.int16)
                if "traj_manipulator_cluster_id" in data
                else np.full(num_tracks, -1, dtype=np.int16)
            ),
            "traj_manipulator_component_size": (
                data["traj_manipulator_component_size"].astype(np.uint16)
                if "traj_manipulator_component_size" in data
                else np.zeros(num_tracks, dtype=np.uint16)
            ),
            "traj_manipulator_cluster_fallback_used": (
                bool(np.asarray(data["traj_manipulator_cluster_fallback_used"]).reshape(-1)[0])
                if "traj_manipulator_cluster_fallback_used" in data
                else False
            ),
            "valid_steps": valid_steps,
            "frame_aligned": frame_aligned,
        }
    finally:
        data.close()


def _apply_render_step_mask(traj: np.ndarray, render_step_mask: np.ndarray) -> np.ndarray:
    traj = np.array(traj, dtype=np.float32, copy=True)
    render_step_mask = np.asarray(render_step_mask, dtype=bool)
    if traj.shape[:2] != render_step_mask.shape:
        raise ValueError(
            f"Expected render_step_mask shape {traj.shape[:2]}, got {render_step_mask.shape}"
        )
    traj[~render_step_mask] = np.nan
    return traj


def build_sample_visualization_view(sample: dict[str, Any]) -> dict[str, Any]:
    """Return a wrist-aware visualization view over a normalized sample payload."""
    traj_uvz = np.asarray(sample["traj_uvz"], dtype=np.float32)
    traj_2d = np.asarray(sample["traj_2d"], dtype=np.float32)
    keypoints = np.asarray(sample["keypoints"], dtype=np.float32)
    segment_frame_indices = np.asarray(sample["segment_frame_indices"], dtype=np.int32)
    traj_valid_mask = np.asarray(sample["traj_valid_mask"]).astype(bool, copy=False)
    traj_supervision_mask = sample.get("traj_supervision_mask")
    if traj_supervision_mask is not None:
        traj_supervision_mask = np.asarray(traj_supervision_mask).astype(bool, copy=False)

    if traj_uvz.ndim != 3 or traj_uvz.shape[-1] != 3:
        raise ValueError(f"Expected traj_uvz shape (N,T,3), got {traj_uvz.shape}")
    if traj_2d.shape != (*traj_uvz.shape[:2], 2):
        raise ValueError(f"Expected traj_2d shape {(*traj_uvz.shape[:2], 2)}, got {traj_2d.shape}")
    if keypoints.shape != (traj_uvz.shape[0], 2):
        raise ValueError(f"Expected keypoints shape {(traj_uvz.shape[0], 2)}, got {keypoints.shape}")
    if traj_valid_mask.shape != (traj_uvz.shape[0],):
        raise ValueError(f"Expected traj_valid_mask shape {(traj_uvz.shape[0],)}, got {traj_valid_mask.shape}")

    if sample.get("frame_aligned", False) and len(segment_frame_indices) < traj_uvz.shape[1]:
        segment_len = int(len(segment_frame_indices))
        traj_uvz = traj_uvz[:, :segment_len]
        traj_2d = traj_2d[:, :segment_len]
        if traj_supervision_mask is not None and traj_supervision_mask.ndim == 2:
            traj_supervision_mask = traj_supervision_mask[:, :segment_len]

    raw_num_tracks = int(traj_uvz.shape[0])
    num_frames = int(traj_uvz.shape[1])
    traj_uvz = traj_uvz[traj_valid_mask]
    traj_2d = traj_2d[traj_valid_mask]
    keypoints = keypoints[traj_valid_mask]
    kept_num_tracks = int(traj_uvz.shape[0])

    finite_step_mask = np.isfinite(traj_uvz).all(axis=-1)
    render_step_mask: np.ndarray | None = None
    if traj_supervision_mask is not None:
        if traj_supervision_mask.shape == (raw_num_tracks, num_frames):
            render_step_mask = np.asarray(traj_supervision_mask[traj_valid_mask], dtype=bool)
        elif traj_supervision_mask.shape == (kept_num_tracks, num_frames):
            render_step_mask = np.asarray(traj_supervision_mask, dtype=bool)

    if render_step_mask is None:
        valid_steps = sample.get("valid_steps")
        if valid_steps is not None:
            valid_steps = np.asarray(valid_steps).astype(bool, copy=False).reshape(-1)[:num_frames]
            if valid_steps.shape == (num_frames,):
                render_step_mask = np.broadcast_to(valid_steps[None, :], traj_uvz.shape[:2]).copy()

    if render_step_mask is None:
        render_step_mask = finite_step_mask.copy()
    else:
        render_step_mask &= finite_step_mask

    return {
        "traj_uvz": _apply_render_step_mask(traj_uvz, render_step_mask),
        "traj_2d": _apply_render_step_mask(traj_2d, render_step_mask),
        "keypoints": keypoints,
        "segment_frame_indices": segment_frame_indices,
        "render_step_mask": render_step_mask.astype(bool),
        "rendered_frame_count": render_step_mask.sum(axis=1).astype(np.uint16),
        "raw_num_tracks": raw_num_tracks,
        "kept_num_tracks": kept_num_tracks,
    }


def list_sample_query_frames(episode_dir: str | Path, video_name: str | None = None) -> list[int]:
    episode_dir = _to_path(episode_dir)
    samples_dir = episode_dir / "samples"
    if not samples_dir.is_dir():
        return []

    pattern = f"{video_name}_*.npz" if video_name else "*.npz"
    frames: list[int] = []
    for sample_path in sorted(samples_dir.glob(pattern)):
        suffix = sample_path.stem.split("_")[-1]
        try:
            frames.append(int(suffix))
        except ValueError:
            continue
    return frames


def traj_uvz_to_world(
    traj_uvz: np.ndarray,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
) -> np.ndarray:
    traj_uvz = np.asarray(traj_uvz, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    u = traj_uvz[..., 0]
    v = traj_uvz[..., 1]
    z = traj_uvz[..., 2]
    valid = np.isfinite(traj_uvz).all(axis=-1) & (z > 0.01) & (z < 50.0)

    x_cam = np.where(valid, (u - cx) * z / (fx + 1e-8), np.nan)
    y_cam = np.where(valid, (v - cy) * z / (fy + 1e-8), np.nan)
    z_cam = np.where(valid, z, np.nan)

    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    pts_cam_h = np.concatenate(
        [pts_cam, np.ones((*pts_cam.shape[:2], 1), dtype=np.float32)],
        axis=-1,
    )
    c2w = np.linalg.inv(w2c).astype(np.float32)
    pts_world = (c2w @ pts_cam_h.reshape(-1, 4).T).T.reshape(*pts_cam.shape[:2], 4)
    pts_world = pts_world[..., :3].astype(np.float32)
    pts_world[~valid] = np.nan
    return pts_world


def build_pointcloud_from_frame(
    *,
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    downsample: int = 4,
    depth_min: float = 0.01,
    depth_max: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world-space points filtered by query-frame camera depth."""
    from utils.threed_utils import unproject_by_depth

    depth = np.asarray(depth, dtype=np.float32)
    rgb = ensure_uint8_video(np.asarray(rgb)[None])[0]
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    w2c = np.asarray(w2c, dtype=np.float32)
    c2w = np.linalg.inv(w2c).astype(np.float32)

    xyz = unproject_by_depth(depth[None, None], intrinsics[None], c2w[None])[0].transpose(1, 2, 0)
    pts = xyz[::downsample, ::downsample].reshape(-1, 3)
    colors = rgb[::downsample, ::downsample].reshape(-1, 3).astype(np.float32) / 255.0
    depth_ds = depth[::downsample, ::downsample].reshape(-1)

    valid = (
        np.isfinite(pts).all(axis=1)
        & np.isfinite(colors).all(axis=1)
        & np.isfinite(depth_ds)
        & (depth_ds > depth_min)
        & (depth_ds < depth_max)
    )
    return pts[valid].astype(np.float32), colors[valid].astype(np.float32)


class SceneReader:
    def __init__(self, episode_dir: str | Path):
        self.episode_dir = _to_path(episode_dir)
        self.layout = detect_output_layout(self.episode_dir)
        self.video_name = self.episode_dir.name
        self._scene_meta: dict[str, Any] | None = None
        self._scene_h5: h5py.File | None = None
        self._scene_reader = None
        self._source_rgb_reader = None
        self._source_depth_reader = None
        self._main_npz = None
        self._source_rgb_files: list[Path] | None = None
        self._source_depth_files: list[Path] | None = None
        self._source_frame_indices: np.ndarray | None = None
        self._source_intrinsics: np.ndarray | None = None
        self._source_extrinsics: np.ndarray | None = None

    def close(self) -> None:
        if self._scene_h5 is not None:
            self._scene_h5.close()
            self._scene_h5 = None
        if self._scene_reader is not None:
            self._scene_reader.close()
            self._scene_reader = None
        if self._source_rgb_reader is not None:
            self._source_rgb_reader.close()
            self._source_rgb_reader = None
        if self._source_depth_reader is not None:
            self._source_depth_reader.close()
            self._source_depth_reader = None
        if self._main_npz is not None:
            self._main_npz.close()
            self._main_npz = None

    def __enter__(self) -> "SceneReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_scene_meta(self) -> dict[str, Any]:
        if self._scene_meta is None:
            self._scene_meta = load_scene_meta(self.episode_dir)
        if self._scene_meta is None:
            raise FileNotFoundError(f"Missing {SCENE_META_NAME} under {self.episode_dir}")
        return self._scene_meta

    def _get_scene_storage_mode(self) -> str:
        if self.layout != V2_LAYOUT:
            return SCENE_STORAGE_CACHE
        return get_scene_storage_mode(self._require_scene_meta())

    def _require_scene_h5(self) -> h5py.File:
        if self._scene_h5 is None:
            meta = self._require_scene_meta()
            scene_h5_path = _resolve_meta_path(self.episode_dir, meta.get("scene_h5_path", SCENE_H5_NAME))
            if scene_h5_path is None or not scene_h5_path.is_file():
                raise FileNotFoundError(
                    f"Missing scene cache H5 for cache-backed v2 output: {scene_h5_path}"
                )
            self._scene_h5 = h5py.File(scene_h5_path, "r")
        return self._scene_h5

    def _require_scene_reader(self):
        if self._scene_reader is None:
            meta = self._require_scene_meta()
            rgb_cache_path = _resolve_meta_path(self.episode_dir, meta.get("rgb_cache_path", SCENE_RGB_NAME))
            if rgb_cache_path is None or not rgb_cache_path.is_file():
                raise FileNotFoundError(
                    f"Missing scene RGB cache for cache-backed v2 output: {rgb_cache_path}"
                )
            self._scene_reader = imageio.get_reader(str(rgb_cache_path))
        return self._scene_reader

    def _require_main_npz(self):
        if self._main_npz is None:
            self._main_npz = np.load(self.episode_dir / f"{self.video_name}.npz")
        return self._main_npz

    def _require_source_frame_indices(self) -> np.ndarray:
        if self._source_frame_indices is None:
            meta = self._require_scene_meta()
            raw_indices = meta.get("source_frame_indices")
            if raw_indices is None:
                frame_count = int(meta.get("frame_count", 0))
                self._source_frame_indices = np.arange(frame_count, dtype=np.int32)
            else:
                self._source_frame_indices = np.asarray(raw_indices, dtype=np.int32).reshape(-1)
            if np.any(self._source_frame_indices < 0):
                raise ValueError(f"source_frame_indices must be non-negative: {self.episode_dir}")
        return self._source_frame_indices

    def _map_source_frame_index(self, frame_idx: int) -> int:
        frame_idx = int(frame_idx)
        source_frame_indices = self._require_source_frame_indices()
        if frame_idx < 0 or frame_idx >= len(source_frame_indices):
            raise IndexError(
                f"frame_idx={frame_idx} exceeds available frames ({len(source_frame_indices)}) "
                f"for {self.episode_dir}"
            )
        return int(source_frame_indices[frame_idx])

    def _require_source_rgb_path(self) -> Path:
        meta = self._require_scene_meta()
        source_rgb_path = _resolve_meta_path(self.episode_dir, meta.get("source_rgb_path"))
        if source_rgb_path is None or not source_rgb_path.exists():
            raise FileNotFoundError(
                f"source_ref output is missing source_rgb_path under {self.episode_dir}"
            )
        return source_rgb_path

    def _require_source_depth_path(self) -> Path:
        meta = self._require_scene_meta()
        source_depth_path = _resolve_meta_path(self.episode_dir, meta.get("source_depth_path"))
        if source_depth_path is None or not source_depth_path.exists():
            raise FileNotFoundError(
                f"source_ref output is missing source_depth_path under {self.episode_dir}"
            )
        return source_depth_path

    def _require_source_geom_path(self) -> Path:
        meta = self._require_scene_meta()
        source_geom_path = _resolve_meta_path(self.episode_dir, meta.get("source_geom_path"))
        if source_geom_path is None or not source_geom_path.exists():
            raise FileNotFoundError(
                f"source_ref output is missing source_geom_path under {self.episode_dir}"
            )
        return source_geom_path

    def _require_source_rgb_files(self) -> list[Path]:
        if self._source_rgb_files is None:
            source_rgb_path = self._require_source_rgb_path()
            if not source_rgb_path.is_dir():
                raise FileNotFoundError(
                    f"source_rgb_path must be a directory for frame-backed source_ref outputs: {source_rgb_path}"
                )
            self._source_rgb_files = _collect_and_sort_frame_files(source_rgb_path, _RGB_FRAME_EXTS)
            if not self._source_rgb_files:
                raise FileNotFoundError(f"No RGB frames found under {source_rgb_path}")
        return self._source_rgb_files

    def _require_source_depth_files(self) -> list[Path]:
        if self._source_depth_files is None:
            source_depth_path = self._require_source_depth_path()
            if not source_depth_path.is_dir():
                raise FileNotFoundError(
                    f"source_depth_path must be a directory for frame-backed source_ref outputs: {source_depth_path}"
                )
            self._source_depth_files = _collect_and_sort_frame_files(source_depth_path, _DEPTH_FRAME_EXTS)
            if not self._source_depth_files:
                raise FileNotFoundError(f"No depth frames found under {source_depth_path}")
        return self._source_depth_files

    def _require_source_rgb_reader(self):
        if self._source_rgb_reader is None:
            source_rgb_path = self._require_source_rgb_path()
            if source_rgb_path.suffix.lower() not in _VIDEO_EXTS:
                raise FileNotFoundError(
                    f"Unsupported source RGB file for source_ref output: {source_rgb_path}"
                )
            self._source_rgb_reader = imageio.get_reader(str(source_rgb_path))
        return self._source_rgb_reader

    def _require_source_depth_reader(self):
        if self._source_depth_reader is None:
            source_depth_path = self._require_source_depth_path()
            if source_depth_path.suffix.lower() not in _VIDEO_EXTS:
                raise FileNotFoundError(
                    f"Unsupported source depth file for source_ref output: {source_depth_path}"
                )
            self._source_depth_reader = imageio.get_reader(str(source_depth_path))
        return self._source_depth_reader

    def _require_source_geom_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._source_intrinsics is None or self._source_extrinsics is None:
            meta = self._require_scene_meta()
            camera_name = str(meta.get("source_camera_name") or self.video_name)
            extr_mode = str(meta.get("source_extrinsics_mode") or "w2c")
            source_geom_path = self._require_source_geom_path()
            self._source_intrinsics, self._source_extrinsics = _load_external_geom_arrays(
                geom_path=source_geom_path,
                camera_name=camera_name,
                extr_mode=extr_mode,
            )
        return self._source_intrinsics, self._source_extrinsics

    def get_camera_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.layout == V2_LAYOUT:
            if self._get_scene_storage_mode() == SCENE_STORAGE_CACHE:
                scene_h5 = self._require_scene_h5()
                intrinsics = scene_h5["camera/intrinsics"][:].astype(np.float32)
                extrinsics = scene_h5["camera/extrinsics_w2c"][:].astype(np.float32)
                return intrinsics, extrinsics

            source_indices = self._require_source_frame_indices()
            intrinsics_all, extrinsics_all = self._require_source_geom_arrays()
            if len(source_indices) == 0:
                return (
                    np.zeros((0, 3, 3), dtype=np.float32),
                    np.zeros((0, 4, 4), dtype=np.float32),
                )
            if int(source_indices.max()) >= len(intrinsics_all) or int(source_indices.max()) >= len(extrinsics_all):
                raise IndexError(
                    f"source_frame_indices exceed source geometry length for {self.episode_dir}: "
                    f"max={int(source_indices.max())}, intrinsics={len(intrinsics_all)}, extrinsics={len(extrinsics_all)}"
                )
            return (
                intrinsics_all[source_indices].astype(np.float32),
                extrinsics_all[source_indices].astype(np.float32),
            )

        data = self._require_main_npz()
        return data["intrinsics"].astype(np.float32), data["extrinsics"].astype(np.float32)

    def get_image_size(self) -> tuple[int, int]:
        if self.layout == V2_LAYOUT:
            meta = load_scene_meta(self.episode_dir)
            if meta is None:
                raise FileNotFoundError(f"Missing {SCENE_META_NAME} under {self.episode_dir}")
            return int(meta["height"]), int(meta["width"])

        data = self._require_main_npz()
        return int(data["height"]), int(data["width"])

    def get_depth_frame(self, frame_idx: int) -> np.ndarray:
        if self.layout == V2_LAYOUT:
            if self._get_scene_storage_mode() == SCENE_STORAGE_CACHE:
                scene_h5 = self._require_scene_h5()
                return scene_h5["dense/depth"][frame_idx].astype(np.float32)

            source_frame_idx = self._map_source_frame_index(frame_idx)
            source_depth_path = self._require_source_depth_path()
            if source_depth_path.is_dir():
                source_depth_files = self._require_source_depth_files()
                if source_frame_idx >= len(source_depth_files):
                    raise IndexError(
                        f"source depth frame {source_frame_idx} exceeds available files ({len(source_depth_files)}) "
                        f"under {source_depth_path}"
                    )
                return _load_depth_image(source_depth_files[source_frame_idx]).astype(np.float32)

            if source_depth_path.suffix.lower() in _VIDEO_EXTS:
                reader = self._require_source_depth_reader()
                frame = np.asarray(reader.get_data(source_frame_idx), dtype=np.float32)
                if frame.ndim == 3:
                    frame = frame[..., 0]
                return frame.astype(np.float32)
            raise FileNotFoundError(
                f"Unsupported source depth storage for source_ref output: {source_depth_path}"
            )

        depth_raw_path = self.episode_dir / "depth" / f"{self.video_name}_{frame_idx}_raw.npz"
        if depth_raw_path.is_file():
            data = np.load(depth_raw_path)
            try:
                return data["depth"].astype(np.float32)
            finally:
                data.close()

        data = self._require_main_npz()
        if "depths" not in data:
            raise FileNotFoundError(f"Legacy output missing depth for frame {frame_idx}: {self.episode_dir}")
        return data["depths"][frame_idx].astype(np.float32)

    def get_rgb_frame(self, frame_idx: int) -> np.ndarray:
        if self.layout == V2_LAYOUT:
            if self._get_scene_storage_mode() == SCENE_STORAGE_CACHE:
                reader = self._require_scene_reader()
                return np.asarray(reader.get_data(frame_idx), dtype=np.uint8)

            source_frame_idx = self._map_source_frame_index(frame_idx)
            source_rgb_path = self._require_source_rgb_path()
            if source_rgb_path.is_dir():
                source_rgb_files = self._require_source_rgb_files()
                if source_frame_idx >= len(source_rgb_files):
                    raise IndexError(
                        f"source RGB frame {source_frame_idx} exceeds available files ({len(source_rgb_files)}) "
                        f"under {source_rgb_path}"
                    )
                return _load_rgb_image(source_rgb_files[source_frame_idx])

            if source_rgb_path.suffix.lower() in _VIDEO_EXTS:
                reader = self._require_source_rgb_reader()
                return np.asarray(reader.get_data(source_frame_idx), dtype=np.uint8)
            raise FileNotFoundError(
                f"Unsupported source RGB storage for source_ref output: {source_rgb_path}"
            )

        image_path = self.episode_dir / "images" / f"{self.video_name}_{frame_idx}.png"
        if image_path.is_file():
            return np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)

        data = self._require_main_npz()
        if "video" not in data:
            raise FileNotFoundError(f"Legacy output missing RGB frame {frame_idx}: {self.episode_dir}")
        video = data["video"]
        frame = video[frame_idx]
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = frame.transpose(1, 2, 0)
        if frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[-1] in (1, 3):
            return frame
        return ensure_uint8_video(frame[None])[0]

    def load_segment_rgb_frames(self, frame_indices: np.ndarray) -> np.ndarray:
        frame_indices = np.asarray(frame_indices, dtype=np.int32)
        return np.stack([self.get_rgb_frame(int(idx)) for idx in frame_indices], axis=0)

    def load_segment_depth_frames(self, frame_indices: np.ndarray) -> np.ndarray:
        frame_indices = np.asarray(frame_indices, dtype=np.int32)
        return np.stack([self.get_depth_frame(int(idx)) for idx in frame_indices], axis=0)
