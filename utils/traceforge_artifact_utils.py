from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np


V2_LAYOUT = "v2"
LEGACY_LAYOUT = "legacy"
SCENE_H5_NAME = "scene.h5"
SCENE_META_NAME = "scene_meta.json"
SCENE_RGB_NAME = "scene_rgb.mp4"


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def detect_output_layout(episode_dir: str | Path) -> str:
    episode_dir = _to_path(episode_dir)
    if (episode_dir / SCENE_H5_NAME).is_file() and (episode_dir / "samples").is_dir():
        return V2_LAYOUT
    return LEGACY_LAYOUT


def is_traceforge_output_complete(output_dir: str | Path) -> bool:
    output_dir = _to_path(output_dir)
    samples_dir = output_dir / "samples"
    has_samples = samples_dir.is_dir() and any(samples_dir.glob("*.npz"))
    if not has_samples:
        return False

    if detect_output_layout(output_dir) == V2_LAYOUT:
        return (
            (output_dir / SCENE_H5_NAME).is_file()
            and (output_dir / SCENE_META_NAME).is_file()
            and (output_dir / SCENE_RGB_NAME).is_file()
        )

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
            traj_valid_mask = (
                np.asarray(data["traj_valid_mask"]).astype(bool, copy=False)
                if "traj_valid_mask" in data
                else np.ones(traj_uvz.shape[0], dtype=bool)
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
            return {
                "layout": V2_LAYOUT,
                "traj_uvz": traj_uvz,
                "traj_2d": traj_uvz[..., :2].astype(np.float32, copy=False),
                "keypoints": keypoints,
                "query_frame_index": query_frame_index,
                "segment_frame_indices": segment_frame_indices,
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
        if "traj_valid_mask" in data:
            traj_valid_mask = np.asarray(data["traj_valid_mask"]).astype(bool, copy=False)
        else:
            traj_valid_mask = np.ones(traj_uvz.shape[0], dtype=bool)

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
            "valid_steps": valid_steps,
            "frame_aligned": frame_aligned,
        }
    finally:
        data.close()


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
        self._scene_h5: h5py.File | None = None
        self._scene_reader = None
        self._main_npz = None

    def close(self) -> None:
        if self._scene_h5 is not None:
            self._scene_h5.close()
            self._scene_h5 = None
        if self._scene_reader is not None:
            self._scene_reader.close()
            self._scene_reader = None
        if self._main_npz is not None:
            self._main_npz.close()
            self._main_npz = None

    def __enter__(self) -> "SceneReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_scene_h5(self) -> h5py.File:
        if self._scene_h5 is None:
            self._scene_h5 = h5py.File(self.episode_dir / SCENE_H5_NAME, "r")
        return self._scene_h5

    def _require_scene_reader(self):
        if self._scene_reader is None:
            self._scene_reader = imageio.get_reader(str(self.episode_dir / SCENE_RGB_NAME))
        return self._scene_reader

    def _require_main_npz(self):
        if self._main_npz is None:
            self._main_npz = np.load(self.episode_dir / f"{self.video_name}.npz")
        return self._main_npz

    def get_camera_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.layout == V2_LAYOUT:
            scene_h5 = self._require_scene_h5()
            intrinsics = scene_h5["camera/intrinsics"][:].astype(np.float32)
            extrinsics = scene_h5["camera/extrinsics_w2c"][:].astype(np.float32)
            return intrinsics, extrinsics

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
            scene_h5 = self._require_scene_h5()
            return scene_h5["dense/depth"][frame_idx].astype(np.float32)

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
            reader = self._require_scene_reader()
            return np.asarray(reader.get_data(frame_idx), dtype=np.uint8)

        image_path = self.episode_dir / "images" / f"{self.video_name}_{frame_idx}.png"
        if image_path.is_file():
            import PIL.Image

            return np.array(PIL.Image.open(image_path).convert("RGB"), dtype=np.uint8)

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
