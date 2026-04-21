from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
from PIL import Image


XPERIENCE_ADAPTER_TYPE = "xperience_v1"
XPERIENCE_SUPPORTED_CAMERAS = ("stereo_left",)


@dataclass(frozen=True)
class XperienceEpisodePaths:
    episode_dir: Path
    annotation_path: Path
    video_path: Path
    camera_name: str


def decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_scalar(value[()])
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parse_ffprobe_rate(raw_rate: str | None) -> float | None:
    if not raw_rate:
        return None
    if "/" in raw_rate:
        numerator_text, denominator_text = raw_rate.split("/", 1)
        numerator = float(numerator_text)
        denominator = float(denominator_text)
        if denominator == 0.0:
            return None
        return numerator / denominator
    return float(raw_rate)


def probe_video_metadata(path: str | Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe returned no video streams for {path}")
    stream = streams[0]
    metadata: dict[str, Any] = {}
    width = stream.get("width")
    height = stream.get("height")
    if width is not None and height is not None:
        metadata["size"] = [int(width), int(height)]
    fps = _parse_ffprobe_rate(stream.get("r_frame_rate"))
    if fps is not None:
        metadata["fps"] = fps
    return metadata


def calibration_vector_to_matrix(calibration: np.ndarray) -> np.ndarray:
    values = np.asarray(calibration, dtype=np.float64).reshape(-1)
    if values.shape != (4,):
        raise ValueError(f"Expected calibration vector [fx, fy, cx, cy], got {values.shape}")
    fx, fy, cx, cy = values.tolist()
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def scale_intrinsics(
    intrinsics: np.ndarray,
    *,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    if source_h <= 0 or source_w <= 0:
        raise ValueError(f"Invalid source_hw: {source_hw}")
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target_hw: {target_hw}")

    scaled = np.asarray(intrinsics, dtype=np.float32).copy()
    scaled[0, 0] *= float(target_w) / float(source_w)
    scaled[0, 2] *= float(target_w) / float(source_w)
    scaled[1, 1] *= float(target_h) / float(source_h)
    scaled[1, 2] *= float(target_h) / float(source_h)
    return scaled


def quat_wxyz_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion [w, x, y, z], got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def slam_pose_to_w2c(translation_xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = quat_wxyz_to_rotation_matrix(quat_wxyz)
    matrix[:3, 3] = np.asarray(translation_xyz, dtype=np.float32).reshape(3)
    return matrix


def build_stereo_left_intrinsics(
    handle: h5py.File,
    *,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    if "calibration/cam01/K" not in handle:
        raise KeyError("Missing calibration/cam01/K in annotation.hdf5")
    base_intrinsics = calibration_vector_to_matrix(handle["calibration/cam01/K"][()])
    return scale_intrinsics(base_intrinsics, source_hw=source_hw, target_hw=target_hw)


def build_stereo_left_extrinsics(
    handle: h5py.File,
    *,
    source_frame_indices: Sequence[int],
) -> np.ndarray:
    source_frame_indices_np = np.asarray(source_frame_indices, dtype=np.int32).reshape(-1)
    if source_frame_indices_np.size == 0:
        return np.zeros((0, 4, 4), dtype=np.float32)

    # h5py requires monotonic increasing fancy indices, but TraceForge scene metadata
    # may preserve arbitrary source frame order for windowed sampling.
    order = np.argsort(source_frame_indices_np, kind="stable")
    sorted_indices = source_frame_indices_np[order]
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.shape[0], dtype=order.dtype)

    translations_sorted = np.asarray(handle["slam/trans_xyz"][sorted_indices], dtype=np.float32)
    quaternions_sorted = np.asarray(handle["slam/quat_wxyz"][sorted_indices], dtype=np.float32)
    translations = translations_sorted[inverse_order]
    quaternions = quaternions_sorted[inverse_order]
    extrinsics = np.empty((translations.shape[0], 4, 4), dtype=np.float32)
    for frame_idx, (translation, quaternion) in enumerate(zip(translations, quaternions)):
        extrinsics[frame_idx] = slam_pose_to_w2c(translation, quaternion)
    return extrinsics


def aligned_video_fps(handle: h5py.File) -> float:
    timestamps = np.asarray(
        [int(decode_scalar(value)) for value in handle["video/device_timestamp"][:]],
        dtype=np.int64,
    )
    if timestamps.size >= 2:
        deltas_sec = np.diff(timestamps).astype(np.float64) / 1e9
        positive = deltas_sec[deltas_sec > 0]
        if positive.size:
            return round(1.0 / float(np.median(positive)), 6)

    video_length_sec = float(handle["video/length_sec"][()])
    if video_length_sec <= 0:
        raise ValueError(f"video_length_sec must be positive, got {video_length_sec}")
    frame_count = int(handle["video/frame_number"].shape[0])
    return round(float(frame_count) / float(video_length_sec), 6)


def read_caption_main_task(handle: h5py.File) -> str:
    if "caption" not in handle:
        return "xperience task"
    raw_caption = decode_scalar(handle["caption"][()])
    if not raw_caption:
        return "xperience task"
    try:
        payload = json.loads(raw_caption)
    except Exception:
        return "xperience task"
    return str(payload.get("config", {}).get("Main Task") or "xperience task").strip()


def resize_rgb_frame(frame: np.ndarray, *, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    if image.size != (target_w, target_h):
        image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _load_video_frames_ffmpeg(video_path: str | Path, frame_indices: Sequence[int]) -> np.ndarray:
    requested_indices = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    if requested_indices.size == 0:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)

    metadata = probe_video_metadata(video_path)
    width, height = map(int, metadata.get("size") or [0, 0])
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not determine video size for {video_path}")

    unique_indices = np.unique(requested_indices)
    select_terms = [f"eq(n\\,{int(frame_idx)})" for frame_idx in unique_indices.tolist()]
    filter_expr = "select=" + "+".join(select_terms)
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(video_path),
            "-vf",
            filter_expr,
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    )

    raw = result.stdout or b""
    frame_bytes = height * width * 3
    if frame_bytes <= 0 or len(raw) % frame_bytes != 0:
        raise RuntimeError(
            f"Unexpected ffmpeg rawvideo payload for {video_path}: got {len(raw)} bytes for frame size {height}x{width}"
        )

    decoded_count = len(raw) // frame_bytes
    if decoded_count != unique_indices.shape[0]:
        raise RuntimeError(
            f"Requested {unique_indices.shape[0]} frames from {video_path}, but ffmpeg returned {decoded_count}"
        )

    decoded = np.frombuffer(raw, dtype=np.uint8).reshape(decoded_count, height, width, 3)
    index_map = {int(frame_idx): idx for idx, frame_idx in enumerate(unique_indices.tolist())}
    ordered = decoded[[index_map[int(frame_idx)] for frame_idx in requested_indices.tolist()]]
    return np.asarray(ordered, dtype=np.uint8)


def load_video_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    return load_video_frames(video_path, [int(frame_index)])[0]


def load_video_frames(
    video_path: str | Path,
    frame_indices: Sequence[int],
    *,
    target_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    requested_indices = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
    if requested_indices.size == 0:
        target_h, target_w = target_hw or (0, 0)
        return np.zeros((0, target_h, target_w, 3), dtype=np.uint8)

    frames = _load_video_frames_ffmpeg(video_path, requested_indices)
    if target_hw is not None:
        frames = np.stack(
            [resize_rgb_frame(frame, target_hw=target_hw) for frame in frames],
            axis=0,
        )
    return np.asarray(frames, dtype=np.uint8)


def open_xperience_episode(
    episode_dir: str | Path,
    *,
    camera_name: str = "stereo_left",
) -> XperienceEpisodePaths:
    episode_dir = Path(episode_dir).expanduser().resolve()
    if camera_name not in XPERIENCE_SUPPORTED_CAMERAS:
        raise ValueError(
            f"camera_name={camera_name!r} is not supported yet; expected one of {XPERIENCE_SUPPORTED_CAMERAS}"
        )
    annotation_path = episode_dir / "annotation.hdf5"
    video_path = episode_dir / f"{camera_name}.mp4"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotation.hdf5 under {episode_dir}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing {camera_name}.mp4 under {episode_dir}")
    return XperienceEpisodePaths(
        episode_dir=episode_dir,
        annotation_path=annotation_path,
        video_path=video_path,
        camera_name=camera_name,
    )


def build_xperience_source_descriptor(
    *,
    dataset_root: str | Path,
    episode_dir: str | Path,
    camera_name: str,
    window_start: int,
    window_stop: int,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> dict[str, Any]:
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    episode_dir_path = Path(episode_dir).expanduser().resolve()
    try:
        episode_relpath = str(episode_dir_path.relative_to(dataset_root_path))
    except ValueError:
        episode_relpath = episode_dir_path.name
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    return {
        "adapter_type": XPERIENCE_ADAPTER_TYPE,
        "dataset_type": "xperience",
        "adapter_version": 1,
        "dataset_root": str(dataset_root_path),
        "episode_dir": str(episode_dir_path),
        "episode_relpath": episode_relpath,
        "annotation_path": str(episode_dir_path / "annotation.hdf5"),
        "video_path": str(episode_dir_path / f"{camera_name}.mp4"),
        "camera_name": str(camera_name),
        "window_start": int(window_start),
        "window_stop": int(window_stop),
        "source_height": int(source_h),
        "source_width": int(source_w),
        "target_height": int(target_h),
        "target_width": int(target_w),
    }


def resolve_xperience_source_descriptor(descriptor: dict[str, Any]) -> XperienceEpisodePaths:
    if not isinstance(descriptor, dict):
        raise TypeError(f"source_descriptor must be a dict, got {type(descriptor).__name__}")
    if str(descriptor.get("adapter_type") or "") != XPERIENCE_ADAPTER_TYPE:
        raise ValueError(
            f"Unsupported Xperience adapter_type={descriptor.get('adapter_type')!r}; expected {XPERIENCE_ADAPTER_TYPE!r}"
        )

    episode_dir_raw = descriptor.get("episode_dir")
    if episode_dir_raw:
        episode_dir = Path(str(episode_dir_raw)).expanduser().resolve()
    else:
        dataset_root = descriptor.get("dataset_root")
        episode_relpath = descriptor.get("episode_relpath")
        if not dataset_root or not episode_relpath:
            raise ValueError("Xperience source_descriptor must contain episode_dir or dataset_root + episode_relpath")
        episode_dir = Path(str(dataset_root)).expanduser().resolve() / str(episode_relpath)

    camera_name = str(descriptor.get("camera_name") or "stereo_left")
    return open_xperience_episode(episode_dir, camera_name=camera_name)
