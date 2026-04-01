#!/usr/bin/env python3
"""
Utilities for inspecting and loading the Xperience-10M sample dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import h5py
import imageio.v3 as iio
import numpy as np


XPERIENCE_SAMPLE_DIR_ENV = "XPERIENCE_SAMPLE_DIR"
DEFAULT_XPERIENCE_SAMPLE_DIR = Path(
    os.environ.get(XPERIENCE_SAMPLE_DIR_ENV, "/DATA/disk0/shared/datasets/xperience-10m-sample")
)


def resolve_dataset_dir(dataset_dir: str | Path | None) -> Path:
    resolved = Path(dataset_dir or DEFAULT_XPERIENCE_SAMPLE_DIR).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {resolved}. "
            f"Pass --dataset-dir or set {XPERIENCE_SAMPLE_DIR_ENV}."
        )
    return resolved


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


def bytes_to_gib(size_bytes: int) -> float:
    return size_bytes / (1024 ** 3)


def collect_file_inventory(dataset_dir: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(item for item in dataset_dir.iterdir() if item.is_file()):
        size_bytes = path.stat().st_size
        files.append(
            {
                "name": path.name,
                "size_bytes": int(size_bytes),
                "size_gib": round(bytes_to_gib(size_bytes), 3),
            }
        )
    return files


def collect_hdf5_schema(handle: h5py.File) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            entries.append(
                {
                    "kind": "dataset",
                    "name": name,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }
            )
        else:
            entries.append({"kind": "group", "name": name})

    handle.visititems(visit)
    return entries


def write_schema_tsv(schema: list[dict[str, Any]], output_path: Path) -> None:
    lines = ["kind\tname\tshape\tdtype"]
    for entry in schema:
        lines.append(
            "\t".join(
                [
                    entry["kind"],
                    entry["name"],
                    json.dumps(entry.get("shape", [])),
                    entry.get("dtype", ""),
                ]
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class CaptionBound:
    raw: Any
    kind: str
    value: int | None


@dataclass(frozen=True)
class CaptionSegment:
    segment_id: int
    sub_task: str
    start: CaptionBound
    end: CaptionBound
    action_labels: tuple[str, ...]
    raw_segment: dict[str, Any]

    def contains(self, frame_index: int, timestamp_ns: int | None) -> bool:
        lower_ok = True
        upper_ok = True

        if self.start.kind == "timestamp" and self.start.value is not None and timestamp_ns is not None:
            lower_ok = timestamp_ns >= self.start.value
        elif self.start.kind == "frame_index" and self.start.value is not None:
            lower_ok = frame_index >= self.start.value

        if self.end.kind == "timestamp" and self.end.value is not None and timestamp_ns is not None:
            upper_ok = timestamp_ns <= self.end.value
        elif self.end.kind == "frame_index" and self.end.value is not None:
            upper_ok = frame_index <= self.end.value

        return lower_ok and upper_ok


@dataclass
class FrameSample:
    index: int
    frame_number: int
    timestamp_ns: int
    relative_time_sec: float
    active_segment: CaptionSegment | None
    video_frames: dict[str, np.ndarray]
    depth: np.ndarray | None
    depth_confidence: np.ndarray | None
    slam_translation: np.ndarray
    slam_quat_wxyz: np.ndarray
    full_body_keypoints: np.ndarray | None
    body_quats: np.ndarray | None
    contacts: np.ndarray | None
    left_hand_joints: np.ndarray | None
    right_hand_joints: np.ndarray | None
    imu: dict[str, np.ndarray] | None

    def summary(self) -> dict[str, Any]:
        payload = {
            "index": int(self.index),
            "frame_number": int(self.frame_number),
            "timestamp_ns": int(self.timestamp_ns),
            "relative_time_sec": round(float(self.relative_time_sec), 3),
            "active_sub_task": self.active_segment.sub_task if self.active_segment else None,
            "video_streams": sorted(self.video_frames.keys()),
            "slam_translation": np.round(self.slam_translation, 6).tolist(),
        }
        if self.depth is not None:
            valid = self.depth[self.depth > 0]
            payload["depth_valid_min_m"] = round(float(np.min(valid)), 6) if valid.size else None
            payload["depth_valid_max_m"] = round(float(np.max(valid)), 6) if valid.size else None
        if self.imu is not None:
            payload["imu_center_index"] = int(self.imu["center_index"])
            payload["imu_window_length"] = int(len(self.imu["device_timestamp_ns"]))
        return payload


def parse_caption_bound(value: Any) -> CaptionBound:
    if value is None:
        return CaptionBound(raw=None, kind="unknown", value=None)

    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return CaptionBound(raw=value, kind="timestamp", value=int(text))
        if text.startswith("frame_"):
            digits = "".join(character for character in text if character.isdigit())
            if digits:
                return CaptionBound(raw=value, kind="frame_index", value=int(digits))
        return CaptionBound(raw=value, kind="unknown", value=None)

    if isinstance(value, (int, np.integer)):
        return CaptionBound(raw=int(value), kind="timestamp", value=int(value))

    return CaptionBound(raw=value, kind="unknown", value=None)


def parse_caption_segments(raw_caption: str) -> tuple[dict[str, Any], list[CaptionSegment]]:
    caption = json.loads(raw_caption)
    segments = []
    for raw_segment in caption.get("segments", []):
        action_labels = tuple(action.get("label", "") for action in raw_segment.get("Current Action", []))
        segments.append(
            CaptionSegment(
                segment_id=int(raw_segment.get("segment_id", -1)),
                sub_task=raw_segment.get("Sub Task", ""),
                start=parse_caption_bound(raw_segment.get("start_frame")),
                end=parse_caption_bound(raw_segment.get("end_frame")),
                action_labels=action_labels,
                raw_segment=raw_segment,
            )
        )
    return caption, segments


def summarize_caption(raw_caption: str) -> dict[str, Any]:
    caption, segments = parse_caption_segments(raw_caption)
    return {
        "main_task": caption.get("config", {}).get("Main Task"),
        "total_frames_config": caption.get("config", {}).get("total_frames"),
        "total_tokens": caption.get("config", {}).get("total_tokens"),
        "segment_count": len(segments),
        "subtasks": [segment.sub_task for segment in segments],
    }


def summarize_health_report(raw_health_report: str) -> dict[str, Any] | None:
    if not raw_health_report:
        return None
    report = json.loads(raw_health_report)
    recordings = report.get("recordings", [])
    if not recordings:
        return None
    recording = recordings[0]
    mjpeg = recording.get("mjpeg_corruption", {})
    frame_drops = recording.get("frame_drops", {})
    return {
        "generated_at": report.get("generated_at"),
        "root": report.get("root"),
        "raw_recording_duration_sec": mjpeg.get("duration_sec"),
        "raw_recording_total_frames": mjpeg.get("total_frames"),
        "cam_expected_fps": frame_drops.get("cam", {}).get("expected_fps"),
        "imu_expected_hz": frame_drops.get("imu", {}).get("expected_fps"),
        "file_integrity_complete": recording.get("file_integrity", {}).get("complete"),
    }


def collect_video_metadata(dataset_dir: Path) -> list[dict[str, Any]]:
    videos = []
    for path in sorted(dataset_dir.glob("*.mp4")):
        meta = iio.immeta(path)
        resolution = None
        if "size" in meta:
            width, height = meta["size"]
            resolution = [int(height), int(width)]
        videos.append(
            {
                "name": path.name,
                "size_bytes": int(path.stat().st_size),
                "size_gib": round(bytes_to_gib(path.stat().st_size), 3),
                "fps": meta.get("fps"),
                "duration_sec": meta.get("duration"),
                "codec": meta.get("codec"),
                "resolution": resolution,
            }
        )
    return videos


def summarize_dataset_dir(dataset_dir: str | Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_path = resolve_dataset_dir(dataset_dir)
    annotation_path = dataset_path / "annotation.hdf5"
    with h5py.File(annotation_path, "r") as handle:
        schema = collect_hdf5_schema(handle)
        metadata = {key: decode_scalar(value[()]) for key, value in handle["metadata"].items()}
        caption_summary = summarize_caption(decode_scalar(handle["caption"][()]))
        health_report_summary = summarize_health_report(metadata.get("health_report", ""))

        frame_count = int(handle["video/frame_number"].shape[0])
        video_length_sec = float(handle["video/length_sec"][()])
        imu_count = int(handle["imu/accel_xyz"].shape[0])
        depth = handle["depth/depth"]
        point_cloud = handle["slam/point_cloud"][:]
        slam_translations = handle["slam/trans_xyz"][:]

        summary = {
            "dataset_dir": str(dataset_path),
            "root_groups": list(handle.keys()),
            "frame_count": frame_count,
            "frame_range": [int(handle["video/frame_number"][0]), int(handle["video/frame_number"][-1])],
            "video_length_sec": round(video_length_sec, 3),
            "effective_video_fps": round(frame_count / video_length_sec, 3),
            "imu_count": imu_count,
            "effective_imu_hz": round(imu_count / video_length_sec, 3),
            "caption": caption_summary,
            "metadata": {
                "body_height_m": metadata.get("body_height"),
                "device_id": metadata.get("device_id"),
                "device_version": metadata.get("device_version"),
                "time_created": metadata.get("time_created"),
            },
            "health_report": health_report_summary,
            "depth": {
                "shape": list(depth.shape),
                "confidence_shape": list(handle["depth/confidence"].shape),
                "depth_min": round(float(handle["depth/depth_min"][()]), 6),
                "depth_max": round(float(handle["depth/depth_max"][()]), 6),
                "depth_scale": float(handle["depth/scale"][()]),
            },
            "slam": {
                "trajectory_shape": list(handle["slam/trans_xyz"].shape),
                "point_cloud_shape": list(point_cloud.shape),
                "trajectory_min_xyz": np.min(slam_translations, axis=0).round(6).tolist(),
                "trajectory_max_xyz": np.max(slam_translations, axis=0).round(6).tolist(),
            },
            "full_body_mocap": {
                "keypoints_shape": list(handle["full_body_mocap/keypoints"].shape),
                "body_quats_shape": list(handle["full_body_mocap/body_quats"].shape),
                "contacts_shape": list(handle["full_body_mocap/contacts"].shape),
            },
            "hand_mocap": {
                "left_joints_shape": list(handle["hand_mocap/left_joints_3d"].shape),
                "right_joints_shape": list(handle["hand_mocap/right_joints_3d"].shape),
            },
        }

    summary["files"] = collect_file_inventory(dataset_path)
    summary["videos"] = collect_video_metadata(dataset_path)
    summary["dataset_size_gib"] = round(sum(item["size_bytes"] for item in summary["files"]) / (1024 ** 3), 3)
    return summary, schema


class XperienceSampleDataset:
    def __init__(self, dataset_dir: str | Path | None = None) -> None:
        self.dataset_dir = resolve_dataset_dir(dataset_dir)
        self.annotation_path = self.dataset_dir / "annotation.hdf5"
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {self.annotation_path}")

        self.video_paths = {path.stem: path for path in sorted(self.dataset_dir.glob("*.mp4"))}
        if not self.video_paths:
            raise FileNotFoundError(f"No MP4 files found under {self.dataset_dir}")

        self._h5 = h5py.File(self.annotation_path, "r")
        self.frame_numbers = self._h5["video/frame_number"][:].astype(np.int64)
        self.device_timestamps_ns = np.asarray(
            [int(decode_scalar(value)) for value in self._h5["video/device_timestamp"][:]],
            dtype=np.int64,
        )
        self.video_length_sec = float(self._h5["video/length_sec"][()])
        self.imu_keyframe_indices = self._h5["imu/keyframe_indices"][:].astype(np.int64)
        self.imu_count = int(self._h5["imu/accel_xyz"].shape[0])
        self.metadata = {key: decode_scalar(value[()]) for key, value in self._h5["metadata"].items()}
        self.raw_caption = decode_scalar(self._h5["caption"][()])
        self.caption_config, self.caption_segments = parse_caption_segments(self.raw_caption)
        self.slam_point_cloud = self._h5["slam/point_cloud"][:]
        self.slam_translations = self._h5["slam/trans_xyz"][:]
        self.slam_quats_wxyz = self._h5["slam/quat_wxyz"][:]

        first_stream = next(iter(self.video_paths))
        first_meta = iio.immeta(self.video_paths[first_stream])
        self.video_fps = float(first_meta.get("fps", len(self.frame_numbers) / self.video_length_sec))
        self.stream_names = tuple(sorted(self.video_paths))

    def __len__(self) -> int:
        return int(self.frame_numbers.shape[0])

    def close(self) -> None:
        if getattr(self, "_h5", None) is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> "XperienceSampleDataset":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def validate_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Frame index out of range: {index}")
        return int(index)

    def frame_timestamp_ns(self, index: int) -> int:
        return int(self.device_timestamps_ns[self.validate_index(index)])

    def frame_time_sec(self, index: int) -> float:
        normalized = self.validate_index(index)
        return float(self.device_timestamps_ns[normalized] - self.device_timestamps_ns[0]) / 1e9

    def active_segment_for_index(self, index: int) -> CaptionSegment | None:
        normalized = self.validate_index(index)
        timestamp_ns = self.frame_timestamp_ns(normalized)
        for segment in self.caption_segments:
            if segment.contains(normalized, timestamp_ns):
                return segment
        return None

    def get_video_frame(self, index: int, stream: str) -> np.ndarray:
        normalized = self.validate_index(index)
        if stream not in self.video_paths:
            raise KeyError(f"Unknown video stream: {stream}")
        return iio.imread(self.video_paths[stream], index=normalized)

    def get_imu_window(self, index: int, radius: int = 10) -> dict[str, np.ndarray]:
        normalized = self.validate_index(index)
        center = int(self.imu_keyframe_indices[normalized])
        start = max(center - radius, 0)
        stop = min(center + radius + 1, self.imu_count)
        return {
            "center_index": np.array(center, dtype=np.int64),
            "slice_bounds": np.array([start, stop], dtype=np.int64),
            "device_timestamp_ns": self._h5["imu/device_timestamp_ns"][start:stop],
            "accel_xyz": self._h5["imu/accel_xyz"][start:stop],
            "gyro_xyz": self._h5["imu/gyro_xyz"][start:stop],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "frame_count": len(self),
            "video_length_sec": round(self.video_length_sec, 3),
            "video_fps": self.video_fps,
            "imu_count": self.imu_count,
            "streams": list(self.stream_names),
            "caption_main_task": self.caption_config.get("config", {}).get("Main Task"),
            "caption_segment_count": len(self.caption_segments),
        }

    def get_frame(
        self,
        index: int,
        video_streams: Sequence[str] | None = None,
        load_video: bool = True,
        load_depth: bool = True,
        load_mocap: bool = True,
        load_imu: bool = True,
        imu_radius: int = 10,
    ) -> FrameSample:
        normalized = self.validate_index(index)
        selected_streams = tuple(video_streams or ("stereo_left",))
        video_frames: dict[str, np.ndarray] = {}
        if load_video:
            for stream in selected_streams:
                video_frames[stream] = self.get_video_frame(normalized, stream)

        return FrameSample(
            index=normalized,
            frame_number=int(self.frame_numbers[normalized]),
            timestamp_ns=int(self.device_timestamps_ns[normalized]),
            relative_time_sec=self.frame_time_sec(normalized),
            active_segment=self.active_segment_for_index(normalized),
            video_frames=video_frames,
            depth=self._h5["depth/depth"][normalized] if load_depth else None,
            depth_confidence=self._h5["depth/confidence"][normalized] if load_depth else None,
            slam_translation=self.slam_translations[normalized],
            slam_quat_wxyz=self.slam_quats_wxyz[normalized],
            full_body_keypoints=self._h5["full_body_mocap/keypoints"][normalized] if load_mocap else None,
            body_quats=self._h5["full_body_mocap/body_quats"][normalized] if load_mocap else None,
            contacts=self._h5["full_body_mocap/contacts"][normalized] if load_mocap else None,
            left_hand_joints=self._h5["hand_mocap/left_joints_3d"][normalized] if load_mocap else None,
            right_hand_joints=self._h5["hand_mocap/right_joints_3d"][normalized] if load_mocap else None,
            imu=self.get_imu_window(normalized, radius=imu_radius) if load_imu else None,
        )

    def iter_samples(
        self,
        *,
        step: int = 1,
        start: int = 0,
        stop: int | None = None,
        **kwargs: Any,
    ) -> Iterator[FrameSample]:
        upper = len(self) if stop is None else min(stop, len(self))
        for index in range(start, upper, step):
            yield self.get_frame(index, **kwargs)
