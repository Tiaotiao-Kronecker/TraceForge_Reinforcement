import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

from scripts.data_analysis.xperience_sample_utils import (
    XperienceSampleDataset,
    parse_caption_bound,
    summarize_dataset_dir,
)


def _write_minimal_dataset(root: Path) -> Path:
    dataset_dir = root / "xperience_sample"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name in ("stereo_left.mp4", "stereo_right.mp4", "fisheye_cam1.mp4"):
        (dataset_dir / name).write_bytes(b"placeholder")

    caption = {
        "config": {
            "Main Task": "Make coffee",
            "total_frames": 4,
            "total_tokens": 123,
        },
        "segments": [
            {
                "segment_id": 0,
                "start_frame": "1000",
                "end_frame": "2500",
                "Sub Task": "Prepare kettle",
                "Current Action": [{"label": "lift"}],
            },
            {
                "segment_id": 1,
                "start_frame": "frame_0000003",
                "end_frame": "frame_0000003",
                "Sub Task": "Finish",
                "Current Action": [{"label": "pour"}],
            },
        ],
    }

    with h5py.File(dataset_dir / "annotation.hdf5", "w") as handle:
        calibration = handle.create_group("calibration")
        cam0 = calibration.create_group("cam0")
        cam0.create_dataset("K", data=np.array([1.0, 2.0, 3.0, 4.0]))
        cam0.create_dataset("D", data=np.array([0.1, 0.2, 0.3, 0.4]))

        video = handle.create_group("video")
        video.create_dataset("frame_number", data=np.arange(4, dtype=np.int64))
        video.create_dataset(
            "device_timestamp",
            data=np.asarray([b"1000", b"2000", b"3000", b"4000"]),
        )
        video.create_dataset("length_sec", data=np.float64(0.2))

        depth = handle.create_group("depth")
        depth.create_dataset("depth", data=np.full((4, 2, 2), 0.5, dtype=np.float32))
        depth.create_dataset("confidence", data=np.full((4, 2, 2), 40, dtype=np.uint8))
        depth.create_dataset("depth_min", data=np.float64(0.1))
        depth.create_dataset("depth_max", data=np.float64(2.0))
        depth.create_dataset("scale", data=np.float64(0.5))

        slam = handle.create_group("slam")
        slam.create_dataset("trans_xyz", data=np.arange(12, dtype=np.float32).reshape(4, 3))
        slam.create_dataset("quat_wxyz", data=np.full((4, 4), 1.0, dtype=np.float32))
        slam.create_dataset("point_cloud", data=np.arange(15, dtype=np.float32).reshape(5, 3))

        full_body = handle.create_group("full_body_mocap")
        full_body.create_dataset("keypoints", data=np.zeros((4, 52, 3), dtype=np.float32))
        full_body.create_dataset("body_quats", data=np.zeros((4, 21, 4), dtype=np.float32))
        full_body.create_dataset("contacts", data=np.zeros((4, 21), dtype=np.float32))

        hands = handle.create_group("hand_mocap")
        hands.create_dataset("left_joints_3d", data=np.zeros((4, 21, 3), dtype=np.float32))
        hands.create_dataset("right_joints_3d", data=np.ones((4, 21, 3), dtype=np.float32))

        imu = handle.create_group("imu")
        imu.create_dataset("accel_xyz", data=np.ones((40, 3), dtype=np.float32))
        imu.create_dataset("gyro_xyz", data=np.full((40, 3), 2.0, dtype=np.float32))
        imu.create_dataset("device_timestamp_ns", data=np.arange(40, dtype=np.int64))
        imu.create_dataset("keyframe_indices", data=np.array([0, 10, 20, 30], dtype=np.int64))

        metadata = handle.create_group("metadata")
        metadata.create_dataset("body_height", data=np.float64(1.78))
        metadata.create_dataset("device_id", data=np.bytes_("device-1"))
        metadata.create_dataset("device_version", data=np.bytes_("v2"))
        metadata.create_dataset("health_report", data=np.bytes_(json.dumps({"recordings": []})))
        metadata.create_dataset("other", data=np.bytes_("{}"))
        metadata.create_dataset("time_created", data=np.bytes_("2026-03-09T18:06:14Z"))

        handle.create_dataset("caption", data=np.bytes_(json.dumps(caption)))

    return dataset_dir


def _mock_video_meta(_: Path) -> dict[str, object]:
    return {
        "fps": 20.0,
        "duration": 0.2,
        "codec": "h264",
        "size": (4, 4),
    }


def _mock_video_frame(_: Path, *, index: int) -> np.ndarray:
    return np.full((4, 4, 3), index, dtype=np.uint8)


class CaptionParsingTests(unittest.TestCase):
    def test_parse_caption_bound_supports_timestamp_and_frame_index(self) -> None:
        timestamp = parse_caption_bound("12345")
        frame_index = parse_caption_bound("frame_0000015")
        self.assertEqual(timestamp.kind, "timestamp")
        self.assertEqual(timestamp.value, 12345)
        self.assertEqual(frame_index.kind, "frame_index")
        self.assertEqual(frame_index.value, 15)


class XperienceSampleDatasetTests(unittest.TestCase):
    @mock.patch("scripts.data_analysis.xperience_sample_utils.iio.imread", side_effect=_mock_video_frame)
    @mock.patch("scripts.data_analysis.xperience_sample_utils.iio.immeta", side_effect=_mock_video_meta)
    def test_dataset_aligns_modalities_on_frame_index(self, _meta: mock.Mock, _read: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = _write_minimal_dataset(Path(tmpdir))
            with XperienceSampleDataset(dataset_dir) as dataset:
                self.assertEqual(dataset.summary()["caption_main_task"], "Make coffee")
                sample = dataset.get_frame(
                    1,
                    video_streams=("stereo_left", "stereo_right"),
                    load_video=True,
                    load_depth=True,
                    load_mocap=True,
                    load_imu=True,
                    imu_radius=1,
                )

                self.assertEqual(sample.active_segment.sub_task, "Prepare kettle")
                self.assertEqual(sample.video_frames["stereo_left"].shape, (4, 4, 3))
                self.assertEqual(sample.depth.shape, (2, 2))
                self.assertEqual(sample.left_hand_joints.shape, (21, 3))
                self.assertEqual(sample.imu["accel_xyz"].shape, (3, 3))
                self.assertEqual(sample.summary()["imu_center_index"], 10)

    @mock.patch("scripts.data_analysis.xperience_sample_utils.iio.immeta", side_effect=_mock_video_meta)
    def test_summarize_dataset_dir_reports_expected_counts(self, _meta: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = _write_minimal_dataset(Path(tmpdir))
            summary, schema = summarize_dataset_dir(dataset_dir)
            self.assertEqual(summary["frame_count"], 4)
            self.assertEqual(summary["caption"]["segment_count"], 2)
            self.assertEqual(summary["videos"][0]["fps"], 20.0)
            self.assertTrue(any(entry["name"] == "video/frame_number" for entry in schema if entry["kind"] == "dataset"))


if __name__ == "__main__":
    unittest.main()
