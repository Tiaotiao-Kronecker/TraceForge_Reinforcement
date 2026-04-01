import unittest
from dataclasses import dataclass

import numpy as np

from scripts.data_analysis.xperience_sample_utils import FrameSample
from scripts.visualization.xperience_sample_visualization_utils import (
    build_storyboard,
    colorize_confidence,
    colorize_depth,
    render_frame_dashboard,
)


@dataclass
class _FakeSegment:
    sub_task: str
    action_labels: tuple[str, ...]


class _FakeDataset:
    def __init__(self) -> None:
        self.caption_config = {"config": {"Main Task": "Make coffee"}}
        self.slam_point_cloud = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 0.5],
            ],
            dtype=np.float32,
        )
        self.slam_translations = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.samples = {
            index: FrameSample(
                index=index,
                frame_number=index,
                timestamp_ns=1000 + index,
                relative_time_sec=index * 0.1,
                active_segment=_FakeSegment("segment", ("lift",)),
                video_frames={
                    "stereo_left": np.full((64, 64, 3), 10 + index, dtype=np.uint8),
                    "stereo_right": np.full((64, 64, 3), 20 + index, dtype=np.uint8),
                    "fisheye_cam1": np.full((64, 64, 3), 30 + index, dtype=np.uint8),
                },
                depth=np.full((16, 16), 0.5, dtype=np.float32),
                depth_confidence=np.full((16, 16), 40, dtype=np.uint8),
                slam_translation=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                slam_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                full_body_keypoints=np.zeros((52, 3), dtype=np.float32),
                body_quats=np.zeros((21, 4), dtype=np.float32),
                contacts=np.zeros((21,), dtype=np.float32),
                left_hand_joints=np.zeros((21, 3), dtype=np.float32),
                right_hand_joints=np.ones((21, 3), dtype=np.float32),
                imu={
                    "center_index": np.array(10, dtype=np.int64),
                    "slice_bounds": np.array([8, 13], dtype=np.int64),
                    "device_timestamp_ns": np.arange(5, dtype=np.int64),
                    "accel_xyz": np.ones((5, 3), dtype=np.float32),
                    "gyro_xyz": np.full((5, 3), 2.0, dtype=np.float32),
                },
            )
            for index in range(3)
        }

    def __len__(self) -> int:
        return 3

    def get_frame(self, index: int, **_: object) -> FrameSample:
        return self.samples[index]


class XperienceVisualizationUtilsTests(unittest.TestCase):
    def test_colorize_helpers_produce_expected_sizes(self) -> None:
        depth = np.full((8, 8), 0.5, dtype=np.float32)
        confidence = np.full((8, 8), 40, dtype=np.uint8)
        self.assertEqual(colorize_depth(depth).size, (320, 320))
        self.assertEqual(colorize_confidence(confidence).size, (320, 320))

    def test_render_frame_dashboard_returns_expected_canvas_size(self) -> None:
        dataset = _FakeDataset()
        dashboard = render_frame_dashboard(dataset, dataset.samples[1])
        self.assertEqual(dashboard.size, (1600, 980))

    def test_build_storyboard_returns_grid_image(self) -> None:
        dataset = _FakeDataset()
        storyboard = build_storyboard(dataset, [0, 1, 2], main_camera="stereo_left")
        self.assertGreater(storyboard.size[0], 1000)
        self.assertGreater(storyboard.size[1], 200)


if __name__ == "__main__":
    unittest.main()
