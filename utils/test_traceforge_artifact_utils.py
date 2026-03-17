import sys
import types
import unittest

import numpy as np

from utils.traceforge_artifact_utils import (
    build_pointcloud_from_frame,
    build_sample_visualization_view,
)


def _camera_z_to_world_x_transform() -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return c2w


class BuildPointcloudFromFrameTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)
        self.intrinsics = np.eye(3, dtype=np.float32)
        self._original_torch = sys.modules.get("torch")
        if self._original_torch is None:
            sys.modules["torch"] = types.SimpleNamespace()

    def tearDown(self):
        if self._original_torch is None:
            sys.modules.pop("torch", None)

    def test_keeps_point_with_valid_camera_depth_even_if_world_z_is_out_of_range(self):
        depth = np.array([[1.0]], dtype=np.float32)
        c2w = _camera_z_to_world_x_transform()
        w2c = np.linalg.inv(c2w).astype(np.float32)

        points, colors = build_pointcloud_from_frame(
            depth=depth,
            rgb=self.rgb,
            intrinsics=self.intrinsics,
            w2c=w2c,
            downsample=1,
            depth_min=0.5,
            depth_max=1.5,
        )

        self.assertEqual(points.shape, (1, 3))
        self.assertAlmostEqual(float(points[0, 2]), 0.0, places=6)
        self.assertEqual(colors.shape, (1, 3))

    def test_filters_point_with_invalid_camera_depth_even_if_world_z_is_in_range(self):
        depth = np.array([[0.2]], dtype=np.float32)
        c2w = _camera_z_to_world_x_transform()
        c2w[:3, 3] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        w2c = np.linalg.inv(c2w).astype(np.float32)

        points, colors = build_pointcloud_from_frame(
            depth=depth,
            rgb=self.rgb,
            intrinsics=self.intrinsics,
            w2c=w2c,
            downsample=1,
            depth_min=0.5,
            depth_max=1.5,
        )

        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(colors.shape, (0, 3))

    def test_filters_depth_values_on_open_interval_boundaries(self):
        depth = np.array([[0.5, 1.0, 1.5]], dtype=np.float32)
        rgb = np.tile(self.rgb, (1, 3, 1))
        w2c = np.eye(4, dtype=np.float32)

        points, colors = build_pointcloud_from_frame(
            depth=depth,
            rgb=rgb,
            intrinsics=self.intrinsics,
            w2c=w2c,
            downsample=1,
            depth_min=0.5,
            depth_max=1.5,
        )

        self.assertEqual(points.shape, (1, 3))
        self.assertEqual(colors.shape, (1, 3))
        self.assertAlmostEqual(float(points[0, 2]), 1.0, places=6)


class BuildSampleVisualizationViewTests(unittest.TestCase):
    def test_v2_supervision_mask_hides_wrist_tail_after_track_filtering(self):
        traj_uvz = np.array(
            [
                [[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [3.0, 3.0, 1.0], [4.0, 4.0, 1.0]],
                [[10.0, 1.0, 1.0], [11.0, 2.0, 1.0], [12.0, 3.0, 1.0], [13.0, 4.0, 1.0]],
                [[20.0, 1.0, 1.0], [21.0, 2.0, 1.0], [22.0, 3.0, 1.0], [23.0, 4.0, 1.0]],
            ],
            dtype=np.float32,
        )
        sample = {
            "layout": "v2",
            "traj_uvz": traj_uvz,
            "traj_2d": traj_uvz[..., :2].copy(),
            "keypoints": np.array([[1.0, 1.0], [10.0, 1.0], [20.0, 1.0]], dtype=np.float32),
            "segment_frame_indices": np.array([0, 1, 2], dtype=np.int32),
            "traj_valid_mask": np.array([True, False, True]),
            "traj_supervision_mask": np.array(
                [
                    [True, True, False, False],
                    [True, False, False, False],
                    [True, False, True, True],
                ],
                dtype=bool,
            ),
            "frame_aligned": True,
        }

        view = build_sample_visualization_view(sample)

        np.testing.assert_array_equal(view["segment_frame_indices"], np.array([0, 1, 2], dtype=np.int32))
        np.testing.assert_array_equal(view["keypoints"], np.array([[1.0, 1.0], [20.0, 1.0]], dtype=np.float32))
        np.testing.assert_array_equal(
            view["render_step_mask"],
            np.array([[True, True, False], [True, False, True]], dtype=bool),
        )
        self.assertTrue(np.isnan(view["traj_uvz"][0, 2]).all())
        self.assertTrue(np.isnan(view["traj_uvz"][1, 1]).all())
        np.testing.assert_array_equal(view["rendered_frame_count"], np.array([2, 2], dtype=np.uint16))
        self.assertEqual(view["raw_num_tracks"], 3)
        self.assertEqual(view["kept_num_tracks"], 2)

    def test_legacy_valid_steps_are_broadcast_when_supervision_mask_is_missing(self):
        traj_uvz = np.array(
            [
                [[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [3.0, 3.0, 1.0], [4.0, 4.0, 1.0]],
                [[5.0, 5.0, 1.0], [6.0, 6.0, 1.0], [7.0, 7.0, 1.0], [8.0, 8.0, 1.0]],
            ],
            dtype=np.float32,
        )
        sample = {
            "layout": "legacy",
            "traj_uvz": traj_uvz,
            "traj_2d": traj_uvz[..., :2].copy(),
            "keypoints": np.array([[1.0, 1.0], [5.0, 5.0]], dtype=np.float32),
            "segment_frame_indices": np.array([7, 8, 9, 10], dtype=np.int32),
            "traj_valid_mask": np.array([True, False]),
            "valid_steps": np.array([True, True, False, False]),
            "frame_aligned": False,
        }

        view = build_sample_visualization_view(sample)

        np.testing.assert_array_equal(
            view["render_step_mask"],
            np.array([[True, True, False, False]], dtype=bool),
        )
        self.assertTrue(np.isnan(view["traj_2d"][0, 2]).all())
        self.assertTrue(np.isnan(view["traj_uvz"][0, 3]).all())
        np.testing.assert_array_equal(view["rendered_frame_count"], np.array([2], dtype=np.uint16))
        self.assertEqual(view["raw_num_tracks"], 2)
        self.assertEqual(view["kept_num_tracks"], 1)

    def test_falls_back_to_finite_steps_when_no_supervision_or_valid_steps_exist(self):
        traj_uvz = np.array(
            [[[1.0, 1.0, 1.0], [np.nan, np.nan, np.nan], [3.0, 3.0, 1.0]]],
            dtype=np.float32,
        )
        sample = {
            "layout": "v2",
            "traj_uvz": traj_uvz,
            "traj_2d": traj_uvz[..., :2].copy(),
            "keypoints": np.array([[1.0, 1.0]], dtype=np.float32),
            "segment_frame_indices": np.array([0, 1, 2], dtype=np.int32),
            "traj_valid_mask": np.array([True]),
            "frame_aligned": True,
        }

        view = build_sample_visualization_view(sample)

        np.testing.assert_array_equal(
            view["render_step_mask"],
            np.array([[True, False, True]], dtype=bool),
        )
        self.assertTrue(np.isnan(view["traj_uvz"][0, 1]).all())
        np.testing.assert_array_equal(view["rendered_frame_count"], np.array([2], dtype=np.uint16))


if __name__ == "__main__":
    unittest.main()
