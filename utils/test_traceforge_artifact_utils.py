import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from utils.traceforge_artifact_utils import (
    SCENE_STORAGE_SOURCE_REF,
    SceneReader,
    build_pointcloud_from_frame,
    build_sample_visualization_view,
    detect_output_layout,
    is_traceforge_output_complete,
    normalize_sample_data,
    write_scene_meta,
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


class NormalizeSampleDataTests(unittest.TestCase):
    def test_reads_query_depth_edge_debug_fields_from_v2_sample(self):
        traj_uvz = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = Path(tmpdir) / "sample.npz"
            np.savez(
                sample_path,
                traj_uvz=traj_uvz,
                keypoints=np.array([[1.0, 1.0]], dtype=np.float32),
                query_frame_index=np.array([0], dtype=np.int32),
                segment_frame_indices=np.array([0], dtype=np.int32),
                traj_query_depth_edge_mask=np.array([True], dtype=bool),
                traj_query_depth_patch_valid_ratio=np.array([1.0], dtype=np.float16),
                traj_query_depth_patch_std=np.array([0.01], dtype=np.float16),
                traj_query_depth_edge_risk_mask=np.array([True], dtype=bool),
            )

            sample = normalize_sample_data(sample_path)

        np.testing.assert_array_equal(sample["traj_query_depth_edge_mask"], np.array([True]))
        np.testing.assert_array_equal(sample["traj_query_depth_edge_risk_mask"], np.array([True]))
        np.testing.assert_array_equal(
            sample["traj_query_depth_patch_valid_ratio"],
            np.array([1.0], dtype=np.float16),
        )
        np.testing.assert_array_equal(
            sample["traj_query_depth_patch_std"],
            np.array([0.01], dtype=np.float16),
        )


def _write_rgb_png(path: Path, value: int, *, hw: tuple[int, int] = (2, 3)) -> None:
    image = np.full((*hw, 3), value, dtype=np.uint8)
    Image.fromarray(image).save(path)


class SourceRefArtifactTests(unittest.TestCase):
    def _build_source_ref_episode(self, tmpdir: str) -> tuple[Path, np.ndarray, np.ndarray]:
        root = Path(tmpdir)
        episode_dir = root / "camera"
        samples_dir = episode_dir / "samples"
        rgb_dir = root / "rgb"
        depth_dir = root / "depth"
        geom_path = root / "geom.npz"

        samples_dir.mkdir(parents=True, exist_ok=True)
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        for idx, value in enumerate((10, 20, 30)):
            _write_rgb_png(rgb_dir / f"{idx:05d}.png", value)
            np.save(depth_dir / f"{idx:05d}.npy", np.full((2, 3), float(idx + 1), dtype=np.float32))

        intrinsics = np.stack(
            [
                np.array([[1.0 + idx, 0.0, 2.0], [0.0, 1.5 + idx, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                for idx in range(3)
            ],
            axis=0,
        )
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
        extrinsics[:, 0, 3] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        np.savez(geom_path, intrinsics=intrinsics, extrinsics=extrinsics)

        np.savez(
            samples_dir / "camera_0.npz",
            traj_uvz=np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32),
            keypoints=np.array([[1.0, 1.0]], dtype=np.float32),
            query_frame_index=np.array([0], dtype=np.int32),
            segment_frame_indices=np.array([0], dtype=np.int32),
        )
        write_scene_meta(
            episode_dir / "scene_meta.json",
            {
                "layout_version": 2,
                "video_name": "camera",
                "frame_count": 2,
                "height": 2,
                "width": 3,
                "extrinsics_mode": "w2c",
                "frame_drop_rate": 1,
                "future_len": 16,
                "original_filenames": ["00002", "00000"],
                "scene_storage_mode": SCENE_STORAGE_SOURCE_REF,
                "scene_h5_path": None,
                "rgb_cache_path": None,
                "source_rgb_path": str(rgb_dir),
                "source_depth_path": str(depth_dir),
                "source_geom_path": str(geom_path),
                "source_camera_name": "camera",
                "source_extrinsics_mode": "w2c",
                "depth_pose_method": "external",
                "source_frame_indices": [2, 0],
            },
        )
        return episode_dir, intrinsics, extrinsics

    def test_detect_output_layout_accepts_source_ref_v2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir, _, _ = self._build_source_ref_episode(tmpdir)
            self.assertEqual(detect_output_layout(episode_dir), "v2")
            self.assertTrue(is_traceforge_output_complete(episode_dir))

    def test_source_ref_complete_requires_existing_source_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir, _, _ = self._build_source_ref_episode(tmpdir)
            geom_path = Path(tmpdir) / "geom.npz"
            geom_path.unlink()
            self.assertFalse(is_traceforge_output_complete(episode_dir))

    def test_scene_reader_reads_source_ref_rgb_depth_and_geometry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir, intrinsics, extrinsics = self._build_source_ref_episode(tmpdir)
            with SceneReader(episode_dir) as reader:
                intrinsics_sel, extrinsics_sel = reader.get_camera_arrays()
                rgb0 = reader.get_rgb_frame(0)
                rgb1 = reader.get_rgb_frame(1)
                depth0 = reader.get_depth_frame(0)
                depth1 = reader.get_depth_frame(1)

            np.testing.assert_allclose(intrinsics_sel[0], intrinsics[2])
            np.testing.assert_allclose(intrinsics_sel[1], intrinsics[0])
            np.testing.assert_allclose(extrinsics_sel[0], extrinsics[2])
            np.testing.assert_allclose(extrinsics_sel[1], extrinsics[0])
            self.assertEqual(int(rgb0[0, 0, 0]), 30)
            self.assertEqual(int(rgb1[0, 0, 0]), 10)
            np.testing.assert_allclose(depth0, np.full((2, 3), 3.0, dtype=np.float32))
            np.testing.assert_allclose(depth1, np.full((2, 3), 1.0, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
