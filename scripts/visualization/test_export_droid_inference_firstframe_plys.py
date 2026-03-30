import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from scripts.visualization.export_droid_inference_firstframe_plys import load_camera_bundle
from utils.traceforge_artifact_utils import write_scene_meta


def _write_rgb(path: Path, value: int) -> None:
    rgb = np.full((2, 3, 3), value, dtype=np.uint8)
    Image.fromarray(rgb).save(path)


def _write_depth(path: Path, value: float) -> None:
    np.save(path, np.full((2, 3), value, dtype=np.float32))


class ExportDroidInferenceFirstframePlysTests(unittest.TestCase):
    def _build_source_ref_episode(self, root: Path, *, geom_kind: str, extr_mode: str) -> tuple[Path, np.ndarray, np.ndarray]:
        episode_dir = root / "hand_camera"
        rgb_dir = root / "rgb"
        depth_dir = root / "depth"
        samples_dir = episode_dir / "samples"
        episode_dir.mkdir()
        rgb_dir.mkdir()
        depth_dir.mkdir()
        samples_dir.mkdir()

        _write_rgb(rgb_dir / "00000.png", 10)
        _write_rgb(rgb_dir / "00001.png", 30)
        _write_depth(depth_dir / "00000.npy", 1.0)
        _write_depth(depth_dir / "00001.npy", 3.0)

        intrinsics = np.stack(
            [
                np.array([[1.0, 0.0, 0.5], [0.0, 1.5, 0.25], [0.0, 0.0, 1.0]], dtype=np.float32),
                np.array([[2.0, 0.0, 1.5], [0.0, 2.5, 1.25], [0.0, 0.0, 1.0]], dtype=np.float32),
            ],
            axis=0,
        )
        extrinsics = np.stack(
            [
                np.eye(4, dtype=np.float32),
                np.array(
                    [
                        [1.0, 0.0, 0.0, 0.5],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.5],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            ],
            axis=0,
        )

        if geom_kind == "npz":
            geom_path = root / "geom.npz"
            np.savez(geom_path, intrinsics=intrinsics, extrinsics=extrinsics)
        elif geom_kind == "h5":
            geom_path = root / "trajectory_valid.h5"
            with h5py.File(geom_path, "w") as f:
                f.create_dataset("observation/camera/intrinsics/hand_camera_left", data=intrinsics)
                f.create_dataset("observation/camera/extrinsics/hand_camera_left", data=extrinsics)
        else:
            raise ValueError(f"Unsupported geom_kind: {geom_kind}")

        np.savez(
            samples_dir / "hand_camera_0.npz",
            traj_uvz=np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32),
            keypoints=np.array([[1.0, 1.0]], dtype=np.float32),
            query_frame_index=np.array([0], dtype=np.int32),
            segment_frame_indices=np.array([0], dtype=np.int32),
        )

        write_scene_meta(
            episode_dir / "scene_meta.json",
            {
                "layout_version": 2,
                "video_name": "hand_camera",
                "frame_count": 1,
                "height": 2,
                "width": 3,
                "extrinsics_mode": "w2c",
                "frame_drop_rate": 1,
                "future_len": 16,
                "original_filenames": ["00001"],
                "scene_storage_mode": "source_ref",
                "scene_h5_path": None,
                "rgb_cache_path": None,
                "source_rgb_path": str(rgb_dir),
                "source_depth_path": str(depth_dir),
                "source_geom_path": str(geom_path),
                "source_camera_name": "hand_camera",
                "source_extrinsics_mode": extr_mode,
                "depth_pose_method": "external",
                "source_frame_indices": [1],
            },
        )
        return episode_dir, intrinsics, extrinsics

    def test_load_camera_bundle_reads_npz_source_ref_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir, intrinsics, extrinsics = self._build_source_ref_episode(
                Path(tmpdir),
                geom_kind="npz",
                extr_mode="c2w",
            )

            bundle = load_camera_bundle(episode_dir, 0)

            self.assertEqual(bundle["stored_extrinsics_mode"], "c2w")
            self.assertEqual(bundle["source_frame_index"], 1)
            self.assertEqual(int(bundle["rgb"][0, 0, 0]), 30)
            np.testing.assert_allclose(bundle["depth"], np.full((2, 3), 3.0, dtype=np.float32))
            np.testing.assert_allclose(bundle["intrinsics"], intrinsics[1])
            np.testing.assert_allclose(bundle["extrinsics"], extrinsics[1])

    def test_load_camera_bundle_reads_h5_source_ref_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir, intrinsics, extrinsics = self._build_source_ref_episode(
                Path(tmpdir),
                geom_kind="h5",
                extr_mode="w2c",
            )

            bundle = load_camera_bundle(episode_dir, 0)

            self.assertEqual(bundle["stored_extrinsics_mode"], "w2c")
            self.assertEqual(bundle["source_frame_index"], 1)
            self.assertEqual(int(bundle["rgb"][0, 0, 0]), 30)
            np.testing.assert_allclose(bundle["depth"], np.full((2, 3), 3.0, dtype=np.float32))
            np.testing.assert_allclose(bundle["intrinsics"], intrinsics[1])
            np.testing.assert_allclose(bundle["extrinsics"], extrinsics[1])


if __name__ == "__main__":
    unittest.main()
