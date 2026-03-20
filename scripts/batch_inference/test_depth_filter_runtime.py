import unittest
from unittest import mock

import numpy as np
import torch

from datasets.data_ops import _build_depth_filter_rays, _filter_one_depth
from scripts.batch_inference import infer


def _make_intrinsics(fx: float = 120.0, fy: float = 120.0, cx: float = 3.0, cy: float = 3.0) -> np.ndarray:
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _make_runtime_fixture() -> tuple[np.ndarray, np.ndarray]:
    depths = np.stack(
        [
            np.full((4, 4), 1.0, dtype=np.float32),
            np.full((4, 4), 2.0, dtype=np.float32),
            np.full((4, 4), 3.0, dtype=np.float32),
        ],
        axis=0,
    )
    intrinsics = np.repeat(_make_intrinsics()[None], depths.shape[0], axis=0)
    return depths, intrinsics


class DepthFilterPrimitiveTests(unittest.TestCase):
    def test_filter_one_depth_matches_cached_rays_path(self):
        depth = np.array(
            [
                [1.0, 1.0, 1.1, 0.0, 1.2, 1.2],
                [1.0, 1.0, 1.1, 0.0, 1.2, 1.2],
                [1.0, 1.0, 1.4, 1.4, 1.2, 1.2],
                [1.0, 1.0, 1.4, 1.4, 1.2, 1.2],
                [0.9, 0.9, 1.4, 1.4, 1.2, 1.2],
                [0.9, 0.9, 1.4, 1.4, 1.2, 1.2],
            ],
            dtype=np.float32,
        )
        intrinsics = _make_intrinsics(cx=2.5, cy=2.5)
        rays = _build_depth_filter_rays(depth.shape, intrinsics)

        filtered_default = _filter_one_depth(depth, 0.08, 15, intrinsics)
        filtered_cached = _filter_one_depth(depth, 0.08, 15, intrinsics, rays=rays)

        np.testing.assert_array_equal(filtered_default, filtered_cached)


class DepthFilterRuntimeTests(unittest.TestCase):
    def test_overlapping_segments_only_filter_unique_frames_once(self):
        depths, intrinsics = _make_runtime_fixture()
        profile_stats: dict[str, float] = {}

        def fake_filter(depth, depth_rtol, normal_tol, intrinsic, rays=None):
            self.assertIsNotNone(rays)
            return depth + 0.5

        with mock.patch.object(infer, "_filter_one_depth", side_effect=fake_filter) as filter_mock:
            with infer._DepthFilterRuntime(
                depths,
                intrinsics,
                [(0, 2), (1, 3)],
                profile_stats=profile_stats,
                max_workers=2,
            ) as runtime:
                first_segment = runtime.get_filtered_depth_segment(0, 2)
                runtime.release_segment_frames(0, 2)
                second_segment = runtime.get_filtered_depth_segment(1, 3)

        self.assertEqual(filter_mock.call_count, 3)
        np.testing.assert_array_equal(first_segment[1], second_segment[0])
        self.assertEqual(profile_stats["prepare_depth_filter_cache_miss_frames"], 3.0)
        self.assertEqual(profile_stats["prepare_depth_filter_cache_hit_frames"], 1.0)

    def test_release_segment_frames_evicts_completed_frames(self):
        depths, intrinsics = _make_runtime_fixture()

        with mock.patch.object(
            infer,
            "_filter_one_depth",
            side_effect=lambda depth, depth_rtol, normal_tol, intrinsic, rays=None: depth,
        ):
            with infer._DepthFilterRuntime(
                depths,
                intrinsics,
                [(0, 2), (1, 3)],
                profile_stats={},
                max_workers=2,
            ) as runtime:
                runtime.get_filtered_depth_segment(0, 2)
                self.assertIn(0, runtime._filtered_depth_cache)
                self.assertIn(1, runtime._filtered_depth_cache)

                runtime.release_segment_frames(0, 2)
                self.assertNotIn(0, runtime._filtered_depth_cache)
                self.assertIn(1, runtime._filtered_depth_cache)

                runtime.get_filtered_depth_segment(1, 3)
                runtime.release_segment_frames(1, 3)
                self.assertEqual(runtime._filtered_depth_cache, {})

    def test_ray_cache_reuses_identical_intrinsics_and_shape(self):
        depths, intrinsics = _make_runtime_fixture()
        profile_stats: dict[str, float] = {}

        with mock.patch.object(
            infer,
            "_build_depth_filter_rays",
            return_value=np.ones((4, 4, 3), dtype=np.float32),
        ) as ray_builder, mock.patch.object(
            infer,
            "_filter_one_depth",
            side_effect=lambda depth, depth_rtol, normal_tol, intrinsic, rays=None: depth,
        ):
            with infer._DepthFilterRuntime(
                depths,
                intrinsics,
                [(0, 3)],
                profile_stats=profile_stats,
                max_workers=2,
            ) as runtime:
                runtime.get_filtered_depth_segment(0, 3)
                self.assertEqual(len(runtime._ray_cache), 1)

        self.assertEqual(ray_builder.call_count, 1)
        self.assertEqual(profile_stats["prepare_depth_filter_ray_cache_miss_frames"], 1.0)
        self.assertEqual(profile_stats["prepare_depth_filter_ray_cache_hit_frames"], 2.0)


class PrepareInputsTests(unittest.TestCase):
    def test_prepare_inputs_does_not_mutate_intrinsics(self):
        video_ten = torch.ones((2, 3, 4, 4), dtype=torch.float32)
        depths = np.ones((2, 4, 4), dtype=np.float32)
        intrinsics = np.repeat(_make_intrinsics()[None], 2, axis=0)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        query_point = [np.array([[0.0, 1.0, 1.0]], dtype=np.float32)]
        intrinsics_before = intrinsics.copy()

        infer.prepare_inputs(
            video_ten,
            depths,
            intrinsics,
            extrinsics,
            query_point,
            inference_res=(4, 4),
            support_grid_size=0,
            device="cpu",
            profile_stats={},
        )

        np.testing.assert_array_equal(intrinsics, intrinsics_before)


if __name__ == "__main__":
    unittest.main()
