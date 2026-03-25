import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from datasets.data_ops import _build_depth_filter_rays, _filter_one_depth, _filter_one_depth_profiled
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

    def test_filter_one_depth_profiled_reports_nonnegative_stage_times(self):
        depth = np.full((4, 4), 1.0, dtype=np.float32)
        intrinsics = _make_intrinsics()

        filtered_depth, profile = _filter_one_depth_profiled(
            depth,
            0.08,
            15,
            intrinsics,
        )

        self.assertEqual(filtered_depth.shape, depth.shape)
        expected_keys = {
            "ray_scale_seconds",
            "points_to_normals_seconds",
            "edge_mask_seconds",
            "distance_transform_seconds",
            "fill_seconds",
            "total_seconds",
        }
        self.assertEqual(set(profile.keys()), expected_keys)
        for value in profile.values():
            self.assertGreaterEqual(float(value), 0.0)


class DepthFilterRuntimeTests(unittest.TestCase):
    def test_overlapping_segments_only_filter_unique_frames_once(self):
        depths, intrinsics = _make_runtime_fixture()
        profile_stats: dict[str, float] = {}

        def fake_filter(depth, depth_rtol, normal_tol, intrinsic, rays=None):
            self.assertIsNotNone(rays)
            return depth + 0.5, {
                "ray_scale_seconds": 0.01,
                "points_to_normals_seconds": 0.02,
                "edge_mask_seconds": 0.03,
                "distance_transform_seconds": 0.04,
                "fill_seconds": 0.05,
                "total_seconds": 0.15,
            }

        with mock.patch.object(infer, "_filter_one_depth_profiled", side_effect=fake_filter) as filter_mock:
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
        self.assertEqual(profile_stats["prepare_depth_filter_unique_frame_count"], 3.0)
        self.assertAlmostEqual(profile_stats["prepare_depth_filter_worker_total_seconds"], 0.45)
        self.assertAlmostEqual(profile_stats["prepare_depth_filter_distance_transform_seconds"], 0.12)
        self.assertGreaterEqual(profile_stats["prepare_depth_filter_stack_seconds"], 0.0)

    def test_release_segment_frames_evicts_completed_frames(self):
        depths, intrinsics = _make_runtime_fixture()

        with mock.patch.object(
            infer,
            "_filter_one_depth_profiled",
            side_effect=lambda depth, depth_rtol, normal_tol, intrinsic, rays=None: (depth, {"total_seconds": 0.0}),
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
            "_filter_one_depth_profiled",
            side_effect=lambda depth, depth_rtol, normal_tol, intrinsic, rays=None: (depth, {"total_seconds": 0.0}),
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


class ShortTailSkipTests(unittest.TestCase):
    def test_apply_short_tail_skip_to_query_frames_updates_metadata(self):
        query_frames, metadata = infer._apply_short_tail_skip_to_query_frames(
            query_frames=[0, 7, 8, 10],
            query_frame_metadata={
                "query_frame_sampling_mode": "shared_schedule",
                "query_frame_indices_local": [0, 7, 8, 10],
                "query_frame_source_indices": [100, 107, 108, 110],
            },
            source_frame_indices=np.arange(100, 116, dtype=np.int32),
            video_length=16,
            future_len=12,
        )

        self.assertEqual(query_frames, [0, 7])
        self.assertEqual(metadata["requested_query_frame_indices_local"], [0, 7, 8, 10])
        self.assertEqual(metadata["requested_query_frame_source_indices"], [100, 107, 108, 110])
        self.assertEqual(metadata["query_frame_indices_local"], [0, 7])
        self.assertEqual(metadata["query_frame_source_indices"], [100, 107])
        self.assertEqual(metadata["skipped_short_tail_query_frame_indices_local"], [8, 10])
        self.assertEqual(metadata["skipped_short_tail_query_frame_source_indices"], [108, 110])
        self.assertEqual(metadata["skipped_short_tail_segment_lengths"], [8, 6])
        self.assertEqual(
            metadata["short_tail_skip_max_segment_len"],
            infer.SHORT_TAIL_SKIP_MAX_SEGMENT_LEN,
        )

    def test_cleanup_skipped_short_tail_artifacts_removes_stale_v2_sample(self):
        with mock.patch.object(infer, "logger"), mock.patch.object(
            infer,
            "_accumulate_profile_stat",
        ) as accumulate_mock, mock.patch.object(infer, "time") as time_mock:
            with TemporaryDirectory() as tmpdir:
                sample_path = (
                    Path(tmpdir)
                    / "varied_camera_1"
                    / "samples"
                    / "varied_camera_1_8.npz"
                )
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(sample_path, value=np.array([1], dtype=np.int32))

                time_mock.perf_counter.side_effect = [1.0, 1.2]
                removed_frames = infer._cleanup_skipped_short_tail_artifacts(
                    video_name="varied_camera_1",
                    output_dir=tmpdir,
                    layout=infer.V2_LAYOUT,
                    query_frame_metadata={
                        "skipped_short_tail_query_frame_indices_local": [8],
                    },
                    profile_stats={},
                )

        self.assertEqual(removed_frames, [8])
        self.assertFalse(sample_path.exists())
        accumulate_mock.assert_called_once()


class V2PaddingTests(unittest.TestCase):
    def test_pad_v2_sample_data_extends_tail_to_future_len_with_inf_and_false_steps(self):
        sample_data = {
            "traj_uvz": np.array(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                ],
                dtype=np.float32,
            ),
            "traj_supervision_mask": np.array(
                [
                    [True, True, False],
                    [True, False, False],
                ],
                dtype=bool,
            ),
            "visibility": np.array(
                [
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=np.float16,
            ),
        }

        padded = infer._pad_v2_sample_data_to_future_len(
            sample_data=sample_data,
            future_len=5,
        )

        self.assertEqual(padded["traj_uvz"].shape, (2, 5, 3))
        self.assertEqual(padded["traj_supervision_mask"].shape, (2, 5))
        self.assertEqual(padded["visibility"].shape, (2, 5))
        np.testing.assert_array_equal(
            padded["valid_steps"],
            np.array([True, True, True, False, False]),
        )
        self.assertTrue(np.isinf(padded["traj_uvz"][:, 3:]).all())
        np.testing.assert_array_equal(
            padded["traj_supervision_mask"][:, 3:],
            np.zeros((2, 2), dtype=bool),
        )
        np.testing.assert_array_equal(
            padded["visibility"][:, 3:],
            np.zeros((2, 2), dtype=np.float16),
        )

    def test_build_v2_sample_data_pads_tail_segment_and_preserves_real_segment_indices(self):
        prepared_bundle = {
            "traj_uvz": np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32),
            "traj_2d": np.array([[[1.0, 2.0], [4.0, 5.0]]], dtype=np.float32),
            "keypoints": np.array([[10.0, 10.0]], dtype=np.float32),
            "dense_keypoints": np.array([[10.0, 10.0]], dtype=np.float32),
            "tracked_query_indices": np.array([0], dtype=np.int32),
            "prefilter_result": None,
            "visibs": np.array([[1.0], [1.0]], dtype=np.float32),
            "query_frame_idx": 7,
            "query_frame_depth": np.ones((2, 2), dtype=np.float32),
            "raw_depths_segment": np.ones((2, 2, 2), dtype=np.float32),
            "intrinsics_segment": np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
            "extrinsics_segment": np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
            "temporal_compare_context": None,
            "support_grid_size": None,
            "query_frame_img": None,
        }
        filter_result = {
            "traj_valid_mask": np.array([True]),
            "traj_depth_consistency_ratio": np.array([1.0], dtype=np.float32),
            "traj_stable_depth_consistency_ratio": np.array([1.0], dtype=np.float32),
            "traj_high_volatility_hit": np.array([False]),
            "traj_volatility_exposure_ratio": np.array([0.0], dtype=np.float32),
            "traj_compare_frame_count": np.array([2], dtype=np.uint16),
            "traj_stable_compare_frame_count": np.array([2], dtype=np.uint16),
            "traj_mask_reason_bits": np.array([0], dtype=np.uint8),
            "traj_supervision_mask": np.array([[True, False]], dtype=bool),
            "traj_supervision_prefix_len": np.array([1], dtype=np.uint16),
            "traj_supervision_count": np.array([1], dtype=np.uint16),
            "traj_wrist_seed_mask": np.array([True]),
            "traj_query_depth_rank": np.array([0.1], dtype=np.float32),
            "traj_query_depth_edge_mask": np.array([False]),
            "traj_query_depth_patch_valid_ratio": np.array([1.0], dtype=np.float32),
            "traj_query_depth_patch_std": np.array([0.0], dtype=np.float32),
            "traj_query_depth_edge_risk_mask": np.array([False]),
            "traj_motion_extent": np.array([0.5], dtype=np.float32),
            "traj_motion_step_median": np.array([0.2], dtype=np.float32),
            "traj_motion_extent_all_valid": np.array([0.5], dtype=np.float32),
            "traj_motion_step_median_all_valid": np.array([0.2], dtype=np.float32),
            "traj_manipulator_candidate_mask": np.array([True]),
            "traj_manipulator_cluster_id": np.array([0], dtype=np.int16),
            "traj_manipulator_component_size": np.array([1], dtype=np.uint16),
            "traj_manipulator_cluster_fallback_used": np.asarray(False, dtype=bool),
        }

        with mock.patch.object(
            infer,
            "build_query_frame_sample_data",
            return_value={
                "sample_payload": {
                    "traj_uvz": prepared_bundle["traj_uvz"],
                    "keypoints": prepared_bundle["keypoints"],
                    "query_frame_index": np.array([7], dtype=np.int32),
                    "segment_frame_indices": np.array([7, 8], dtype=np.int32),
                    "traj_valid_mask": filter_result["traj_valid_mask"],
                    "traj_depth_consistency_ratio": filter_result["traj_depth_consistency_ratio"],
                    "traj_stable_depth_consistency_ratio": filter_result["traj_stable_depth_consistency_ratio"],
                    "traj_high_volatility_hit": filter_result["traj_high_volatility_hit"],
                    "traj_volatility_exposure_ratio": filter_result["traj_volatility_exposure_ratio"],
                    "traj_compare_frame_count": filter_result["traj_compare_frame_count"],
                    "traj_stable_compare_frame_count": filter_result["traj_stable_compare_frame_count"],
                    "traj_mask_reason_bits": filter_result["traj_mask_reason_bits"],
                    "traj_supervision_mask": filter_result["traj_supervision_mask"],
                    "traj_supervision_prefix_len": filter_result["traj_supervision_prefix_len"],
                    "traj_supervision_count": filter_result["traj_supervision_count"],
                    "traj_wrist_seed_mask": filter_result["traj_wrist_seed_mask"],
                    "traj_query_depth_rank": filter_result["traj_query_depth_rank"],
                    "traj_query_depth_edge_mask": filter_result["traj_query_depth_edge_mask"],
                    "traj_query_depth_patch_valid_ratio": filter_result["traj_query_depth_patch_valid_ratio"],
                    "traj_query_depth_patch_std": filter_result["traj_query_depth_patch_std"],
                    "traj_query_depth_edge_risk_mask": filter_result["traj_query_depth_edge_risk_mask"],
                    "traj_motion_extent": filter_result["traj_motion_extent"],
                    "traj_motion_step_median": filter_result["traj_motion_step_median"],
                    "traj_motion_extent_all_valid": filter_result["traj_motion_extent_all_valid"],
                    "traj_motion_step_median_all_valid": filter_result["traj_motion_step_median_all_valid"],
                    "traj_manipulator_candidate_mask": filter_result["traj_manipulator_candidate_mask"],
                    "traj_manipulator_cluster_id": filter_result["traj_manipulator_cluster_id"],
                    "traj_manipulator_component_size": filter_result["traj_manipulator_component_size"],
                    "traj_manipulator_cluster_fallback_used": filter_result["traj_manipulator_cluster_fallback_used"],
                },
                "traj_filter_result": filter_result,
            },
        ):
            built = infer.build_v2_sample_data(
                prepared_bundle=prepared_bundle,
                filter_args=SimpleNamespace(future_len=4, save_visibility=False),
                high_volatility_mask=None,
                save_profile_stats=None,
            )

        sample_data = built["sample_data"]
        self.assertEqual(sample_data["traj_uvz"].shape, (1, 4, 3))
        np.testing.assert_array_equal(sample_data["segment_frame_indices"], np.array([7, 8], dtype=np.int32))
        np.testing.assert_array_equal(
            sample_data["valid_steps"],
            np.array([True, True, False, False]),
        )
        self.assertTrue(np.isinf(sample_data["traj_uvz"][:, 2:]).all())
        np.testing.assert_array_equal(
            sample_data["traj_supervision_mask"],
            np.array([[True, False, False, False]], dtype=bool),
        )


if __name__ == "__main__":
    unittest.main()
