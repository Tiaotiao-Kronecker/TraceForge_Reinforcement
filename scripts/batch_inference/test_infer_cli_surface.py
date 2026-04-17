import ast
import unittest
from pathlib import Path

import numpy as np


_SOURCE_PATH = Path(__file__).resolve().with_name("infer.py")
_SOURCE_AST = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(_SOURCE_PATH))


def _collect_cli_flags(func_ast: ast.FunctionDef) -> set[str]:
    flags: set[str] = set()
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flags.add(arg.value)
    return flags


def _collect_cli_defaults(func_ast: ast.FunctionDef) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        flag = None
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flag = arg.value
                break
        if flag is None:
            continue
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                defaults[flag] = keyword.value.value
    return defaults


def _collect_cli_choices(func_ast: ast.FunctionDef) -> dict[str, tuple[str, ...]]:
    choices: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(func_ast):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        flag = None
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                flag = arg.value
                break
        if flag is None:
            continue
        for keyword in node.keywords:
            if keyword.arg != "choices":
                continue
            if not isinstance(keyword.value, (ast.List, ast.Tuple)):
                continue
            values: list[str] = []
            for element in keyword.value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    values.append(element.value)
            if values:
                choices[flag] = tuple(values)
    return choices


_PARSE_ARGS_FUNC_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
)
_RESOLVE_SUPPORT_GRID_SIZE_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_support_grid_size"
)
_BUILD_GRID_KEYPOINTS_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "_build_grid_keypoints"
)
_BUILD_DENSE_SAMPLE_PAYLOAD_AST = next(
    node
    for node in _SOURCE_AST.body
    if isinstance(node, ast.FunctionDef) and node.name == "_build_dense_sample_payload_from_tracked_subset"
)

_HELPER_MODULE = ast.Module(
    body=[_BUILD_GRID_KEYPOINTS_AST, _RESOLVE_SUPPORT_GRID_SIZE_AST, _BUILD_DENSE_SAMPLE_PAYLOAD_AST],
    type_ignores=[],
)
_HELPER_NAMESPACE: dict[str, object] = {"np": np}
exec(compile(_HELPER_MODULE, str(_SOURCE_PATH), "exec"), _HELPER_NAMESPACE)
build_grid_keypoints = _HELPER_NAMESPACE["_build_grid_keypoints"]
resolve_support_grid_size = _HELPER_NAMESPACE["_resolve_support_grid_size"]
build_dense_sample_payload_from_tracked_subset = _HELPER_NAMESPACE["_build_dense_sample_payload_from_tracked_subset"]

_CLI_FLAGS = _collect_cli_flags(_PARSE_ARGS_FUNC_AST)
_CLI_DEFAULTS = _collect_cli_defaults(_PARSE_ARGS_FUNC_AST)
_CLI_CHOICES = _collect_cli_choices(_PARSE_ARGS_FUNC_AST)


class InferCliSurfaceTests(unittest.TestCase):
    def test_exposes_query_prefilter_and_support_ratio_flags(self):
        self.assertIn("--query_sampler_mode", _CLI_FLAGS)
        self.assertIn("--query_prefilter_mode", _CLI_FLAGS)
        self.assertIn("--query_prefilter_wrist_rank_keep_ratio", _CLI_FLAGS)
        self.assertIn("--query_visibility_gate_mode", _CLI_FLAGS)
        self.assertIn("--query_visibility_gate_min_border_dist_px", _CLI_FLAGS)
        self.assertIn("--query_fixed_view_depth_gate_mode", _CLI_FLAGS)
        self.assertIn("--query_fixed_view_depth_gate_uv_threshold_px", _CLI_FLAGS)
        self.assertIn("--query_fixed_view_depth_gate_depth_threshold_m", _CLI_FLAGS)
        self.assertIn("--traj_uvd_gate_mode", _CLI_FLAGS)
        self.assertIn("--traj_uvd_gate_uv_mean_threshold_px", _CLI_FLAGS)
        self.assertIn("--traj_uvd_gate_depth_std_threshold_m", _CLI_FLAGS)
        self.assertIn("--traj_uvd_gate_max_depth_threshold_m", _CLI_FLAGS)
        self.assertIn("--query_depth_stabilization_mode", _CLI_FLAGS)
        self.assertIn("--query_depth_stabilization_reproj_tol_px", _CLI_FLAGS)
        self.assertIn("--query_depth_stabilization_min_support", _CLI_FLAGS)
        self.assertIn("--dense_depth_stabilization_mode", _CLI_FLAGS)
        self.assertIn("--dense_depth_stabilization_radius", _CLI_FLAGS)
        self.assertIn("--dense_depth_stabilization_min_support", _CLI_FLAGS)
        self.assertIn("--tracker_precision_mode", _CLI_FLAGS)
        self.assertIn("--support_grid_ratio", _CLI_FLAGS)
        self.assertIn("--grid_border_trim_right", _CLI_FLAGS)
        self.assertIn("--traj_filter_ablation_mode", _CLI_FLAGS)
        self.assertIn("--collect_profile_stats", _CLI_FLAGS)
        self.assertIn("--depth_filter_workers", _CLI_FLAGS)
        self.assertIn("--depth_filter_blas_threads", _CLI_FLAGS)
        self.assertIn("--resize_width", _CLI_FLAGS)
        self.assertIn("--resize_height", _CLI_FLAGS)

    def test_exposes_external_depth_static_query_prefilter_mode(self):
        self.assertIn("external_depth_static_v1", _CLI_CHOICES.get("--query_prefilter_mode", ()))

    def test_exposes_temporal_query_depth_stabilization_mode(self):
        self.assertIn(
            "all_future_v1",
            _CLI_CHOICES.get("--query_visibility_gate_mode", ()),
        )
        self.assertIn(
            "first_frame_uvd_v1",
            _CLI_CHOICES.get("--query_fixed_view_depth_gate_mode", ()),
        )
        self.assertIn(
            "delta_uv_depth_v1",
            _CLI_CHOICES.get("--traj_uvd_gate_mode", ()),
        )
        self.assertIn(
            "temporal_median_world_v1",
            _CLI_CHOICES.get("--query_depth_stabilization_mode", ()),
        )
        self.assertIn(
            "temporal_median_reproject_v1",
            _CLI_CHOICES.get("--dense_depth_stabilization_mode", ()),
        )
        self.assertIn("deep_bf16", _CLI_CHOICES.get("--tracker_precision_mode", ()))

    def test_num_iters_default_is_three(self):
        self.assertEqual(_CLI_DEFAULTS.get("--num_iters"), 3)

    def test_defaults_to_grid_and_no_filter(self):
        self.assertEqual(_CLI_DEFAULTS.get("--query_sampler_mode"), "grid")
        self.assertEqual(_CLI_DEFAULTS.get("--filter_level"), "none")
        self.assertEqual(_CLI_DEFAULTS.get("--grid_border_trim_left"), 30)
        self.assertEqual(_CLI_DEFAULTS.get("--grid_border_trim_right"), 30)
        self.assertEqual(_CLI_DEFAULTS.get("--grid_border_trim_top"), 30)
        self.assertEqual(_CLI_DEFAULTS.get("--grid_border_trim_bottom"), 10)

    def test_support_grid_ratio_default_is_zero(self):
        self.assertEqual(_CLI_DEFAULTS.get("--support_grid_ratio"), 0.0)
        self.assertEqual(_CLI_DEFAULTS.get("--tracker_precision_mode"), "fp32")
        self.assertEqual(_CLI_DEFAULTS.get("--query_visibility_gate_mode"), "all_future_v1")
        self.assertEqual(_CLI_DEFAULTS.get("--query_visibility_gate_min_border_dist_px"), 0.0)
        self.assertEqual(_CLI_DEFAULTS.get("--query_visibility_gate_near_depth_exempt_threshold_m"), 0.0)
        self.assertEqual(_CLI_DEFAULTS.get("--query_fixed_view_depth_gate_mode"), "first_frame_uvd_v1")
        self.assertEqual(_CLI_DEFAULTS.get("--query_fixed_view_depth_gate_uv_threshold_px"), 1.0)
        self.assertEqual(_CLI_DEFAULTS.get("--query_fixed_view_depth_gate_depth_threshold_m"), 0.1)
        self.assertEqual(_CLI_DEFAULTS.get("--traj_uvd_gate_mode"), "delta_uv_depth_v1")
        self.assertEqual(_CLI_DEFAULTS.get("--traj_uvd_gate_uv_mean_threshold_px"), 3.0)
        self.assertEqual(_CLI_DEFAULTS.get("--traj_uvd_gate_depth_std_threshold_m"), 0.01)
        self.assertEqual(_CLI_DEFAULTS.get("--traj_uvd_gate_max_depth_threshold_m"), 1.5)
        self.assertEqual(_CLI_DEFAULTS.get("--query_depth_stabilization_mode"), "off")
        self.assertEqual(_CLI_DEFAULTS.get("--dense_depth_stabilization_mode"), "off")

    def test_depth_filter_workers_default_is_eight(self):
        self.assertEqual(_CLI_DEFAULTS.get("--depth_filter_workers"), 8)

    def test_depth_filter_blas_threads_default_is_one(self):
        self.assertEqual(_CLI_DEFAULTS.get("--depth_filter_blas_threads"), 1)

    def test_exposes_wrist_pick_place_no_heatmap_profile(self):
        self.assertIn("wrist_pick_place_no_heatmap", _CLI_CHOICES.get("--traj_filter_profile", ()))

    def test_support_grid_ratio_uses_rounded_nonnegative_size(self):
        self.assertEqual(resolve_support_grid_size(80, 0.8), 64)
        self.assertEqual(resolve_support_grid_size(80, 0.4), 32)
        self.assertEqual(resolve_support_grid_size(80, 0.0), 0)
        self.assertEqual(resolve_support_grid_size(3, 0.8), 2)
        self.assertEqual(resolve_support_grid_size(3, -1.0), 0)

    def test_grid_keypoints_reject_overlarge_grid_border_trim(self):
        with self.assertRaises(ValueError):
            build_grid_keypoints(100, 200, 5, trim_left=20, trim_right=20, trim_top=20)

    def test_grid_keypoints_support_right_border_trim(self):
        keypoints = build_grid_keypoints(100, 200, 5, trim_right=2)
        self.assertEqual(keypoints.shape, (15, 2))
        np.testing.assert_allclose(np.unique(keypoints[:, 0]), np.array([0.0, 49.75, 99.5], dtype=np.float32))

    def test_dense_scatter_preserves_shape_and_marks_untracked_queries_invalid(self):
        tracked_sample_payload = {
            "traj_uvz": np.array(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
                dtype=np.float32,
            ),
            "traj_2d": np.array(
                [
                    [[1.0, 2.0], [4.0, 5.0]],
                    [[7.0, 8.0], [10.0, 11.0]],
                ],
                dtype=np.float32,
            ),
            "keypoints": np.array([[10.0, 10.0], [30.0, 30.0]], dtype=np.float32),
            "query_frame_index": np.array([3], dtype=np.int32),
            "segment_frame_indices": np.array([3, 4], dtype=np.int32),
            "traj_valid_mask": np.array([True, False]),
            "traj_depth_consistency_ratio": np.array([1.0, np.nan], dtype=np.float16),
            "traj_stable_depth_consistency_ratio": np.array([1.0, np.nan], dtype=np.float16),
            "traj_high_volatility_hit": np.array([False, False]),
            "traj_volatility_exposure_ratio": np.array([0.0, np.nan], dtype=np.float16),
            "traj_compare_frame_count": np.array([2, 0], dtype=np.uint16),
            "traj_stable_compare_frame_count": np.array([2, 0], dtype=np.uint16),
            "traj_mask_reason_bits": np.array([0, 4], dtype=np.uint8),
            "traj_supervision_mask": np.array([[True, True], [False, False]]),
            "traj_supervision_prefix_len": np.array([2, 0], dtype=np.uint16),
            "traj_supervision_count": np.array([2, 0], dtype=np.uint16),
            "traj_wrist_seed_mask": np.array([True, False]),
            "traj_query_depth_rank": np.array([0.1, np.nan], dtype=np.float16),
            "traj_query_depth_edge_mask": np.array([False, False]),
            "traj_query_depth_patch_valid_ratio": np.array([1.0, np.nan], dtype=np.float16),
            "traj_query_depth_patch_std": np.array([0.0, np.nan], dtype=np.float16),
            "traj_query_depth_edge_risk_mask": np.array([False, False]),
            "traj_motion_extent": np.array([0.5, np.nan], dtype=np.float16),
            "traj_motion_step_median": np.array([0.1, np.nan], dtype=np.float16),
            "traj_motion_extent_all_valid": np.array([0.6, np.nan], dtype=np.float16),
            "traj_motion_step_median_all_valid": np.array([0.2, np.nan], dtype=np.float16),
            "traj_manipulator_candidate_mask": np.array([True, False]),
            "traj_manipulator_cluster_id": np.array([0, -1], dtype=np.int16),
            "traj_manipulator_component_size": np.array([2, 0], dtype=np.uint16),
            "traj_manipulator_cluster_fallback_used": np.asarray(False, dtype=bool),
            "traj_pick_place_heatmap_hit_count": np.array([3, 0], dtype=np.uint16),
            "traj_pick_place_heatmap_support_mask": np.array([True, False]),
            "traj_pick_place_min_manipulator_distance": np.array([0.1, np.nan], dtype=np.float16),
            "traj_pick_place_contact_mask": np.array([True, False]),
            "traj_pick_place_depth_guard_mask": np.array([True, False]),
            "traj_pick_place_object_mask": np.array([True, False]),
            "traj_uvd_gate_reliable_mask": np.array([True, False]),
            "traj_uvd_gate_removed_mask": np.array([False, True]),
            "traj_uvd_gate_uv_depth_anomaly_mask": np.array([False, True]),
            "traj_uvd_gate_far_depth_mask": np.array([False, False]),
            "traj_uvd_gate_uv_mean_delta_px": np.array([2.0, 0.5], dtype=np.float16),
            "traj_uvd_gate_depth_delta_std_m": np.array([0.0, 0.02], dtype=np.float16),
            "traj_uvd_gate_max_depth_m": np.array([1.0, 1.2], dtype=np.float16),
            "traj_uvd_gate_uv_pair_valid_count": np.array([2, 2], dtype=np.uint16),
            "traj_uvd_gate_depth_pair_valid_count": np.array([2, 2], dtype=np.uint16),
            "visibility": np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float16),
        }
        dense_keypoints = np.array(
            [
                [0.0, 0.0],
                [10.0, 10.0],
                [20.0, 20.0],
                [30.0, 30.0],
            ],
            dtype=np.float32,
        )
        prefilter_result = {
            "reason_bits": np.array([2, 0, 128, 0], dtype=np.uint8),
            "query_depth_rank": np.array([np.nan, 0.1, np.nan, 0.2], dtype=np.float32),
            "query_depth_edge_mask": np.array([False, False, True, False]),
            "query_depth_patch_valid_ratio": np.array([0.2, 1.0, 0.7, 1.0], dtype=np.float32),
            "query_depth_patch_std": np.array([np.nan, 0.0, 0.02, 0.0], dtype=np.float32),
            "query_depth_edge_risk_mask": np.array([False, False, True, False]),
        }
        fixed_view_gate_result = {
            "reliable_track_mask": np.array([False, True, False, True], dtype=bool),
            "compare_frame_count": np.array([2, 2, 1, 2], dtype=np.uint16),
            "uv_stable_hit_count": np.array([2, 2, 0, 1], dtype=np.uint16),
            "depth_jump_hit_count": np.array([1, 0, 0, 1], dtype=np.uint16),
            "depth_anomaly_hit_count": np.array([1, 0, 0, 1], dtype=np.uint16),
            "first_anomaly_step": np.array([1, -1, -1, 1], dtype=np.int16),
            "max_depth_delta_m": np.array([0.2, 0.0, np.nan, 0.3], dtype=np.float32),
            "min_uv_delta_px": np.array([0.2, 0.0, np.nan, 0.5], dtype=np.float32),
        }
        query_visibility_gate_result = {
            "reliable_track_mask": np.array([False, True, False, True], dtype=bool),
            "ever_out_of_view_mask": np.array([True, False, False, True], dtype=bool),
            "future_visible_ratio": np.array([0.0, 1.0, 0.5, 1.0], dtype=np.float32),
            "first_invalid_step": np.array([1, -1, 2, -1], dtype=np.int16),
        }

        dense_payload = build_dense_sample_payload_from_tracked_subset(
            dense_keypoints=dense_keypoints,
            tracked_query_indices=np.array([1, 3], dtype=np.int32),
            tracked_sample_payload=tracked_sample_payload,
            prefilter_result=prefilter_result,
            query_visibility_gate_result=query_visibility_gate_result,
            query_fixed_view_depth_gate_result=fixed_view_gate_result,
        )

        self.assertEqual(dense_payload["traj_uvz"].shape, (4, 2, 3))
        self.assertEqual(dense_payload["keypoints"].shape, (4, 2))
        self.assertEqual(dense_payload["traj_valid_mask"].shape, (4,))
        self.assertTrue(np.isnan(dense_payload["traj_uvz"][0]).all())
        self.assertTrue(np.isnan(dense_payload["traj_uvz"][2]).all())
        np.testing.assert_array_equal(dense_payload["traj_valid_mask"], np.array([False, True, False, False]))
        np.testing.assert_array_equal(
            dense_payload["traj_pick_place_object_mask"],
            np.array([False, True, False, False]),
        )
        np.testing.assert_array_equal(dense_payload["traj_mask_reason_bits"], np.array([2, 0, 128, 4], dtype=np.uint8))
        np.testing.assert_array_equal(dense_payload["traj_supervision_mask"][0], np.array([False, False]))
        np.testing.assert_array_equal(dense_payload["traj_query_depth_edge_risk_mask"], np.array([False, False, True, False]))
        np.testing.assert_array_equal(dense_payload["keypoints"], dense_keypoints)
        np.testing.assert_array_equal(dense_payload["visibility"][0], np.array([0.0, 0.0], dtype=np.float16))
        np.testing.assert_array_equal(dense_payload["visibility"][1], np.array([1.0, 1.0], dtype=np.float16))
        np.testing.assert_array_equal(
            dense_payload["traj_query_fixed_view_depth_consistency_mask"],
            np.array([False, True, False, True]),
        )
        np.testing.assert_array_equal(
            dense_payload["traj_query_fixed_view_depth_anomaly_mask"],
            np.array([True, False, False, True]),
        )
        np.testing.assert_array_equal(
            dense_payload["traj_query_visibility_reliable_mask"],
            np.array([False, True, False, True]),
        )
        np.testing.assert_array_equal(
            dense_payload["traj_query_visibility_removed_mask"],
            np.array([True, False, True, False]),
        )
        np.testing.assert_array_equal(
            dense_payload["traj_query_visibility_first_invalid_step"],
            np.array([1, -1, 2, -1], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            dense_payload["traj_uvd_gate_reliable_mask"],
            np.array([False, True, False, False]),
        )
        np.testing.assert_array_equal(
            dense_payload["traj_uvd_gate_removed_mask"],
            np.array([False, False, False, True]),
        )


if __name__ == "__main__":
    unittest.main()
