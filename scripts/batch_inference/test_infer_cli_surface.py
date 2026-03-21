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


_PARSE_ARGS_FUNC_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
)
_RESOLVE_SUPPORT_GRID_SIZE_AST = next(
    node for node in _SOURCE_AST.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_support_grid_size"
)
_BUILD_DENSE_SAMPLE_PAYLOAD_AST = next(
    node
    for node in _SOURCE_AST.body
    if isinstance(node, ast.FunctionDef) and node.name == "_build_dense_sample_payload_from_tracked_subset"
)

_HELPER_MODULE = ast.Module(
    body=[_RESOLVE_SUPPORT_GRID_SIZE_AST, _BUILD_DENSE_SAMPLE_PAYLOAD_AST],
    type_ignores=[],
)
_HELPER_NAMESPACE: dict[str, object] = {"np": np}
exec(compile(_HELPER_MODULE, str(_SOURCE_PATH), "exec"), _HELPER_NAMESPACE)
resolve_support_grid_size = _HELPER_NAMESPACE["_resolve_support_grid_size"]
build_dense_sample_payload_from_tracked_subset = _HELPER_NAMESPACE["_build_dense_sample_payload_from_tracked_subset"]

_CLI_FLAGS = _collect_cli_flags(_PARSE_ARGS_FUNC_AST)
_CLI_DEFAULTS = _collect_cli_defaults(_PARSE_ARGS_FUNC_AST)


class InferCliSurfaceTests(unittest.TestCase):
    def test_exposes_query_prefilter_and_support_ratio_flags(self):
        self.assertIn("--query_prefilter_mode", _CLI_FLAGS)
        self.assertIn("--query_prefilter_wrist_rank_keep_ratio", _CLI_FLAGS)
        self.assertIn("--support_grid_ratio", _CLI_FLAGS)
        self.assertIn("--traj_filter_ablation_mode", _CLI_FLAGS)

    def test_num_iters_default_is_five(self):
        self.assertEqual(_CLI_DEFAULTS.get("--num_iters"), 5)

    def test_support_grid_ratio_uses_rounded_nonnegative_size(self):
        self.assertEqual(resolve_support_grid_size(80, 0.8), 64)
        self.assertEqual(resolve_support_grid_size(80, 0.4), 32)
        self.assertEqual(resolve_support_grid_size(80, 0.0), 0)
        self.assertEqual(resolve_support_grid_size(3, 0.8), 2)
        self.assertEqual(resolve_support_grid_size(3, -1.0), 0)

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
            "traj_manipulator_candidate_mask": np.array([True, False]),
            "traj_manipulator_cluster_id": np.array([0, -1], dtype=np.int16),
            "traj_manipulator_component_size": np.array([2, 0], dtype=np.uint16),
            "traj_manipulator_cluster_fallback_used": np.asarray(False, dtype=bool),
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

        dense_payload = build_dense_sample_payload_from_tracked_subset(
            dense_keypoints=dense_keypoints,
            tracked_query_indices=np.array([1, 3], dtype=np.int32),
            tracked_sample_payload=tracked_sample_payload,
            prefilter_result=prefilter_result,
        )

        self.assertEqual(dense_payload["traj_uvz"].shape, (4, 2, 3))
        self.assertEqual(dense_payload["keypoints"].shape, (4, 2))
        self.assertEqual(dense_payload["traj_valid_mask"].shape, (4,))
        self.assertTrue(np.isnan(dense_payload["traj_uvz"][0]).all())
        self.assertTrue(np.isnan(dense_payload["traj_uvz"][2]).all())
        np.testing.assert_array_equal(dense_payload["traj_valid_mask"], np.array([False, True, False, False]))
        np.testing.assert_array_equal(dense_payload["traj_mask_reason_bits"], np.array([2, 0, 128, 4], dtype=np.uint8))
        np.testing.assert_array_equal(dense_payload["traj_supervision_mask"][0], np.array([False, False]))
        np.testing.assert_array_equal(dense_payload["traj_query_depth_edge_risk_mask"], np.array([False, False, True, False]))
        np.testing.assert_array_equal(dense_payload["keypoints"], dense_keypoints)
        np.testing.assert_array_equal(dense_payload["visibility"][0], np.array([0.0, 0.0], dtype=np.float16))
        np.testing.assert_array_equal(dense_payload["visibility"][1], np.array([1.0, 1.0], dtype=np.float16))


if __name__ == "__main__":
    unittest.main()
