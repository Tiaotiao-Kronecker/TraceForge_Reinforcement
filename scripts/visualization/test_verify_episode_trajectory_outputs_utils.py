import ast
import types
import unittest
from pathlib import Path

import numpy as np


_SOURCE_PATH = Path(__file__).resolve().with_name("verify_episode_trajectory_outputs.py")
_SOURCE_AST = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(_SOURCE_PATH))
_TARGET_NAMES = {
    "choose_track_indices",
    "choose_gif_candidate_indices",
    "choose_spatial_subset",
    "choose_balanced_subset",
    "choose_gif_track_indices",
}
_TARGET_BODY = []
for node in _SOURCE_AST.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.startswith("GIF_TRACK_SAMPLING_"):
                _TARGET_BODY.append(node)
                break
    elif isinstance(node, ast.FunctionDef) and node.name in _TARGET_NAMES:
        _TARGET_BODY.append(node)
_TARGET_MODULE = ast.Module(body=_TARGET_BODY, type_ignores=[])

_NAMESPACE = {"np": np}
exec(compile(_TARGET_MODULE, str(_SOURCE_PATH), "exec"), _NAMESPACE)
choose_track_indices = _NAMESPACE["choose_track_indices"]
choose_gif_candidate_indices = _NAMESPACE["choose_gif_candidate_indices"]
choose_spatial_subset = _NAMESPACE["choose_spatial_subset"]
choose_gif_track_indices = _NAMESPACE["choose_gif_track_indices"]
GIF_TRACK_SAMPLING_SHARED = _NAMESPACE["GIF_TRACK_SAMPLING_SHARED"]
GIF_TRACK_SAMPLING_BALANCED = _NAMESPACE["GIF_TRACK_SAMPLING_BALANCED"]
GIF_TRACK_SAMPLING_TOP_MOTION = _NAMESPACE["GIF_TRACK_SAMPLING_TOP_MOTION"]
GIF_TRACK_SAMPLING_SPATIAL = _NAMESPACE["GIF_TRACK_SAMPLING_SPATIAL"]


class GifTrackSamplingTests(unittest.TestCase):
    def test_choose_gif_candidate_indices_uses_all_tracks_for_spatial_sampling(self) -> None:
        traj_world = np.zeros((4, 2, 3), dtype=np.float32)
        traj_world[0, 1, 0] = 4.0
        traj_world[1, 1, 0] = 3.0
        traj_world[2, 1, 0] = 2.0
        traj_world[3, 1, 0] = 1.0

        candidate = choose_gif_candidate_indices(
            traj_world=traj_world,
            max_tracks=2,
            gif_track_sampling=GIF_TRACK_SAMPLING_SPATIAL,
            shared_track_indices=None,
        )

        np.testing.assert_array_equal(candidate, np.array([0, 1, 2, 3], dtype=np.int32))

    def test_choose_gif_candidate_indices_keeps_top_motion_for_top_motion(self) -> None:
        traj_world = np.zeros((4, 2, 3), dtype=np.float32)
        traj_world[0, 1, 0] = 1.0
        traj_world[1, 1, 0] = 4.0
        traj_world[2, 1, 0] = 3.0
        traj_world[3, 1, 0] = 2.0

        candidate = choose_gif_candidate_indices(
            traj_world=traj_world,
            max_tracks=2,
            gif_track_sampling=GIF_TRACK_SAMPLING_TOP_MOTION,
            shared_track_indices=None,
        )

        np.testing.assert_array_equal(candidate, np.array([1, 2], dtype=np.int32))

    def test_choose_spatial_subset_spreads_tracks_across_query_frame(self) -> None:
        track_indices = np.array([0, 1, 2, 3], dtype=np.int32)
        query_points = np.array(
            [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
            dtype=np.float32,
        )

        selected = choose_spatial_subset(track_indices, max_tracks=2, query_points=query_points)

        np.testing.assert_array_equal(selected, np.array([0, 3], dtype=np.int32))

    def test_choose_gif_track_indices_spatial_requires_query_points(self) -> None:
        with self.assertRaises(ValueError):
            choose_gif_track_indices(
                track_indices=np.array([0, 1, 2], dtype=np.int32),
                max_gif_tracks=2,
                gif_track_sampling=GIF_TRACK_SAMPLING_SPATIAL,
                group_labels=[("manip", -1)] * 3,
                query_points=None,
            )


if __name__ == "__main__":
    unittest.main()
