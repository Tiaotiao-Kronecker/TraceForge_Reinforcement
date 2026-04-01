import ast
import types
import unittest
from pathlib import Path

import numpy as np


_SOURCE_PATH = Path(__file__).resolve().with_name("visualize_3d_keypoint_animation.py")
_SOURCE_AST = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"), filename=str(_SOURCE_PATH))
_TARGET_NAMES = {"compute_scene_bounds", "set_initial_camera_from_scene"}
_TARGET_BODY = [
    node
    for node in _SOURCE_AST.body
    if isinstance(node, ast.FunctionDef) and node.name in _TARGET_NAMES
]
_TARGET_MODULE = ast.Module(body=_TARGET_BODY, type_ignores=[])


class _FakeLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


_LOGGER = _FakeLogger()
_NAMESPACE = {
    "np": np,
    "logger": _LOGGER,
    "viser": types.SimpleNamespace(ViserServer=object),
}
exec(compile(_TARGET_MODULE, str(_SOURCE_PATH), "exec"), _NAMESPACE)
compute_scene_bounds = _NAMESPACE["compute_scene_bounds"]
set_initial_camera_from_scene = _NAMESPACE["set_initial_camera_from_scene"]


class ComputeSceneBoundsTests(unittest.TestCase):
    def test_merges_multiple_point_sets_and_ignores_invalid_rows(self) -> None:
        center_radius = compute_scene_bounds(
            [
                np.array([[0.0, 0.0, 0.0], [np.inf, 1.0, 1.0]], dtype=np.float32),
                np.array([[2.0, 4.0, 6.0]], dtype=np.float32),
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
            ]
        )

        self.assertIsNotNone(center_radius)
        center, radius = center_radius
        self.assertTrue(np.allclose(center, np.array([1.0, 2.0, 3.0], dtype=np.float64)))
        self.assertAlmostEqual(radius, 0.5 * np.linalg.norm([2.0, 4.0, 6.0]), places=6)

    def test_returns_none_when_no_valid_xyz_points_exist(self) -> None:
        self.assertIsNone(
            compute_scene_bounds(
                [
                    np.array([[np.nan, 0.0, 0.0]], dtype=np.float32),
                    np.empty((0, 3), dtype=np.float32),
                    np.array([[1.0, 2.0]], dtype=np.float32),
                ]
            )
        )


class _FakeInitialCamera:
    def __init__(self) -> None:
        self.look_at = None
        self.position = None
        self.near = None
        self.far = None


class _FakeServer:
    def __init__(self) -> None:
        self.initial_camera = _FakeInitialCamera()


class SetInitialCameraFromSceneTests(unittest.TestCase):
    def test_sets_camera_position_clip_planes_and_look_at(self) -> None:
        server = _FakeServer()
        scene_center = np.array([1.0, -2.0, 0.5], dtype=np.float64)
        scene_radius = 1.25

        set_initial_camera_from_scene(
            server,
            scene_center=scene_center,
            scene_radius=scene_radius,
        )

        direction = np.array([2.2, 1.4, 2.2], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        expected_distance = scene_radius * 4.0

        self.assertTrue(np.allclose(server.initial_camera.look_at, scene_center))
        self.assertTrue(
            np.allclose(
                server.initial_camera.position,
                scene_center + direction * expected_distance,
            )
        )
        self.assertAlmostEqual(server.initial_camera.near, scene_radius * 0.02)
        self.assertAlmostEqual(server.initial_camera.far, scene_radius * 40.0)
        self.assertTrue(_LOGGER.messages)

    def test_enforces_minimum_distance_and_clip_planes(self) -> None:
        server = _FakeServer()

        set_initial_camera_from_scene(
            server,
            scene_center=np.zeros(3, dtype=np.float64),
            scene_radius=0.01,
        )

        self.assertAlmostEqual(np.linalg.norm(server.initial_camera.position), 0.35)
        self.assertAlmostEqual(server.initial_camera.near, 1e-3)
        self.assertAlmostEqual(server.initial_camera.far, 5.0)


if __name__ == "__main__":
    unittest.main()
