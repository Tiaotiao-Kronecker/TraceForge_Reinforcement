import unittest

import numpy as np

from utils.moge_utils3d import depth_edge, normals_edge, points_to_normals


def _legacy_sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    axis = axis % x.ndim
    shape = (*x.shape[:axis], (x.shape[axis] - window_size + 1) // stride, *x.shape[axis + 1 :], window_size)
    strides = (*x.strides[:axis], stride * x.strides[axis], *x.strides[axis + 1 :], x.strides[axis])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def _legacy_sliding_window_nd(
    x: np.ndarray,
    window_size: tuple[int, ...],
    stride: tuple[int, ...],
    axis: tuple[int, ...],
) -> np.ndarray:
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = _legacy_sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x


def _legacy_sliding_window_2d(
    x: np.ndarray,
    window_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    axis: tuple[int, int] = (-2, -1),
) -> np.ndarray:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return _legacy_sliding_window_nd(x, window_size, stride, axis)


def _legacy_max_pool_1d(x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == "f" else np.iinfo(x.dtype).min
        padding_arr = np.full(
            (*x.shape[:axis], padding, *x.shape[axis + 1 :]),
            fill_value=fill_value,
            dtype=x.dtype,
        )
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = _legacy_sliding_window_1d(x, kernel_size, stride, axis)
    return np.nanmax(a_sliding, axis=-1)


def _legacy_max_pool_2d(
    x: np.ndarray,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    axis: tuple[int, int] = (-2, -1),
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    x = _legacy_max_pool_1d(x, kernel_size[0], stride[0], padding[0], axis[0])
    x = _legacy_max_pool_1d(x, kernel_size[1], stride[1], padding[1], axis[1])
    return x


def _legacy_depth_edge(
    depth: np.ndarray,
    atol: float | None = None,
    rtol: float | None = None,
    kernel_size: int = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if mask is None:
        diff = _legacy_max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + _legacy_max_pool_2d(
            -depth,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
    else:
        diff = _legacy_max_pool_2d(
            np.where(mask, depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ) + _legacy_max_pool_2d(
            np.where(mask, -depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


def _legacy_normals_edge(
    normals: np.ndarray,
    tol: float,
    kernel_size: int = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
    padding = kernel_size // 2
    normals_window = _legacy_sliding_window_2d(
        np.pad(
            normals,
            (*([(0, 0)] * (normals.ndim - 3)), (padding, padding), (padding, padding), (0, 0)),
            mode="edge",
        ),
        window_size=kernel_size,
        stride=1,
        axis=(-3, -2),
    )
    if mask is None:
        angle_diff = np.arccos((normals[..., None, None] * normals_window).sum(axis=-3)).max(axis=(-2, -1))
    else:
        mask_window = _legacy_sliding_window_2d(
            np.pad(
                mask,
                (*([(0, 0)] * (mask.ndim - 3)), (padding, padding), (padding, padding)),
                mode="edge",
            ),
            window_size=kernel_size,
            stride=1,
            axis=(-3, -2),
        )
        angle_diff = np.where(
            mask_window,
            np.arccos((normals[..., None, None] * normals_window).sum(axis=-3)),
            0,
        ).max(axis=(-2, -1))

    angle_diff = _legacy_max_pool_2d(angle_diff, kernel_size, stride=1, padding=kernel_size // 2)
    return angle_diff > np.deg2rad(tol)


def _legacy_points_to_normals(point: np.ndarray, mask: np.ndarray | None = None):
    height, width = point.shape[-3:-1]
    has_mask = mask is not None

    if mask is None:
        mask = np.ones_like(point[..., 0], dtype=bool)
    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask
    mask = mask_pad

    pts = np.zeros((height + 2, width + 2, 3), dtype=point.dtype)
    pts[1:-1, 1:-1, :] = point
    up = pts[:-2, 1:-1, :] - pts[1:-1, 1:-1, :]
    left = pts[1:-1, :-2, :] - pts[1:-1, 1:-1, :]
    down = pts[2:, 1:-1, :] - pts[1:-1, 1:-1, :]
    right = pts[1:-1, 2:, :] - pts[1:-1, 1:-1, :]
    normal = np.stack(
        [
            np.cross(up, left, axis=-1),
            np.cross(left, down, axis=-1),
            np.cross(down, right, axis=-1),
            np.cross(right, up, axis=-1),
        ]
    )
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)
    valid = (
        np.stack(
            [
                mask[:-2, 1:-1] & mask[1:-1, :-2],
                mask[1:-1, :-2] & mask[2:, 1:-1],
                mask[2:, 1:-1] & mask[1:-1, 2:],
                mask[1:-1, 2:] & mask[:-2, 1:-1],
            ]
        )
        & mask[None, 1:-1, 1:-1]
    )
    normal = (normal * valid[..., None]).sum(axis=0)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    if has_mask:
        normal_mask = valid.any(axis=0)
        normal = np.where(normal_mask[..., None], normal, 0)
        return normal, normal_mask
    return normal


class MogeUtils3dTests(unittest.TestCase):
    def test_depth_edge_matches_legacy_without_mask(self):
        rng = np.random.default_rng(0)
        depth = rng.uniform(0.2, 3.0, size=(2, 7, 9)).astype(np.float32)

        expected = _legacy_depth_edge(depth, rtol=0.08, kernel_size=3)
        actual = depth_edge(depth, rtol=0.08, kernel_size=3)

        np.testing.assert_array_equal(actual, expected)

    def test_depth_edge_matches_legacy_with_mask(self):
        rng = np.random.default_rng(1)
        depth = rng.uniform(0.2, 3.0, size=(7, 9)).astype(np.float32)
        mask = rng.random((7, 9)) > 0.25

        expected = _legacy_depth_edge(depth, rtol=0.08, kernel_size=3, mask=mask)
        actual = depth_edge(depth, rtol=0.08, kernel_size=3, mask=mask)

        np.testing.assert_array_equal(actual, expected)

    def test_normals_edge_matches_legacy_without_mask(self):
        rng = np.random.default_rng(2)
        normals = rng.normal(size=(6, 8, 3)).astype(np.float32)

        expected = _legacy_normals_edge(normals, tol=15.0, kernel_size=3)
        actual = normals_edge(normals, tol=15.0, kernel_size=3)

        np.testing.assert_array_equal(actual, expected)

    def test_normals_edge_matches_legacy_with_mask(self):
        rng = np.random.default_rng(3)
        normals = rng.normal(size=(6, 8, 3)).astype(np.float32)
        mask = rng.random((6, 8)) > 0.3

        expected = _legacy_normals_edge(normals, tol=15.0, kernel_size=3, mask=mask)
        actual = normals_edge(normals, tol=15.0, kernel_size=3, mask=mask)

        np.testing.assert_array_equal(actual, expected)

    def test_points_to_normals_matches_legacy_without_mask(self):
        rng = np.random.default_rng(4)
        point = rng.normal(size=(9, 11, 3)).astype(np.float32)

        expected = _legacy_points_to_normals(point)
        actual = points_to_normals(point)

        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_points_to_normals_matches_legacy_with_mask(self):
        rng = np.random.default_rng(5)
        point = rng.normal(size=(9, 11, 3)).astype(np.float32)
        mask = rng.random((9, 11)) > 0.25

        expected_normal, expected_mask = _legacy_points_to_normals(point, mask=mask)
        actual_normal, actual_mask = points_to_normals(point, mask=mask)

        np.testing.assert_array_equal(actual_mask, expected_mask)
        np.testing.assert_allclose(actual_normal, expected_normal, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
