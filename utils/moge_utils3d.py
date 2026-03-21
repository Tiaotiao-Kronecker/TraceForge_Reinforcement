# Copied from https://github.com/EasternJournalist/utils3d/blob/3913c65d81e05e47b9f367250cf8c0f7462a0900/utils3d/numpy/utils.py

import numpy as np
from typing import Tuple, Union
from numbers import Number
from scipy import ndimage

def sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1):
    """
    Return x view of the input array with x sliding window of the given kernel size and stride.
    The sliding window is performed over the given axis, and the window dimension is append to the end of the output array's shape.

    Args:
        x (np.ndarray): input array with shape (..., axis_size, ...)
        kernel_size (int): size of the sliding window
        stride (int): stride of the sliding window
        axis (int): axis to perform sliding window over
    
    Returns:
        a_sliding (np.ndarray): view of the input array with shape (..., n_windows, ..., kernel_size), where n_windows = (axis_size - kernel_size + 1) // stride
    """
    assert x.shape[axis] >= window_size, f"kernel_size ({window_size}) is larger than axis_size ({x.shape[axis]})"
    axis = axis % x.ndim
    shape = (*x.shape[:axis], (x.shape[axis] - window_size + 1) // stride, *x.shape[axis + 1:], window_size)
    strides = (*x.strides[:axis], stride * x.strides[axis], *x.strides[axis + 1:], x.strides[axis])
    x_sliding = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return x_sliding

def max_pool_1d(x: np.ndarray, kernel_size: int, stride: int, padding: int = 0, axis: int = -1):
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == 'f' else np.iinfo(x.dtype).min
        padding_arr = np.full((*x.shape[:axis], padding, *x.shape[axis + 1:]), fill_value=fill_value, dtype=x.dtype)
        x = np.concatenate([padding_arr, x, padding_arr], axis=axis)
    a_sliding = sliding_window_1d(x, kernel_size, stride, axis)
    max_pool = np.nanmax(a_sliding, axis=-1)
    return max_pool

def sliding_window_nd(x: np.ndarray, window_size: Tuple[int,...], stride: Tuple[int,...], axis: Tuple[int,...]) -> np.ndarray:
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    for i in range(len(axis)):
        x = sliding_window_1d(x, window_size[i], stride[i], axis[i])
    return x

def sliding_window_2d(x: np.ndarray, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, axis)

def max_pool_nd(x: np.ndarray, kernel_size: Tuple[int,...], stride: Tuple[int,...], padding: Tuple[int,...], axis: Tuple[int,...]) -> np.ndarray:
    for i in range(len(axis)):
        x = max_pool_1d(x, kernel_size[i], stride[i], padding[i], axis[i])
    return x

def max_pool_2d(x: np.ndarray, kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], axis: Tuple[int, int] = (-2, -1)):
    if isinstance(kernel_size, Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, Number):
        stride = (stride, stride)
    if isinstance(padding, Number):
        padding = (padding, padding)
    axis = tuple(axis)
    return max_pool_nd(x, kernel_size, stride, padding, axis)


def _last2d_filter_size(ndim: int, kernel_size: int) -> tuple[int, ...]:
    if ndim < 2:
        raise ValueError(f"Expected array with at least 2 dims, got ndim={ndim}")
    return (1,) * (ndim - 2) + (int(kernel_size), int(kernel_size))

def depth_edge(depth: np.ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: np.ndarray = None) -> np.ndarray:
    """
    Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth.
    
    Args:
        depth (np.ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    filter_size = _last2d_filter_size(depth.ndim, kernel_size)
    if mask is None:
        local_max = ndimage.maximum_filter(depth, size=filter_size, mode="constant", cval=-np.inf)
        local_min = ndimage.minimum_filter(depth, size=filter_size, mode="constant", cval=np.inf)
    else:
        local_max = ndimage.maximum_filter(
            np.where(mask, depth, -np.inf),
            size=filter_size,
            mode="constant",
            cval=-np.inf,
        )
        local_min = ndimage.minimum_filter(
            np.where(mask, depth, np.inf),
            size=filter_size,
            mode="constant",
            cval=np.inf,
        )

    diff = local_max - local_min

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol
    
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge

def normals_edge(normals: np.ndarray, tol: float, kernel_size: int = 3, mask: np.ndarray = None) -> np.ndarray:
    """
    Compute the edge mask from normal map.

    Args:
        normal (np.ndarray): shape (..., height, width, 3), normal map
        tol (float): tolerance in degrees
   
    Returns:
        edge (np.ndarray): shape (..., height, width) of dtype torch.bool
    """
    assert normals.ndim >= 3 and normals.shape[-1] == 3, "normal should be of shape (..., height, width, 3)"
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)

    height, width = normals.shape[-3:-1]
    padding = kernel_size // 2
    normals_pad = np.pad(
        normals,
        (*([(0, 0)] * (normals.ndim - 3)), (padding, padding), (padding, padding), (0, 0)),
        mode="edge",
    )
    mask_pad = None
    if mask is not None:
        mask_pad = np.pad(
            mask,
            (*([(0, 0)] * (mask.ndim - 2)), (padding, padding), (padding, padding)),
            mode="edge",
        )

    cos_tol = np.cos(np.deg2rad(tol))
    base_edge = np.zeros(normals.shape[:-1], dtype=bool)
    for dy in range(kernel_size):
        y_slice = slice(dy, dy + height)
        for dx in range(kernel_size):
            x_slice = slice(dx, dx + width)
            neighbor_normals = normals_pad[..., y_slice, x_slice, :]
            dot = (normals * neighbor_normals).sum(axis=-1)
            neighbor_edge = (dot >= -1.0) & (dot <= 1.0) & (dot < cos_tol)
            if mask_pad is not None:
                neighbor_edge &= mask_pad[..., y_slice, x_slice]
            base_edge |= neighbor_edge

    edge = ndimage.maximum_filter(
        base_edge,
        size=_last2d_filter_size(base_edge.ndim, kernel_size),
        mode="constant",
        cval=False,
    )
    return edge.astype(bool, copy=False)

def _accumulate_normalized_cross(
    accum: np.ndarray,
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    valid: np.ndarray,
) -> None:
    cross = np.cross(vec_a, vec_b, axis=-1)
    cross /= np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-12
    accum += cross * valid[..., None]

def points_to_normals(point: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Calculate normal map from point map. Value range is [-1, 1]. Normal direction in OpenGL identity camera's coordinate system.

    Args:
        point (np.ndarray): shape (height, width, 3), point map
    Returns:
        normal (np.ndarray): shape (height, width, 3), normal map. 
    """
    height, width = point.shape[-3:-1]
    has_mask = mask is not None

    if mask is None:
        mask = np.ones_like(point[..., 0], dtype=bool)
    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask
    mask = mask_pad

    pts = np.zeros((height + 2, width + 2, 3), dtype=point.dtype)
    pts[1:-1, 1:-1, :] = point
    center = pts[1:-1, 1:-1, :]
    up = pts[:-2, 1:-1, :] - center
    left = pts[1:-1, :-2, :] - center
    down = pts[2:, 1:-1, :] - center
    right = pts[1:-1, 2:, :] - center

    center_mask = mask[1:-1, 1:-1]
    up_mask = mask[:-2, 1:-1]
    left_mask = mask[1:-1, :-2]
    down_mask = mask[2:, 1:-1]
    right_mask = mask[1:-1, 2:]

    normal = np.zeros_like(center)
    valid_any = np.zeros((height, width), dtype=bool) if has_mask else None

    valid = center_mask & up_mask & left_mask
    _accumulate_normalized_cross(normal, up, left, valid)
    if has_mask:
        valid_any |= valid

    valid = center_mask & left_mask & down_mask
    _accumulate_normalized_cross(normal, left, down, valid)
    if has_mask:
        valid_any |= valid

    valid = center_mask & down_mask & right_mask
    _accumulate_normalized_cross(normal, down, right, valid)
    if has_mask:
        valid_any |= valid

    valid = center_mask & right_mask & up_mask
    _accumulate_normalized_cross(normal, right, up, valid)
    if has_mask:
        valid_any |= valid

    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)
    
    if has_mask:
        normal_mask = valid_any
        normal = np.where(normal_mask[..., None], normal, 0)
        return normal, normal_mask
    else:
        return normal
