from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_TRAJ_UVD_GATE_UV_MEAN_THRESHOLD_PX = 3.0
DEFAULT_TRAJ_UVD_GATE_DEPTH_STD_THRESHOLD_M = 0.01
DEFAULT_TRAJ_UVD_GATE_MAX_DEPTH_THRESHOLD_M = 1.5
DEFAULT_TRAJ_UVD_GATE_NEAR_DEPTH_THRESHOLD_M = 0.0
DEFAULT_TRAJ_UVD_GATE_NEAR_DEPTH_RELAXED_STD_THRESHOLD_M = 0.0
DEFAULT_TRAJ_UVD_GATE_NEAR_DEPTH_EXEMPT_THRESHOLD_M = 0.0


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {"finite_count": 0, "median": None, "p95": None, "max": None}
    valid = arr[finite].astype(np.float64)
    return {
        "finite_count": int(valid.size),
        "median": float(np.median(valid)),
        "p95": float(np.percentile(valid, 95)),
        "max": float(np.max(valid)),
    }


def _nanmean_per_track(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    finite_mask = np.isfinite(values)
    counts = finite_mask.sum(axis=1).astype(np.uint16)
    result = np.full(values.shape[0], np.nan, dtype=np.float32)
    valid_tracks = counts > 0
    if np.any(valid_tracks):
        result[valid_tracks] = np.nanmean(values[valid_tracks], axis=1).astype(np.float32)
    return result, counts


def _nanstd_per_track(values: np.ndarray, *, ddof: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    finite_mask = np.isfinite(values)
    counts = finite_mask.sum(axis=1).astype(np.uint16)
    result = np.full(values.shape[0], np.nan, dtype=np.float32)
    valid_tracks = counts > ddof
    if np.any(valid_tracks):
        result[valid_tracks] = np.nanstd(values[valid_tracks], axis=1, ddof=ddof).astype(np.float32)
    return result, counts


def _nanmax_per_track(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    finite_mask = np.isfinite(values)
    counts = finite_mask.sum(axis=1).astype(np.uint16)
    result = np.full(values.shape[0], np.nan, dtype=np.float32)
    valid_tracks = counts > 0
    if np.any(valid_tracks):
        result[valid_tracks] = np.nanmax(values[valid_tracks], axis=1).astype(np.float32)
    return result, counts


def compute_traj_uvd_motion_gate(
    traj_uvz: np.ndarray,
    *,
    uv_mean_threshold_px: float = DEFAULT_TRAJ_UVD_GATE_UV_MEAN_THRESHOLD_PX,
    depth_std_threshold_m: float = DEFAULT_TRAJ_UVD_GATE_DEPTH_STD_THRESHOLD_M,
    max_depth_threshold_m: float = DEFAULT_TRAJ_UVD_GATE_MAX_DEPTH_THRESHOLD_M,
    near_depth_threshold_m: float = DEFAULT_TRAJ_UVD_GATE_NEAR_DEPTH_THRESHOLD_M,
    near_depth_relaxed_std_threshold_m: float = DEFAULT_TRAJ_UVD_GATE_NEAR_DEPTH_RELAXED_STD_THRESHOLD_M,
    near_depth_exempt_threshold_m: float = DEFAULT_TRAJ_UVD_GATE_NEAR_DEPTH_EXEMPT_THRESHOLD_M,
    depth_std_ddof: int = 1,
) -> dict[str, Any]:
    traj_uvz = np.asarray(traj_uvz, dtype=np.float32)
    if traj_uvz.ndim != 3 or traj_uvz.shape[-1] != 3:
        raise ValueError(f"Expected traj_uvz shape (N,T,3), got {traj_uvz.shape}")
    if float(uv_mean_threshold_px) < 0.0:
        raise ValueError(f"Expected uv_mean_threshold_px >= 0, got {uv_mean_threshold_px}")
    if float(depth_std_threshold_m) < 0.0:
        raise ValueError(f"Expected depth_std_threshold_m >= 0, got {depth_std_threshold_m}")
    if float(max_depth_threshold_m) < 0.0:
        raise ValueError(f"Expected max_depth_threshold_m >= 0, got {max_depth_threshold_m}")
    if float(near_depth_threshold_m) < 0.0:
        raise ValueError(f"Expected near_depth_threshold_m >= 0, got {near_depth_threshold_m}")
    if float(near_depth_relaxed_std_threshold_m) < 0.0:
        raise ValueError(
            "Expected near_depth_relaxed_std_threshold_m >= 0, "
            f"got {near_depth_relaxed_std_threshold_m}"
        )
    if float(near_depth_exempt_threshold_m) < 0.0:
        raise ValueError(
            "Expected near_depth_exempt_threshold_m >= 0, "
            f"got {near_depth_exempt_threshold_m}"
        )
    if (
        float(near_depth_relaxed_std_threshold_m) > 0.0
        and float(near_depth_relaxed_std_threshold_m) < float(depth_std_threshold_m)
    ):
        raise ValueError(
            "Expected near_depth_relaxed_std_threshold_m to be >= depth_std_threshold_m "
            f"when enabled, got {near_depth_relaxed_std_threshold_m} < {depth_std_threshold_m}"
        )

    track_count, frame_count, _ = traj_uvz.shape
    if frame_count <= 1:
        empty_counts = np.zeros(track_count, dtype=np.uint16)
        nan_tracks = np.full(track_count, np.nan, dtype=np.float32)
        false_mask = np.zeros(track_count, dtype=bool)
        reliable_track_mask = np.ones(track_count, dtype=bool)
        return {
            "track_count": int(track_count),
            "frame_count": int(frame_count),
            "uv_mean_threshold_px": float(uv_mean_threshold_px),
            "depth_std_threshold_m": float(depth_std_threshold_m),
            "max_depth_threshold_m": float(max_depth_threshold_m),
            "near_depth_threshold_m": float(near_depth_threshold_m),
            "near_depth_relaxed_std_threshold_m": float(near_depth_relaxed_std_threshold_m),
            "near_depth_exempt_threshold_m": float(near_depth_exempt_threshold_m),
            "depth_std_ddof": int(depth_std_ddof),
            "uv_mean_delta_px": nan_tracks.copy(),
            "depth_delta_std_m": nan_tracks.copy(),
            "max_depth_m": _nanmax_per_track(traj_uvz[..., 2])[0].astype(np.float32, copy=False),
            "effective_depth_std_threshold_m": np.full(
                track_count,
                float(depth_std_threshold_m),
                dtype=np.float32,
            ),
            "uv_pair_valid_count": empty_counts.copy(),
            "depth_pair_valid_count": empty_counts.copy(),
            "depth_valid_count": empty_counts.copy(),
            "near_depth_relaxed_mask": false_mask.copy(),
            "near_depth_exempt_mask": false_mask.copy(),
            "uv_depth_anomaly_mask": false_mask.copy(),
            "far_depth_mask": false_mask.copy(),
            "removed_track_mask": false_mask.copy(),
            "reliable_track_mask": reliable_track_mask,
            "summary": {
                "track_count": int(track_count),
                "removed_track_count": 0,
                "near_depth_relaxed_count": 0,
                "near_depth_exempt_count": 0,
                "uv_depth_anomaly_count": 0,
                "far_depth_count": 0,
                "uv_mean_delta_px_summary": _finite_summary(nan_tracks),
                "depth_delta_std_m_summary": _finite_summary(nan_tracks),
                "max_depth_m_summary": _finite_summary(_nanmax_per_track(traj_uvz[..., 2])[0]),
            },
        }

    delta_uvz = traj_uvz[:, 1:, :] - traj_uvz[:, :-1, :]

    uv_pair_valid = np.isfinite(traj_uvz[:, 1:, 0:2]).all(axis=-1) & np.isfinite(traj_uvz[:, :-1, 0:2]).all(axis=-1)
    delta_uv = np.full((track_count, frame_count - 1), np.nan, dtype=np.float32)
    if np.any(uv_pair_valid):
        delta_uv[uv_pair_valid] = np.linalg.norm(delta_uvz[:, :, 0:2][uv_pair_valid], axis=-1).astype(np.float32)
    uv_mean_delta_px, uv_pair_valid_count = _nanmean_per_track(delta_uv)

    depth_pair_valid = np.isfinite(traj_uvz[:, 1:, 2]) & np.isfinite(traj_uvz[:, :-1, 2])
    delta_depth = np.full((track_count, frame_count - 1), np.nan, dtype=np.float32)
    if np.any(depth_pair_valid):
        delta_depth[depth_pair_valid] = np.abs(delta_uvz[:, :, 2][depth_pair_valid]).astype(np.float32)
    depth_delta_std_m, depth_pair_valid_count = _nanstd_per_track(delta_depth, ddof=int(depth_std_ddof))

    max_depth_m, depth_valid_count = _nanmax_per_track(traj_uvz[..., 2])
    effective_depth_std_threshold_m = np.full(track_count, float(depth_std_threshold_m), dtype=np.float32)
    near_depth_relaxed_mask = np.zeros(track_count, dtype=bool)
    if float(near_depth_threshold_m) > 0.0 and float(near_depth_relaxed_std_threshold_m) > 0.0:
        near_depth_relaxed_mask = np.isfinite(max_depth_m) & (max_depth_m < float(near_depth_threshold_m))
        effective_depth_std_threshold_m[near_depth_relaxed_mask] = float(near_depth_relaxed_std_threshold_m)
    near_depth_exempt_mask = np.zeros(track_count, dtype=bool)
    if float(near_depth_exempt_threshold_m) > 0.0:
        near_depth_exempt_mask = np.isfinite(max_depth_m) & (max_depth_m < float(near_depth_exempt_threshold_m))

    uv_depth_anomaly_mask = (
        np.isfinite(uv_mean_delta_px)
        & np.isfinite(depth_delta_std_m)
        & (uv_mean_delta_px < float(uv_mean_threshold_px))
        & (depth_delta_std_m > effective_depth_std_threshold_m)
        & (~near_depth_exempt_mask)
    )
    far_depth_mask = np.isfinite(max_depth_m) & (max_depth_m > float(max_depth_threshold_m))
    removed_track_mask = uv_depth_anomaly_mask | far_depth_mask
    reliable_track_mask = ~removed_track_mask

    summary = {
        "track_count": int(track_count),
        "removed_track_count": int(np.count_nonzero(removed_track_mask)),
        "near_depth_relaxed_count": int(np.count_nonzero(near_depth_relaxed_mask)),
        "near_depth_exempt_count": int(np.count_nonzero(near_depth_exempt_mask)),
        "uv_depth_anomaly_count": int(np.count_nonzero(uv_depth_anomaly_mask)),
        "far_depth_count": int(np.count_nonzero(far_depth_mask)),
        "uv_mean_delta_px_summary": _finite_summary(uv_mean_delta_px),
        "depth_delta_std_m_summary": _finite_summary(depth_delta_std_m),
        "max_depth_m_summary": _finite_summary(max_depth_m),
    }

    return {
        "track_count": int(track_count),
        "frame_count": int(frame_count),
        "uv_mean_threshold_px": float(uv_mean_threshold_px),
        "depth_std_threshold_m": float(depth_std_threshold_m),
        "max_depth_threshold_m": float(max_depth_threshold_m),
        "near_depth_threshold_m": float(near_depth_threshold_m),
        "near_depth_relaxed_std_threshold_m": float(near_depth_relaxed_std_threshold_m),
        "near_depth_exempt_threshold_m": float(near_depth_exempt_threshold_m),
        "depth_std_ddof": int(depth_std_ddof),
        "uv_mean_delta_px": uv_mean_delta_px.astype(np.float32, copy=False),
        "depth_delta_std_m": depth_delta_std_m.astype(np.float32, copy=False),
        "max_depth_m": max_depth_m.astype(np.float32, copy=False),
        "effective_depth_std_threshold_m": effective_depth_std_threshold_m.astype(np.float32, copy=False),
        "uv_pair_valid_count": uv_pair_valid_count.astype(np.uint16, copy=False),
        "depth_pair_valid_count": depth_pair_valid_count.astype(np.uint16, copy=False),
        "depth_valid_count": depth_valid_count.astype(np.uint16, copy=False),
        "near_depth_relaxed_mask": near_depth_relaxed_mask.astype(bool, copy=False),
        "near_depth_exempt_mask": near_depth_exempt_mask.astype(bool, copy=False),
        "uv_depth_anomaly_mask": uv_depth_anomaly_mask.astype(bool, copy=False),
        "far_depth_mask": far_depth_mask.astype(bool, copy=False),
        "removed_track_mask": removed_track_mask.astype(bool, copy=False),
        "reliable_track_mask": reliable_track_mask.astype(bool, copy=False),
        "summary": summary,
    }


def summarize_traj_uvd_motion_gate(
    *,
    gate_result: dict[str, Any],
    traj_valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    reliable_track_mask = np.asarray(gate_result["reliable_track_mask"], dtype=bool)
    removed_track_mask = np.asarray(gate_result["removed_track_mask"], dtype=bool)
    uv_depth_anomaly_mask = np.asarray(gate_result["uv_depth_anomaly_mask"], dtype=bool)
    far_depth_mask = np.asarray(gate_result["far_depth_mask"], dtype=bool)
    uv_mean_delta_px = np.asarray(gate_result["uv_mean_delta_px"], dtype=np.float32)
    depth_delta_std_m = np.asarray(gate_result["depth_delta_std_m"], dtype=np.float32)
    max_depth_m = np.asarray(gate_result["max_depth_m"], dtype=np.float32)

    if traj_valid_mask is None:
        effective_valid_mask = np.ones(reliable_track_mask.shape[0], dtype=bool)
    else:
        effective_valid_mask = np.asarray(traj_valid_mask, dtype=bool).reshape(-1)
        if effective_valid_mask.shape != reliable_track_mask.shape:
            raise ValueError(
                f"Expected traj_valid_mask shape {reliable_track_mask.shape}, got {effective_valid_mask.shape}"
            )

    removed_valid_mask = effective_valid_mask & removed_track_mask
    return {
        "track_count": int(reliable_track_mask.shape[0]),
        "valid_track_count": int(np.count_nonzero(effective_valid_mask)),
        "reliable_track_count": int(np.count_nonzero(reliable_track_mask)),
        "removed_track_count": int(np.count_nonzero(removed_track_mask)),
        "removed_valid_count": int(np.count_nonzero(removed_valid_mask)),
        "uv_depth_anomaly_count": int(np.count_nonzero(uv_depth_anomaly_mask)),
        "far_depth_count": int(np.count_nonzero(far_depth_mask)),
        "removed_valid_uv_mean_delta_px_summary": _finite_summary(uv_mean_delta_px[removed_valid_mask]),
        "removed_valid_depth_delta_std_m_summary": _finite_summary(depth_delta_std_m[removed_valid_mask]),
        "removed_valid_max_depth_m_summary": _finite_summary(max_depth_m[removed_valid_mask]),
    }
