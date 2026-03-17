import os
import re
import numpy as np
import cv2
import mediapy as media
import torch
from PIL import Image
import math
import tqdm
import glob
from rich import print
import argparse
from loguru import logger
import json
import sys
from pathlib import Path

from utils.video_depth_pose_utils import video_depth_pose_dict
from utils.traceforge_artifact_utils import (
    LEGACY_LAYOUT,
    V2_LAYOUT,
    is_traceforge_output_complete,
    path_kind,
    write_scene_h5,
    write_scene_meta,
    write_scene_rgb_mp4,
)

from datasets.data_ops import _filter_one_depth
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from utils.inference_utils import load_model, inference
from utils.threed_utils import (
    project_tracks_3d_to_2d,
    project_tracks_3d_to_3d,
)
from utils.traj_filter_utils import build_traj_filter_result, compute_depth_volatility_map

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to video directory (for batch processing) or single video folder",
    )
    parser.add_argument(
        "--depth_path",
        type=str,
        default=None,
        help="Path to depth directory (if known depth is provided) for batch processing or single video folder",
    )
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/tapip3d_final.pth"
    )
    parser.add_argument('--depth_pose_method', type=str, default='vggt4', choices=video_depth_pose_dict.keys())
    parser.add_argument(
        "--external_extr_mode",
        type=str,
        default="w2c",
        choices=["w2c", "c2w"],
        help="For depth_pose_method='external': how to interpret extrinsics in external_geom_npz. "
             "'w2c' means matrices map world→camera (TraceForge default, will be used directly); "
             "'c2w' means matrices map camera→world and will be inverted internally to obtain w2c.",
    )
    parser.add_argument(
        "--external_geom_npz",
        type=str,
        default=None,
        help=(
            "Path to NPZ/H5 with external geometry. "
            "For --depth_pose_method external: uses external intrinsics+extrinsics and skips VGGT. "
            "For --depth_pose_method vggt4: only replaces VGGT extrinsics."
        ),
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="hand_camera",
        help=(
            "Camera name for H5 file (e.g., hand_camera, varied_camera_1). "
            "Used with --external_geom_npz."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_iters", type=int, default=6)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument(
        "--video_name",
        type=str,
        default=None,
        help="Optional output video name override. If set, output folder/file names use this value.",
    )
    parser.add_argument("--max_num_frames", type=int, default=384)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument(
        "--output_layout",
        type=str,
        default=V2_LAYOUT,
        choices=[V2_LAYOUT, LEGACY_LAYOUT],
        help="Artifact layout: v2(scene cache + samples) or legacy(main NPZ + query frame images/depth).",
    )
    parser.add_argument(
        "--save_visibility",
        action="store_true",
        default=False,
        help="Store per-query visibility arrays in sample NPZ files.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
        help="Trajectory horizon length for each sample",
    )
    parser.add_argument(
        "--batch_process",
        action="store_true",
        default=False,
        help="Process all video folders in the given directory",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip processing if output already exists",
    )
    parser.add_argument(
        "--use_all_trajectories",
        action="store_true",
        default=True,
        help="Include all visible trajectories in each frame (default: True)",
    )
    parser.add_argument(
        "--frame_drop_rate",
        type=int,
        default=1,
        help="Query uniform grid points every N frames (default: 1, query every frame)",
    )
    parser.add_argument(
        "--scan_depth",
        type=int,
        default=2,  # default depth changed to 2
        help="How many directory levels below --video_path to scan for subfolders "
            "when --batch_process is enabled. Default is 2 (e.g., P02_02_01)."
    )
    parser.add_argument(
        "--future_len",
        type=int,
        default=128,
        help="Tracking window length (number of frames) per query frame in offline mode",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=50,
        help="Target max frames to keep per episode. If --fps <= 0, use stride = ceil(N / max_frames_per_video).",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=20,
        help="Grid size for uniform keypoint sampling (grid_size x grid_size points per frame). Default is 20 (400 points).",
    )
    parser.add_argument(
        "--filter_level",
        type=str,
        default="standard",
        choices=["none", "basic", "standard", "strict"],
        help="Trajectory filtering level: none(no filter), basic(basic checks), standard(recommended), strict(high quality)",
    )
    parser.add_argument(
        "--traj_filter_profile",
        type=str,
        default="external",
        choices=["external", "external_manipulator", "external_manipulator_v2", "wrist", "wrist_manipulator"],
        help=(
            "Trajectory filter profile: external keeps the current strict full-track mask; "
            "external_manipulator keeps external as the seed and then prunes to manipulator-like tracks; "
            "external_manipulator_v2 is a looser external manipulator profile that keeps major manipulator components; "
            "wrist keeps partial trajectories with per-frame supervision masks; "
            "wrist_manipulator adds near-field and motion-aware pruning on top of wrist."
        ),
    )
    parser.add_argument(
        "--min_valid_frames",
        type=int,
        default=None,
        help="Minimum valid frames per trajectory (overrides filter_level default)",
    )
    parser.add_argument(
        "--visibility_threshold",
        type=float,
        default=None,
        help="Minimum visibility ratio (overrides filter_level default)",
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.01,
        help="Minimum depth value in meters",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=10.0,
        help="Maximum depth value in meters",
    )
    parser.add_argument(
        "--boundary_margin",
        type=int,
        default=None,
        help="Projection boundary margin in pixels (overrides filter_level default)",
    )
    parser.add_argument(
        "--depth_change_threshold",
        type=float,
        default=None,
        help="Depth change std threshold in meters (overrides filter_level default)",
    )
    return parser.parse_args()

def retarget_trajectories(
    trajectory: np.ndarray,
    interval: float = 0.05,
    max_length: int = 64,
    top_percent: float = 0.02,
):
    """
    Dormant helper for synchronous arc-length retargeting.

    This helper is intentionally kept for reference, but the current TraceForge
    inference/save path does not invoke it.

    Steps:
      1) Global normalize x,y by (trajectory[-1,0,0], trajectory[-1,0,1]), then clip x,y to [0,1].
      2) For each time segment t: compute lengths for all tracks; take mean of top `top_percent`
         → robust_seglen[t].
      3) Build cumulative arc-length from robust_seglen and place targets every `interval`.
         (Long segments get subdivided; short ones merge implicitly.)
      4) For each target in segment t with fraction alpha, interpolate *all* tracks
         between frames t and t+1 with the same alpha (synchronous).
      5) Denormalize x,y only; z (if present) is linearly interpolated without scaling.

    Args:
        trajectory: (N, H, D) with D in {2,3}
        interval: target arc-length step
        max_length: output max length
        top_percent: fraction (0,1] for robust top-k mean per segment (e.g., 0.02 = top 2%)

    Returns:
        retargeted: (N, max_length, D), padded with -np.inf
        valid_mask: (max_length) bool
    """
    assert trajectory.ndim == 3, "trajectory must be (N, H, D)"
    N, H, D = trajectory.shape
    assert D in (2, 3), "D must be 2 or 3"
    if not (0 < top_percent <= 1.0):
        raise ValueError("top_percent must be in (0, 1].")
    if interval <= 0:
        raise ValueError("interval must be > 0")
    if H < 2:
        # If H==1, there is no segment to interpolate → return only the first frame
        ret = np.full((N, max_length, D), -np.inf, dtype=trajectory.dtype)
        mask = np.zeros((max_length), dtype=bool)
        ret[:, 0, :] = trajectory[:, 0, :]
        mask[0] = True
        return ret, mask

    eps = 1e-12

    # ---- 1) Global normalization (x,y) & clipping ----
    scale_x = float(trajectory[-1, 0, 0])
    scale_y = float(trajectory[-1, 0, 1])
    if abs(scale_x) < eps: scale_x = 1.0
    if abs(scale_y) < eps: scale_y = 1.0

    traj_norm = trajectory.astype(np.float64, copy=True)
    traj_norm[:, :, 0] /= scale_x
    traj_norm[:, :, 1] /= scale_y
    # clip x,y to [0,1]
    np.clip(traj_norm[:, :, 0], 0.0, 1.0, out=traj_norm[:, :, 0])
    np.clip(traj_norm[:, :, 1], 0.0, 1.0, out=traj_norm[:, :, 1])
    # z is not scaled/clipped

    # ---- 2) Robust length per segment t: mean of top k% ----
    # seglens_all: (N, H-1)
    diffs_all = traj_norm[:, 1:, :] - traj_norm[:, :-1, :]
    seglens_all = np.linalg.norm(diffs_all, axis=2)

    k = max(1, int(np.ceil(top_percent * N)))
    # Use np.partition to get per-segment (column-wise) top-k without full sorting
    # Values below index N-k are smaller; values at/above are larger
    part = np.partition(seglens_all, N - k, axis=0)      # (N, H-1)
    topk = part[N - k:, :]                                # (k, H-1)
    robust_seglen = topk.mean(axis=0)                     # (H-1,)

    total_len = float(robust_seglen.sum())
    # Output buffers
    retargeted = np.full((N, max_length, D), -np.inf, dtype=trajectory.dtype)
    valid_mask = np.zeros((max_length), dtype=bool)

    # ---- 3) Create targets at 'interval' along the robust cumulative length ----
    k_max = int(np.floor(total_len / interval))
    num_samples = min(k_max + 1, max_length)
    targets = interval * np.arange(num_samples, dtype=np.float64)
    targets[-1] = min(targets[-1], total_len)

    # Cumulative length s (vertex-based): s[0]=0, s[i]=sum_{j<i} robust_seglen[j]
    s = np.zeros((H,), dtype=np.float64)
    s[1:] = np.cumsum(robust_seglen, dtype=np.float64)

    # Segment index and in-segment fraction alpha for each target
    idx_seq = np.searchsorted(s, targets, side='right') - 1   # (num_samples,)
    idx_seq = np.clip(idx_seq, 0, H - 2)
    denom = np.maximum(robust_seglen[idx_seq], eps)           # (num_samples,)
    alpha = (targets - s[idx_seq]) / denom                    # (num_samples,)
    alpha_seq = alpha.reshape(-1, 1)                          # (num_samples,1)

    # ---- 4) Synchronous interpolation: apply the same (idx, alpha) to all tracks ----
    left = traj_norm[:, idx_seq, :]           # (N, num_samples, D)
    right = traj_norm[:, idx_seq + 1, :]      # (N, num_samples, D)
    samples_norm = left + alpha_seq[None, :, :] * (right - left)  # (N, num_samples, D)

    # ---- 5) Denormalize: scale only x,y back ----
    samples_out = samples_norm.astype(trajectory.dtype, copy=True)
    samples_out[:, :, 0] *= scale_x
    samples_out[:, :, 1] *= scale_y
    # Keep z as the linear interpolation result

    L = num_samples
    retargeted[:, :L, :] = samples_out
    valid_mask[:L] = True
    return retargeted, valid_mask


def filter_anomalous_trajectories(
    traj: np.ndarray,
    direction_threshold: float = 0.3,
    accel_threshold: float = 3.0,
) -> np.ndarray:
    """
    No-op placeholder for an older trajectory cleanup experiment.

    The historical implementation was intentionally removed from the live path to
    avoid leaving dead filtering logic in the file. Keep the helper so the call
    site remains explicit that no extra anomaly pruning is applied after
    projection.
    """
    del direction_threshold, accel_threshold
    return traj


def _tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _tensor_video_to_uint8_hwc(video_tensor):
    video_np = _tensor_to_numpy(video_tensor)
    if video_np.ndim != 4 or video_np.shape[1] not in (1, 3):
        raise ValueError(f"Expected video tensor with shape (T,3,H,W), got {video_np.shape}")
    video_np = np.clip(np.round(video_np * 255.0), 0, 255).astype(np.uint8)
    return video_np.transpose(0, 2, 3, 1)


def _build_grid_keypoints(frame_h: int, frame_w: int, grid_size: int) -> np.ndarray:
    y_coords = np.linspace(0, frame_h - 1, grid_size)
    x_coords = np.linspace(0, frame_w - 1, grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    return np.stack([xx.flatten(), yy.flatten()], axis=1).astype(np.float32)


def build_query_frame_sample_data(
    *,
    query_frame_idx: int,
    frame_data,
    grid_size: int,
    filter_args=None,
    save_visibility: bool = False,
    raw_depths_segment: np.ndarray | None = None,
    depth_volatility_map: np.ndarray | None = None,
):
    coords_np = _tensor_to_numpy(frame_data["coords"])
    visibs_np = _tensor_to_numpy(frame_data["visibs"])
    if visibs_np.ndim == 3 and visibs_np.shape[-1] == 1:
        visibs_np = visibs_np.squeeze(-1)
    video_segment = _tensor_video_to_uint8_hwc(frame_data["video_segment"])
    intrinsics_np = _tensor_to_numpy(frame_data["intrinsics_segment"]).astype(np.float32)
    extrinsics_np = _tensor_to_numpy(frame_data["extrinsics_segment"]).astype(np.float32)
    depths_segment = _tensor_to_numpy(frame_data["depths_segment"]).astype(np.float32)

    frame_h, frame_w = video_segment.shape[1:3]
    keypoints = _build_grid_keypoints(frame_h, frame_w, grid_size)
    query_frame_depth = depths_segment[0]
    fixed_camera_view = {
        "c2w": np.linalg.inv(extrinsics_np[0]),
        "K": intrinsics_np[0],
        "height": frame_h,
        "width": frame_w,
    }

    try:
        tracks2d_fixed = project_tracks_3d_to_2d(
            tracks3d=coords_np,
            camera_views=[fixed_camera_view] * len(coords_np),
        ).transpose(1, 0, 2).astype(np.float32)
        traj_uvz = project_tracks_3d_to_3d(
            tracks3d=coords_np,
            camera_views=[fixed_camera_view] * len(coords_np),
        ).transpose(1, 0, 2).astype(np.float32)
    except Exception as e:
        logger.error(f"Error projecting tracks for frame {query_frame_idx}: {e}")
        tracks2d_fixed = coords_np[:, :, :2].transpose(1, 0, 2).astype(np.float32)
        traj_uvz = coords_np.transpose(1, 0, 2).astype(np.float32)

    traj_uvz = filter_anomalous_trajectories(traj_uvz)
    traj_filter_result = build_traj_filter_result(
        traj=traj_uvz,
        visibs=visibs_np,
        image_width=frame_w,
        image_height=frame_h,
        filter_args=filter_args,
        keypoints=keypoints,
        query_depth=query_frame_depth,
        raw_depths_segment=raw_depths_segment,
        intrinsics_segment=intrinsics_np,
        extrinsics_segment=extrinsics_np,
        depth_volatility_map=depth_volatility_map,
    )
    sample_payload = {
        "traj_uvz": traj_uvz.astype(np.float32),
        "traj_2d": tracks2d_fixed.astype(np.float32),
        "keypoints": keypoints,
        "query_frame_index": np.array([query_frame_idx], dtype=np.int32),
        "segment_frame_indices": query_frame_idx + np.arange(traj_uvz.shape[1], dtype=np.int32),
        "traj_valid_mask": traj_filter_result["traj_valid_mask"].astype(bool),
        "traj_depth_consistency_ratio": traj_filter_result["traj_depth_consistency_ratio"].astype(np.float16),
        "traj_stable_depth_consistency_ratio": traj_filter_result["traj_stable_depth_consistency_ratio"].astype(
            np.float16
        ),
        "traj_high_volatility_hit": traj_filter_result["traj_high_volatility_hit"].astype(bool),
        "traj_volatility_exposure_ratio": traj_filter_result["traj_volatility_exposure_ratio"].astype(np.float16),
        "traj_compare_frame_count": traj_filter_result["traj_compare_frame_count"].astype(np.uint16),
        "traj_stable_compare_frame_count": traj_filter_result["traj_stable_compare_frame_count"].astype(np.uint16),
        "traj_mask_reason_bits": traj_filter_result["traj_mask_reason_bits"].astype(np.uint8),
        "traj_supervision_mask": traj_filter_result["traj_supervision_mask"].astype(bool),
        "traj_supervision_prefix_len": traj_filter_result["traj_supervision_prefix_len"].astype(np.uint16),
        "traj_supervision_count": traj_filter_result["traj_supervision_count"].astype(np.uint16),
        "traj_wrist_seed_mask": traj_filter_result["traj_wrist_seed_mask"].astype(bool),
        "traj_query_depth_rank": traj_filter_result["traj_query_depth_rank"].astype(np.float16),
        "traj_motion_extent": traj_filter_result["traj_motion_extent"].astype(np.float16),
        "traj_motion_step_median": traj_filter_result["traj_motion_step_median"].astype(np.float16),
        "traj_manipulator_candidate_mask": traj_filter_result["traj_manipulator_candidate_mask"].astype(bool),
        "traj_manipulator_cluster_id": traj_filter_result["traj_manipulator_cluster_id"].astype(np.int16),
        "traj_manipulator_component_size": traj_filter_result["traj_manipulator_component_size"].astype(
            np.uint16
        ),
        "traj_manipulator_cluster_fallback_used": np.asarray(
            traj_filter_result["traj_manipulator_cluster_fallback_used"], dtype=bool
        ),
    }
    if save_visibility:
        visibility = visibs_np
        if visibility.shape[0] == traj_uvz.shape[1] and visibility.shape[1] == traj_uvz.shape[0]:
            visibility = visibility.T
        sample_payload["visibility"] = visibility.astype(np.float16)

    return {
        "sample_payload": sample_payload,
        "query_frame_img": video_segment[0],
        "query_frame_depth": query_frame_depth,
    }


def save_single_query_frame_legacy(
    *,
    video_name,
    output_dir,
    query_frame_idx,
    frame_data,
    future_len: int,
    grid_size: int,
    filter_args=None,
    full_depths: np.ndarray | None = None,
    depth_volatility_map: np.ndarray | None = None,
):
    video_output_dir = os.path.join(output_dir, video_name)
    images_dir = os.path.join(video_output_dir, "images")
    depth_dir = os.path.join(video_output_dir, "depth")
    samples_dir = os.path.join(video_output_dir, "samples")
    for dir_path in [images_dir, depth_dir, samples_dir]:
        os.makedirs(dir_path, exist_ok=True)

    segment_len = int(frame_data["coords"].shape[0])
    raw_depths_segment = (
        np.asarray(full_depths[query_frame_idx : query_frame_idx + segment_len], dtype=np.float32)
        if full_depths is not None
        else None
    )
    bundle = build_query_frame_sample_data(
        query_frame_idx=query_frame_idx,
        frame_data=frame_data,
        grid_size=grid_size,
        filter_args=filter_args,
        save_visibility=getattr(filter_args, "save_visibility", False),
        raw_depths_segment=raw_depths_segment,
        depth_volatility_map=depth_volatility_map,
    )
    sample_payload = bundle["sample_payload"]
    traj = sample_payload["traj_uvz"]
    traj_2d = sample_payload["traj_2d"]
    traj_supervision_mask = sample_payload["traj_supervision_mask"]
    current_len = traj.shape[1]
    valid_steps = np.zeros(future_len, dtype=bool)
    valid_steps[:current_len] = True

    if current_len < future_len:
        pad_len = future_len - current_len
        traj = np.concatenate(
            [traj, np.full((traj.shape[0], pad_len, traj.shape[2]), np.inf, dtype=traj.dtype)],
            axis=1,
        )
        traj_2d = np.concatenate(
            [traj_2d, np.full((traj_2d.shape[0], pad_len, traj_2d.shape[2]), np.inf, dtype=traj_2d.dtype)],
            axis=1,
        )
        traj_supervision_mask = np.concatenate(
            [traj_supervision_mask, np.zeros((traj_supervision_mask.shape[0], pad_len), dtype=bool)],
            axis=1,
        )

    sample_data = {
        "image_path": np.array([f"images/{video_name}_{query_frame_idx}.png"], dtype="<U64"),
        "frame_index": np.array([query_frame_idx], dtype=np.int32),
        "keypoints": sample_payload["keypoints"],
        "traj": traj.astype(np.float32),
        "traj_2d": traj_2d.astype(np.float32),
        "traj_valid_mask": sample_payload["traj_valid_mask"],
        "traj_depth_consistency_ratio": sample_payload["traj_depth_consistency_ratio"],
        "traj_stable_depth_consistency_ratio": sample_payload["traj_stable_depth_consistency_ratio"],
        "traj_high_volatility_hit": sample_payload["traj_high_volatility_hit"],
        "traj_volatility_exposure_ratio": sample_payload["traj_volatility_exposure_ratio"],
        "traj_compare_frame_count": sample_payload["traj_compare_frame_count"],
        "traj_stable_compare_frame_count": sample_payload["traj_stable_compare_frame_count"],
        "traj_mask_reason_bits": sample_payload["traj_mask_reason_bits"],
        "traj_supervision_mask": traj_supervision_mask,
        "traj_supervision_prefix_len": sample_payload["traj_supervision_prefix_len"],
        "traj_supervision_count": sample_payload["traj_supervision_count"],
        "traj_wrist_seed_mask": sample_payload["traj_wrist_seed_mask"],
        "traj_query_depth_rank": sample_payload["traj_query_depth_rank"],
        "traj_motion_extent": sample_payload["traj_motion_extent"],
        "traj_motion_step_median": sample_payload["traj_motion_step_median"],
        "traj_manipulator_candidate_mask": sample_payload["traj_manipulator_candidate_mask"],
        "traj_manipulator_cluster_id": sample_payload["traj_manipulator_cluster_id"],
        "traj_manipulator_component_size": sample_payload["traj_manipulator_component_size"],
        "traj_manipulator_cluster_fallback_used": sample_payload["traj_manipulator_cluster_fallback_used"],
        "valid_steps": valid_steps,
    }
    if "visibility" in sample_payload:
        sample_data["visibility"] = sample_payload["visibility"]

    img_path = os.path.join(images_dir, f"{video_name}_{query_frame_idx}.png")
    Image.fromarray(bundle["query_frame_img"]).save(img_path)

    query_frame_depth = bundle["query_frame_depth"]
    depth_valid = query_frame_depth[query_frame_depth > 0]
    if len(depth_valid) > 0:
        depth_min = depth_valid.min()
        depth_max = depth_valid.max()
        if depth_max > depth_min:
            depth_normalized = ((query_frame_depth - depth_min) / (depth_max - depth_min) * 65535.0).clip(0, 65535).astype(np.uint16)
        else:
            depth_normalized = np.full_like(query_frame_depth, 32767, dtype=np.uint16)
    else:
        depth_normalized = np.zeros_like(query_frame_depth, dtype=np.uint16)
    Image.fromarray(depth_normalized, mode="I;16").save(
        os.path.join(depth_dir, f"{video_name}_{query_frame_idx}.png")
    )
    np.savez(os.path.join(depth_dir, f"{video_name}_{query_frame_idx}_raw.npz"), depth=query_frame_depth)
    np.savez(os.path.join(samples_dir, f"{video_name}_{query_frame_idx}.npz"), **sample_data)
    logger.info(f"Saved legacy query frame {query_frame_idx}")


def save_single_query_frame_v2(
    *,
    video_name,
    output_dir,
    query_frame_idx,
    frame_data,
    grid_size: int,
    filter_args=None,
    full_depths: np.ndarray | None = None,
    depth_volatility_map: np.ndarray | None = None,
):
    video_output_dir = os.path.join(output_dir, video_name)
    samples_dir = os.path.join(video_output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    segment_len = int(frame_data["coords"].shape[0])
    raw_depths_segment = (
        np.asarray(full_depths[query_frame_idx : query_frame_idx + segment_len], dtype=np.float32)
        if full_depths is not None
        else None
    )
    bundle = build_query_frame_sample_data(
        query_frame_idx=query_frame_idx,
        frame_data=frame_data,
        grid_size=grid_size,
        filter_args=filter_args,
        save_visibility=getattr(filter_args, "save_visibility", False),
        raw_depths_segment=raw_depths_segment,
        depth_volatility_map=depth_volatility_map,
    )
    sample_payload = bundle["sample_payload"]
    sample_data = {
        "traj_uvz": sample_payload["traj_uvz"],
        "keypoints": sample_payload["keypoints"],
        "query_frame_index": sample_payload["query_frame_index"],
        "segment_frame_indices": sample_payload["segment_frame_indices"],
        "traj_valid_mask": sample_payload["traj_valid_mask"],
        "traj_depth_consistency_ratio": sample_payload["traj_depth_consistency_ratio"],
        "traj_stable_depth_consistency_ratio": sample_payload["traj_stable_depth_consistency_ratio"],
        "traj_high_volatility_hit": sample_payload["traj_high_volatility_hit"],
        "traj_volatility_exposure_ratio": sample_payload["traj_volatility_exposure_ratio"],
        "traj_compare_frame_count": sample_payload["traj_compare_frame_count"],
        "traj_stable_compare_frame_count": sample_payload["traj_stable_compare_frame_count"],
        "traj_mask_reason_bits": sample_payload["traj_mask_reason_bits"],
        "traj_supervision_mask": sample_payload["traj_supervision_mask"],
        "traj_supervision_prefix_len": sample_payload["traj_supervision_prefix_len"],
        "traj_supervision_count": sample_payload["traj_supervision_count"],
        "traj_wrist_seed_mask": sample_payload["traj_wrist_seed_mask"],
        "traj_query_depth_rank": sample_payload["traj_query_depth_rank"],
        "traj_motion_extent": sample_payload["traj_motion_extent"],
        "traj_motion_step_median": sample_payload["traj_motion_step_median"],
        "traj_manipulator_candidate_mask": sample_payload["traj_manipulator_candidate_mask"],
        "traj_manipulator_cluster_id": sample_payload["traj_manipulator_cluster_id"],
        "traj_manipulator_component_size": sample_payload["traj_manipulator_component_size"],
        "traj_manipulator_cluster_fallback_used": sample_payload["traj_manipulator_cluster_fallback_used"],
    }
    if "visibility" in sample_payload:
        sample_data["visibility"] = sample_payload["visibility"]

    np.savez(os.path.join(samples_dir, f"{video_name}_{query_frame_idx}.npz"), **sample_data)
    logger.info(f"Saved v2 query frame {query_frame_idx}")


def save_structured_data(
    video_name,
    output_dir,
    video_tensor,
    depths,
    coords,
    visibs,
    intrinsics,
    extrinsics,
    query_points_per_frame,
    horizon,
    original_filenames,
    use_all_trajectories=True,
    query_frame_results=None,
    future_len: int = 128,
    grid_size: int = 20,
    filter_args=None,
    *,
    full_video_tensor=None,
    full_depths=None,
    full_intrinsics=None,
    full_extrinsics=None,
    depth_conf=None,
    video_source_path: str | None = None,
    depth_source_path: str | None = None,
):
    """Save TraceForge inference artifacts."""
    del intrinsics, extrinsics, query_points_per_frame, horizon, use_all_trajectories

    layout = getattr(filter_args, "output_layout", V2_LAYOUT) if filter_args is not None else V2_LAYOUT
    video_output_dir = Path(output_dir) / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    if query_frame_results is None:
        logger.warning(f"No query frame results to save for {video_name}")
        return

    logger.info(f"Saving {len(query_frame_results)} query frame results using layout={layout}")
    if layout == V2_LAYOUT:
        if full_video_tensor is None or full_depths is None or full_intrinsics is None or full_extrinsics is None:
            raise ValueError("v2 output requires full_video_tensor/full_depths/full_intrinsics/full_extrinsics")

        full_video_uint8 = _tensor_video_to_uint8_hwc(full_video_tensor)
        full_depths_np = _tensor_to_numpy(full_depths).astype(np.float32)
        full_intrinsics_np = _tensor_to_numpy(full_intrinsics).astype(np.float32)
        full_extrinsics_np = _tensor_to_numpy(full_extrinsics).astype(np.float32)
        depth_volatility_map = compute_depth_volatility_map(
            full_depths_np,
            min_depth=float(getattr(filter_args, "min_depth", 0.01)),
            max_depth=float(getattr(filter_args, "max_depth", 10.0)),
        )
        write_scene_h5(
            video_output_dir / "scene.h5",
            depths=full_depths_np,
            intrinsics=full_intrinsics_np,
            extrinsics_w2c=full_extrinsics_np,
        )
        write_scene_rgb_mp4(video_output_dir / "scene_rgb.mp4", video_frames=full_video_uint8, fps=10)
        write_scene_meta(
            video_output_dir / "scene_meta.json",
            {
                "layout_version": 2,
                "video_name": video_name,
                "frame_count": int(full_video_uint8.shape[0]),
                "height": int(full_video_uint8.shape[1]),
                "width": int(full_video_uint8.shape[2]),
                "extrinsics_mode": "w2c",
                "frame_drop_rate": int(getattr(filter_args, "frame_drop_rate", 1)),
                "future_len": int(future_len),
                "original_filenames": list(map(str, original_filenames)),
                "rgb_cache_path": "scene_rgb.mp4",
                "source_rgb_path": video_source_path,
                "source_rgb_kind": path_kind(video_source_path),
                "source_depth_path": depth_source_path,
                "source_depth_kind": path_kind(depth_source_path),
            },
        )
        for query_frame_idx, frame_data in query_frame_results.items():
            save_single_query_frame_v2(
                video_name=video_name,
                output_dir=output_dir,
                query_frame_idx=query_frame_idx,
                frame_data=frame_data,
                grid_size=grid_size,
                filter_args=filter_args,
                full_depths=full_depths_np,
                depth_volatility_map=depth_volatility_map,
            )
        return

    full_depths_np = _tensor_to_numpy(full_depths).astype(np.float32) if full_depths is not None else None
    depth_volatility_map = (
        compute_depth_volatility_map(
            full_depths_np,
            min_depth=float(getattr(filter_args, "min_depth", 0.01)),
            max_depth=float(getattr(filter_args, "max_depth", 10.0)),
        )
        if full_depths_np is not None
        else None
    )
    for query_frame_idx, frame_data in query_frame_results.items():
        save_single_query_frame_legacy(
            video_name=video_name,
            output_dir=output_dir,
            query_frame_idx=query_frame_idx,
            frame_data=frame_data,
            future_len=future_len,
            grid_size=grid_size,
            filter_args=filter_args,
            full_depths=full_depths_np,
            depth_volatility_map=depth_volatility_map,
        )

    main_npz = {
        "coords": _tensor_to_numpy(coords),
        "extrinsics": _tensor_to_numpy(full_extrinsics if full_extrinsics is not None else extrinsics),
        "intrinsics": _tensor_to_numpy(full_intrinsics if full_intrinsics is not None else intrinsics),
        "height": int(_tensor_to_numpy(video_tensor).shape[-2]),
        "width": int(_tensor_to_numpy(video_tensor).shape[-1]),
        "depths": _tensor_to_numpy(depths).astype(np.float16),
        "visibs": _tensor_to_numpy(visibs)[..., None],
    }
    if depth_conf is not None:
        main_npz["unc_metric"] = np.asarray(depth_conf).astype(np.float16)
    if getattr(filter_args, "save_video", False):
        main_npz["video"] = _tensor_to_numpy(video_tensor)
    save_path = video_output_dir / f"{video_name}.npz"
    np.savez(save_path, **main_npz)
    logger.info(f"Traditional visualization NPZ saved to {save_path}")


def process_single_video(video_path, depth_path, args, model_3dtracker, model_depth_pose, video_name=None, output_dir=None):
    """Process a single video and return the processed data"""
    logger.info(f"Processing video: {video_path}")

    # --- NEW: per-episode stride based on frame count when --fps <= 0 ---
    # If user set --fps > 0, use that fixed stride; otherwise auto-compute from N.
    if args.fps and int(args.fps) > 0:
        stride = int(args.fps)
        n_frames = 0  # unknown/not needed in fixed stride mode
    else:
        stride = 1
        n_frames = 0
        if os.path.isdir(video_path):
            # Count frames（与 load_video_and_mask 相同的收集与排序逻辑，保证 stride 一致）
            img_files = _collect_and_sort_frame_files(video_path, ["jpg", "jpeg", "png"])
            n_frames = len(img_files)

            # Auto stride: ceil(N / target), where target = --max_frames_per_video
            target = max(1, int(getattr(args, "max_frames_per_video", 150)))
            stride = max(1, math.ceil(n_frames / target)) if n_frames > 0 else 1
        else:
            # For video files (.mp4, etc.), we keep stride=1 (or you can extend to probe length)
            stride = 1

    logger.info(
        f"[{os.path.basename(video_path)}] frames={n_frames if n_frames else 'n/a'} "
        f"target={getattr(args, 'max_frames_per_video', 150)} -> stride={stride}"
    )

    # Load RGB with computed stride
    video_tensor, video_mask, original_filenames = load_video_and_mask(
        video_path, args.mask_dir, stride, args.max_num_frames
    )

    # Load depth (if provided) with the SAME stride and same frame order ( _frame_sort_key ) as RGB
    depth_tensor = None
    if depth_path is not None:
        depth_tensor, _, _ = load_video_and_mask(
            depth_path, None, stride, args.max_num_frames, is_depth=True
        )  # [T, H, W]
        if len(depth_tensor) != len(video_tensor):
            logger.warning(
                f"Depth frame count ({len(depth_tensor)}) != RGB frame count ({len(video_tensor)}); "
                "aligning to min length."
            )
            min_len = min(len(depth_tensor), len(video_tensor))
            depth_tensor = depth_tensor[:min_len]
            video_tensor = video_tensor[:min_len]
            original_filenames = original_filenames[:min_len]
        valid_depth = (depth_tensor > 0)
        depth_tensor[~valid_depth] = 0  # Invalidate bad depth values

    video_length = len(video_tensor)

    # 外部几何按 stride 采样对齐（外部几何通常是全量帧，需要和 RGB/深度同步）
    _original_extrs = None
    _original_intrs = None
    if hasattr(model_depth_pose, 'external_extrs') and model_depth_pose.external_extrs is not None and stride > 1:
        _original_extrs = model_depth_pose.external_extrs
        model_depth_pose.external_extrs = _original_extrs[::stride]
        logger.info(
            f"外部外参按 stride={stride} 采样: "
            f"{_original_extrs.shape[0]} -> {model_depth_pose.external_extrs.shape[0]} 帧"
        )
    if hasattr(model_depth_pose, 'external_intrs') and model_depth_pose.external_intrs is not None and stride > 1:
        _original_intrs = model_depth_pose.external_intrs
        model_depth_pose.external_intrs = _original_intrs[::stride]
        logger.info(
            f"外部内参按 stride={stride} 采样: "
            f"{_original_intrs.shape[0]} -> {model_depth_pose.external_intrs.shape[0]} 帧"
        )

    # obtain video depth and pose
    (
        video_ten, depth_npy, depth_conf, extrs_npy, intrs_npy
    ) = model_depth_pose(
        video_tensor,
        known_depth=depth_tensor,  # can be None
        stationary_camera=False,
        replace_with_known_depth=False,  # if known depth is given, always replace
    )

    # Keep depth_conf for visualization NPZ
    if isinstance(depth_conf, torch.Tensor):
        depth_conf_npy = depth_conf.squeeze().cpu().numpy()
    else:
        depth_conf_npy = np.asarray(depth_conf).squeeze()

    # 恢复外部几何（避免影响后续视频处理）
    if _original_extrs is not None:
        model_depth_pose.external_extrs = _original_extrs
    if _original_intrs is not None:
        model_depth_pose.external_intrs = _original_intrs

    video_length = video_ten.shape[0]
    if len(original_filenames) != video_length:
        logger.warning(
            f"原始文件名数量({len(original_filenames)})与有效帧数({video_length})不一致，"
            f"按前 {video_length} 帧对齐。"
        )
        original_filenames = original_filenames[:video_length]

    frame_H, frame_W = video_ten.shape[-2:]

    # Sample query points using uniform grid and store which frame they belong to
    query_points_per_frame = {}

    # Use uniform grid sampling (grid_size x grid_size points per frame)
    query_point = []
    tracking_segments = []  # Store info about which frames to track for each segment

    # Determine which frames to query based on frame_drop_rate
    query_frames = list(range(0, video_length, args.frame_drop_rate))
    logger.info(
        f"Using uniform grid sampling on frames: {query_frames} (frame_drop_rate={args.frame_drop_rate})"
    )
    logger.info(f"Tracking up to {args.future_len} frames from each query frame")

    for frame_idx in query_frames:
        # Calculate the end frame for this tracking segment (16 frames max)
        end_frame = min(frame_idx + args.future_len, video_length)
        tracking_segments.append((frame_idx, end_frame))

        # Create uniform grid for this frame (grid_size x grid_size points)
        grid_points = (
            create_uniform_grid_points(
                height=frame_H, width=frame_W, grid_size=args.grid_size, device="cpu"
            )
            .squeeze(0)
            .numpy()
        )  # Remove batch dimension and convert to numpy

        # Set the correct frame index for all points
        grid_points[:, 0] = frame_idx

        query_point.append(grid_points)

    # Group query points by frame
    for query_frame_points in query_point:
        if len(query_frame_points) > 0:
            frame_idx = int(query_frame_points[0, 0])
            points_xy = query_frame_points[:, 1:3]  # Extract x, y coordinates
            query_points_per_frame[frame_idx] = points_xy

    # All wrappers now return w2c directly; keep that convention unchanged.

    # Store results for each query frame
    query_frame_results = {}

    logger.info(f"Processing {len(tracking_segments)} independent tracking segments")

    for seg_idx, (start_frame, end_frame) in enumerate(tracking_segments):
        logger.info(
            f"Processing query frame {start_frame}: tracking {end_frame - start_frame} frames"
        )

        # Clear CUDA cache before each segment to avoid fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Extract video segment (16 frames starting from query frame)
        video_segment = video_ten[start_frame:end_frame]
        depth_segment = depth_npy[start_frame:end_frame]
        intrs_segment = intrs_npy[start_frame:end_frame]
        extrs_segment = extrs_npy[start_frame:end_frame]

        # Get query points for this segment (only from the starting frame)
        # Need to adjust the frame index to be relative to segment start (0)
        segment_query_point = [query_point[seg_idx].copy()]
        segment_query_point[0][:, 0] = 0  # Set frame index to 0 for segment start

        # Calculate support_grid_size proportionally to grid_size
        # Original: grid_size=20 -> support_grid_size=16 (ratio = 0.8)
        support_grid_size = int(args.grid_size * 0.8)
        
        video, depths, intrinsics, extrinsics, query_point_tensor, support_grid_size = (
            prepare_inputs(
                video_segment,
                depth_segment,
                intrs_segment,
                extrs_segment,
                segment_query_point,
                inference_res=(frame_H, frame_W),
                support_grid_size=support_grid_size,
                device=args.device,
            )
        )

        model_3dtracker.set_image_size((frame_H, frame_W))

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                coords_seg, visibs_seg = inference(
                    model=model_3dtracker,
                    video=video,
                    depths=depths,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    query_point=query_point_tensor,
                    num_iters=args.num_iters,
                    grid_size=support_grid_size,
                    bidrectional=False,  # Disable backward tracking
                )

        # Validate inference results before storing
        logger.debug(
            f"Query frame {start_frame}: coords_seg shape = {coords_seg.shape}, visibs_seg shape = {visibs_seg.shape}"
        )

        # Check if results have expected dimensions
        if len(coords_seg.shape) != 3 or len(visibs_seg.shape) != 2:
            logger.error(
                f"Query frame {start_frame}: Invalid result shapes - coords: {coords_seg.shape}, visibs: {visibs_seg.shape}"
            )
            continue

        # Check if we have the expected number of trajectories
        expected_trajectories = args.grid_size * args.grid_size
        if coords_seg.shape[1] != expected_trajectories:
            logger.warning(
                f"Query frame {start_frame}: Expected {expected_trajectories} trajectories, got {coords_seg.shape[1]}"
            )

        # Store results for this query frame (move to CPU to avoid GPU memory accumulation)
        query_frame_results[start_frame] = {
            "coords": coords_seg.cpu(),  # Shape: (T, grid_size*grid_size, 3)
            "visibs": visibs_seg.cpu(),  # Shape: (T, grid_size*grid_size)
            "video_segment": video.cpu(),
            "depths_segment": depths.cpu(),
            "intrinsics_segment": intrinsics.cpu(),
            "extrinsics_segment": extrinsics.cpu(),
        }

        logger.info(
            f"Query frame {start_frame}: tracked {coords_seg.shape[1]} trajectories for {coords_seg.shape[0]} frames"
        )

        # Clear cache after inference to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # For compatibility with the rest of the pipeline, use the first segment as the main result
    # But we'll save each segment independently in save_structured_data
    if query_frame_results:
        first_frame = min(query_frame_results.keys())
        coords = query_frame_results[first_frame]["coords"]
        visibs = query_frame_results[first_frame]["visibs"]
        video = query_frame_results[first_frame]["video_segment"]
        depths = query_frame_results[first_frame]["depths_segment"]
        intrinsics = query_frame_results[first_frame]["intrinsics_segment"]
        extrinsics = query_frame_results[first_frame]["extrinsics_segment"]
    else:
        flen = min(args.future_len, len(video_ten))
        coords = torch.empty((0, 0, 3))
        visibs = torch.empty((0, 0))
        video = video_ten[:flen]
        depths = torch.from_numpy(depth_npy[:flen]).float().to(args.device)
        intrinsics = torch.from_numpy(intrs_npy[:flen]).float().to(args.device)
        extrinsics = torch.from_numpy(extrs_npy[:flen]).float().to(args.device)

    # Validate tensor shapes after inference
    logger.debug(
        f"After inference - coords shape: {coords.shape}, visibs shape: {visibs.shape}"
    )

    # Ensure visibs has the expected dimensions
    if visibs.dim() == 3 and visibs.shape[-1] == 1:
        visibs = visibs.squeeze(-1)  # Remove last dimension if it's 1
        logger.debug(f"Squeezed visibs shape: {visibs.shape}")

    # Validate final shapes
    expected_frames = video.shape[0]
    expected_points = coords.shape[1] if coords.dim() >= 2 else 0
    if coords.dim() != 3 or visibs.dim() != 2:
        logger.error(
            f"Unexpected tensor dimensions - coords: {coords.shape}, visibs: {visibs.shape}"
        )
        raise ValueError(f"Invalid tensor shapes after inference")

    if coords.shape[0] != expected_frames or visibs.shape[0] != expected_frames:
        logger.error(
            f"Frame count mismatch - expected {expected_frames}, got coords: {coords.shape[0]}, visibs: {visibs.shape[0]}"
        )
        raise ValueError(f"Frame count mismatch in inference results")

    return {
        "video_tensor": video,
        "full_video_tensor": video_ten,
        "depths": depths,
        "full_depths": depth_npy.astype(np.float32),
        "coords": coords,
        "visibs": visibs,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "query_points_per_frame": query_points_per_frame,
        "original_filenames": original_filenames,
        "depth_conf": depth_conf_npy,
        "query_frame_results": query_frame_results,  # Add individual frame results
        "full_intrinsics": torch.from_numpy(intrs_npy)
        .float()
        .to(args.device),  # Full video intrinsics
        "full_extrinsics": torch.from_numpy(extrs_npy)
        .float()
        .to(args.device),  # Full video extrinsics
    }


def find_video_folders(base_path: str, scan_depth: int = 2):
    """
    Recursively scan subfolders up to a given depth and return inputs
    that contain images (.jpg/.jpeg/.png) or stand-alone video files
    (.mp4/.webm/etc.).

    Args:
        base_path: Root directory to scan
        scan_depth: Number of directory levels to traverse

    Returns:
        List of folder paths containing image files at the target depth
    """
    img_exts = (".jpg", ".jpeg", ".png")
    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg")

    # Normalize the base path
    base_path = os.path.abspath(base_path.rstrip(os.sep))
    base_depth = base_path.count(os.sep)
    target_depth = base_depth + scan_depth

    video_folders = []

    for root, dirs, files in os.walk(base_path):
        current_depth = os.path.abspath(root.rstrip(os.sep)).count(os.sep)

        # Skip folders above the target depth
        if current_depth < target_depth:
            continue

        # Select only folders/files exactly at the target depth
        if current_depth == target_depth:
            has_images = any(f.lower().endswith(img_exts) for f in files)
            if has_images:
                video_folders.append(root)
            # Also collect individual video files at this depth
            for f in files:
                if f.lower().endswith(video_exts):
                    video_folders.append(os.path.join(root, f))

        # Skip deeper folders for performance (no need to go further)
        if current_depth > target_depth:
            dirs[:] = []  # prevent os.walk from descending further

    # Deduplicate and sort for stable ordering
    video_folders = sorted(list(dict.fromkeys(video_folders)))
    return video_folders


def _frame_sort_key(path):
    """
    为帧文件路径生成排序键，自动兼容两种常见命名规则，按时间顺序排序：
    - 规则1：纯数字文件名，如 00000.png, 00001.png, 00100.png → 按数值 0,1,100 排序
    - 规则2：前缀+数字，如 im_0.jpg, im_1.jpg, im_10.jpg, frame_00001.png → 按末尾数字 0,1,10,1 排序
    避免字典序导致 im_10 排在 im_2 前。
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.isdigit():
        return (int(stem), path)
    numbers = re.findall(r"\d+", stem)
    if numbers:
        return (int(numbers[-1]), path)
    return (0, path)


def _collect_and_sort_frame_files(video_path, extensions):
    """收集目录下指定扩展名的帧文件，并按 _frame_sort_key 排序（与 RGB/深度 对齐）。"""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(video_path, f"*.{ext}")))

    # 兼容 DROID 相机目录结构：若相机根目录无帧文件，回退读取 left 子目录。
    if not files:
        left_dir = os.path.join(video_path, "left")
        if os.path.isdir(left_dir):
            for ext in extensions:
                files.extend(glob.glob(os.path.join(left_dir, f"*.{ext}")))

    files.sort(key=_frame_sort_key)
    return files


def load_video_and_mask(video_path, mask_dir=None, fps=1, max_num_frames=384, is_depth=False):
    original_filenames = []

    if os.path.isdir(video_path):
        # RGB 支持 jpg/jpeg/png；深度图通常为 png，统一用同一排序逻辑保证与 RGB 逐帧对齐
        exts = ["jpg", "jpeg", "png"] if not is_depth else ["npy", "png", "jpg", "jpeg"]
        img_files = _collect_and_sort_frame_files(video_path, exts)

        # IMPORTANT: Subsample the file list BEFORE loading to save memory
        img_files = img_files[::fps]
        if max_num_frames is not None and max_num_frames > 0:
            img_files = img_files[:max_num_frames]

        video_tensor = []
        for img_file in tqdm.tqdm(img_files, desc="Loading images" if not is_depth else "Loading depth"):
            if is_depth:
                # 支持 .npy 和 .png 两种深度格式
                if img_file.endswith('.npy'):
                    # .npy 格式：直接加载，假定单位已经是米 (m)
                    depth_array = np.load(img_file).astype(np.float32)
                    video_tensor.append(torch.from_numpy(depth_array))
                else:
                    # .png 格式：从 16-bit PNG 加载，单位为毫米 (mm)，转换为米 (m)
                    img = Image.open(img_file)
                    img = img.convert("I;16")
                    depth_array = np.array(img).astype(np.float32) / 1000.0
                    video_tensor.append(torch.from_numpy(depth_array))
            else:
                img = Image.open(img_file)
                img = img.convert("RGB")
                video_tensor.append(
                    torch.from_numpy(np.array(img)).float()
                )
            # Extract original filename without extension
            filename = os.path.splitext(os.path.basename(img_file))[0]
            original_filenames.append(filename)
        video_tensor = torch.stack(video_tensor)  # (N, H, W, 3)
    elif video_path.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        # simple video reading. Please modify it if it causes OOM
        video_tensor = torch.from_numpy(media.read_video(video_path))
        # Generate frame names for video files
        for i in range(len(video_tensor)):
            original_filenames.append(f"frame_{i:010d}")
        # For video files, subsample after loading
        video_tensor = video_tensor[::fps]
        original_filenames = original_filenames[::fps]
        if max_num_frames is not None and max_num_frames > 0:
            video_tensor = video_tensor[:max_num_frames]
            original_filenames = original_filenames[:max_num_frames]

    if not is_depth:
        video_tensor = video_tensor.permute(
            0, 3, 1, 2
        )  # Convert to tensor and permute to (N, C, H, W)
    video_tensor = video_tensor.float()
    if max_num_frames is not None and max_num_frames > 0:
        video_tensor = video_tensor[:max_num_frames]
        original_filenames = original_filenames[:max_num_frames]
    video_length = len(video_tensor)
    logger.debug(f"Loaded video with {video_length} frames from {video_path}")
    frame_h, frame_w = video_tensor.shape[-2:]

    video_mask_npy = None
    if mask_dir is not None:
        video_mask_npy = []
        mask_files = _collect_and_sort_frame_files(mask_dir, ["png"])

        for mask_file in mask_files:
            mask = media.read_image(mask_file)
            mask = cv2.resize(mask, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
            video_mask_npy.append(mask)
        video_mask_npy = np.stack(video_mask_npy)

    if not is_depth:
        video_tensor /= 255.
    return video_tensor, video_mask_npy, original_filenames


def create_uniform_grid_points(height, width, grid_size=20, device="cuda"):
    """Create uniform grid points across the image.

    Args:
        height (int): Image height
        width (int): Image width
        grid_size (int): Grid size (grid_size x grid_size points)
        device (str): Device for tensor

    Returns:
        torch.Tensor: Grid points [1, grid_size*grid_size, 3] where each point is [t, x, y]
    """
    # Create uniform grid
    y_coords = np.linspace(0, height - 1, grid_size)
    x_coords = np.linspace(0, width - 1, grid_size)

    # Create meshgrid
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Flatten and create points [N, 2]
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)

    # Add time dimension (t=0 for all points) -> [N, 3]
    time_col = np.zeros((grid_points.shape[0], 1))
    grid_points_3d = np.concatenate([time_col, grid_points], axis=1)

    # Convert to tensor and add batch dimension -> [1, N, 3]
    grid_tensor = torch.tensor(
        grid_points_3d, dtype=torch.float32, device=device
    ).unsqueeze(0)

    return grid_tensor

def prepare_query_points(query_xyt, depths, intrinsics, extrinsics):
    final_queries = []
    for query_i in query_xyt:
        if len(query_i) == 0:
            continue

        t = int(query_i[0, 0])
        depth_t = depths[t]
        K_inv_t = np.linalg.inv(intrinsics[t])
        c2w_t = np.linalg.inv(extrinsics[t])

        xy = query_i[:, 1:]
        ji = np.round(xy).astype(int)
        d = depth_t[ji[..., 1], ji[..., 0]]
        xy_homo = np.concatenate([xy, np.ones_like(xy[:, :1])], axis=-1)
        local_coords = K_inv_t @ xy_homo.T  # (3, N)
        local_coords = local_coords * d[None, :]  # (3, N)
        world_coords = c2w_t[:3, :3] @ local_coords + c2w_t[:3, 3:]
        final_queries.append(np.concatenate([query_i[:, :1], world_coords.T], axis=-1))
    return np.concatenate(final_queries, axis=0)  # (N, 4)


def prepare_inputs(
    video_ten,
    depths,
    intrinsics,
    extrinsics,
    query_point,
    inference_res: Tuple[int, int],
    support_grid_size: int,
    num_threads: int = 8,
    device: str = "cuda",
):
    _original_res = depths.shape[1:3]
    inference_res = _original_res  # fix as the same

    intrinsics[:, 0, :] *= (inference_res[1] - 1) / (_original_res[1] - 1)
    intrinsics[:, 1, :] *= (inference_res[0] - 1) / (_original_res[0] - 1)

    # resize & remove edges
    with ThreadPoolExecutor(num_threads) as executor:
        depths_futures = [
            executor.submit(_filter_one_depth, depth, 0.08, 15, intrinsic)
            for depth, intrinsic in zip(depths, intrinsics)
        ]
        depths = np.stack([future.result() for future in depths_futures])

    query_point = prepare_query_points(query_point, depths, intrinsics, extrinsics)
    query_point = torch.from_numpy(query_point).float().to(device)
    video = (video_ten.float()).to(device).clamp(0, 1)
    depths = torch.from_numpy(depths).float().to(device)
    intrinsics = torch.from_numpy(intrinsics).float().to(device)
    extrinsics = torch.from_numpy(extrinsics).float().to(device)

    return video, depths, intrinsics, extrinsics, query_point, support_grid_size


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # initialize 3D models
    model_depth_pose = video_depth_pose_dict[args.depth_pose_method](args)
    model_3dtracker = load_model(args.checkpoint).to(args.device)

    # Determine video paths to process
    if args.batch_process:
        video_folders = find_video_folders(args.video_path, args.scan_depth)
        if args.depth_path is not None:
            depth_folders = find_video_folders(args.depth_path)
            if len(depth_folders) != len(video_folders):
                logger.error(
                    f"Number of depth folders ({len(depth_folders)}) does not match number of video folders ({len(video_folders)})"
                )
                exit(1)
        else:
            depth_folders = [None] * len(video_folders)

        logger.info(f"Found {len(video_folders)} video folders to process")
        if not video_folders:
            logger.error(f"No video folders found in {args.video_path}")
            exit(1)
    else:
        video_folders = [args.video_path]
        depth_folders = [args.depth_path]

    # Process each video
    failed_videos = 0
    for video_path, depth_path in zip(video_folders, depth_folders):
        # 输出目录命名优先级：
        # 1) 显式 --video_name
        # 2) 传入外部外参时使用 --camera_name
        # 3) 默认使用输入路径名
        if args.video_name:
            video_name = args.video_name
        elif getattr(args, 'external_geom_npz', None) and hasattr(args, 'camera_name'):
            video_name = args.camera_name
        else:
            video_name = os.path.basename(video_path.rstrip("/"))

        # Check if output already exists and skip if requested
        if args.skip_existing:
            output_path = os.path.join(out_dir, video_name)
            if os.path.exists(output_path):
                if is_traceforge_output_complete(output_path):
                    logger.info(f"Skipping {video_name} - output already exists and is complete")
                    continue
                logger.warning(f"Output directory for {video_name} exists but is incomplete, will reprocess")

        try:
            # Clear CUDA cache before processing each video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Process video
            result = process_single_video(video_path, depth_path, args, model_3dtracker, model_depth_pose, video_name, out_dir)

            # Save structured data
            save_structured_data(
                video_name=video_name,
                output_dir=out_dir,
                video_tensor=result["video_tensor"],
                depths=result["depths"],
                coords=result["coords"],
                visibs=result["visibs"],
                intrinsics=result["intrinsics"],
                extrinsics=result["extrinsics"],
                query_points_per_frame=result["query_points_per_frame"],
                horizon=args.horizon,
                original_filenames=result["original_filenames"],
                use_all_trajectories=args.use_all_trajectories,
                query_frame_results=result.get("query_frame_results"),
                future_len=args.future_len,
                grid_size=args.grid_size,
                filter_args=args,
                full_video_tensor=result["full_video_tensor"],
                full_depths=result["full_depths"],
                full_intrinsics=result["full_intrinsics"],
                full_extrinsics=result["full_extrinsics"],
                depth_conf=result["depth_conf"],
                video_source_path=video_path,
                depth_source_path=depth_path,
            )

        except Exception as e:
            import traceback

            logger.error(f"Failed to process {video_name}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            failed_videos += 1
            continue

    # Cleanup
    del model_3dtracker
    del model_depth_pose
    torch.cuda.empty_cache()
    if failed_videos > 0:
        logger.error(f"Batch processing completed with failures: {failed_videos} video(s) failed.")
        sys.exit(1)
    logger.info("Batch processing completed!")
