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

from utils.video_depth_pose_utils import video_depth_pose_dict

from datasets.data_ops import _filter_one_depth
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from utils.inference_utils import load_model, inference
from utils.threed_utils import (
    project_tracks_3d_to_2d,
    project_tracks_3d_to_3d,
)

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
    return parser.parse_args()

def retarget_trajectories(
    trajectory: np.ndarray,
    interval: float = 0.05,
    max_length: int = 64,
    top_percent: float = 0.02,
):
    """
    Synchronous arc-length retargeting using per-segment robust speeds.

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
    Filter out anomalous trajectories based on velocity direction consistency and acceleration.

    Detects false trajectories caused by depth instability using two criteria:
    1. Random direction changes (back-and-forth jumps)
    2. Sudden velocity changes (high acceleration)

    Args:
        traj: (N, T, 3) trajectory array
        direction_threshold: Threshold for mean cosine similarity (default: 0.3)
        accel_threshold: IQR multiplier for acceleration outlier detection (default: 3.0)

    Returns:
        filtered_traj: (N, T, 3) with anomalous trajectories set to inf
    """
    # FILTERING DISABLED - return original trajectory
    return traj

    N, T, D = traj.shape
    if T < 3:
        return traj

    # Compute valid mask (non-inf points)
    valid = np.isfinite(traj).all(axis=-1)  # (N, T)

    # Compute direction consistency and acceleration for each trajectory
    mean_cos_sim = np.ones(N)  # Default to 1 (consistent)
    max_accel = np.zeros(N)

    for i in range(N):
        valid_idx = np.where(valid[i])[0]
        if len(valid_idx) < 3:
            continue

        # Compute velocity vectors
        vel = np.diff(traj[i, valid_idx], axis=0)  # (T_valid-1, 3)
        vel_norm = np.linalg.norm(vel, axis=1, keepdims=True)

        # Skip if velocities are too small (static points)
        valid_vel = vel_norm.flatten() > 1e-4
        if np.sum(valid_vel) < 2:
            continue

        vel = vel[valid_vel]
        vel_norm = vel_norm[valid_vel]

        # 1. Direction consistency
        vel_normalized = vel / (vel_norm + 1e-8)
        cos_sim = np.sum(vel_normalized[:-1] * vel_normalized[1:], axis=1)
        mean_cos_sim[i] = np.mean(cos_sim) if len(cos_sim) > 0 else 1.0

        # 2. Acceleration (second-order difference)
        if len(vel) >= 2:
            accel = np.diff(vel, axis=0)  # (T_valid-2, 3)
            accel_norm = np.linalg.norm(accel, axis=1)
            max_accel[i] = np.max(accel_norm) if len(accel_norm) > 0 else 0

    # Filter by direction consistency
    outliers_direction = mean_cos_sim < direction_threshold

    # Filter by acceleration using IQR
    valid_accel = max_accel[max_accel > 0]
    outliers_accel = np.zeros(N, dtype=bool)
    if len(valid_accel) >= 4:
        q25, q75 = np.percentile(valid_accel, [25, 75])
        iqr = q75 - q25
        if iqr > 1e-6:
            upper_bound = q75 + accel_threshold * iqr
            outliers_accel = max_accel > upper_bound

    # Combine filters (OR logic)
    outliers = outliers_direction | outliers_accel

    # Filter trajectories
    filtered_traj = traj.copy()
    filtered_traj[outliers] = np.inf

    n_dir = np.sum(outliers_direction)
    n_accel = np.sum(outliers_accel)
    n_total = np.sum(outliers)
    if n_total > 0:
        logger.info(f"Filtered {n_total}/{N} trajectories (direction: {n_dir}, accel: {n_accel}, overlap: {n_dir + n_accel - n_total})")

    return filtered_traj


def save_single_query_frame(
    video_name,
    output_dir,
    query_frame_idx,
    frame_data,
    future_len: int,
    grid_size: int,
):
    """Save single query frame result"""
    video_output_dir = os.path.join(output_dir, video_name)
    images_dir = os.path.join(video_output_dir, "images")
    depth_dir = os.path.join(video_output_dir, "depth")
    samples_dir = os.path.join(video_output_dir, "samples")

    for dir_path in [images_dir, depth_dir, samples_dir]:
        os.makedirs(dir_path, exist_ok=True)

    coords_np = frame_data["coords"].cpu().numpy()
    visibs_np = frame_data["visibs"].cpu().numpy()
    video_segment = frame_data["video_segment"].cpu().numpy() * 255
    video_segment = video_segment.astype(np.uint8).transpose(0, 2, 3, 1)
    intrinsics_np = frame_data["intrinsics_segment"].cpu().numpy()
    extrinsics_np = frame_data["extrinsics_segment"].cpu().numpy()

    frame_h, frame_w = video_segment.shape[1:3]
    y_coords = np.linspace(0, frame_h - 1, grid_size)
    x_coords = np.linspace(0, frame_w - 1, grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    keypoints = np.stack([xx.flatten(), yy.flatten()], axis=1)

    sample_data = {
        "image_path": np.array([f"images/{video_name}_{query_frame_idx}.png"], dtype="<U50"),
        "frame_index": np.array([query_frame_idx]),
        "keypoints": keypoints.astype(np.float32),
    }

    camera_views_segment = []
    for t in range(len(intrinsics_np)):
        camera_views_segment.append({
            "c2w": np.linalg.inv(extrinsics_np[t]),
            "K": intrinsics_np[t],
            "height": frame_h,
            "width": frame_w,
        })

    fixed_camera_view = camera_views_segment[0]
    coords_3d_for_projection = coords_np

    try:
        tracks2d_fixed = project_tracks_3d_to_2d(
            tracks3d=coords_3d_for_projection,
            camera_views=[fixed_camera_view] * len(coords_3d_for_projection),
        )
        tracks3d_fixed = project_tracks_3d_to_3d(
            tracks3d=coords_3d_for_projection,
            camera_views=[fixed_camera_view] * len(coords_3d_for_projection),
        )
        sample_data["traj_2d"] = tracks2d_fixed.transpose(1, 0, 2).astype(np.float32)
        sample_data["traj"] = tracks3d_fixed.transpose(1, 0, 2).astype(np.float32)
    except Exception as e:
        logger.error(f"Error projecting tracks for frame {query_frame_idx}: {e}")
        sample_data["traj_2d"] = coords_np[:, :, :2].transpose(1, 0, 2).astype(np.float32)
        sample_data["traj"] = coords_np.transpose(1, 0, 2).astype(np.float32)

    # Filter anomalous trajectories
    sample_data["traj"] = filter_anomalous_trajectories(sample_data["traj"])

    # Pad trajectories to future_len
    current_len = sample_data["traj"].shape[1]
    if current_len < future_len:
        pad_len = future_len - current_len
        pad_shape = (sample_data["traj"].shape[0], pad_len, sample_data["traj"].shape[2])
        pad_array = np.full(pad_shape, np.inf, dtype=sample_data["traj"].dtype)
        sample_data["traj"] = np.concatenate([sample_data["traj"], pad_array], axis=1)

        pad_shape_2d = (sample_data["traj_2d"].shape[0], pad_len, sample_data["traj_2d"].shape[2])
        pad_array_2d = np.full(pad_shape_2d, np.inf, dtype=sample_data["traj_2d"].dtype)
        sample_data["traj_2d"] = np.concatenate([sample_data["traj_2d"], pad_array_2d], axis=1)

    query_frame_img = video_segment[0]
    query_frame_depth = frame_data["depths_segment"].cpu().numpy()[0]

    img_filename = f"{video_name}_{query_frame_idx}.png"
    img_path = os.path.join(images_dir, img_filename)
    Image.fromarray(query_frame_img).save(img_path)

    depth_filename = f"{video_name}_{query_frame_idx}.png"
    depth_path = os.path.join(depth_dir, depth_filename)
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
    Image.fromarray(depth_normalized, mode="I;16").save(depth_path)

    depth_raw_filename = f"{video_name}_{query_frame_idx}_raw.npz"
    depth_raw_path = os.path.join(depth_dir, depth_raw_filename)
    np.savez(depth_raw_path, depth=query_frame_depth)

    # Retarget disabled - use original trajectory
    # retargeted, valid_mask = retarget_trajectories(sample_data["traj"], max_length=future_len)
    # sample_data["traj"] = retargeted
    # sample_data["valid_steps"] = valid_mask

    sample_filename = f"{video_name}_{query_frame_idx}.npz"
    sample_path = os.path.join(samples_dir, sample_filename)
    np.savez(sample_path, **sample_data)

    logger.info(f"Saved query frame {query_frame_idx}")


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
):
    """Save data in the structured format"""

    # Create output directories
    video_output_dir = os.path.join(output_dir, video_name)
    images_dir = os.path.join(video_output_dir, "images")
    depth_dir = os.path.join(video_output_dir, "depth")
    samples_dir = os.path.join(video_output_dir, "samples")

    # Save structured data in the new format
    for dir_path in [images_dir, depth_dir, samples_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # If we have query_frame_results, save each query frame's results independently
    if query_frame_results is not None:
        logger.info(f"Processing {len(query_frame_results)} query frame results")

        saved_count = 0

        for query_frame_idx, frame_data in query_frame_results.items():
            coords_np = frame_data["coords"].cpu().numpy()  # (T, grid_size*grid_size, 3)

            # Save RGB images for this segment
            video_segment = frame_data["video_segment"].cpu().numpy() * 255
            video_segment = video_segment.astype(np.uint8).transpose(
                0, 2, 3, 1
            )  # (T, H, W, 3)


            # Save sample data for this query frame
            coords_np = frame_data["coords"].cpu().numpy()  # (T, grid_size*grid_size, 3) where T <= future_len
            visibs_np = frame_data["visibs"].cpu().numpy()  # (T, grid_size*grid_size)
            intrinsics_np = frame_data["intrinsics_segment"].cpu().numpy()  # (T, 3, 3)
            extrinsics_np = frame_data["extrinsics_segment"].cpu().numpy()  # (T, 4, 4)

            # Debug: Check shapes
            logger.debug(
                f"Query frame {query_frame_idx}: coords_np shape = {coords_np.shape}"
            )
            logger.debug(
                f"Query frame {query_frame_idx}: visibs_np shape = {visibs_np.shape}"
            )

            # Handle edge cases where coords might not have expected dimensions
            if len(coords_np.shape) != 3:
                logger.error(
                    f"Unexpected coords shape for frame {query_frame_idx}: {coords_np.shape}"
                )
                continue

            # Get actual number of frames in this segment
            actual_frames = coords_np.shape[0]

            # Create sample data for this query frame
            sample_data = {}

            # Grid points (grid_size x grid_size points) for this query frame
            frame_h, frame_w = video_segment.shape[1:3]
            y_coords = np.linspace(0, frame_h - 1, grid_size)
            x_coords = np.linspace(0, frame_w - 1, grid_size)
            xx, yy = np.meshgrid(x_coords, y_coords)
            keypoints = np.stack([xx.flatten(), yy.flatten()], axis=1)  # (grid_size*grid_size, 2)

            sample_data["image_path"] = np.array(
                [f"images/{video_name}_{query_frame_idx}.png"], dtype="<U50"
            )
            sample_data["frame_index"] = np.array([query_frame_idx])
            sample_data["keypoints"] = keypoints.astype(np.float32)  # (grid_size*grid_size, 2)

            # trajectories: (grid_size*grid_size, T, 3) - grid_size*grid_size tracks, T frames (T <= future_len), xyz coordinates
            try:
                sample_data["traj"] = coords_np.transpose(1, 0, 2).astype(
                    np.float32
                )  # (grid_size*grid_size, T, 3)
            except ValueError as e:
                logger.error(
                    f"Error transposing coords for frame {query_frame_idx}: {e}"
                )
                logger.error(f"coords_np shape: {coords_np.shape}")
                # Skip this frame and continue
                continue

            # Project 3D coordinates to 2D for traj_2d
            camera_views_segment = []
            for t in range(len(intrinsics_np)):
                camera_views_segment.append(
                    {
                        "c2w": np.linalg.inv(extrinsics_np[t]),
                        "K": intrinsics_np[t],
                        "height": frame_h,
                        "width": frame_w,
                    }
                )

            # Use the first frame's camera for consistent projection
            fixed_camera_view = camera_views_segment[0]

            # Project to 2D using the same camera view
            coords_3d_for_projection = coords_np  # (T, grid_size*grid_size, 3)
            try:
                tracks2d_fixed = project_tracks_3d_to_2d(
                    tracks3d=coords_3d_for_projection,
                    camera_views=[fixed_camera_view] * len(coords_3d_for_projection),
                )  # (T, grid_size*grid_size, 2)
                tracks3d_fixed = project_tracks_3d_to_3d(
                    tracks3d=coords_3d_for_projection,
                    camera_views=[fixed_camera_view] * len(coords_3d_for_projection),
                )  # (T, grid_size*grid_size, 3)

                sample_data["traj_2d"] = tracks2d_fixed.transpose(1, 0, 2).astype(
                    np.float32
                )  # (grid_size*grid_size, T, 2)
                sample_data["traj"] = tracks3d_fixed.transpose(1, 0, 2).astype(
                    np.float32
                )  # (grid_size*grid_size, T, 3)
            except Exception as e:
                logger.error(
                    f"Error projecting tracks for frame {query_frame_idx}: {e}"
                )
                # Fallback: use original coordinates
                sample_data["traj_2d"] = (
                    coords_np[:, :, :2].transpose(1, 0, 2).astype(np.float32)
                )
                sample_data["traj"] = coords_np.transpose(1, 0, 2).astype(np.float32)

            # Filter anomalous trajectories
            sample_data["traj"] = filter_anomalous_trajectories(sample_data["traj"])

            # Pad trajectories to future_len
            current_len = sample_data["traj"].shape[1]
            if current_len < args.future_len:
                pad_len = args.future_len - current_len
                pad_shape = (sample_data["traj"].shape[0], pad_len, sample_data["traj"].shape[2])
                pad_array = np.full(pad_shape, np.inf, dtype=sample_data["traj"].dtype)
                sample_data["traj"] = np.concatenate([sample_data["traj"], pad_array], axis=1)

                pad_shape_2d = (sample_data["traj_2d"].shape[0], pad_len, sample_data["traj_2d"].shape[2])
                pad_array_2d = np.full(pad_shape_2d, np.inf, dtype=sample_data["traj_2d"].dtype)
                sample_data["traj_2d"] = np.concatenate([sample_data["traj_2d"], pad_array_2d], axis=1)

            # Only save image and depth for the query frame itself, not the entire segment
            query_frame_img = video_segment[
                0
            ]  # First frame in segment is the query frame
            query_frame_depth = (
                frame_data["depths_segment"].cpu().numpy()[0]
            )  # First depth

            img_filename = f"{video_name}_{query_frame_idx}.png"
            img_path = os.path.join(images_dir, img_filename)
            if not os.path.exists(img_path):  # Avoid duplicate saves
                Image.fromarray(query_frame_img).save(img_path)

            # Save depth image for query frame only
            depth_filename = f"{video_name}_{query_frame_idx}.png"
            depth_path = os.path.join(depth_dir, depth_filename)
            if not os.path.exists(depth_path):  # Avoid duplicate saves
                # Save depth as 16-bit PNG with normalization for better visualization
                # Normalize depth to use full 16-bit range (0-65535) for better contrast
                # The raw depth values are saved in NPZ for accurate reconstruction
                depth_valid = query_frame_depth[query_frame_depth > 0]
                if len(depth_valid) > 0:
                    depth_min = depth_valid.min()
                    depth_max = depth_valid.max()
                    # Normalize to 0-65535 range, with a small margin to avoid clipping
                    if depth_max > depth_min:
                        depth_normalized = ((query_frame_depth - depth_min) / (depth_max - depth_min) * 65535.0).clip(0, 65535).astype(np.uint16)
                    else:
                        # If all values are the same, set to a middle value
                        depth_normalized = np.full_like(query_frame_depth, 32767, dtype=np.uint16)
                else:
                    # No valid depth values
                    depth_normalized = np.zeros_like(query_frame_depth, dtype=np.uint16)
                
                Image.fromarray(depth_normalized, mode="I;16").save(depth_path)

                # save depth raw value as npz (for accurate reconstruction)
                depth_raw_filename = f"{video_name}_{query_frame_idx}_raw.npz"
                depth_raw_path = os.path.join(depth_dir, depth_raw_filename)
                np.savez(depth_raw_path, depth=query_frame_depth)

            # Retarget disabled - use original trajectory
            # retargeted, valid_mask = retarget_trajectories(sample_data["traj"], max_length=args.future_len)
            # sample_data["traj"] = retargeted
            # sample_data["valid_steps"] = valid_mask

            # Save sample NPZ for this query frame
            sample_filename = f"{video_name}_{query_frame_idx}.npz"
            sample_path = os.path.join(samples_dir, sample_filename)
            np.savez(sample_path, **sample_data)

            logger.info(
                f"Saved query frame {query_frame_idx} with {grid_size * grid_size} trajectories tracked for {actual_frames} frames"
            )
            saved_count += 1

        logger.info(f"Saved {saved_count} frames")


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

        # Save immediately after processing each query frame
        if video_name is not None and output_dir is not None:
            save_single_query_frame(
                video_name=video_name,
                output_dir=output_dir,
                query_frame_idx=start_frame,
                frame_data=query_frame_results[start_frame],
                future_len=args.future_len,
                grid_size=args.grid_size,
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
        "depths": depths,
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
                # 检查输出目录是否有实际内容（不仅仅是空目录）
                images_dir = os.path.join(output_path, "images")
                depth_dir = os.path.join(output_path, "depth")
                samples_dir = os.path.join(output_path, "samples")
                
                # 检查是否有文件（至少检查images和samples目录）
                has_images = False
                has_samples = False
                
                if os.path.exists(images_dir):
                    try:
                        has_images = any(
                            os.path.isfile(os.path.join(images_dir, f))
                            for f in os.listdir(images_dir)
                        )
                    except (OSError, PermissionError):
                        pass
                
                if os.path.exists(samples_dir):
                    try:
                        has_samples = any(
                            os.path.isfile(os.path.join(samples_dir, f))
                            for f in os.listdir(samples_dir)
                        )
                    except (OSError, PermissionError):
                        pass
                
                # 如果images和samples都有内容，认为输出完整，跳过
                if has_images and has_samples:
                    logger.info(f"Skipping {video_name} - output already exists and is complete")
                    continue
                else:
                    # 目录存在但内容不完整，重新处理
                    logger.warning(f"Output directory for {video_name} exists but is incomplete, will reprocess")
                    # 可以选择删除不完整的目录，或者直接覆盖
                    # import shutil
                    # shutil.rmtree(output_path)

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
            )

            # Always save traditional visualization NPZ in video directory root
            video_dir = os.path.join(out_dir, video_name)
            data_npz_load = {}
            data_npz_load["coords"] = result["coords"].cpu().numpy()
            # Use full video camera parameters instead of segmented ones
            data_npz_load["extrinsics"] = result["full_extrinsics"].cpu().numpy()
            data_npz_load["intrinsics"] = result["full_intrinsics"].cpu().numpy()
            data_npz_load["height"] = result["video_tensor"].shape[-2]
            data_npz_load["width"] = result["video_tensor"].shape[-1]
            data_npz_load["depths"] = result["depths"].cpu().numpy().astype(np.float16)
            data_npz_load["unc_metric"] = result["depth_conf"].astype(np.float16)
            data_npz_load["visibs"] = result["visibs"][..., None].cpu().numpy()
            if args.save_video:
                data_npz_load["video"] = result["video_tensor"].cpu().numpy()

            save_path = os.path.join(video_dir, video_name + ".npz")
            np.savez(save_path, **data_npz_load)
            logger.info(f"Traditional visualization NPZ saved to {save_path}")

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
