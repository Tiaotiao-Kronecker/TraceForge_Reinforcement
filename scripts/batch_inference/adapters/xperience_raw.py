from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch
from loguru import logger


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.batch_inference import infer
from utils.inference_utils import load_model
from utils.keyframe_schedule_utils import build_candidate_source_frame_indices
from utils.traceforge_artifact_utils import (
    SCENE_STORAGE_ADAPTER_REF,
    SCENE_STORAGE_CACHE,
    V2_LAYOUT,
    is_traceforge_output_complete,
)
from utils.xperience_adapter_utils import (
    XPERIENCE_SUPPORTED_CAMERAS,
    aligned_video_fps,
    build_stereo_left_extrinsics,
    build_stereo_left_intrinsics,
    build_xperience_source_descriptor,
    load_video_frame,
    load_video_frames,
    open_xperience_episode,
    read_caption_main_task,
)


ADAPTER_NAME = "xperience_raw"
DEFAULT_XPERIENCE_EPISODE_GLOB = "*/*"
DEFAULT_XPERIENCE_WINDOW_SIZE = 512


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Native Xperience batch inference")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root of the raw Xperience dataset tree")
    parser.add_argument(
        "--episode_glob",
        type=str,
        default=DEFAULT_XPERIENCE_EPISODE_GLOB,
        help="Glob under dataset_root used to discover episodes. Expected matches look like <uuid>/epN.",
    )
    parser.add_argument("--episode_limit", type=int, default=None, help="Optional max number of episodes to process")
    parser.add_argument(
        "--camera_name",
        type=str,
        default="stereo_left",
        choices=list(XPERIENCE_SUPPORTED_CAMERAS),
        help="Raw Xperience camera to process. First maintained release only supports stereo_left.",
    )
    parser.add_argument("--start_index", type=int, default=0, help="Raw start frame within each episode")
    parser.add_argument("--stop_index", type=int, default=None, help="Optional raw stop frame within each episode")
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_XPERIENCE_WINDOW_SIZE,
        help="Raw episode window size before stride/cap sampling. <=0 means use one whole-episode window.",
    )
    parser.add_argument(
        "--window_step",
        type=int,
        default=None,
        help="Raw window step. Defaults to window_size for non-overlapping windows.",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_existing", action="store_true", default=False)
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--fps", type=int, default=1, help="Raw frame stride applied within each window")
    parser.add_argument("--max_num_frames", type=int, default=512)
    parser.add_argument("--frame_drop_rate", type=int, default=1)
    parser.add_argument("--future_len", type=int, default=32)
    parser.add_argument("--grid_size", type=int, default=80)
    parser.add_argument("--grid_width", type=int, default=None)
    parser.add_argument("--grid_height", type=int, default=None)
    parser.add_argument("--support_grid_ratio", type=float, default=0.0)
    parser.add_argument("--query_candidate_grid_factor", type=float, default=2.0)
    parser.add_argument("--grid_border_trim_left", type=int, default=30)
    parser.add_argument("--grid_border_trim_right", type=int, default=30)
    parser.add_argument("--grid_border_trim_top", type=int, default=30)
    parser.add_argument("--grid_border_trim_bottom", type=int, default=10)
    parser.add_argument("--query_sampler_mode", type=str, default="grid", choices=["auto", "grid", "relevance_first_v1"])
    parser.add_argument("--query_prefilter_mode", type=str, default="off", choices=["off", "profile_aware_static_v1", "external_depth_static_v1"])
    parser.add_argument("--query_prefilter_wrist_rank_keep_ratio", type=float, default=0.30)
    parser.add_argument("--query_visibility_gate_mode", type=str, default="all_future_v1", choices=["off", "all_future_v1"])
    parser.add_argument("--query_visibility_gate_min_border_dist_px", type=float, default=0.0)
    parser.add_argument("--query_visibility_gate_near_depth_exempt_threshold_m", type=float, default=0.0)
    parser.add_argument("--query_fixed_view_depth_gate_mode", type=str, default="first_frame_uvd_v1", choices=["off", "first_frame_uvd_v1"])
    parser.add_argument("--query_fixed_view_depth_gate_uv_threshold_px", type=float, default=1.0)
    parser.add_argument("--query_fixed_view_depth_gate_depth_threshold_m", type=float, default=0.10)
    parser.add_argument("--traj_uvd_gate_mode", type=str, default="delta_uv_depth_v1", choices=["off", "delta_uv_depth_v1"])
    parser.add_argument("--traj_uvd_gate_uv_mean_threshold_px", type=float, default=3.0)
    parser.add_argument("--traj_uvd_gate_depth_std_threshold_m", type=float, default=0.01)
    parser.add_argument("--traj_uvd_gate_max_depth_threshold_m", type=float, default=1.5)
    parser.add_argument("--traj_uvd_gate_near_depth_threshold_m", type=float, default=0.0)
    parser.add_argument("--traj_uvd_gate_near_depth_relaxed_std_threshold_m", type=float, default=0.0)
    parser.add_argument("--traj_uvd_gate_near_depth_exempt_threshold_m", type=float, default=0.0)
    parser.add_argument("--query_depth_stabilization_mode", type=str, default="off", choices=["off", "temporal_median_world_v1"])
    parser.add_argument("--query_depth_stabilization_reproj_tol_px", type=float, default=3.0)
    parser.add_argument("--query_depth_stabilization_min_support", type=int, default=3)
    parser.add_argument("--query_depth_stabilization_min_query_depth_m", type=float, default=0.01)
    parser.add_argument("--query_depth_stabilization_min_border_dist_px", type=float, default=0.0)
    parser.add_argument("--dense_depth_stabilization_mode", type=str, default="off", choices=["off", "temporal_median_reproject_v1"])
    parser.add_argument("--dense_depth_stabilization_radius", type=int, default=2)
    parser.add_argument("--dense_depth_stabilization_min_support", type=int, default=3)
    parser.add_argument("--tracker_precision_mode", type=str, default="fp32", choices=["fp32", "autocast_bf16", "deep_bf16", "bf16"])
    parser.add_argument("--filter_level", type=str, default="none", choices=["none", "basic", "standard", "strict"])
    parser.add_argument("--traj_filter_profile", type=str, default="external")
    parser.add_argument("--traj_filter_ablation_mode", type=str, default="none")
    parser.add_argument("--collect_profile_stats", action="store_true", default=False)
    parser.add_argument("--collect_filter_stage_diagnostics", action="store_true", default=False)
    parser.add_argument("--depth_filter_workers", type=int, default=8)
    parser.add_argument("--depth_filter_blas_threads", type=int, default=1)
    parser.add_argument("--resize_width", type=int, default=None)
    parser.add_argument("--resize_height", type=int, default=None)
    parser.add_argument("--save_visibility", action="store_true", default=False)
    parser.add_argument("--output_layout", type=str, default=V2_LAYOUT, choices=[V2_LAYOUT])
    parser.add_argument(
        "--scene_storage_mode",
        type=str,
        default=SCENE_STORAGE_ADAPTER_REF,
        choices=[SCENE_STORAGE_ADAPTER_REF, SCENE_STORAGE_CACHE],
    )
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.fps <= 0:
        raise ValueError("--fps must be >= 1 for native Xperience loading")
    if args.window_step is not None and args.window_step <= 0:
        raise ValueError("--window_step must be > 0 when provided")
    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0")
    if args.stop_index is not None and args.stop_index <= args.start_index:
        raise ValueError("--stop_index must be greater than --start_index")
    if args.depth_filter_workers <= 0:
        raise ValueError("--depth_filter_workers must be >= 1")
    if args.depth_filter_blas_threads < 0:
        raise ValueError("--depth_filter_blas_threads must be >= 0")
    args.processing_resize_hw = infer._resolve_processing_resize_hw(args)
    args.query_grid_hw = infer._resolve_query_grid_hw(args)
    args.depth_pose_method = "external"
    args.external_geom_npz = None
    args.external_extr_mode = "w2c"
    args.mask_dir = None
    args.video_name = None
    args.batch_process = False
    args.scan_depth = 0
    return args


def _find_episode_dirs(
    dataset_root: Path,
    *,
    episode_glob: str,
    episode_limit: int | None,
) -> list[Path]:
    annotation_paths = sorted(dataset_root.glob(f"{episode_glob}/annotation.hdf5"))
    if not annotation_paths:
        annotation_paths = sorted(dataset_root.rglob("annotation.hdf5"))
    episode_dirs = [path.parent.resolve() for path in annotation_paths]
    if episode_limit is not None and episode_limit > 0:
        episode_dirs = episode_dirs[: int(episode_limit)]
    return episode_dirs


def _resolve_frame_count(handle: h5py.File) -> int:
    return int(
        min(
            handle["video/frame_number"].shape[0],
            handle["depth/depth"].shape[0],
            handle["slam/trans_xyz"].shape[0],
            handle["slam/quat_wxyz"].shape[0],
        )
    )


def _build_window_ranges(
    *,
    frame_count: int,
    start_index: int,
    stop_index: int | None,
    window_size: int,
    window_step: int | None,
) -> list[tuple[int, int]]:
    resolved_start = max(0, int(start_index))
    resolved_stop = frame_count if stop_index is None else min(int(stop_index), int(frame_count))
    if resolved_start >= resolved_stop:
        raise ValueError(
            f"Invalid Xperience window range [{resolved_start}, {resolved_stop}) for frame_count={frame_count}"
        )
    if window_size <= 0:
        return [(resolved_start, resolved_stop)]
    step = int(window_step if window_step is not None else window_size)
    ranges: list[tuple[int, int]] = []
    current = resolved_start
    while current < resolved_stop:
        window_stop = min(current + int(window_size), resolved_stop)
        ranges.append((current, window_stop))
        current += step
    return ranges


def _build_video_name(episode_dir: Path, camera_name: str, start_index: int, stop_index: int) -> str:
    return f"{episode_dir.parent.name}__{episode_dir.name}__{camera_name}__{start_index:06d}_{stop_index:06d}"


def _load_scene_bundle(
    episode_dir: Path,
    *,
    dataset_root: Path,
    args: argparse.Namespace,
    start_index: int,
    stop_index: int,
) -> dict[str, object]:
    paths = open_xperience_episode(episode_dir, camera_name=args.camera_name)
    with h5py.File(paths.annotation_path, "r") as handle:
        relative_indices = build_candidate_source_frame_indices(
            stop_index - start_index,
            stride=int(args.fps),
            max_num_frames=args.max_num_frames,
        )
        if relative_indices.size == 0:
            raise ValueError(f"No frames selected from {episode_dir} in range [{start_index}, {stop_index})")
        source_frame_indices = (relative_indices + int(start_index)).astype(np.int32, copy=False)
        first_frame = load_video_frame(paths.video_path, int(source_frame_indices[0]))
        depth_frames = np.asarray(handle["depth/depth"][source_frame_indices], dtype=np.float32)
        depth_conf = np.asarray(handle["depth/confidence"][source_frame_indices], dtype=np.float32)
        target_hw = tuple(int(value) for value in depth_frames.shape[1:3])
        source_hw = tuple(int(value) for value in first_frame.shape[:2])
        rgb_frames = load_video_frames(paths.video_path, source_frame_indices, target_hw=target_hw)
        intrinsics_single = build_stereo_left_intrinsics(handle, source_hw=source_hw, target_hw=target_hw)
        intrinsics = np.repeat(intrinsics_single[None, :, :], source_frame_indices.shape[0], axis=0)
        extrinsics = build_stereo_left_extrinsics(handle, source_frame_indices=source_frame_indices)
        language_text = read_caption_main_task(handle)
        episode_fps = float(aligned_video_fps(handle))

    source_descriptor = build_xperience_source_descriptor(
        dataset_root=dataset_root,
        episode_dir=episode_dir,
        camera_name=args.camera_name,
        window_start=start_index,
        window_stop=stop_index,
        source_hw=source_hw,
        target_hw=target_hw,
    )
    source_descriptor["episode_fps"] = float(episode_fps)
    source_descriptor["selected_frame_count"] = int(source_frame_indices.shape[0])
    source_descriptor["language_text"] = str(language_text)
    return {
        "video_ten": torch.from_numpy(rgb_frames).permute(0, 3, 1, 2).float() / 255.0,
        "depth_npy": depth_frames.astype(np.float32, copy=False),
        "depth_conf": depth_conf.astype(np.float32, copy=False),
        "intrs_npy": intrinsics.astype(np.float32, copy=False),
        "extrs_npy": extrinsics.astype(np.float32, copy=False),
        "original_filenames": [f"{int(frame_idx):06d}" for frame_idx in source_frame_indices.tolist()],
        "source_frame_indices": source_frame_indices.astype(np.int32, copy=False),
        "camera_name": str(args.camera_name),
        "scene_id": _build_video_name(episode_dir, args.camera_name, start_index, stop_index),
        "language_text": str(language_text),
        "source_descriptor": source_descriptor,
        "video_path": None,
        "depth_path": None,
        "stride": int(args.fps),
    }


def _write_language_file(output_root: Path, video_name: str, language_text: str) -> None:
    if not language_text:
        return
    output_dir = output_root / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lang.txt").write_text(str(language_text).strip() + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Xperience dataset_root does not exist: {dataset_root}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = _find_episode_dirs(
        dataset_root,
        episode_glob=args.episode_glob,
        episode_limit=args.episode_limit,
    )
    if not episode_dirs:
        raise SystemExit(f"No Xperience episodes found under {dataset_root} with pattern {args.episode_glob!r}")

    logger.info(f"Discovered {len(episode_dirs)} Xperience episodes under {dataset_root}")
    model_3dtracker = load_model(args.checkpoint).to(args.device)
    failures = 0
    processed = 0
    skipped = 0

    try:
        for episode_dir in episode_dirs:
            with h5py.File(episode_dir / "annotation.hdf5", "r") as handle:
                frame_count = _resolve_frame_count(handle)
            windows = _build_window_ranges(
                frame_count=frame_count,
                start_index=args.start_index,
                stop_index=args.stop_index,
                window_size=args.window_size,
                window_step=args.window_step,
            )
            for window_start, window_stop in windows:
                video_name = _build_video_name(episode_dir, args.camera_name, window_start, window_stop)
                output_path = out_dir / video_name
                if args.skip_existing and output_path.exists() and is_traceforge_output_complete(output_path):
                    skipped += 1
                    logger.info(f"Skipping {video_name} because output already exists and is complete")
                    continue
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    scene_bundle = _load_scene_bundle(
                        episode_dir,
                        dataset_root=dataset_root,
                        args=args,
                        start_index=window_start,
                        stop_index=window_stop,
                    )
                    result = infer.process_single_scene_bundle(
                        scene_bundle,
                        args,
                        model_3dtracker,
                        video_name=video_name,
                        output_dir=str(out_dir),
                    )
                    infer.save_structured_data(
                        video_name=video_name,
                        output_dir=str(out_dir),
                        video_tensor=result["video_tensor"],
                        depths=result["depths"],
                        coords=result["coords"],
                        visibs=result["visibs"],
                        intrinsics=result["intrinsics"],
                        extrinsics=result["extrinsics"],
                        query_points_per_frame=result["query_points_per_frame"],
                        original_filenames=result["original_filenames"],
                        query_frame_results=result.get("query_frame_results"),
                        future_len=args.future_len,
                        grid_size=args.grid_size,
                        filter_args=args,
                        full_video_tensor=result["full_video_tensor"],
                        full_depths=result["full_depths"],
                        full_intrinsics=result["full_intrinsics"],
                        full_extrinsics=result["full_extrinsics"],
                        depth_conf=result["depth_conf"],
                        video_source_path=None,
                        depth_source_path=None,
                        source_descriptor=scene_bundle["source_descriptor"],
                        source_frame_indices=result["source_frame_indices"],
                        query_frame_metadata=result.get("query_frame_metadata"),
                        original_frame_height=result.get("original_frame_height"),
                        original_frame_width=result.get("original_frame_width"),
                        processing_frame_height=result.get("processing_frame_height"),
                        processing_frame_width=result.get("processing_frame_width"),
                    )
                    _write_language_file(out_dir, video_name, str(scene_bundle.get("language_text") or ""))
                    processed += 1
                    logger.info(f"Finished Xperience window {video_name}")
                except Exception:
                    failures += 1
                    logger.error(f"Failed Xperience window {video_name}")
                    logger.error(traceback.format_exc())
    finally:
        del model_3dtracker
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"Xperience batch completed: processed={processed} skipped={skipped} failures={failures}")
    return 1 if failures else 0
