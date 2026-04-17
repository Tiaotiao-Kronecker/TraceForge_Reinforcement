#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.batch_inference.infer import _resize_scene_inputs, load_video_and_mask
from scripts.minimal_tracker_inference.minimal_tracker_core import (
    build_world_grid_queries,
    load_tracker_model,
)
from utils.extrinsics_utils import normalize_extrinsics_to_w2c
from utils.inference_utils import inference, normalize_tracker_precision_mode
from utils.video_depth_pose_utils import _load_external_geom


@dataclass(frozen=True)
class LoadedRealCase:
    case_name: str
    rgb_dir: str
    depth_dir: str
    geom_path: str
    camera_name: str
    source_frame_indices: np.ndarray
    video: torch.Tensor
    depths: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    video_np: np.ndarray
    depths_np: np.ndarray
    intrinsics_np: np.ndarray
    extrinsics_np: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fp32 / bf16 tracker A/B regression on a real external-only case "
            "without invoking the full batch orchestration path."
        )
    )
    parser.add_argument("--episode_dir", type=str, required=True)
    parser.add_argument("--camera_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="../TraceForge_Reinforcement/checkpoints/tapip3d_final.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision_modes", type=str, default="fp32,deep_bf16")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max_num_frames", type=int, default=64)
    parser.add_argument("--resize_width", type=int, default=None)
    parser.add_argument("--resize_height", type=int, default=None)
    parser.add_argument("--external_geom_path", type=str, default=None)
    parser.add_argument("--external_extr_mode", type=str, default="w2c", choices=["w2c", "c2w"])
    parser.add_argument("--query_grid_size", type=int, default=80)
    parser.add_argument("--support_grid_size", type=int, default=0)
    parser.add_argument("--query_frames", type=str, default="0,16,32")
    parser.add_argument("--num_iters", type=int, default=3)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _parse_precision_modes(raw: str) -> list[str]:
    modes = [normalize_tracker_precision_mode(item) for item in str(raw).split(",") if item.strip()]
    if not modes:
        raise ValueError("precision_modes must contain at least one mode")
    return modes


def _parse_query_frames(raw: str, num_frames: int) -> list[int]:
    if str(raw).strip().lower() == "auto3":
        candidates = [0, num_frames // 4, num_frames // 2]
    else:
        candidates = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    query_frames = sorted({frame for frame in candidates if 0 <= frame < num_frames})
    if not query_frames:
        raise ValueError(f"No valid query_frames remain after clipping to [0, {num_frames - 1}]")
    return query_frames


def _resolve_resize_hw(args: argparse.Namespace) -> tuple[int, int] | None:
    if args.resize_width is None and args.resize_height is None:
        return None
    if args.resize_width is None or args.resize_height is None:
        raise ValueError("resize_width and resize_height must either both be set or both be omitted")
    if args.resize_width <= 0 or args.resize_height <= 0:
        raise ValueError("resize_width and resize_height must be > 0")
    return int(args.resize_height), int(args.resize_width)


def _load_real_case(args: argparse.Namespace) -> LoadedRealCase:
    episode_dir = Path(args.episode_dir)
    rgb_dir = episode_dir / "rgb" / args.camera_name
    depth_dir = episode_dir / "depth" / args.camera_name
    geom_path = Path(args.external_geom_path) if args.external_geom_path else episode_dir / "trajectory_valid.h5"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
    if not geom_path.is_file():
        raise FileNotFoundError(f"Geometry file not found: {geom_path}")

    video_tensor, _, _, source_frame_indices = load_video_and_mask(
        str(rgb_dir),
        None,
        int(args.fps),
        int(args.max_num_frames),
    )
    depth_tensor, _, _, depth_source_frame_indices = load_video_and_mask(
        str(depth_dir),
        None,
        int(args.fps),
        int(args.max_num_frames),
        is_depth=True,
    )
    source_frame_indices = np.asarray(source_frame_indices, dtype=np.int32)
    depth_source_frame_indices = np.asarray(depth_source_frame_indices, dtype=np.int32)
    if (
        source_frame_indices.shape == depth_source_frame_indices.shape
        and not np.array_equal(source_frame_indices, depth_source_frame_indices)
    ):
        raise ValueError("RGB and depth source frame indices diverged after stride/cap sampling")

    intrinsics, extrinsics_raw = _load_external_geom(str(geom_path), args.camera_name)
    extrinsics = normalize_extrinsics_to_w2c(
        extrinsics_raw,
        extr_mode=str(args.external_extr_mode),
        context="run_real_case_precision_ab",
    )
    intrinsics = np.asarray(intrinsics[:: int(args.fps)], dtype=np.float32)
    extrinsics = np.asarray(extrinsics[:: int(args.fps)], dtype=np.float32)
    if int(args.max_num_frames) > 0:
        intrinsics = intrinsics[: int(args.max_num_frames)]
        extrinsics = extrinsics[: int(args.max_num_frames)]

    min_len = min(len(video_tensor), len(depth_tensor), intrinsics.shape[0], extrinsics.shape[0])
    if min_len <= 0:
        raise ValueError("No usable frames remain after RGB/depth/geometry alignment")
    video_tensor = video_tensor[:min_len].float() / 255.0
    depth_np = depth_tensor[:min_len].cpu().numpy().astype(np.float32, copy=False)
    intrinsics = intrinsics[:min_len]
    extrinsics = extrinsics[:min_len]
    source_frame_indices = source_frame_indices[:min_len]

    resize_hw = _resolve_resize_hw(args)
    if resize_hw is not None:
        video_tensor, depth_np, _, intrinsics = _resize_scene_inputs(
            video_ten=video_tensor,
            depth_npy=depth_np,
            depth_conf_npy=(depth_np > 0).astype(np.float32, copy=False),
            intrs_npy=intrinsics,
            target_hw=resize_hw,
        )

    video_np = video_tensor.cpu().numpy().astype(np.float32, copy=False)
    depths_np = np.asarray(depth_np, dtype=np.float32)
    intrinsics_np = np.asarray(intrinsics, dtype=np.float32)
    extrinsics_np = np.asarray(extrinsics, dtype=np.float32)
    device = torch.device(args.device)

    return LoadedRealCase(
        case_name=f"{episode_dir.name}/{args.camera_name}",
        rgb_dir=str(rgb_dir),
        depth_dir=str(depth_dir),
        geom_path=str(geom_path),
        camera_name=str(args.camera_name),
        source_frame_indices=source_frame_indices,
        video=torch.from_numpy(video_np).to(device=device, dtype=torch.float32),
        depths=torch.from_numpy(depths_np).to(device=device, dtype=torch.float32),
        intrinsics=torch.from_numpy(intrinsics_np).to(device=device, dtype=torch.float32),
        extrinsics=torch.from_numpy(extrinsics_np).to(device=device, dtype=torch.float32),
        video_np=video_np,
        depths_np=depths_np,
        intrinsics_np=intrinsics_np,
        extrinsics_np=extrinsics_np,
    )


def _project_world_tracks(
    coords_world: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coords_world = np.asarray(coords_world, dtype=np.float32)
    track_len = coords_world.shape[0]
    intrinsics = np.asarray(intrinsics[:track_len], dtype=np.float32)
    extrinsics = np.asarray(extrinsics[:track_len], dtype=np.float32)
    coords_homo = np.concatenate(
        [coords_world, np.ones_like(coords_world[..., :1])],
        axis=-1,
    )
    camera_coords_homo = np.einsum("tij,tnj->tni", extrinsics, coords_homo)
    camera_coords = camera_coords_homo[..., :3] / np.clip(camera_coords_homo[..., 3:], a_min=1e-6, a_max=None)
    image_coords_homo = np.einsum("tij,tnj->tni", intrinsics, camera_coords)
    uv = image_coords_homo[..., :2] / np.clip(image_coords_homo[..., 2:3], a_min=1e-6, a_max=None)
    valid = np.isfinite(uv).all(axis=-1) & np.isfinite(camera_coords).all(axis=-1) & (camera_coords[..., 2] > 1e-6)
    return uv.astype(np.float32, copy=False), valid


def _summarize_distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _run_single_query(
    *,
    model: torch.nn.Module,
    case: LoadedRealCase,
    query_frame: int,
    query_grid_size: int,
    support_grid_size: int,
    num_iters: int,
    precision_mode: str,
    warmup_runs: int,
) -> dict[str, Any]:
    query_points_world = build_world_grid_queries(
        depths=case.depths_np,
        intrinsics=case.intrinsics_np,
        extrinsics=case.extrinsics_np,
        grid_size=query_grid_size,
        query_frame=query_frame,
    )
    query_point = torch.from_numpy(query_points_world).to(device=case.video.device, dtype=torch.float32)

    for _ in range(max(0, int(warmup_runs))):
        inference(
            model=model,
            video=case.video,
            depths=case.depths,
            intrinsics=case.intrinsics,
            extrinsics=case.extrinsics,
            query_point=query_point,
            num_iters=num_iters,
            grid_size=support_grid_size,
            bidrectional=False,
            tracker_precision_mode=precision_mode,
        )

    if case.video.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(case.video.device)
    _sync_if_cuda(case.video.device)
    started = time.perf_counter()
    coords, visibs, metadata = inference(
        model=model,
        video=case.video,
        depths=case.depths,
        intrinsics=case.intrinsics,
        extrinsics=case.extrinsics,
        query_point=query_point,
        num_iters=num_iters,
        grid_size=support_grid_size,
        bidrectional=False,
        tracker_precision_mode=precision_mode,
        return_metadata=True,
    )
    _sync_if_cuda(case.video.device)
    wall_time_seconds = time.perf_counter() - started
    peak_memory_gb = None
    if case.video.device.type == "cuda":
        peak_memory_gb = float(torch.cuda.max_memory_allocated(case.video.device) / (1024**3))
    return {
        "query_frame": int(query_frame),
        "query_count": int(query_point.shape[0]),
        "wall_time_seconds": float(wall_time_seconds),
        "peak_memory_gb": peak_memory_gb,
        "effective_support_query_count": int(metadata.get("effective_support_query_count", 0)),
        "coords": coords.cpu().numpy().astype(np.float32, copy=False),
        "visibs": visibs.cpu().numpy().astype(bool, copy=False),
    }


def _compare_mode_against_fp32(
    *,
    baseline_runs: list[dict[str, Any]],
    candidate_runs: list[dict[str, Any]],
    case: LoadedRealCase,
) -> dict[str, Any]:
    if len(baseline_runs) != len(candidate_runs):
        raise ValueError("Baseline and candidate run counts differ")

    all_coord_dists = []
    both_visible_coord_dists = []
    all_uv_dists = []
    both_visible_uv_dists = []
    endpoint_dists = []
    visibility_disagreements = []
    per_query = []

    for baseline_run, candidate_run in zip(baseline_runs, candidate_runs):
        if int(baseline_run["query_frame"]) != int(candidate_run["query_frame"]):
            raise ValueError("Query-frame ordering mismatch between baseline and candidate runs")
        baseline_coords = np.asarray(baseline_run["coords"], dtype=np.float32)
        candidate_coords = np.asarray(candidate_run["coords"], dtype=np.float32)
        baseline_visibs = np.asarray(baseline_run["visibs"], dtype=bool)
        candidate_visibs = np.asarray(candidate_run["visibs"], dtype=bool)
        if baseline_coords.shape != candidate_coords.shape:
            raise ValueError("Tracker output shapes differ between baseline and candidate runs")

        coord_dist = np.linalg.norm(candidate_coords - baseline_coords, axis=-1)
        finite_mask = np.isfinite(coord_dist)
        both_visible_mask = finite_mask & baseline_visibs & candidate_visibs
        all_coord_dists.append(coord_dist[finite_mask])
        both_visible_coord_dists.append(coord_dist[both_visible_mask])

        baseline_uv, baseline_uv_valid = _project_world_tracks(
            baseline_coords,
            case.intrinsics_np,
            case.extrinsics_np,
        )
        candidate_uv, candidate_uv_valid = _project_world_tracks(
            candidate_coords,
            case.intrinsics_np,
            case.extrinsics_np,
        )
        uv_dist = np.linalg.norm(candidate_uv - baseline_uv, axis=-1)
        uv_valid = np.isfinite(uv_dist) & baseline_uv_valid & candidate_uv_valid
        both_visible_uv_mask = uv_valid & baseline_visibs & candidate_visibs
        all_uv_dists.append(uv_dist[uv_valid])
        both_visible_uv_dists.append(uv_dist[both_visible_uv_mask])

        endpoint_mask = finite_mask[-1]
        endpoint_dists.append(coord_dist[-1][endpoint_mask])
        visibility_disagreements.append((baseline_visibs ^ candidate_visibs).reshape(-1).astype(np.float32))

        per_query.append(
            {
                "query_frame": int(baseline_run["query_frame"]),
                "coord_l2_m": _summarize_distribution(coord_dist[finite_mask]),
                "coord_l2_m_both_visible": _summarize_distribution(coord_dist[both_visible_mask]),
                "uv_l2_px": _summarize_distribution(uv_dist[uv_valid]),
                "uv_l2_px_both_visible": _summarize_distribution(uv_dist[both_visible_uv_mask]),
                "endpoint_coord_l2_m": _summarize_distribution(coord_dist[-1][endpoint_mask]),
                "visibility_disagreement_ratio": float(
                    np.mean((baseline_visibs ^ candidate_visibs).astype(np.float32))
                ),
            }
        )

    def _concat(parts: list[np.ndarray]) -> np.ndarray:
        non_empty = [np.asarray(part, dtype=np.float64).reshape(-1) for part in parts if np.asarray(part).size > 0]
        if not non_empty:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(non_empty, axis=0)

    visibility_disagreement = _concat(visibility_disagreements)
    return {
        "coord_l2_m": _summarize_distribution(_concat(all_coord_dists)),
        "coord_l2_m_both_visible": _summarize_distribution(_concat(both_visible_coord_dists)),
        "uv_l2_px": _summarize_distribution(_concat(all_uv_dists)),
        "uv_l2_px_both_visible": _summarize_distribution(_concat(both_visible_uv_dists)),
        "endpoint_coord_l2_m": _summarize_distribution(_concat(endpoint_dists)),
        "visibility_disagreement_ratio": (
            float(visibility_disagreement.mean()) if visibility_disagreement.size > 0 else None
        ),
        "per_query_frame": per_query,
    }


def main() -> None:
    args = parse_args()
    precision_modes = _parse_precision_modes(args.precision_modes)
    case = _load_real_case(args)
    query_frames = _parse_query_frames(args.query_frames, case.video.shape[0])
    device = torch.device(args.device)

    mode_runs: dict[str, list[dict[str, Any]]] = {}
    mode_summaries: dict[str, dict[str, Any]] = {}
    for precision_mode in precision_modes:
        model = load_tracker_model(args.checkpoint, device=device)
        per_query_runs = []
        for query_frame in query_frames:
            per_query_runs.append(
                _run_single_query(
                    model=model,
                    case=case,
                    query_frame=query_frame,
                    query_grid_size=int(args.query_grid_size),
                    support_grid_size=int(args.support_grid_size),
                    num_iters=int(args.num_iters),
                    precision_mode=precision_mode,
                    warmup_runs=int(args.warmup_runs),
                )
            )
        mode_runs[precision_mode] = per_query_runs
        mode_summaries[precision_mode] = {
            "total_wall_time_seconds": float(sum(item["wall_time_seconds"] for item in per_query_runs)),
            "mean_wall_time_seconds": float(np.mean([item["wall_time_seconds"] for item in per_query_runs])),
            "max_peak_memory_gb": (
                float(max(item["peak_memory_gb"] for item in per_query_runs if item["peak_memory_gb"] is not None))
                if any(item["peak_memory_gb"] is not None for item in per_query_runs)
                else None
            ),
            "per_query_frame": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in {"coords", "visibs"}
                }
                for item in per_query_runs
            ],
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparisons = {}
    baseline_runs = mode_runs.get("fp32")
    if baseline_runs is not None:
        for precision_mode, runs in mode_runs.items():
            if precision_mode == "fp32":
                continue
            comparisons[precision_mode] = _compare_mode_against_fp32(
                baseline_runs=baseline_runs,
                candidate_runs=runs,
                case=case,
            )

    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)

    summary = {
        "case_name": case.case_name,
        "camera_name": case.camera_name,
        "rgb_dir": case.rgb_dir,
        "depth_dir": case.depth_dir,
        "geom_path": case.geom_path,
        "device": str(device),
        "gpu_name": gpu_name,
        "query_frames": query_frames,
        "query_grid_size": int(args.query_grid_size),
        "support_grid_size": int(args.support_grid_size),
        "num_iters": int(args.num_iters),
        "fps": int(args.fps),
        "max_num_frames": int(args.max_num_frames),
        "resize_width": args.resize_width,
        "resize_height": args.resize_height,
        "source_frame_indices": case.source_frame_indices.tolist(),
        "mode_summaries": mode_summaries,
        "comparisons_vs_fp32": comparisons,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
