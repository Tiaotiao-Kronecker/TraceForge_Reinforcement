#!/usr/bin/env python3
"""
BridgeV2 数据集专用推理脚本。

在标准推理功能基础上增加：
1. 将数据集中该条轨迹的 lang.txt、obs_dict.pkl、policy_out.pkl 复制到对应输出目录；
2. 每条数据支持多相机（通常 3 个）：对 images0+depth_images0、images1+depth_images1、
   images2+depth_images2 分别做推理，输出到 out_dir/{traj_id}/images0、images1、images2。

数据集目录结构预期（与 bridge_depth / bridge_data_v2 一致）：
    base_path/
        {traj_id}/
            lang.txt
            obs_dict.pkl
            policy_out.pkl
            images0/      # 相机0 RGB
            images1/
            images2/
            depth_images0/
            depth_images1/
            depth_images2/

用法示例：
    # 建议在 conda 环境 traceforge 下运行
    python scripts/batch_inference/infer_bridge_v2.py \
        --base_path /data1/dataset/dataset/opt/dataset_temp/bridge_depth \
        --out_dir ./output_bridge_v2 \
        --frame_drop_rate 5 \
        --skip_existing

    # 只处理前 2 条轨迹、仅相机 0（测试）
    python scripts/batch_inference/infer_bridge_v2.py \
        --base_path /path/to/bridge_depth \
        --out_dir ./out_test \
        --max_trajs 2 \
        --max_cameras 1
"""

import os
import sys
import shutil
import argparse
import numpy as np
import torch
from pathlib import Path
from loguru import logger

# 项目根目录加入 path，保证可导入 infer 及 utils/models
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 同目录下的 infer 模块
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import infer


# 需要复制到输出目录的 BridgeV2 元数据文件
BRIDGE_V2_META_FILES = ["lang.txt", "obs_dict.pkl", "policy_out.pkl"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="BridgeV2 数据集推理：多相机 + 复制 lang/obs_dict/policy_out"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="BridgeV2 数据集根目录（其下为 traj_id 子目录，如 00000, 00001）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs_bridge_v2",
        help="推理结果输出根目录",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若某条轨迹的某相机输出已存在且完整则跳过该相机",
    )
    # 与 infer.py 一致的推理参数
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/tapip3d_final.pth")
    parser.add_argument(
        "--depth_pose_method",
        type=str,
        default="vggt4",
        choices=infer.video_depth_pose_dict.keys(),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_iters", type=int, default=6)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--max_num_frames", type=int, default=384)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument(
        "--use_all_trajectories",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--frame_drop_rate",
        type=int,
        default=1,
        help="每隔多少帧做一次 query",
    )
    parser.add_argument(
        "--future_len",
        type=int,
        default=128,
        help="每个 query 帧的跟踪长度（帧数）",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=20,
        help="每帧网格采样点数 grid_size x grid_size",
    )
    parser.add_argument(
        "--max_cameras",
        type=int,
        default=3,
        help="每条轨迹最多处理的相机数（0/1/2 或 0/1/2/3）",
    )
    parser.add_argument(
        "--max_trajs",
        type=int,
        default=None,
        help="最多处理多少条轨迹（用于测试），默认不限制",
    )
    parser.add_argument(
        "--traj_id",
        type=str,
        default=None,
        help="只处理指定轨迹 ID（如 05798），用于单条对比或重跑",
    )
    return parser.parse_args()


def find_bridge_v2_trajs(base_path):
    """
    扫描 base_path 下所有满足 BridgeV2 结构的轨迹目录。
    要求：至少存在一对 images* 与 depth_images*（如 images0 与 depth_images0）。
    返回: list of Path，每个为一条轨迹的目录路径。
    """
    base = Path(base_path).resolve()
    if not base.is_dir():
        return []
    trajs = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        # 至少有一个相机：imagesN 与 depth_imagesN 同时存在
        for cam in range(3):  # 0, 1, 2
            im_dir = d / f"images{cam}"
            dep_dir = d / f"depth_images{cam}"
            if im_dir.is_dir() and dep_dir.is_dir():
                has_im = any(
                    f.suffix.lower() in (".jpg", ".jpeg", ".png")
                    for f in im_dir.iterdir() if f.is_file()
                )
                has_dep = any(
                    f.suffix.lower() in (".png", ".jpg", ".jpeg")
                    for f in dep_dir.iterdir() if f.is_file()
                )
                if has_im and has_dep:
                    trajs.append(d)
                    break
    return trajs


def copy_bridge_v2_meta(traj_dir: Path, out_traj_dir: Path):
    """将 lang.txt、obs_dict.pkl、policy_out.pkl 从 traj_dir 复制到 out_traj_dir。"""
    out_traj_dir = Path(out_traj_dir)
    out_traj_dir.mkdir(parents=True, exist_ok=True)
    for name in BRIDGE_V2_META_FILES:
        src = Path(traj_dir) / name
        if src.is_file():
            dst = out_traj_dir / name
            shutil.copy2(src, dst)
            logger.debug(f"Copied {name} -> {dst}")
        else:
            logger.debug(f"Skip (not found): {src}")


def camera_output_complete(out_traj_dir: Path, camera_name: str) -> bool:
    """判断某相机输出是否已存在且完整（images 与 samples 均有文件）。"""
    cam_dir = out_traj_dir / camera_name
    images_dir = cam_dir / "images"
    samples_dir = cam_dir / "samples"
    try:
        has_images = (
            images_dir.is_dir()
            and any((images_dir / f).is_file() for f in os.listdir(images_dir))
        )
        has_samples = (
            samples_dir.is_dir()
            and any((samples_dir / f).is_file() for f in os.listdir(samples_dir))
        )
    except OSError:
        return False
    return has_images and has_samples


def run_traj(
    traj_dir: Path,
    out_traj_dir: Path,
    args,
    model_3dtracker,
    model_depth_pose,
):
    """
    对一条轨迹：先复制 lang/obs_dict/policy_out，再对每个存在的相机做推理并保存。
    """
    traj_id = traj_dir.name
    copy_bridge_v2_meta(traj_dir, out_traj_dir)

    max_cams = getattr(args, "max_cameras", 3)
    for cam in range(max_cams):
        camera_name = f"images{cam}"
        video_path = traj_dir / camera_name
        depth_path = traj_dir / f"depth_images{cam}"
        if not video_path.is_dir() or not depth_path.is_dir():
            logger.debug(f"{traj_id}: skip {camera_name} (dir missing)")
            continue

        video_path = str(video_path)
        depth_path = str(depth_path)
        cam_out = Path(out_traj_dir) / camera_name

        if args.skip_existing and camera_output_complete(Path(out_traj_dir), camera_name):
            logger.info(f"{traj_id}/{camera_name} - skip (existing)")
            continue

        logger.info(f"{traj_id}/{camera_name} - run inference")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            result = infer.process_single_video(
                video_path,
                depth_path,
                args,
                model_3dtracker,
                model_depth_pose,
            )
        except Exception as e:
            logger.error(f"{traj_id}/{camera_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        out_dir_str = str(out_traj_dir)
        # save_structured_data 内部使用 args.future_len，需注入
        infer.args = args
        infer.save_structured_data(
            video_name=camera_name,
            output_dir=out_dir_str,
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

        video_dir = os.path.join(out_dir_str, camera_name)
        data_npz_load = {
            "coords": result["coords"].cpu().numpy(),
            "extrinsics": result["full_extrinsics"].cpu().numpy(),
            "intrinsics": result["full_intrinsics"].cpu().numpy(),
            "height": result["video_tensor"].shape[-2],
            "width": result["video_tensor"].shape[-1],
            "depths": result["depths"].cpu().numpy().astype(np.float16),
            "unc_metric": result["depth_conf"].astype(np.float16),
            "visibs": result["visibs"][..., None].cpu().numpy(),
        }
        if args.save_video:
            data_npz_load["video"] = result["video_tensor"].cpu().numpy()
        save_path = os.path.join(video_dir, camera_name + ".npz")
        np.savez(save_path, **data_npz_load)
        logger.info(f"Saved {save_path}")


def main():
    args = parse_args()
    base_path = Path(args.base_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs = find_bridge_v2_trajs(base_path)
    if not trajs:
        logger.warning(f"No BridgeV2 trajs found under {base_path}")
        return
    if getattr(args, "traj_id", None):
        trajs = [t for t in trajs if t.name == args.traj_id]
        if not trajs:
            logger.warning(f"未找到轨迹 {args.traj_id}")
            return
        logger.info(f"仅处理轨迹: {args.traj_id}")
    elif getattr(args, "max_trajs", None) is not None and args.max_trajs > 0:
        trajs = trajs[: args.max_trajs]
        logger.info(f"Limiting to first {len(trajs)} trajs (--max_trajs)")

    logger.info(f"Found {len(trajs)} trajs, loading models (device={args.device})")
    model_depth_pose = infer.video_depth_pose_dict[args.depth_pose_method](args)
    model_3dtracker = infer.load_model(args.checkpoint).to(args.device)

    for i, traj_dir in enumerate(trajs):
        traj_id = traj_dir.name
        out_traj_dir = out_dir / traj_id
        logger.info(f"[{i+1}/{len(trajs)}] {traj_id}")
        run_traj(traj_dir, out_traj_dir, args, model_3dtracker, model_depth_pose)

    del model_3dtracker
    del model_depth_pose
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("BridgeV2 inference done.")


if __name__ == "__main__":
    main()
