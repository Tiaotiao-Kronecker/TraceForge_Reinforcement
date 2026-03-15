#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import viser
import viser.transforms as tf
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.traceforge_artifact_utils import (
    SceneReader,
    build_pointcloud_from_frame,
    normalize_sample_data,
    traj_uvz_to_world,
)
from utils.viser_utils import define_track_colors

RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def load_depth_override(depth_path: str | Path) -> np.ndarray:
    depth_path = Path(depth_path)
    if depth_path.suffix == ".npz":
        depth_data = np.load(depth_path)
        try:
            return depth_data["depth"].astype(np.float32)
        finally:
            depth_data.close()

    if depth_path.suffix != ".png":
        raise ValueError(f"Unsupported depth file format: {depth_path}")

    raw_npz_path = depth_path.with_suffix("")
    raw_npz_path = raw_npz_path.parent / f"{raw_npz_path.name}_raw.npz"
    if raw_npz_path.is_file():
        depth_data = np.load(raw_npz_path)
        try:
            return depth_data["depth"].astype(np.float32)
        finally:
            depth_data.close()

    raise ValueError(
        f"PNG depth file found but no corresponding raw NPZ file: {raw_npz_path}. "
        "Please use the _raw.npz file for accurate depth values."
    )


def load_rgb_override(image_path: str | Path) -> np.ndarray:
    return np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def resize_rgb_if_needed(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if rgb.shape[:2] == (target_h, target_w):
        return rgb
    resized = Image.fromarray(rgb).resize((target_w, target_h), RESAMPLE_LANCZOS)
    return np.array(resized, dtype=np.uint8)


def resolve_scene_inputs(
    npz_path: str | Path,
    image_path: str | None = None,
    depth_path: str | None = None,
) -> dict:
    sample = normalize_sample_data(npz_path)
    episode_dir = Path(npz_path).resolve().parent.parent
    query_frame_idx = int(sample["query_frame_index"])

    with SceneReader(episode_dir) as scene_reader:
        intrinsics_all, extrinsics_all = scene_reader.get_camera_arrays()
        intrinsics = intrinsics_all[query_frame_idx].astype(np.float32)
        w2c = extrinsics_all[query_frame_idx].astype(np.float32)
        rgb = (
            load_rgb_override(image_path)
            if image_path is not None
            else scene_reader.get_rgb_frame(query_frame_idx)
        )
        depth = (
            load_depth_override(depth_path)
            if depth_path is not None
            else scene_reader.get_depth_frame(query_frame_idx)
        )

    rgb = resize_rgb_if_needed(rgb, depth.shape)

    traj_uvz = sample["traj_uvz"].astype(np.float32)
    keypoints = sample["keypoints"].astype(np.float32)
    traj_valid_mask = sample["traj_valid_mask"].astype(bool, copy=False)
    segment_frame_indices = np.asarray(sample["segment_frame_indices"], dtype=np.int32)

    if sample.get("frame_aligned", False) and len(segment_frame_indices) < traj_uvz.shape[1]:
        traj_uvz = traj_uvz[:, : len(segment_frame_indices)]

    raw_num_trajectories = int(traj_uvz.shape[0])
    traj_uvz = traj_uvz[traj_valid_mask]
    keypoints = keypoints[traj_valid_mask]
    traj_world = traj_uvz_to_world(traj_uvz, intrinsics, w2c)
    point_cloud_xyz, point_cloud_rgb = build_pointcloud_from_frame(
        depth=depth,
        rgb=rgb,
        intrinsics=intrinsics,
        w2c=w2c,
        downsample=4,
    )

    return {
        "episode_dir": episode_dir,
        "query_frame_idx": query_frame_idx,
        "traj_world": traj_world,
        "keypoints": keypoints,
        "raw_num_trajectories": raw_num_trajectories,
        "segment_frame_count": len(segment_frame_indices),
        "rgb": rgb,
        "depth": depth,
        "intrinsics": intrinsics,
        "w2c": w2c,
        "point_cloud_xyz": point_cloud_xyz,
        "point_cloud_rgb": point_cloud_rgb,
    }


def visualize_single_image(
    npz_path: str,
    image_path: str | None = None,
    depth_path: str | None = None,
    port: int = 8080,
) -> None:
    scene = resolve_scene_inputs(npz_path, image_path=image_path, depth_path=depth_path)
    traj_world = scene["traj_world"]
    raw_num_trajectories = scene["raw_num_trajectories"]
    rgb = scene["rgb"].astype(np.float32) / 255.0
    intrinsics = scene["intrinsics"]
    w2c = scene["w2c"]
    point_cloud_xyz = scene["point_cloud_xyz"]
    point_cloud_rgb = scene["point_cloud_rgb"]
    keypoints_world = traj_world[:, 0] if traj_world.shape[1] > 0 else np.zeros((0, 3), dtype=np.float32)

    logger.info(
        f"Loaded frame {scene['query_frame_idx']} from {scene['episode_dir']}: "
        f"{len(traj_world)}/{raw_num_trajectories} valid trajectories"
    )
    logger.info(f"Point cloud: {len(point_cloud_xyz)} points after filtering")

    track_colors = define_track_colors(traj_world, colormap="turbo")

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")
    logger.info(f"Started Viser server at http://localhost:{port}")

    num_trajectories = len(traj_world)
    if num_trajectories == 0:
        logger.warning("No valid trajectories remain after traj_valid_mask filtering")

    with server.gui.add_folder("Visualization"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.02, step=1e-3, initial_value=0.006
        )
        gui_track_width = server.gui.add_slider(
            "Track width", min=0.5, max=10.0, step=0.5, initial_value=4.0
        )
        gui_track_length = server.gui.add_slider(
            "Track length",
            min=0,
            max=traj_world.shape[1],
            step=1,
            initial_value=min(scene["segment_frame_count"], 30, traj_world.shape[1]),
        )
        gui_num_trajectories = server.gui.add_slider(
            "Number of trajectories",
            min=0,
            max=num_trajectories,
            step=1,
            initial_value=min(100, num_trajectories),
        )
        gui_num_keypoints = server.gui.add_slider(
            "Number of keypoints",
            min=0,
            max=num_trajectories,
            step=1,
            initial_value=min(100, num_trajectories),
        )
        gui_show_pointcloud = server.gui.add_checkbox("Show point cloud", True)
        gui_show_tracks = server.gui.add_checkbox("Show tracks", True)
        gui_show_keypoints = server.gui.add_checkbox("Show keypoints", False)
        gui_keypoint_size = server.gui.add_slider(
            "Keypoint size", min=0.001, max=0.1, step=0.001, initial_value=0.005
        )
        gui_show_frustum = server.gui.add_checkbox("Show camera frustum", True)
        gui_show_axes = server.gui.add_checkbox("Show world axes", True)

    point_cloud_handle = server.scene.add_point_cloud(
        name="point_cloud",
        points=point_cloud_xyz,
        colors=point_cloud_rgb,
        point_size=gui_point_size.value,
        point_shape="rounded",
    )

    track_handles = []
    keypoint_handles = []

    def update_trajectories() -> None:
        for handle in track_handles:
            handle.remove()
        track_handles.clear()

        num_traj_to_show = min(int(gui_num_trajectories.value), num_trajectories)
        step_limit = int(gui_track_length.value)
        for i, (traj, color) in enumerate(
            zip(traj_world[:num_traj_to_show], track_colors[:num_traj_to_show])
        ):
            valid_traj = traj[:step_limit]
            finite_steps = np.isfinite(valid_traj).all(axis=1)
            if len(valid_traj) <= 1 or np.count_nonzero(finite_steps) <= 1:
                continue

            segments = []
            seg_colors = []
            for j in range(len(valid_traj) - 1):
                if not (finite_steps[j] and finite_steps[j + 1]):
                    continue
                segments.append([valid_traj[j], valid_traj[j + 1]])
                seg_colors.append([color, color])

            if not segments:
                continue

            track_handle = server.scene.add_line_segments(
                name=f"track_{i}",
                points=np.array(segments),
                colors=np.array(seg_colors),
                line_width=gui_track_width.value,
            )
            track_handle.visible = gui_show_tracks.value
            track_handles.append(track_handle)

    def update_keypoints() -> None:
        for handle in keypoint_handles:
            handle.remove()
        keypoint_handles.clear()

        num_kp_to_show = min(int(gui_num_keypoints.value), num_trajectories)
        for i in range(num_kp_to_show):
            kp_world = keypoints_world[i]
            if not np.isfinite(kp_world).all():
                continue
            keypoint_handle = server.scene.add_point_cloud(
                name=f"keypoint_{i}",
                points=kp_world[None],
                colors=track_colors[i][None],
                point_size=gui_keypoint_size.value,
                point_shape="circle",
            )
            keypoint_handle.visible = gui_show_keypoints.value
            keypoint_handles.append(keypoint_handle)

    update_trajectories()
    update_keypoints()

    c2w = np.linalg.inv(w2c)
    fov = 2 * np.arctan2(rgb.shape[0] / 2, intrinsics[0, 0])
    aspect = rgb.shape[1] / rgb.shape[0]
    frustum_handle = server.scene.add_camera_frustum(
        name="camera_frustum",
        fov=fov,
        aspect=aspect,
        scale=0.1,
        image=rgb,
        wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3],
    )
    frustum_handle.visible = gui_show_frustum.value

    axes_handle = server.scene.add_line_segments(
        name="world_axes",
        points=np.array(
            [
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
            ]
        ),
        colors=np.array(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            ]
        ),
        line_width=3.0,
    )
    axes_handle.visible = gui_show_axes.value

    @gui_point_size.on_update
    def _(_) -> None:
        if gui_show_pointcloud.value:
            point_cloud_handle.point_size = gui_point_size.value

    @gui_track_width.on_update
    def _(_) -> None:
        update_trajectories()

    @gui_keypoint_size.on_update
    def _(_) -> None:
        update_keypoints()

    @gui_show_pointcloud.on_update
    def _(_) -> None:
        point_cloud_handle.visible = gui_show_pointcloud.value

    @gui_show_tracks.on_update
    def _(_) -> None:
        for handle in track_handles:
            handle.visible = gui_show_tracks.value

    @gui_show_keypoints.on_update
    def _(_) -> None:
        for handle in keypoint_handles:
            handle.visible = gui_show_keypoints.value

    @gui_show_frustum.on_update
    def _(_) -> None:
        frustum_handle.visible = gui_show_frustum.value

    @gui_show_axes.on_update
    def _(_) -> None:
        axes_handle.visible = gui_show_axes.value

    @gui_track_length.on_update
    def _(_) -> None:
        update_trajectories()

    @gui_num_trajectories.on_update
    def _(_) -> None:
        update_trajectories()

    @gui_num_keypoints.on_update
    def _(_) -> None:
        update_keypoints()

    logger.info("Visualization ready! Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize 3D scene with trajectories for a single query frame"
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help="Path to a sample NPZ file under <episode>/samples/",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Optional RGB image override. If omitted, load from episode artifacts.",
    )
    parser.add_argument(
        "--depth_path",
        type=str,
        default=None,
        help="Optional depth override (.npz or .png with matching _raw.npz). If omitted, load from episode artifacts.",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for Viser server (default: 8080)"
    )

    args = parser.parse_args()
    npz_path = Path(args.npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    if args.image_path is not None and not Path(args.image_path).exists():
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    if args.depth_path is not None and not Path(args.depth_path).exists():
        raise FileNotFoundError(f"Depth file not found: {args.depth_path}")

    visualize_single_image(
        str(npz_path),
        image_path=args.image_path,
        depth_path=args.depth_path,
        port=args.port,
    )
