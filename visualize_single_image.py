import os
import sys
import argparse
import numpy as np
import cv2 as cv
from PIL import Image
import viser
import viser.extras
import viser.transforms as tf
from utils.viser_utils import define_track_colors
from utils.threed_utils import unproject_by_depth, inverse_intrinsic, get_meshgrid
from loguru import logger


"""
Usage:
    python visualize_single_image.py \
        --npz_path <output_dir>/<video_name>/samples/<video_name>_<frame>.npz \
        --image_path <output_dir>/<video_name>/images/<video_name>_<frame>.png \
        --depth_path <output_dir>/<video_name>/depth/<video_name>_<frame>.png \
        --port 8080
"""

def load_depth_from_path(depth_path):
    """Load depth data from either PNG or NPZ file"""
    if depth_path.endswith(".npz"):
        # Load raw depth from NPZ
        depth_data = np.load(depth_path)
        depth = depth_data["depth"]
        depth_data.close()
        return depth
    elif depth_path.endswith(".png"):
        # Check if there's a corresponding raw NPZ file
        base_path = depth_path[:-4]  # Remove .png extension
        raw_npz_path = f"{base_path}_raw.npz"
        if os.path.exists(raw_npz_path):
            depth_data = np.load(raw_npz_path)
            depth = depth_data["depth"]
            depth_data.close()
            return depth
        else:
            # Load from PNG (16-bit, need to convert back from mm)
            depth_img = np.array(Image.open(depth_path))
            depth = depth_img.astype(np.float32) / 10000.0  # Convert back from mm
            return depth
    else:
        raise ValueError(f"Unsupported depth file format: {depth_path}")


def get_camera_params_from_main_npz(episode_dir, frame_idx):
    """Get camera intrinsics and extrinsics from the main NPZ file"""
    # Look for the main NPZ file in the episode directory
    episode_name = os.path.basename(episode_dir)
    main_npz_path = os.path.join(episode_dir, f"{episode_name}.npz")

    if os.path.exists(main_npz_path):
        data = np.load(main_npz_path)

        # Get camera parameters for the specific frame
        intrinsics = data["intrinsics"][frame_idx]  # (3, 3)
        extrinsics = data["extrinsics"][frame_idx]  # (4, 4) - world to camera
        c2w = np.linalg.inv(extrinsics)  # camera to world
        height, width = int(data["height"]), int(data["width"])

        data.close()    
    
    else:
        print(f"Main NPZ file not found: {main_npz_path}")
        # Use hardcoded camera parameters
        intrinsics = np.array([
            [257.91296, 0.0, 259.0],
            [0.0, 261.4576, 161.0],
            [0.0, 0.0, 1.0]
        ])
        
        extrinsics = np.array([
            [1.0000000e+00, 4.0706014e-05, 8.9567264e-05, 9.0881156e-05],
            [-4.0680898e-05, 9.9999994e-01, -2.8039535e-04, 3.5203320e-05],
            [-8.9578680e-05, 2.8039169e-04, 9.9999994e-01, -2.7687754e-04],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        c2w = np.linalg.inv(extrinsics)  # camera to world
        height, width = 322, 518

    return {
        "K": intrinsics,
        "c2w": c2w,
        "w2c": extrinsics,
        "height": height,
        "width": width,
    }


def convert_image_coords_to_world(traj_image_coords, camera_params):
    """
    Convert trajectories from image coordinates (x,y,z) to world coordinates

    Args:
        traj_image_coords: (N, H, 3) trajectories in image coordinates
        camera_params: dict with camera parameters

    Returns:
        traj_world: (N, H, 3) trajectories in world coordinates
    """
    N, H, _ = traj_image_coords.shape
    K = camera_params["K"]
    c2w = camera_params["c2w"]

    # Reshape for batch processing
    traj_flat = traj_image_coords.reshape(N * H, 3)  # (N*H, 3)

    world_points = []
    for i in range(N * H):
        x, y, z = traj_flat[i]

        # Convert pixel coordinates to normalized camera coordinates
        x_norm = (x - K[0, 2]) / K[0, 0]
        y_norm = (y - K[1, 2]) / K[1, 1]

        # Create point in camera coordinates
        cam_point = np.array([x_norm * z, y_norm * z, z, 1.0])

        # Transform to world coordinates
        world_point = c2w @ cam_point
        world_points.append(world_point[:3])

    world_points = np.array(world_points).reshape(N, H, 3)
    return world_points


def visualize_single_image(npz_path, image_path, depth_path, port=8080):
    """Visualize 3D scene with trajectories for a single image"""

    # Parse paths to get episode directory and frame index
    sample_dir = os.path.dirname(npz_path)
    episode_dir = os.path.dirname(sample_dir)

    # Extract frame index from NPZ filename
    npz_filename = os.path.basename(npz_path)
    # Format: P01_101_ep1_40.npz -> frame 40
    frame_idx = int(npz_filename.split("_")[-1].split(".")[0])

    logger.info(f"Loading data for frame {frame_idx} from {episode_dir}")

    # Load sample data
    sample_data = np.load(npz_path)
    traj_image_coords = sample_data[
        "traj"
    ]  # (N, H, 3) - trajectories in image coordinates
    keypoints = sample_data["keypoints"]  # (N, 2) - starting keypoints
    valid_steps = sample_data["valid_steps"]  # (H,) - validity mask
    sample_data.close()

    logger.info(
        f"Loaded {len(traj_image_coords)} trajectories with horizon {traj_image_coords.shape[1]}"
    )

    # Load RGB image
    image = np.array(Image.open(image_path)).astype(np.float32) / 255.0
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)

    # Load depth data
    depth = load_depth_from_path(depth_path)

    logger.info(f"Image shape: {image.shape}, Depth shape: {depth.shape}")

    # Get camera parameters
    camera_params = get_camera_params_from_main_npz(episode_dir, frame_idx)

    # Convert trajectories from image coordinates to world coordinates
    traj_world = convert_image_coords_to_world(traj_image_coords, camera_params)

    # Create point cloud from RGB image and depth
    H, W = depth.shape
    points_xyz = unproject_by_depth(
        depth=depth[None, None],  # (1, 1, H, W)
        K=camera_params["K"][None],  # (1, 3, 3)
        c2w=camera_params["c2w"][None],  # (1, 4, 4)
    )[0].transpose(1, 2, 0)  # (H, W, 3)

    # Downsample point cloud for visualization
    downsample_factor = 4
    points_xyz_ds = points_xyz[::downsample_factor, ::downsample_factor].reshape(-1, 3)
    points_rgb_ds = image[::downsample_factor, ::downsample_factor].reshape(-1, 3)

    # Filter out invalid points (depth = 0 or too far)
    valid_mask = (points_xyz_ds[:, 2] > 0) & (
        points_xyz_ds[:, 2] < 10.0
    )  # Filter points within 10m
    points_xyz_ds = points_xyz_ds[valid_mask]
    points_rgb_ds = points_rgb_ds[valid_mask]

    logger.info(f"Point cloud: {len(points_xyz_ds)} points after filtering")

    # Define colors for trajectories
    track_colors = define_track_colors(traj_world, colormap='turbo')

    # Start Viser server
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")

    logger.info(f"Started Viser server at http://localhost:{port}")

    # Add GUI controls
    with server.gui.add_folder("Visualization"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.02, step=1e-3, initial_value=0.006
        )
        gui_track_width = server.gui.add_slider(
            "Track width", min=0.5, max=5.0, step=0.5, initial_value=4.0
        )
        gui_track_length = server.gui.add_slider(
            "Track length",
            min=1,
            max=traj_world.shape[1],
            step=1,
            initial_value=min(30, traj_world.shape[1]),
        )
        gui_show_pointcloud = server.gui.add_checkbox("Show point cloud", True)
        gui_show_tracks = server.gui.add_checkbox("Show tracks", True)
        gui_show_keypoints = server.gui.add_checkbox("Show keypoints", False)
        gui_keypoint_size = server.gui.add_slider(
            "Keypoint size", min=0.005, max=0.05, step=0.005, initial_value=0.005
        )
        gui_show_frustum = server.gui.add_checkbox("Show camera frustum", True)
        gui_show_axes = server.gui.add_checkbox("Show world axes", True)

    # Add point cloud
    point_cloud_handle = server.scene.add_point_cloud(
        name="point_cloud",
        points=points_xyz_ds,
        colors=points_rgb_ds,
        point_size=gui_point_size.value,
        point_shape="rounded",
    )

    # Add trajectories as line segments
    track_handles = []
    keypoint_handles = []

    for i, (traj, color) in enumerate(zip(traj_world, track_colors)):
        # Create line segments for trajectory
        valid_traj = traj[: gui_track_length.value]  # Use only the first N points
        if len(valid_traj) > 1:
            segments = []
            seg_colors = []
            for j in range(len(valid_traj) - 1):
                segments.append([valid_traj[j], valid_traj[j + 1]])
                seg_colors.append([color, color])

            if segments:
                track_handle = server.scene.add_line_segments(
                    name=f"track_{i}",
                    points=np.array(segments),
                    colors=np.array(seg_colors),
                    line_width=gui_track_width.value,
                )
                track_handles.append(track_handle)

        # Add starting keypoint in 3D (convert from image coordinates to world)
        kp_x, kp_y = keypoints[i]
        kp_depth = (
            depth[int(kp_y), int(kp_x)] if 0 <= kp_x < W and 0 <= kp_y < H else 1.0
        )

        # Convert keypoint to world coordinates
        kp_world = convert_image_coords_to_world(
            np.array([[[kp_x, kp_y, kp_depth]]]), camera_params
        )[0, 0]

        keypoint_handle = server.scene.add_point_cloud(
            name=f"keypoint_{i}",
            points=kp_world[None],
            colors=color[None],
            point_size=gui_keypoint_size.value,
            point_shape="circle",
        )
        keypoint_handle.visible = False  # Initially hidden
        keypoint_handles.append(keypoint_handle)

    # Add camera frame
    c2w = camera_params["c2w"]
    fov = 2 * np.arctan2(camera_params["height"] / 2, camera_params["K"][0, 0])
    aspect = camera_params["width"] / camera_params["height"]

    frustum_handle = server.scene.add_camera_frustum(
        name="camera_frustum",
        fov=fov,
        aspect=aspect,
        scale=0.1,
        image=image,
        wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
        position=c2w[:3, 3],
    )
    frustum_handle.visible = gui_show_frustum.value 

    # Add coordinate axes at world origin
    axes_handle = server.scene.add_line_segments(
        name="world_axes",
        points=np.array([
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
        ]),
        colors=np.array([
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ]),
        line_width=3.0,
    )
    axes_handle.visible = gui_show_axes.value

    # Update callbacks for GUI controls
    @gui_point_size.on_update
    def _(_) -> None:
        if gui_show_pointcloud.value:
            point_cloud_handle.point_size = gui_point_size.value

    @gui_track_width.on_update
    def _(_) -> None:
        if gui_show_tracks.value:
            for handle in track_handles:
                handle.line_width = gui_track_width.value

    @gui_keypoint_size.on_update
    def _(_) -> None:
        if gui_show_keypoints.value:
            for handle in keypoint_handles:
                handle.point_size = gui_keypoint_size.value

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
    def _(_ev):
        frustum_handle.visible = gui_show_frustum.value

    @gui_show_axes.on_update
    def _(_ev):
        axes_handle.visible = gui_show_axes.value

    @gui_track_length.on_update
    def _(_) -> None:
        # Remove old track handles
        for handle in track_handles:
            handle.remove()
        track_handles.clear()

        # Create new tracks with updated length
        for i, (traj, color) in enumerate(zip(traj_world, track_colors)):
            valid_traj = traj[: gui_track_length.value]
            if len(valid_traj) > 1:
                segments = []
                seg_colors = []
                for j in range(len(valid_traj) - 1):
                    segments.append([valid_traj[j], valid_traj[j + 1]])
                    seg_colors.append([color, color])

                if segments:
                    track_handle = server.scene.add_line_segments(
                        name=f"track_{i}_updated",
                        points=np.array(segments),
                        colors=np.array(seg_colors),
                        line_width=gui_track_width.value,
                    )
                    track_handle.visible = gui_show_tracks.value
                    track_handles.append(track_handle)

    logger.info("Visualization ready! Press Ctrl+C to exit.")

    # Keep the server running
    try:
        while True:
            import time

            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize 3D scene with trajectories for a single image"
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help="Path to the NPZ file containing trajectory data",
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the RGB image"
    )
    parser.add_argument(
        "--depth_path", type=str, required=True, help="Path to the depth image/data"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for Viser server (default: 8080)"
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.npz_path):
        raise FileNotFoundError(f"NPZ file not found: {args.npz_path}")
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")
    if not os.path.exists(args.depth_path):
        raise FileNotFoundError(f"Depth file not found: {args.depth_path}")

    visualize_single_image(args.npz_path, args.image_path, args.depth_path, args.port)
