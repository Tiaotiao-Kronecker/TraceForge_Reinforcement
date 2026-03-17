#!/usr/bin/env python3
"""
Batch Process Result Checker (uvz/depth-colored trajectory overlay).

Supports both TraceForge `v2` and `legacy` layouts.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.traceforge_artifact_utils import (
    SceneReader,
    build_sample_visualization_view,
    detect_output_layout,
    list_sample_query_frames,
    normalize_sample_data,
)

warnings.filterwarnings("ignore")


class BatchProcessChecker:
    def __init__(self, output_root: str, max_videos_to_check: int = 3, max_samples_per_video: int = 3):
        self.output_root = Path(output_root)
        self.max_videos_to_check = max_videos_to_check
        self.max_samples_per_video = max_samples_per_video
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def log_error(self, message: str) -> None:
        self.errors.append(message)
        print(f"ERROR: {message}")

    def log_warning(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARNING: {message}")

    def log_info(self, message: str) -> None:
        print(f"INFO: {message}")

    def log_success(self, message: str) -> None:
        print(f"OK: {message}")

    def iter_video_dirs(self) -> list[Path]:
        return [d for d in self.output_root.iterdir() if d.is_dir() and d.name != "visualizations"]

    def check_overall_structure(self) -> bool:
        print("\n" + "=" * 60)
        print("CHECKING OVERALL STRUCTURE")
        print("=" * 60)

        if not self.output_root.exists():
            self.log_error(f"Output root directory does not exist: {self.output_root}")
            return False

        video_dirs = self.iter_video_dirs()
        if not video_dirs:
            self.log_error(f"No video directories found in {self.output_root}")
            return False

        self.log_success(f"Found {len(video_dirs)} video directories")
        valid_videos = [video_dir for video_dir in video_dirs if self.check_video_structure(video_dir)]
        if not valid_videos:
            self.log_error("No valid video directories found")
            return False

        self.log_success(f"{len(valid_videos)}/{len(video_dirs)} video directories have valid structure")
        return True

    def check_video_structure(self, video_dir: Path) -> bool:
        video_name = video_dir.name
        samples_dir = video_dir / "samples"
        if not samples_dir.is_dir():
            self.log_warning(f"Missing samples directory in {video_name}")
            return False

        sample_files = list(samples_dir.glob("*.npz"))
        if not sample_files:
            self.log_warning(f"No NPZ files found in {video_name}/samples/")
            return False

        layout = detect_output_layout(video_dir)
        if layout == "v2":
            required = ["scene.h5", "scene_meta.json", "scene_rgb.mp4"]
            missing = [name for name in required if not (video_dir / name).is_file()]
            if missing:
                self.log_warning(f"Missing v2 artifacts in {video_name}: {missing}")
                return False
        else:
            images_dir = video_dir / "images"
            if not images_dir.is_dir():
                self.log_warning(f"Missing images directory in {video_name}")
                return False
            if not list(images_dir.glob("*.png")):
                self.log_warning(f"No PNG files found in {video_name}/images/")
                return False

        self.log_info(f"Video {video_name}: layout={layout}, samples={len(sample_files)}")
        return True

    def load_and_validate_sample(self, sample_path: Path) -> dict | None:
        try:
            sample = normalize_sample_data(sample_path)
            render_view = build_sample_visualization_view(sample)
            traj = render_view["traj_uvz"]
            keypoints = render_view["keypoints"]
            rendered_frame_count = render_view["rendered_frame_count"]

            if keypoints.ndim != 2 or keypoints.shape[1] != 2:
                self.log_warning(f"Invalid keypoints shape in {sample_path.name}: {keypoints.shape}")
                return None
            if traj.ndim != 3 or traj.shape[0] != len(keypoints) or traj.shape[2] != 3:
                self.log_warning(f"Invalid uvz trajectory shape in {sample_path.name}: {traj.shape}")
                return None

            return {
                "keypoints": keypoints,
                "traj": traj,
                "rendered_frame_count": rendered_frame_count,
                "query_frame_index": int(sample["query_frame_index"]),
            }
        except Exception as exc:
            self.log_error(f"Failed to load sample {sample_path.name}: {exc}")
            return None

    def check_keypoints_in_bounds(self, keypoints: np.ndarray, image_shape: tuple[int, ...], sample_name: str) -> bool:
        h, w = image_shape[:2]
        x_out_of_bounds = (keypoints[:, 0] < 0) | (keypoints[:, 0] >= w)
        y_out_of_bounds = (keypoints[:, 1] < 0) | (keypoints[:, 1] >= h)
        out_of_bounds = x_out_of_bounds | y_out_of_bounds
        if np.any(out_of_bounds):
            self.log_warning(f"{sample_name}: {int(np.sum(out_of_bounds))}/{len(keypoints)} keypoints out of bounds")
            return False
        self.log_info(f"{sample_name}: All keypoints within bounds")
        return True

    def create_visualization(
        self,
        image: np.ndarray,
        data: dict,
        output_path: Path,
        video_name: str,
        frame_idx: int,
    ) -> bool:
        try:
            h, w = image.shape[:2]
            keypoints = data["keypoints"]
            traj = data["traj"]
            rendered_frame_count = np.asarray(data["rendered_frame_count"], dtype=np.uint16)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            keypoint_color = plt.cm.tab20(0)

            ax1.imshow(image)
            ax1.set_title(f"{video_name} Frame {frame_idx} - Keypoints")
            for kp in keypoints:
                ax1.scatter(kp[0], kp[1], c=[keypoint_color], s=50, alpha=0.8, edgecolors="white", linewidths=1)
            ax1.set_xlim(0, w)
            ax1.set_ylim(h, 0)
            ax1.axis("off")

            ax2.imshow(image)
            ax2.set_title(f"{video_name} Frame {frame_idx} - UVZ Trajectories (Depth as Color)")
            for kp, trajectory in zip(keypoints, traj):
                ax2.scatter(kp[0], kp[1], c=["red"], s=100, alpha=0.9, edgecolors="white", linewidths=2, marker="o")
                finite_mask = np.isfinite(trajectory).all(axis=1)
                valid_traj = trajectory[finite_mask]
                if len(valid_traj) < 2:
                    continue

                x_coords = valid_traj[:, 0]
                y_coords = valid_traj[:, 1]
                depth_values = valid_traj[:, 2]
                depth_min = float(np.min(depth_values))
                depth_max = float(np.max(depth_values))
                if depth_max > depth_min:
                    depth_normalized = (depth_values - depth_min) / (depth_max - depth_min)
                else:
                    depth_normalized = np.full_like(depth_values, 0.5)

                for idx in range(len(x_coords) - 1):
                    segment_depth = 0.5 * (depth_normalized[idx] + depth_normalized[idx + 1])
                    ax2.plot(
                        [x_coords[idx], x_coords[idx + 1]],
                        [y_coords[idx], y_coords[idx + 1]],
                        color=plt.cm.viridis(segment_depth),
                        alpha=0.8,
                        linewidth=2,
                    )
                ax2.scatter(x_coords, y_coords, c=plt.cm.viridis(depth_normalized), s=40, alpha=0.9, marker="x", linewidths=2)

            ax2.set_xlim(0, w)
            ax2.set_ylim(h, 0)
            ax2.axis("off")

            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax2, shrink=0.8, aspect=20)
            cbar.set_label("Normalized Depth (0=Near, 1=Far)", rotation=270, labelpad=15)

            median_rendered_frames = float(np.median(rendered_frame_count)) if rendered_frame_count.size > 0 else 0.0
            info_text = f"Keypoints: {len(keypoints)}\nMedian rendered frames/track: {median_rendered_frames:.1f}"
            fig.text(
                0.02,
                0.98,
                info_text,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            self.log_success(f"Created visualization: {output_path}")
            return True
        except Exception as exc:
            self.log_error(f"Failed to create visualization for {video_name} frame {frame_idx}: {exc}")
            return False

    def check_and_visualize_data(self) -> None:
        print("\n" + "=" * 60)
        print("CHECKING DATA AND CREATING VISUALIZATIONS")
        print("=" * 60)

        video_dirs = sorted(self.iter_video_dirs())[: self.max_videos_to_check]
        vis_output_dir = self.output_root / "visualizations"
        vis_output_dir.mkdir(exist_ok=True)

        for video_dir in video_dirs:
            video_name = video_dir.name
            self.log_info(f"Processing video: {video_name}")
            samples_dir = video_dir / "samples"
            if not samples_dir.is_dir():
                continue

            sample_files = sorted(samples_dir.glob("*.npz"))[: self.max_samples_per_video]
            with SceneReader(video_dir) as scene_reader:
                for sample_file in sample_files:
                    data = self.load_and_validate_sample(sample_file)
                    if data is None:
                        continue

                    frame_idx = int(data["query_frame_index"])
                    try:
                        image = scene_reader.get_rgb_frame(frame_idx)
                        self.check_keypoints_in_bounds(data["keypoints"], image.shape, f"{video_name}_{frame_idx}")
                        vis_path = vis_output_dir / f"{video_name}_frame_{frame_idx}_visualization.png"
                        self.create_visualization(image, data, vis_path, video_name, frame_idx)
                    except Exception as exc:
                        self.log_error(f"Failed to process {video_name} frame {frame_idx}: {exc}")

    def generate_summary_report(self) -> None:
        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)
        print(f"Output directory: {self.output_root}")

        video_dirs = self.iter_video_dirs()
        total_samples = sum(len(list((video_dir / "samples").glob("*.npz"))) for video_dir in video_dirs if (video_dir / "samples").is_dir())
        total_query_frames = sum(len(list_sample_query_frames(video_dir, video_dir.name)) for video_dir in video_dirs)

        print(f"Total videos: {len(video_dirs)}")
        print(f"Total samples: {total_samples}")
        print(f"Total query frames: {total_query_frames}")
        print(f"Videos checked: {min(len(video_dirs), self.max_videos_to_check)}")

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        if not self.errors and not self.warnings:
            print("\nAll checks passed successfully.")

        vis_dir = self.output_root / "visualizations"
        if vis_dir.exists():
            vis_files = list(vis_dir.glob("*.png"))
            print(f"\nCreated {len(vis_files)} visualization files in: {vis_dir}")

    def run_checks(self) -> bool:
        print("Starting batch process result checker...")
        if not self.check_overall_structure():
            self.log_error("Structure check failed. Stopping.")
            self.generate_summary_report()
            return False
        self.check_and_visualize_data()
        self.generate_summary_report()
        return len(self.errors) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check batch processing results")
    parser.add_argument("output_root", help="Root directory containing batch processing results")
    parser.add_argument("--max-videos", type=int, default=3, help="Maximum number of videos to check")
    parser.add_argument("--max-samples", type=int, default=3, help="Maximum number of samples per video to visualize")
    args = parser.parse_args()

    checker = BatchProcessChecker(
        args.output_root,
        max_videos_to_check=args.max_videos,
        max_samples_per_video=args.max_samples,
    )
    success = checker.run_checks()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
