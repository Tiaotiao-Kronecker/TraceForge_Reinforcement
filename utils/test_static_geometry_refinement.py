import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from utils.external_wobble_diagnostics import (
    _rasterize_world_points_to_depth_image,
    compute_static_geometry_consistency,
)
from utils.static_geometry_refinement import (
    audit_static_geometry_heavy_tail,
    refine_extrinsics_w2c_static_background,
    summarize_spatial_tail_clusters,
)


def _pose_w2c(tx: float, ty: float, tz: float, *, rot_y_deg: float = 0.0) -> np.ndarray:
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = Rotation.from_euler("y", rot_y_deg, degrees=True).as_matrix().astype(np.float32)
    c2w[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return np.linalg.inv(c2w).astype(np.float32)


def _render_depth_sequence(
    world_points: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    return np.stack(
        [
            _rasterize_world_points_to_depth_image(
                world_points,
                intrinsics=intrinsics[frame_idx],
                w2c=extrinsics[frame_idx],
                height=height,
                width=width,
                min_depth=0.2,
                max_depth=10.0,
            )
            for frame_idx in range(extrinsics.shape[0])
        ],
        axis=0,
    ).astype(np.float32)


class StaticGeometryRefinementTests(unittest.TestCase):
    def test_summarize_spatial_tail_clusters_groups_by_cell(self):
        keypoints = np.array(
            [
                [10.0, 10.0],
                [18.0, 18.0],
                [70.0, 12.0],
                [74.0, 20.0],
            ],
            dtype=np.float32,
        )
        drift_px = np.array([35.0, 28.0, 2.0, 3.0], dtype=np.float32)
        valid_mask = np.array([True, True, True, True], dtype=bool)

        cells = summarize_spatial_tail_clusters(
            keypoints,
            drift_px,
            valid_mask,
            image_height=96,
            image_width=96,
            cell_size_px=32,
            tail_threshold_px=20.0,
            top_k=2,
        )

        self.assertEqual(len(cells), 2)
        self.assertEqual(cells[0]["cell_row"], 0)
        self.assertEqual(cells[0]["cell_col"], 0)
        self.assertEqual(cells[0]["tail_track_count"], 2)

    def test_pose_refinement_reduces_static_geometry_tail_on_synthetic_scene(self):
        height = 64
        width = 64
        fx = fy = 60.0
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        intrinsics_single = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        intrinsics = np.repeat(intrinsics_single[None], 3, axis=0)

        xs = np.linspace(-0.9, 0.9, 180, dtype=np.float32)
        ys = np.linspace(-0.6, 0.6, 120, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        grid_z = (
            1.4
            + 0.18 * grid_x
            - 0.10 * grid_y
            + 0.05 * np.sin(4.0 * grid_x)
            + 0.04 * np.cos(3.0 * grid_y)
        ).astype(np.float32)
        world_points = np.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)],
            axis=1,
        ).astype(np.float32)

        true_extrinsics = np.stack(
            [
                _pose_w2c(0.0, 0.0, 0.0, rot_y_deg=0.0),
                _pose_w2c(0.04, 0.0, 0.0, rot_y_deg=2.0),
                _pose_w2c(0.08, 0.0, 0.0, rot_y_deg=4.0),
            ],
            axis=0,
        )
        depth_frames = _render_depth_sequence(
            world_points,
            intrinsics,
            true_extrinsics,
            height=height,
            width=width,
        )

        perturbed_extrinsics = true_extrinsics.copy()
        perturbed_extrinsics[1] = _pose_w2c(0.075, 0.0, 0.0, rot_y_deg=4.5)
        perturbed_extrinsics[2] = _pose_w2c(0.115, 0.0, 0.0, rot_y_deg=6.5)

        before = compute_static_geometry_consistency(
            depth_frames,
            intrinsics,
            perturbed_extrinsics,
            query_frame=0,
            grid_size=12,
            min_query_depth_m=0.2,
            min_border_dist_px=2.0,
            min_depth=0.2,
            max_depth=10.0,
        )
        result = refine_extrinsics_w2c_static_background(
            depth_frames,
            intrinsics,
            perturbed_extrinsics,
            query_frames=[0],
            grid_size=12,
            min_query_depth_m=0.2,
            min_border_dist_px=2.0,
            min_depth=0.2,
            max_depth=10.0,
            min_target_border_dist_px=1.0,
            max_depth_error_m=0.30,
            max_world_error_m=0.30,
            max_query_reproj_error_px=12.0,
            min_correspondences=20,
            temporal_smooth_radius=0,
            temporal_regularization_weight=0.0,
            max_translation_delta_m=0.08,
            max_rotation_delta_deg=4.0,
        )
        refined_extrinsics = np.asarray(result["extrinsics_w2c"], dtype=np.float32)
        after = compute_static_geometry_consistency(
            depth_frames,
            intrinsics,
            refined_extrinsics,
            query_frame=0,
            grid_size=12,
            min_query_depth_m=0.2,
            min_border_dist_px=2.0,
            min_depth=0.2,
            max_depth=10.0,
        )

        self.assertLess(
            float(after["final_query_reproj_drift_p95_px"]),
            float(before["final_query_reproj_drift_p95_px"]),
        )
        self.assertLess(
            float(after["final_query_reproj_global_disp_px"]),
            float(before["final_query_reproj_global_disp_px"]),
        )
        self.assertGreater(result["support_count_summary"]["finite_count"], 0)

    def test_heavy_tail_audit_marks_worst_frame(self):
        height = 48
        width = 48
        intrinsics_single = np.array(
            [
                [45.0, 0.0, (width - 1) * 0.5],
                [0.0, 45.0, (height - 1) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        intrinsics = np.repeat(intrinsics_single[None], 3, axis=0)
        xs = np.linspace(-0.6, 0.6, 120, dtype=np.float32)
        ys = np.linspace(-0.4, 0.4, 80, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        grid_z = (1.3 + 0.12 * grid_x - 0.08 * grid_y + 0.03 * np.sin(5.0 * grid_x)).astype(np.float32)
        world_points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1), grid_z.reshape(-1)], axis=1).astype(
            np.float32
        )
        true_extrinsics = np.stack([_pose_w2c(0.0, 0.0, 0.0)] * 3, axis=0)
        depth_frames = _render_depth_sequence(
            world_points,
            intrinsics,
            true_extrinsics,
            height=height,
            width=width,
        )
        extrinsics = true_extrinsics.copy()
        extrinsics[1] = _pose_w2c(0.10, 0.0, 0.0, rot_y_deg=3.0)

        report = audit_static_geometry_heavy_tail(
            depth_frames,
            intrinsics,
            extrinsics,
            query_frame=0,
            grid_size=10,
            min_query_depth_m=0.2,
            min_border_dist_px=2.0,
            min_depth=0.2,
            max_depth=10.0,
            cell_size_px=12,
            tail_threshold_px=1.0,
            top_k_frames=2,
            top_k_cells=2,
        )

        self.assertEqual(int(report["worst_frame"]["frame_index"]), 1)
        self.assertGreater(float(report["worst_frame"]["drift_p95_px"]), 0.0)


if __name__ == "__main__":
    unittest.main()
