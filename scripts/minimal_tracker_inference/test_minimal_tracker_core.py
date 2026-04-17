import math
import unittest

import numpy as np

from scripts.minimal_tracker_inference.minimal_tracker_core import (
    build_roofline_estimate,
    build_symbolic_complexity_report,
    build_world_grid_queries,
    create_synthetic_case,
    estimate_streaming_window_count,
    prepare_tracker_case,
)


class MinimalTrackerCoreTests(unittest.TestCase):
    def test_estimate_streaming_window_count_matches_half_overlap_schedule(self):
        self.assertEqual(estimate_streaming_window_count(12, 12), 1)
        self.assertEqual(estimate_streaming_window_count(13, 12), 2)
        self.assertEqual(estimate_streaming_window_count(24, 12), 3)

    def test_build_world_grid_queries_uses_depth_and_identity_camera(self):
        depths = np.ones((2, 4, 4), dtype=np.float32)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        queries = build_world_grid_queries(
            depths=depths,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            grid_size=2,
            query_frame=1,
        )
        self.assertEqual(queries.shape, (4, 4))
        self.assertTrue(np.allclose(queries[:, 0], 1.0))
        self.assertTrue(np.allclose(queries[:, 3], 1.0))

    def test_symbolic_complexity_tracks_window_and_query_scaling(self):
        case = create_synthetic_case(frames=13, height=8, width=8)
        prepared_case = prepare_tracker_case(
            case,
            device="cpu",
            query_grid_size=2,
            query_frame=0,
        )
        report = build_symbolic_complexity_report(
            prepared_case=prepared_case,
            num_iters=3,
            seq_len=12,
            support_query_count=4,
        )
        self.assertEqual(report.window_count, 2)
        self.assertEqual(report.query_count, 4)
        self.assertEqual(report.total_query_count, 8)
        self.assertEqual(report.shared_unprojection_points, 13 * 64)
        self.assertEqual(report.repeated_window_unprojection_points, 2 * 13 * 64)
        self.assertEqual(report.iterative_track_state_updates, 2 * 3 * 12 * 8)

    def test_roofline_estimate_uses_dense_h100_peaks(self):
        estimate = build_roofline_estimate(
            profiled_flops=int(9.895e14),
            h100_variant="sxm",
            precision_mode="bf16",
        )
        self.assertEqual(estimate.peak_tflops, 989.5)
        self.assertTrue(math.isclose(estimate.theoretical_min_seconds or 0.0, 1.0, rel_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
