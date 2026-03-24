import unittest

import numpy as np

from utils.keyframe_schedule_utils import (
    build_candidate_source_frame_indices,
    filter_query_local_indices_by_remaining_frames,
    map_query_source_indices_to_local,
    sample_query_source_indices_per_second,
)


class BuildCandidateSourceFrameIndicesTests(unittest.TestCase):
    def test_applies_stride_then_max_num_frames(self):
        indices = build_candidate_source_frame_indices(
            20,
            stride=3,
            max_num_frames=4,
        )

        np.testing.assert_array_equal(indices, np.array([0, 3, 6, 9], dtype=np.int32))


class SampleQuerySourceIndicesPerSecondTests(unittest.TestCase):
    def test_fixed_count_samples_exact_number_per_second_without_duplicates(self):
        candidate_source_indices = np.arange(90, dtype=np.int32)

        sampled = sample_query_source_indices_per_second(
            candidate_source_indices,
            episode_fps=30.0,
            keyframes_per_sec_min=5,
            keyframes_per_sec_max=5,
            seed=7,
        )

        self.assertIn(0, sampled.tolist())
        self.assertIn(sampled.shape, [(15,), (16,)])
        self.assertEqual(np.unique(sampled).size, sampled.size)
        per_second_counts = [
            int(np.sum((sampled >= sec * 30) & (sampled < (sec + 1) * 30)))
            for sec in range(3)
        ]
        self.assertIn(per_second_counts[0], [5, 6])
        self.assertEqual(per_second_counts[1:], [5, 5])

    def test_variable_count_stays_within_closed_interval_for_each_second(self):
        candidate_source_indices = np.arange(120, dtype=np.int32)

        sampled = sample_query_source_indices_per_second(
            candidate_source_indices,
            episode_fps=30.0,
            keyframes_per_sec_min=2,
            keyframes_per_sec_max=3,
            seed=11,
        )

        self.assertIn(0, sampled.tolist())
        per_second_counts = [
            int(np.sum((sampled >= sec * 30) & (sampled < (sec + 1) * 30)))
            for sec in range(4)
        ]
        self.assertTrue(2 <= per_second_counts[0] <= 4)
        self.assertTrue(all(2 <= count <= 3 for count in per_second_counts[1:]))

    def test_same_seed_is_deterministic(self):
        candidate_source_indices = np.arange(75, dtype=np.int32)

        sampled_a = sample_query_source_indices_per_second(
            candidate_source_indices,
            episode_fps=25.0,
            keyframes_per_sec_min=2,
            keyframes_per_sec_max=4,
            seed=12345,
        )
        sampled_b = sample_query_source_indices_per_second(
            candidate_source_indices,
            episode_fps=25.0,
            keyframes_per_sec_min=2,
            keyframes_per_sec_max=4,
            seed=12345,
        )

        np.testing.assert_array_equal(sampled_a, sampled_b)

    def test_frame_zero_is_forced_when_present_in_candidates(self):
        candidate_source_indices = np.array([0, 7, 14, 21, 28, 35], dtype=np.int32)

        sampled = sample_query_source_indices_per_second(
            candidate_source_indices,
            episode_fps=30.0,
            keyframes_per_sec_min=1,
            keyframes_per_sec_max=1,
            seed=0,
        )

        self.assertEqual(int(sampled[0]), 0)
        self.assertIn(0, sampled.tolist())


class FilterQueryLocalIndicesByRemainingFramesTests(unittest.TestCase):
    def test_drops_query_frames_with_eight_or_fewer_remaining_frames_inclusive(self):
        kept, dropped = filter_query_local_indices_by_remaining_frames(
            np.arange(10, dtype=np.int32),
            video_length=10,
        )

        np.testing.assert_array_equal(kept, np.array([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(dropped, np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32))

    def test_keeps_empty_input_empty(self):
        kept, dropped = filter_query_local_indices_by_remaining_frames(
            np.zeros((0,), dtype=np.int32),
            video_length=4,
        )

        np.testing.assert_array_equal(kept, np.zeros((0,), dtype=np.int32))
        np.testing.assert_array_equal(dropped, np.zeros((0,), dtype=np.int32))


class MapQuerySourceIndicesToLocalTests(unittest.TestCase):
    def test_maps_existing_indices_dedups_and_reports_missing(self):
        local_indices, missing_indices = map_query_source_indices_to_local(
            np.array([0, 2, 4, 6], dtype=np.int32),
            np.array([2, 2, 4, 5, 6], dtype=np.int32),
        )

        np.testing.assert_array_equal(local_indices, np.array([1, 2, 3], dtype=np.int32))
        np.testing.assert_array_equal(missing_indices, np.array([5], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
