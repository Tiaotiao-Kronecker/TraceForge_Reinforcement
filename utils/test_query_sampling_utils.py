import unittest

import numpy as np

from utils.query_sampling_utils import (
    QUERY_RISK_IMAGE_BORDER,
    QueryCueResult,
    RelevanceCueProvider,
    build_relevance_first_query_sampler_result,
)


class _FixedCueProvider(RelevanceCueProvider):
    def __init__(
        self,
        *,
        hand_score: np.ndarray,
        manipulator_score: np.ndarray,
        object_score: np.ndarray,
        context_score: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        self._hand_score = np.asarray(hand_score, dtype=np.float32)
        self._manipulator_score = np.asarray(manipulator_score, dtype=np.float32)
        self._object_score = np.asarray(object_score, dtype=np.float32)
        self._context_score = np.asarray(context_score, dtype=np.float32)
        self._valid_mask = np.asarray(valid_mask, dtype=bool)

    def propose(
        self,
        *,
        query_rgb: np.ndarray,
        query_depth: np.ndarray,
        temporal_rgbs: np.ndarray,
        candidate_keypoints: np.ndarray,
        min_depth: float,
        max_depth: float,
    ) -> QueryCueResult:
        return QueryCueResult(
            hand_score=self._hand_score,
            manipulator_score=self._manipulator_score,
            object_score=self._object_score,
            context_score=self._context_score,
            valid_mask=self._valid_mask,
            provider_name="fixed",
            provider_meta={},
        )


class RelevanceFirstQuerySamplerTests(unittest.TestCase):
    def test_prefers_non_border_candidates_when_budget_is_satisfied_without_edges(self):
        candidate_keypoints = np.array(
            [
                [32.0, 32.0],
                [63.0, 32.0],
            ],
            dtype=np.float32,
        )
        query_rgb = np.zeros((64, 64, 3), dtype=np.float32)
        query_depth = np.ones((64, 64), dtype=np.float32)
        temporal_rgbs = np.repeat(query_rgb[None], 2, axis=0)
        cue_provider = _FixedCueProvider(
            hand_score=np.array([0.8, 1.0], dtype=np.float32),
            manipulator_score=np.array([0.8, 1.0], dtype=np.float32),
            object_score=np.zeros(2, dtype=np.float32),
            context_score=np.zeros(2, dtype=np.float32),
            valid_mask=np.array([True, True]),
        )

        result = build_relevance_first_query_sampler_result(
            candidate_keypoints=candidate_keypoints,
            query_rgb=query_rgb,
            query_depth=query_depth,
            temporal_rgbs=temporal_rgbs,
            target_query_count=1,
            min_depth=0.01,
            max_depth=10.0,
            cue_provider=cue_provider,
        )

        np.testing.assert_allclose(result["keypoints"], candidate_keypoints[[0]])
        self.assertEqual(int(result["query_risk_bits"][0] & QUERY_RISK_IMAGE_BORDER), 0)

    def test_falls_back_to_border_candidates_when_no_safe_candidates_exist(self):
        candidate_keypoints = np.array(
            [
                [0.0, 0.0],
                [63.0, 32.0],
            ],
            dtype=np.float32,
        )
        query_rgb = np.zeros((64, 64, 3), dtype=np.float32)
        query_depth = np.ones((64, 64), dtype=np.float32)
        temporal_rgbs = np.repeat(query_rgb[None], 2, axis=0)
        cue_provider = _FixedCueProvider(
            hand_score=np.array([1.0, 0.5], dtype=np.float32),
            manipulator_score=np.array([1.0, 0.5], dtype=np.float32),
            object_score=np.zeros(2, dtype=np.float32),
            context_score=np.zeros(2, dtype=np.float32),
            valid_mask=np.array([True, True]),
        )

        result = build_relevance_first_query_sampler_result(
            candidate_keypoints=candidate_keypoints,
            query_rgb=query_rgb,
            query_depth=query_depth,
            temporal_rgbs=temporal_rgbs,
            target_query_count=1,
            min_depth=0.01,
            max_depth=10.0,
            cue_provider=cue_provider,
        )

        np.testing.assert_allclose(result["keypoints"], candidate_keypoints[[0]])
        self.assertNotEqual(int(result["query_risk_bits"][0] & QUERY_RISK_IMAGE_BORDER), 0)


if __name__ == "__main__":
    unittest.main()
