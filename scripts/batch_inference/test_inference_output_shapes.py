import unittest
from unittest import mock

import torch

from utils import inference_utils


class _FakePreds:
    def __init__(self, coords: torch.Tensor, visibs: torch.Tensor):
        self.coords = coords
        self.visibs = visibs


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bidirectional = False
        self.image_size = None

    def set_image_size(self, image_size):
        self.image_size = image_size


class InferenceShapeTests(unittest.TestCase):
    def _run_inference(self, num_frames: int):
        num_queries = 4
        model = _FakeModel()

        coords = torch.arange(
            num_frames * num_queries * 3, dtype=torch.float32
        ).reshape(1, num_frames, num_queries, 3)
        visib_logits = torch.full((1, num_frames, num_queries), 10.0)

        video = torch.rand(num_frames, 3, 4, 4)
        depths = torch.ones(num_frames, 4, 4)
        intrinsics = torch.eye(3).repeat(num_frames, 1, 1)
        extrinsics = torch.eye(4).repeat(num_frames, 1, 1)
        query_point = torch.zeros(num_queries, 4)

        with mock.patch.object(
            inference_utils,
            "_inference_with_grid",
            return_value=(_FakePreds(coords, visib_logits), None),
        ):
            return inference_utils.inference(
                model=model,
                video=video,
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                query_point=query_point,
                bidrectional=False,
                grid_size=0,
            )

    def test_single_frame_keeps_time_dimension(self):
        coords, visibs = self._run_inference(num_frames=1)

        self.assertEqual(coords.shape, torch.Size([1, 4, 3]))
        self.assertEqual(visibs.shape, torch.Size([1, 4]))
        self.assertTrue(visibs.all())

    def test_multi_frame_shape_is_unchanged(self):
        coords, visibs = self._run_inference(num_frames=2)

        self.assertEqual(coords.shape, torch.Size([2, 4, 3]))
        self.assertEqual(visibs.shape, torch.Size([2, 4]))
        self.assertTrue(visibs.all())


if __name__ == "__main__":
    unittest.main()
