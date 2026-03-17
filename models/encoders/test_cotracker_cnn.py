import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from models.encoders.cotracker_cnn import CoTrackerCNNEncoder


class _FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded_state_dicts: list[tuple[dict[str, torch.Tensor], bool]] = []

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state_dicts.append((state_dict, strict))

    def forward(self, x):
        return x


class CoTrackerCNNEncoderHubLoadTests(unittest.TestCase):
    def _build_fake_cotracker(self):
        state_dict = {"weight": torch.tensor([1.0], dtype=torch.float32)}
        fake_fnet = types.SimpleNamespace(state_dict=mock.Mock(return_value=state_dict))
        fake_cotracker = types.SimpleNamespace(model=types.SimpleNamespace(fnet=fake_fnet))
        return fake_cotracker, state_dict

    def test_cache_hit_uses_local_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_dir = Path(tmpdir)
            cached_repo = hub_dir / "facebookresearch_co-tracker_main"
            cached_repo.mkdir()
            fake_backbone = _FakeBackbone()
            fake_cotracker, state_dict = self._build_fake_cotracker()

            with mock.patch("models.encoders.cotracker_cnn.BasicEncoder", return_value=fake_backbone) as mock_encoder, \
                mock.patch("models.encoders.cotracker_cnn.torch.hub.get_dir", return_value=str(hub_dir)), \
                mock.patch("models.encoders.cotracker_cnn.torch.hub.load", return_value=fake_cotracker) as mock_load, \
                self.assertLogs("models.encoders.cotracker_cnn", level="INFO") as logs:
                CoTrackerCNNEncoder(
                    resolution=(256, 256),
                    output_dim=128,
                    stride=4,
                    pretrained=True,
                    freeze_mode="none",
                )

        mock_encoder.assert_called_once_with(output_dim=128, stride=4)
        mock_load.assert_called_once_with(str(cached_repo), "cotracker3_offline", source="local")
        self.assertEqual(fake_backbone.loaded_state_dicts, [(state_dict, True)])
        combined_logs = "\n".join(logs.output)
        self.assertIn("local_cache", combined_logs)
        self.assertIn(str(hub_dir), combined_logs)
        self.assertIn(str(cached_repo), combined_logs)

    def test_cache_miss_uses_remote_hub_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hub_dir = Path(tmpdir)
            fake_backbone = _FakeBackbone()
            fake_cotracker, state_dict = self._build_fake_cotracker()

            with mock.patch("models.encoders.cotracker_cnn.BasicEncoder", return_value=fake_backbone) as mock_encoder, \
                mock.patch("models.encoders.cotracker_cnn.torch.hub.get_dir", return_value=str(hub_dir)), \
                mock.patch("models.encoders.cotracker_cnn.torch.hub.load", return_value=fake_cotracker) as mock_load, \
                self.assertLogs("models.encoders.cotracker_cnn", level="INFO") as logs:
                CoTrackerCNNEncoder(
                    resolution=(256, 256),
                    output_dim=128,
                    stride=4,
                    pretrained=True,
                    freeze_mode="none",
                )

        mock_encoder.assert_called_once_with(output_dim=128, stride=4)
        mock_load.assert_called_once_with("facebookresearch/co-tracker", "cotracker3_offline")
        self.assertEqual(fake_backbone.loaded_state_dicts, [(state_dict, True)])
        combined_logs = "\n".join(logs.output)
        self.assertIn("remote_hub", combined_logs)
        self.assertIn(str(hub_dir), combined_logs)
        self.assertIn("facebookresearch/co-tracker", combined_logs)

    def test_pretrained_false_skips_hub_load(self):
        fake_backbone = _FakeBackbone()

        with mock.patch("models.encoders.cotracker_cnn.BasicEncoder", return_value=fake_backbone) as mock_encoder, \
            mock.patch("models.encoders.cotracker_cnn.torch.hub.load") as mock_load:
            CoTrackerCNNEncoder(
                resolution=(256, 256),
                output_dim=128,
                stride=4,
                pretrained=False,
                freeze_mode="none",
            )

        mock_encoder.assert_called_once_with(output_dim=128, stride=4)
        mock_load.assert_not_called()
        self.assertEqual(fake_backbone.loaded_state_dicts, [])


if __name__ == "__main__":
    unittest.main()
