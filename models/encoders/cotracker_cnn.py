import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.cotracker_blocks import BasicEncoder
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CoTrackerHubSource:
    hub_dir: Path
    repo_or_dir: str
    load_kwargs: dict[str, str]
    source_label: str


def _resolve_cotracker_hub_source() -> _CoTrackerHubSource:
    hub_dir = Path(torch.hub.get_dir())
    cached_repo = hub_dir / "facebookresearch_co-tracker_main"
    if cached_repo.is_dir():
        return _CoTrackerHubSource(
            hub_dir=hub_dir,
            repo_or_dir=str(cached_repo),
            load_kwargs={"source": "local"},
            source_label="local_cache",
        )

    return _CoTrackerHubSource(
        hub_dir=hub_dir,
        repo_or_dir="facebookresearch/co-tracker",
        load_kwargs={},
        source_label="remote_hub",
    )


class CoTrackerCNNEncoder(nn.Module):
    def __init__(self, resolution: Tuple[int, int], output_dim: int, stride: int, pretrained: bool, freeze_mode: str):
        super().__init__()
        self.resolution = resolution
        self.stride = stride
        self.output_dim = output_dim
        assert freeze_mode in ["all", "none"], f"Freezing mode {freeze_mode} not supported"

        logger.info(f"Loading CoTracker CNN Encoder")
        logger.info(f"Freezing mode: {freeze_mode}")

        self.backbone = BasicEncoder(output_dim=output_dim, stride=stride)
        if pretrained:
            hub_source = _resolve_cotracker_hub_source()
            logger.info(
                "Loading CoTracker weights via %s (hub_dir=%s, target=%s)",
                hub_source.source_label,
                hub_source.hub_dir,
                hub_source.repo_or_dir,
            )
            cotracker = torch.hub.load(hub_source.repo_or_dir, "cotracker3_offline", **hub_source.load_kwargs)
            backbone_state_dict = cotracker.model.fnet.state_dict()
            self.backbone.load_state_dict(backbone_state_dict, strict=True)
            del cotracker

        if freeze_mode == "all":
            for param in self.backbone.parameters():
                param.requires_grad = False

        H, W = self.resolution
        assert H % self.stride == 0 and W % self.stride == 0, f"Image size {H}x{W} must be divisible by stride {self.stride}"

        self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        assert (H, W) == self.resolution, f"Image size {H}x{W} must be {self.resolution}"

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.transform(x)
        outputs = self.backbone(x)

        return rearrange(outputs, "(b t) c h w -> b t c h w", b=B, t=T)
    
    @property
    def embedding_dim(self):
        return self.output_dim

    def set_image_size(self, image_size: Tuple[int, int]):
        self.resolution = image_size
