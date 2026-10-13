import torch
import torch.nn as nn


class FlattenHead(nn.Module):
    """flatten(C*H*W) -> Linear. Spatially blind but a strong baseline."""

    def __init__(self, in_channels: int, spatial_hw: tuple, out_dim: int = 2):
        super().__init__()
        h, w = spatial_hw
        self.fc = nn.Linear(in_channels * h * w, out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, H, W) -> (B, out_dim)"""
        return self.fc(feat.flatten(1))
