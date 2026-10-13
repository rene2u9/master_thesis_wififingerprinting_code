import torch
import torch.nn as nn


class SqueezeHead(nn.Module):
    """1x1 conv to squeeze_channels -> flatten -> Linear."""

    def __init__(self, in_channels: int, spatial_hw: tuple,
                 squeeze_channels: int = 3, out_dim: int = 2):
        super().__init__()
        h, w = spatial_hw
        self.proj = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.fc = nn.Linear(squeeze_channels * h * w, out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, H, W) -> (B, out_dim)"""
        return self.fc(self.proj(feat).flatten(1))
