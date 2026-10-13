import torch
import torch.nn as nn


class DwConvHead(nn.Module):
    """Depthwise HxW conv (learned spatial pooling per channel) -> Linear."""

    def __init__(self, in_channels: int, spatial_hw: tuple, out_dim: int = 2):
        super().__init__()
        h, w = spatial_hw
        self.proj = nn.Conv2d(
            in_channels, in_channels, kernel_size=(h, w),
            groups=in_channels, padding=0,
        )
        self.fc = nn.Linear(in_channels, out_dim)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, C, H, W) -> (B, out_dim)"""
        return self.fc(self.proj(feat).flatten(1))
