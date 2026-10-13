import torch.nn as nn

from .dwconv import DwConvHead
from .flatten import FlattenHead
from .squeeze import SqueezeHead


def RegressionHead(
    in_channels: int,
    spatial_hw: tuple,
    head_type: str = "flatten",
    out_dim: int = 2,
    squeeze_channels: int = 3,
) -> nn.Module:
    """Factory for the spatial heads. Names proj/fc are checkpoint keys."""
    if head_type == "flatten":
        return FlattenHead(in_channels, spatial_hw, out_dim)
    if head_type == "dwconv":
        return DwConvHead(in_channels, spatial_hw, out_dim)
    if head_type == "squeeze":
        return SqueezeHead(in_channels, spatial_hw, squeeze_channels, out_dim)
    raise ValueError(
        f"Unknown head_type {head_type!r}; expected 'flatten', 'dwconv', or 'squeeze'"
    )
