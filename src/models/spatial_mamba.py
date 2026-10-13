"""Pretrained Spatial-Mamba backbone (vendored) with a regression head on one stage.
Xiao et al., ICLR 2025. https://github.com/EdwardChasel/Spatial-Mamba
Needs the custom CUDA kernels selective_scan_cuda_oflex_rh and dwconv2d.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base_module import BaseWiFiModule
from .heads import RegressionHead


_VARIANTS = {
    'tiny':  {'depths': [2, 4, 8, 4],  'dims': [64, 128, 256, 512]},
    'small': {'depths': [2, 4, 21, 5], 'dims': [64, 128, 256, 512]},
    'base':  {'depths': [2, 4, 21, 5], 'dims': [96, 192, 384, 768]},
}


class SpatialMambaWiFi(nn.Module):
    def __init__(
        self,
        variant: str = "tiny",
        pretrained_path: Optional[str] = None,
        image_shape: tuple = (224, 224, 1),
        out_stage: int = 2,
        freeze_backbone: bool = False,
        head_type: str = "flatten",
        squeeze_channels: int = 3,
    ):
        super().__init__()
        from .spatial_mamba_vendor.spatialmamba import Backbone_SpatialMamba

        cfg = _VARIANTS[variant]
        self.backbone = Backbone_SpatialMamba(
            img_size=image_shape[0],
            patch_size=4,
            in_chans=3,
            num_classes=0,
            depths=cfg['depths'],
            dims=cfg['dims'],
            out_indices=(out_stage,),
            pretrained=pretrained_path,
            norm_layer="ln",
        )

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('pixel_mean', mean)
        self.register_buffer('pixel_std', std)

        # dims computed, a probe forward would need CUDA
        grid = image_shape[0] // (4 * 2 ** out_stage)
        self.head = RegressionHead(
            in_channels=cfg['dims'][out_stage],
            spatial_hw=(grid, grid),
            head_type=head_type,
            squeeze_channels=squeeze_channels,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = (x - self.pixel_mean) / self.pixel_std
        feats = self.backbone(x)[0]
        return self.head(feats)


class SpatialMambaWiFiModule(BaseWiFiModule):

    def __init__(
        self,
        variant: str = "tiny",
        pretrained_path: Optional[str] = None,
        image_shape: tuple = (224, 224, 1),
        out_stage: int = 2,
        freeze_backbone: bool = False,
        head_type: str = "flatten",
        squeeze_channels: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = SpatialMambaWiFi(
            variant=variant,
            pretrained_path=pretrained_path,
            image_shape=image_shape,
            out_stage=out_stage,
            freeze_backbone=freeze_backbone,
            head_type=head_type,
            squeeze_channels=squeeze_channels,
        )
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
