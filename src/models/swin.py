"""Swin Transformer (timm, ImageNet-pretrained).
Liu et al., ICCV 2021. https://arxiv.org/abs/2103.14030
"""

import torch
import torch.nn as nn

from .base_module import BaseWiFiModule
from .heads import RegressionHead


class SwinWiFi(nn.Module):
    def __init__(
        self,
        variant: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        image_shape: tuple = (224, 224, 1),
        freeze_backbone: bool = False,
        out_stage: int = 3,
        head_type: str = "flatten",
        squeeze_channels: int = 3,
    ):
        super().__init__()
        import timm

        self.in_chans = image_shape[2] if len(image_shape) == 3 else 1

        # 3 channels to keep the pretrained patch embedding
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            in_chans=3,
            features_only=True,
            out_indices=(out_stage,),
        )

        # features_only returns NHWC for Swin
        self._channels_last = str(getattr(self.backbone, 'output_fmt', 'NHWC')).split('.')[-1] == 'NHWC'

        data_cfg = timm.data.resolve_data_config({}, model=self.backbone)
        self.register_buffer(
            'pixel_mean', torch.tensor(data_cfg['mean']).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'pixel_std', torch.tensor(data_cfg['std']).view(1, 3, 1, 1)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_shape[0], image_shape[1])
            feat = self._to_nchw(self.backbone(dummy)[-1])

        self.head = RegressionHead(
            in_channels=feat.shape[1],
            spatial_hw=(feat.shape[2], feat.shape[3]),
            head_type=head_type,
            squeeze_channels=squeeze_channels,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _to_nchw(self, feat: torch.Tensor) -> torch.Tensor:
        return feat.permute(0, 3, 1, 2).contiguous() if self._channels_last else feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = (x - self.pixel_mean) / self.pixel_std
        feat = self._to_nchw(self.backbone(x)[-1])
        return self.head(feat)


class SwinWiFiModule(BaseWiFiModule):

    def __init__(
        self,
        variant: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        image_shape: tuple = (224, 224, 1),
        freeze_backbone: bool = False,
        out_stage: int = 3,
        head_type: str = "flatten",
        squeeze_channels: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = SwinWiFi(
            variant=variant,
            pretrained=pretrained,
            image_shape=image_shape,
            freeze_backbone=freeze_backbone,
            out_stage=out_stage,
            head_type=head_type,
            squeeze_channels=squeeze_channels,
        )
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
