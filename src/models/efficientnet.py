"""EfficientNet-B0 / V2-S (timm, ImageNet-pretrained) on the RSSI image.

Tan and Le, ICML 2019. https://arxiv.org/abs/1905.11946
"""

import torch
import torch.nn as nn

from .base_module import BaseWiFiModule
from .heads import RegressionHead


class EfficientNetWiFi(BaseWiFiModule):
    def __init__(
        self,
        variant: str = 'b0',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        image_shape: tuple = (64, 64, 1),
        head_type: str = "flatten",
        squeeze_channels: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        import timm

        model_name = {'b0': 'efficientnet_b0', 'v2_s': 'tf_efficientnetv2_s'}[variant]

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            features_only=True,
            out_indices=(4,),
        )

        data_cfg = timm.data.resolve_data_config({}, model=self.backbone)
        self.register_buffer(
            'pixel_mean', torch.tensor(data_cfg['mean']).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'pixel_std', torch.tensor(data_cfg['std']).view(1, 3, 1, 1)
        )

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_shape[0], image_shape[1])
            feat = self.backbone(dummy)[-1]
        self.backbone.train()

        self.head = RegressionHead(
            in_channels=feat.shape[1],
            spatial_hw=(feat.shape[2], feat.shape[3]),
            head_type=head_type,
            squeeze_channels=squeeze_channels,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        if x.dim() == 2:
            raise ValueError("EfficientNet requires image input, got flat vector")
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = (x - self.pixel_mean) / self.pixel_std
        feat = self.backbone(x)[-1]
        return self.head(feat)
