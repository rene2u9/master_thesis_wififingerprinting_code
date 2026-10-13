"""ResNet (timm, ImageNet-pretrained) on the 1-channel RSSI image.

He et al., CVPR 2016. https://arxiv.org/abs/1512.03385
"""

import torch
import torch.nn as nn

from .base_module import BaseWiFiModule
from .heads import RegressionHead


class ResNetWiFi(nn.Module):
    def __init__(
        self,
        variant: str = "resnet18",
        pretrained: bool = True,
        image_shape: tuple = (16, 16, 1),
        freeze_backbone: bool = False,
        head_type: str = "flatten",
        squeeze_channels: int = 3,
    ):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            variant,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = (x - self.pixel_mean) / self.pixel_std
        feat = self.backbone(x)[-1]
        return self.head(feat)


class ResNetWiFiModule(BaseWiFiModule):

    def __init__(self, variant="resnet18", pretrained=True, image_shape=(16, 16, 1),
                 freeze_backbone=False, head_type="flatten", squeeze_channels=3, **kwargs):
        super().__init__(**kwargs)
        self.model = ResNetWiFi(
            variant=variant,
            pretrained=pretrained,
            image_shape=image_shape,
            freeze_backbone=freeze_backbone,
            head_type=head_type,
            squeeze_channels=squeeze_channels,
        )
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
