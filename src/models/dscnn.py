"""DS-CNN (deep signal-strength CNN) on the RSSI image.

Zhu et al., Sensors 23(15):6992, 2023. https://doi.org/10.3390/s23156992
"""

import torch
import torch.nn as nn

from .base_module import BaseWiFiModule


class DSCNN(BaseWiFiModule):

    def __init__(
        self,
        image_shape: tuple = (16, 16, 1),
        dropout: float = 0.5,
        use_batchnorm: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_shape = image_shape

        # use_batchnorm=True is the regression twin of DSCNNClsWiFiModule
        def conv_block(c_in, c_out, **conv_kw):
            layers = [nn.Conv2d(c_in, c_out, kernel_size=3, **conv_kw)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.features = nn.Sequential(
            *conv_block(1, 32),
            nn.MaxPool2d(2, 2),
            *conv_block(32, 64),
            nn.MaxPool2d(2, 2),
            *conv_block(64, 32, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            *conv_block(32, 64, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, *self.image_shape).permute(0, 3, 1, 2)
        return self.classifier(self.features(x))
