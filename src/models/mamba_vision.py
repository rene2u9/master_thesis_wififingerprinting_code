"""MambaVision (ImageNet-pretrained via HuggingFace) with a flatten head.
Hatamizadeh and Kautz, CVPR 2025. https://arxiv.org/abs/2407.08083
"""

import torch
import torch.nn as nn
from typing import Optional

from .base_module import BaseWiFiModule

# MambaVision requires mamba-ssm CUDA kernels
MAMBAVISION_AVAILABLE = False
try:
    from mambavision import create_model as create_mambavision_model
    MAMBAVISION_AVAILABLE = True
except ImportError:
    create_mambavision_model = None


class MambaVisionWiFi(BaseWiFiModule):

    def __init__(
        self,
        variant: str = "mamba_vision_T",
        pretrained: bool = True,
        image_shape: tuple = (128, 128, 1),
        freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not MAMBAVISION_AVAILABLE:
            raise ImportError(
                "mambavision not available. Install with: pip install mambavision\n"
                "Requires mamba-ssm with CUDA toolkit."
            )

        import argparse
        import torch.serialization
        torch.serialization.add_safe_globals([argparse.Namespace])

        self.model = create_mambavision_model(variant, pretrained=pretrained, num_classes=0)

        if hasattr(self.model, 'avgpool'):
            self.model.avgpool = nn.Identity()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('pixel_mean', mean)
        self.register_buffer('pixel_std', std)

        # selective-scan kernel is CUDA only
        probe_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval().to(probe_device)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_shape[0], image_shape[1], device=probe_device)
            feat = self.model(dummy)
        self.model.train().cpu()
        self.head = nn.Linear(feat.flatten(1).shape[1], 2)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = (x - self.pixel_mean) / self.pixel_std
        feat = self.model(x)
        return self.head(feat.flatten(1))
