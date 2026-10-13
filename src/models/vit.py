"""Small ViT trained from scratch. Dosovitskiy et al., ICLR 2021.
https://arxiv.org/abs/2010.11929
"""

import torch
import torch.nn as nn

from .base_module import BaseWiFiModule


class PatchEmbedding(nn.Module):

    def __init__(self, in_chans: int = 1, embed_dim: int = 128,
                 patch_size: int = 4, img_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)


class WiFiViT(nn.Module):
    def __init__(
        self,
        img_size: int = 16,
        patch_size: int = 4,
        in_chans: int = 1,
        embed_dim: int = 128,
        depth: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(in_chans, embed_dim, patch_size, img_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)

        B = x.size(0)
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.norm(self.encoder(x))
        return self.head(x[:, 0])


class WiFiViTModule(BaseWiFiModule):

    def __init__(
        self,
        img_size: int = 16,
        patch_size: int = 4,
        in_chans: int = 1,
        embed_dim: int = 128,
        depth: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = WiFiViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
