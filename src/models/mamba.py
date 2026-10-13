"""Own single-stage Mamba model with structure-aware state fusion, pure PyTorch.
Adapted from Spatial-Mamba, Xiao et al., ICLR 2025. https://github.com/EdwardChasel/Spatial-Mamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

from timm.models.layers import DropPath, trunc_normal_

from .base_module import BaseWiFiModule


def selective_scan_states(xs, dts, A, Bs, Ds):
    """Selective scan, returns hidden states (B, d_inner, N, L)."""
    B_batch, d_inner, L = xs.shape
    N = A.shape[1]

    h = torch.zeros(B_batch, d_inner, N, device=xs.device, dtype=xs.dtype)
    hs = []
    for t in range(L):
        dt = dts[:, :, t]
        A_bar = torch.exp(dt.unsqueeze(-1) * A)
        B_bar = dt.unsqueeze(-1) * Bs[:, :, t].unsqueeze(1)
        h = A_bar * h + B_bar * xs[:, :, t].unsqueeze(-1)
        hs.append(h)

    return torch.stack(hs, dim=-1)  # (B, d_inner, N, L)


class StateFusion(nn.Module):
    """Depthwise 3x3 convs at dilation 1/3/5, mixed with learned weights."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.kernel_d1 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_d3 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_d5 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_inner, H, W)"""
        h1 = F.conv2d(F.pad(h, (1, 1, 1, 1), mode='replicate'),
                       self.kernel_d1, padding=0, dilation=1, groups=self.dim)
        h2 = F.conv2d(F.pad(h, (3, 3, 3, 3), mode='replicate'),
                       self.kernel_d3, padding=0, dilation=3, groups=self.dim)
        h3 = F.conv2d(F.pad(h, (5, 5, 5, 5), mode='replicate'),
                       self.kernel_d5, padding=0, dilation=5, groups=self.dim)
        return self.alpha[0] * h1 + self.alpha[1] * h2 + self.alpha[2] * h3


class StructureAwareSSM(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = 1,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv // 2, groups=self.d_inner, bias=True,
        )

        self.x_proj_weight = nn.Parameter(
            torch.empty(self.dt_rank + d_state * 2, self.d_inner)
        )
        nn.init.kaiming_uniform_(self.x_proj_weight, a=math.sqrt(5))

        self.dt_proj_weight = nn.Parameter(torch.empty(self.d_inner, self.dt_rank))
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj_weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_proj_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(self.d_inner, -1).clone()
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.state_fusion = StateFusion(self.d_inner)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def ssm(self, x: torch.Tensor):
        """x: (B, d_inner, H, W) -> y: (B, d_inner, L)"""
        B, C, H, W = x.shape
        L = H * W
        xs = x.reshape(B, C, L)

        x_dbl = torch.matmul(self.x_proj_weight.unsqueeze(0), xs)
        dts, Bs, Cs = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=1)

        dts = F.softplus(
            torch.matmul(self.dt_proj_weight.unsqueeze(0), dts)
            + self.dt_proj_bias.view(1, -1, 1)
        )

        A = -torch.exp(self.A_log.float())
        hs = selective_scan_states(xs, dts, A, Bs, self.D)

        hs_2d = hs.squeeze(2) if self.d_state == 1 else hs[:, :, 0]
        hs_fused = self.state_fusion(hs_2d.reshape(B, self.d_inner, H, W))
        hs_fused = hs_fused.reshape(B, self.d_inner, L)

        y = hs_fused * Cs[:, 0:1]
        y = y + xs * self.D.view(1, -1, 1)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, d_model)"""
        B, H, W, _ = x.shape
        xz = self.in_proj(x)
        x_seq, z = xz.chunk(2, dim=-1)
        x_seq = F.silu(self.conv2d(x_seq.permute(0, 3, 1, 2).contiguous()))
        y = self.ssm(x_seq)
        y = self.out_norm(y.reshape(B, self.d_inner, H, W).permute(0, 2, 3, 1))
        y = self.out_proj(y * F.silu(z))
        return self.dropout(y)


class SpatialMambaBlock(nn.Module):
    """Pre-norm block: CPE + SSM + CPE + MLP, each with residual."""

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        d_state: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.cpe1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.cpe2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)
        self.ssm = StructureAwareSSM(d_model=hidden_dim, d_state=d_state, dropout=dropout)
        self.ln_2 = norm_layer(hidden_dim)
        hidden_features = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_dim),
            nn.Dropout(dropout),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)"""
        x = x + self.cpe1(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.ssm(self.ln_1(x)))
        x = x + self.cpe2(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class WiFiSpatialMamba(nn.Module):
    """Single-stage Spatial-Mamba for WiFi RSSI image regression."""

    def __init__(
        self,
        img_size: int = 16,
        patch_size: int = 4,
        in_chans: int = 1,
        embed_dim: int = 96,
        depth: int = 6,
        d_state: int = 1,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size

        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            SpatialMambaBlock(
                hidden_dim=embed_dim,
                drop_path=dpr[i],
                d_state=d_state,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        num_patches = self.grid_size * self.grid_size
        self.head = nn.Linear(num_patches * embed_dim, 2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        x = self.patch_embed(x).permute(0, 2, 3, 1).contiguous()  # (B, G, G, embed_dim)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x.reshape(x.size(0), -1, self.embed_dim))
        return self.head(x.flatten(1))


class WiFiSpatialMambaModule(BaseWiFiModule):

    def __init__(
        self,
        img_size: int = 16,
        patch_size: int = 4,
        in_chans: int = 1,
        embed_dim: int = 96,
        depth: int = 6,
        d_state: int = 1,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = WiFiSpatialMamba(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            dropout=dropout,
        )
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
