"""Vision Mamba (Vim-S) with pretrained weights. Zhu et al., ICML 2024.
https://arxiv.org/abs/2401.09417, https://github.com/hustvl/Vim
Uses mamba_ssm blocks if installed, otherwise a slow pure-PyTorch SSM.
"""

import torch
import torch.nn as nn
from typing import Optional
from timm.models.layers import trunc_normal_, DropPath
import math
import logging
import re

logger = logging.getLogger(__name__)

MAMBA_SSM_AVAILABLE = False
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    MAMBA_SSM_AVAILABLE = True
    logger.info("mamba-ssm available, using official Mamba blocks")
except ImportError:
    Mamba = None
    logger.warning(
        "mamba-ssm not installed. Vision Mamba will use a slow PyTorch fallback. "
        "Install for full pretrained weight support: pip install mamba-ssm causal-conv1d --no-build-isolation"
    )

from .base_module import BaseWiFiModule
from .heads import MLPHead, RegressionHead


class PatchEmbed(nn.Module):

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})"
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x



class BidirectionalMamba(nn.Module):
    """Forward and backward Mamba block."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, **kwargs):
        super().__init__()
        assert Mamba is not None, "mamba-ssm required for BidirectionalMamba"
        self.forward_ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_fwd = self.forward_ssm(x)
        y_bwd = self.backward_ssm(x.flip(1)).flip(1)
        return (y_fwd + y_bwd) / 2


class VimBlock(nn.Module):
    """RMSNorm (as in the Vim checkpoints) -> BidirectionalMamba -> DropPath + residual."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.mixer = BidirectionalMamba(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.mixer(self.norm(x)))



class FallbackSSM(nn.Module):
    """Pure-PyTorch SSM fallback."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self.act(self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2))

        x_dbl = self.x_proj(x)
        delta, B_ssm = x_dbl.chunk(2, dim=-1)
        delta = nn.functional.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            x_t, delta_t, B_t = x[:, t], delta[:, t], B_ssm[:, t]
            h = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0)) * h + \
                delta_t.unsqueeze(-1) * B_t.unsqueeze(1) * x_t.unsqueeze(-1)
            outputs.append(h.sum(dim=-1) + self.D * x_t)

        y = torch.stack(outputs, dim=1)
        return self.out_proj(y * self.act(z))


class FallbackBidirectionalSSM(nn.Module):

    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.forward_ssm = FallbackSSM(d_model, **kwargs)
        self.backward_ssm = FallbackSSM(d_model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.forward_ssm(x) + self.backward_ssm(x.flip(1)).flip(1)) / 2


class FallbackVimBlock(nn.Module):

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-5)
        self.mixer = FallbackBidirectionalSSM(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.mixer(self.norm(x)))



class VisionMamba(nn.Module):
    """PatchEmbed + CLS token + VimBlocks + head."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 2,
        embed_dim: int = 384,
        depth: int = 24,
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_ratio: float = 2.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_cls_token: bool = True,
        head_type: str = 'linear',
        squeeze_channels: int = 3,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=0.02)
            # midclstok checkpoints put the CLS token in the middle of the sequence
            self.token_position = num_patches // 2

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + (1 if use_cls_token else 0), embed_dim)
        )
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        expand = int(ssm_ratio)

        BlockClass = VimBlock if MAMBA_SSM_AVAILABLE else FallbackVimBlock
        if not MAMBA_SSM_AVAILABLE:
            logger.warning("Using FallbackVimBlock (no mamba-ssm). Pretrained weights will NOT load for SSM layers.")

        self.layers = nn.ModuleList([  # name matches the checkpoint keys
            BlockClass(embed_dim, d_state=ssm_d_state, d_conv=ssm_d_conv,
                       expand=expand, drop_path=dpr[i])
            for i in range(depth)
        ])

        self.norm = nn.RMSNorm(embed_dim, eps=1e-5)
        # dwconv/squeeze read the patch tokens as a grid instead of the CLS token
        self.spatial_head = head_type in ('dwconv', 'squeeze')
        if self.spatial_head:
            grid = self.patch_embed.grid_size
            self.head = RegressionHead(
                in_channels=embed_dim,
                spatial_hw=(grid, grid),
                head_type=head_type,
                squeeze_channels=squeeze_channels,
                out_dim=num_classes,
            )
        elif head_type == 'mlp':
            self.head = MLPHead(embed_dim, out_dim=num_classes)
        elif head_type == 'linear':
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            raise ValueError(
                f"Unknown head_type {head_type!r}; expected 'linear', 'mlp', 'dwconv', or 'squeeze'"
            )
        # keep the mamba-ssm init of the mixer (dt_proj bias)
        for name, m in self.named_modules():
            if '.mixer.' not in name + '.':
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.use_cls_token:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat(
                [x[:, :self.token_position], cls, x[:, self.token_position:]], dim=1
            )
        x = self.pos_drop(x + self.pos_embed)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        if self.spatial_head:
            # CLS token stays in the trunk, dropped only here
            if self.use_cls_token:
                x = torch.cat(
                    [x[:, :self.token_position], x[:, self.token_position + 1:]], dim=1
                )
            g = self.patch_embed.grid_size
            x = x.transpose(1, 2).reshape(x.shape[0], self.embed_dim, g, g)
            return self.head(x)
        x = x[:, self.token_position] if self.use_cls_token else x.mean(dim=1)
        return self.head(x)



class VisionMambaWiFi(BaseWiFiModule):
    """Vision Mamba for WiFi fingerprinting with pretrained weight loading."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 24,
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_ratio: float = 2.0,
        drop_path_rate: float = 0.1,
        use_cls_token: bool = True,
        head_type: str = 'linear',
        squeeze_channels: int = 3,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.img_size = img_size

        self.model = VisionMamba(
            img_size=img_size, patch_size=patch_size, in_chans=3,
            num_classes=2, embed_dim=embed_dim, depth=depth,
            ssm_d_state=ssm_d_state, ssm_d_conv=ssm_d_conv,
            ssm_ratio=ssm_ratio, drop_path_rate=drop_path_rate,
            use_cls_token=use_cls_token, head_type=head_type,
            squeeze_channels=squeeze_channels,
        )

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('pixel_mean', mean)
        self.register_buffer('pixel_std', std)

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
            print("Backbone frozen, only training head")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.shape[0], -1, self.img_size, self.img_size)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        x = (x - self.pixel_mean) / self.pixel_std
        return self.model(x)

    def _param_groups(self):
        """Disc-LR split on model.head, there is no 'backbone' in the param names."""
        if self.backbone_lr_mult == 1.0:
            return self.parameters()
        backbone, head = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (head if name.startswith('model.head') else backbone).append(p)
        return [
            {'params': backbone, 'lr': self.learning_rate * self.backbone_lr_mult},
            {'params': head, 'lr': self.learning_rate},
        ]

    def load_pretrained(self, checkpoint_path: str):
        """Load Vim-S weights, mapping mixer.{p} / {p}_b to forward_ssm / backward_ssm.

        in_proj/out_proj are shared in Vim and get copied into both directions.
        """
        print(f"Loading pretrained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('head.'):
                continue

            # A_b_log does not end in _b
            m = re.match(r'^(layers\.\d+\.mixer\.)A_b_log$', k)
            if m:
                remapped[f'{m.group(1)}backward_ssm.A_log'] = v
                continue

            m = re.match(r'^(layers\.\d+\.mixer\.)(\w+)_b(\..*)?$', k)
            if m:
                prefix, param, suffix = m.group(1), m.group(2), m.group(3) or ''
                remapped[f'{prefix}backward_ssm.{param}{suffix}'] = v
                continue

            m = re.match(r'^(layers\.\d+\.mixer\.)(.+)$', k)
            if m:
                prefix, rest = m.group(1), m.group(2)
                remapped[f'{prefix}forward_ssm.{rest}'] = v
                if rest.startswith('in_proj') or rest.startswith('out_proj'):
                    remapped[f'{prefix}backward_ssm.{rest}'] = v.clone()
                continue

            m = re.match(r'^norm_f\.(.+)$', k)
            if m:
                remapped[f'norm.{m.group(1)}'] = v
                continue

            remapped[k] = v

        if 'pos_embed' in remapped and 'pos_embed' in self.model.state_dict():
            pretrained_pe = remapped['pos_embed']
            model_pe = self.model.state_dict()['pos_embed']
            if pretrained_pe.shape != model_pe.shape:
                has_cls = self.model.use_cls_token
                if has_cls:
                    # midclstok checkpoints carry the CLS PE in the middle slot.
                    src_cls_pos = (pretrained_pe.shape[1] - 1) // 2
                    cls_token_pe = pretrained_pe[:, src_cls_pos:src_cls_pos + 1, :]
                    spatial_pe = torch.cat(
                        [pretrained_pe[:, :src_cls_pos], pretrained_pe[:, src_cls_pos + 1:]],
                        dim=1,
                    )
                else:
                    spatial_pe = pretrained_pe

                src_len = spatial_pe.shape[1]
                src_grid = int(src_len ** 0.5)
                tgt_len = model_pe.shape[1] - (1 if has_cls else 0)
                tgt_grid = int(tgt_len ** 0.5)

                print(f"  Interpolating pos_embed: {src_grid}x{src_grid} -> {tgt_grid}x{tgt_grid}")

                spatial_pe = spatial_pe.reshape(1, src_grid, src_grid, -1).permute(0, 3, 1, 2)
                spatial_pe = torch.nn.functional.interpolate(
                    spatial_pe, size=(tgt_grid, tgt_grid), mode='bicubic', align_corners=False
                )
                spatial_pe = spatial_pe.permute(0, 2, 3, 1).reshape(1, tgt_len, -1)

                if has_cls:
                    tgt_pos = self.model.token_position
                    remapped['pos_embed'] = torch.cat(
                        [spatial_pe[:, :tgt_pos], cls_token_pe, spatial_pe[:, tgt_pos:]],
                        dim=1,
                    )
                else:
                    remapped['pos_embed'] = spatial_pe

        pe_key = 'patch_embed.proj.weight'
        if pe_key in remapped and pe_key in self.model.state_dict():
            src_w = remapped[pe_key]
            tgt_w = self.model.state_dict()[pe_key]
            if src_w.shape != tgt_w.shape:
                print(f"  Adapting patch_embed: {list(src_w.shape)} -> {list(tgt_w.shape)}")
                tgt_h, tgt_w_dim = tgt_w.shape[2], tgt_w.shape[3]
                adapted = torch.nn.functional.interpolate(
                    src_w.float(), size=(tgt_h, tgt_w_dim), mode='bilinear', align_corners=False
                )
                if adapted.shape[1] != tgt_w.shape[1]:
                    adapted = adapted.mean(dim=1, keepdim=True)
                remapped[pe_key] = adapted

        model_sd = self.model.state_dict()
        compatible = {
            k: v for k, v in remapped.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        dropped = sorted(k for k in remapped if k not in compatible)
        not_loaded = sorted(k for k in model_sd if k not in compatible)

        model_sd.update(compatible)
        self.model.load_state_dict(model_sd, strict=False)

        def _collapse(keys):
            """Group keys that differ only by layer index: layers.*.mixer.foo x24"""
            counts = {}
            for k in keys:
                pattern = re.sub(r'(^|\.)\d+(\.)', r'\1*\2', k)
                counts[pattern] = counts.get(pattern, 0) + 1
            return [f"{p} (x{c})" if c > 1 else p for p, c in sorted(counts.items())]

        print(f"Loaded {len(compatible)}/{len(model_sd)} model parameters from checkpoint")
        if dropped:
            print(f"  {len(dropped)} remapped checkpoint keys had no matching model param:")
            for p in _collapse(dropped):
                print(f"    {p}")
        if not_loaded:
            print(f"  {len(not_loaded)} model params left at init:")
            for p in _collapse(not_loaded):
                print(f"    {p}")
