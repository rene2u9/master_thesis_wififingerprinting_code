"""Bay-classification (*_cls) variants of the model families.

Same centroid classifier on top of each family trunk, regression head removed.
"""

import torch
import torch.nn as nn

from .base_module import BaseWiFiModule
from .cls_head import ClsHeadMixin
from .efficientnet import EfficientNetWiFi
from .ffn import ReferenceFFN
from .gcn import ReferenceSAGE
from .mamba import WiFiSpatialMamba
from .mamba_vision import MambaVisionWiFi
from .resnet import ResNetWiFi
from .sae import ReferenceSAE
from .spatial_mamba import _VARIANTS as _SM_VARIANTS
from .spatial_mamba import SpatialMambaWiFi
from .swin import SwinWiFi
from .transformer import WiFiTransformerModule
from .vgae import ReferenceGCNVGAE
from .vision_mamba import VisionMambaWiFi
from .vit import WiFiViT


class ResNetClsWiFiModule(ClsHeadMixin, BaseWiFiModule):
    """ResNet trunk, features = flattened backbone output."""

    def __init__(self, variant="resnet50", pretrained=True, image_shape=(16, 16, 1),
                 freeze_backbone=False, label_sigma_m=2.5, readout_k=9,
                 cls_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.model = ResNetWiFi(
            variant=variant,
            pretrained=pretrained,
            image_shape=image_shape,
            freeze_backbone=freeze_backbone,
            head_type="flatten",
        )
        self.model.head = nn.Identity()

        self.model.eval()
        with torch.no_grad():
            feat = self.model(torch.zeros(1, *image_shape))
        self.model.train()
        self.feat_dim = feat.flatten(1).shape[1]

        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).flatten(1)


class VimClsWiFiModule(ClsHeadMixin, VisionMambaWiFi):
    """Vim-S trunk, features = 384-d CLS token."""

    def __init__(self, label_sigma_m=2.5, readout_k=9, cls_hidden=512, **kwargs):
        kwargs.setdefault('head_type', 'linear')  # replaced below
        super().__init__(**kwargs)
        self.model.head = nn.Identity()
        self.feat_dim = self.model.embed_dim
        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, x):
        return VisionMambaWiFi.forward(self, x)

    # parent keys the head on 'model.head', which is Identity here
    _param_groups = ClsHeadMixin._cls_param_groups


class SwinClsWiFiModule(ClsHeadMixin, BaseWiFiModule):
    """Swin-T trunk, GAP over the stage map.

    Flattening the map at 224px would give a ~19M-param classifier.
    """

    def __init__(self, variant="swin_tiny_patch4_window7_224", pretrained=True,
                 image_shape=(224, 224, 1), freeze_backbone=False, out_stage=3,
                 label_sigma_m=2.5, readout_k=9, cls_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.model = SwinWiFi(
            variant=variant,
            pretrained=pretrained,
            image_shape=image_shape,
            freeze_backbone=freeze_backbone,
            out_stage=out_stage,
            head_type="flatten",
        )
        self.model.head = nn.Identity()  # forward yields NCHW map

        self.model.eval()
        with torch.no_grad():
            feat = self.model(torch.zeros(1, *image_shape))
        self.model.train()
        self.feat_dim = feat.shape[1]

        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).mean(dim=(2, 3))


class DSCNNClsWiFiModule(ClsHeadMixin, BaseWiFiModule):
    """DSCNN trunk with BatchNorm after each conv, features = 128-d penultimate layer.

    Without BN the signal on the sparse RSSI image collapses to ~1e-4 relative
    std by the GAP layer and soft-CE stays at chance. Regression twin is
    DSCNN(use_batchnorm=True).
    """

    def __init__(self, image_shape=(16, 16, 1), dropout=0.5,
                 label_sigma_m=2.5, readout_k=9, cls_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.image_shape = image_shape

        self.trunk = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(  # "classifier" is the mixin's name
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.feat_dim = 128

        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, *self.image_shape).permute(0, 3, 1, 2)
        return self.mlp(self.trunk(x))


class EfficientNetClsWiFiModule(ClsHeadMixin, EfficientNetWiFi):
    """EfficientNet-B0 trunk, GAP over the stage-4 map (320 dims)."""

    def __init__(self, label_sigma_m=2.5, readout_k=9, cls_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.head = nn.Identity()

        image_shape = kwargs.get('image_shape', (64, 64, 1))
        self.eval()
        with torch.no_grad():
            feat = EfficientNetWiFi.forward(self, torch.zeros(1, *image_shape))
        self.train()
        self.feat_dim = feat.shape[1]

        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return EfficientNetWiFi.forward(self, x).mean(dim=(2, 3))


class ViTClsWiFiModule(ClsHeadMixin, BaseWiFiModule):
    """WiFiViT trunk, features = post-norm CLS token."""

    def __init__(self, img_size=16, patch_size=4, in_chans=1, embed_dim=128,
                 depth=4, nhead=4, dim_feedforward=256, dropout=0.1,
                 label_sigma_m=2.5, readout_k=9, cls_hidden=512, **kwargs):
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
        self.model.head = nn.Identity()
        self.feat_dim = embed_dim
        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FFNClsWiFiModule(ClsHeadMixin, ReferenceFFN):
    """FFN trunk without the output layer."""

    def __init__(self, label_sigma_m=2.5, readout_k=9, cls_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.feat_dim = self.network.output.in_features
        self.network.output = nn.Identity()
        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SAEClsWiFiModule(ClsHeadMixin, ReferenceSAE):
    """SDAE-pretrained SAE encoder, features = bottleneck code.

    The sigmoid bottleneck has per-dim std ~1e-4, hence the BatchNorm1d on top
    (LayerNorm in the classifier does not fix per-dim scale). The encoder
    diverges above lr ~1e-4, so keep learning_rate * backbone_lr_mult near 1e-4.
    """

    def __init__(self, label_sigma_m=2.5, readout_k=9, cls_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.regression = nn.Identity()
        self.feat_dim = self.encoder_layer_sizes[-1]
        self.feat_norm = nn.BatchNorm1d(self.feat_dim)
        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feat_norm(self.encoder(x))

    # no 'backbone' in param names
    _param_groups = ClsHeadMixin._cls_param_groups
    _head_param_prefixes = ClsHeadMixin._head_param_prefixes + ('feat_norm',)


class GCNClsWiFiModule(ClsHeadMixin, BaseWiFiModule):
    """GraphSAGE trunk, features = pooled graph embedding (2048-d with concat)."""

    def __init__(self, in_channels=1, hidden_channels=128, num_layers=3,
                 dropout=0.3, pooling="concat", num_aps=256, node_proj_dim=8,
                 label_sigma_m=2.5, readout_k=9, cls_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.model = ReferenceSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
            num_aps=num_aps,
            node_proj_dim=node_proj_dim,
        )
        self.model.regression_head = nn.Identity()
        self.feat_dim = (num_aps * node_proj_dim if pooling == "concat"
                         else hidden_channels)
        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, data) -> torch.Tensor:
        return self.model(data)


class VGAEClsWiFiModule(ClsHeadMixin, BaseWiFiModule):
    """VGAE trunk, features = pooled latent z_graph. Loss = soft-CE + masked KL."""

    def __init__(self, in_channels=1, hidden_channels=128, latent_dim=64,
                 dropout=0.3, pooling="concat", kl_weight=0.1, num_aps=256,
                 node_proj_dim=4, label_sigma_m=2.5, readout_k=9,
                 cls_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.kl_weight = kl_weight
        self.model = ReferenceGCNVGAE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            dropout=dropout,
            pooling=pooling,
            kl_weight=kl_weight,
            num_aps=num_aps,
            node_proj_dim=node_proj_dim,
        )
        self.model.regression_head = nn.Identity()
        self.feat_dim = (num_aps * node_proj_dim if pooling == "concat"
                         else latent_dim)
        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, data) -> torch.Tensor:
        return self.model(data, return_variational=False)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        features, targets, metadata = batch
        z_graph, mu, logvar = self.model(features, return_variational=True)
        logits = self.classifier(z_graph)

        ce_loss = self._soft_ce(logits, targets)
        # KL over visible APs only, the ~220 missing-AP nodes would dominate
        kl_per_node = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        node_mask = features.x[:, 0] > 0
        kl_loss = kl_per_node[node_mask].mean() if node_mask.any() else kl_per_node.mean()
        loss = ce_loss + self.kl_weight * kl_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_kl_loss', kl_loss, on_step=False, on_epoch=True)
        with torch.no_grad():
            predictions = self._coords(logits)
        if torch.isfinite(predictions).all():
            self.train_metrics.update(self._to_metres(predictions), self._to_metres(targets))
        return loss


class MambaClsWiFiModule(ClsHeadMixin, BaseWiFiModule):
    """Own Mamba trunk ('mamba'), features = flattened token map (1536-d at 16x16).

    Keep 16x16 input, the Python selective scan is slow for more tokens.
    """

    def __init__(self, img_size=16, patch_size=4, in_chans=1, embed_dim=96,
                 depth=6, d_state=1, mlp_ratio=4.0, drop_path_rate=0.1,
                 dropout=0.0, label_sigma_m=2.5, readout_k=9, cls_hidden=512,
                 **kwargs):
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
        self.feat_dim = self.model.head.in_features
        self.model.head = nn.Identity()
        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MambaVisionClsWiFiModule(ClsHeadMixin, MambaVisionWiFi):
    """MambaVision trunk (CUDA only), features = flattened map (~10k dims at 128px)."""

    def __init__(self, label_sigma_m=2.5, readout_k=9, cls_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.feat_dim = self.head.in_features
        self.head = nn.Identity()
        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return MambaVisionWiFi.forward(self, x)

    # params live under 'model.*', no 'backbone' in the names
    _param_groups = ClsHeadMixin._cls_param_groups


class SpatialMambaClsWiFiModule(ClsHeadMixin, BaseWiFiModule):
    """Spatial-Mamba-T trunk, GAP over the out_stage map (256 dims for tiny)."""

    def __init__(self, variant="tiny", pretrained_path=None,
                 image_shape=(224, 224, 1), out_stage=2, freeze_backbone=False,
                 label_sigma_m=2.5, readout_k=9, cls_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.model = SpatialMambaWiFi(
            variant=variant,
            pretrained_path=pretrained_path,
            image_shape=image_shape,
            out_stage=out_stage,
            freeze_backbone=freeze_backbone,
            head_type="flatten",
        )
        self.model.head = nn.Identity()
        # no probe forward, the vendor kernels need CUDA
        self.feat_dim = _SM_VARIANTS[variant]['dims'][out_stage]
        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).mean(dim=(2, 3))


class TransformerClsWiFiModule(ClsHeadMixin, WiFiTransformerModule):
    """WiFiTransformer trunk, features = pooled [B, d_model]."""

    def __init__(self, label_sigma_m=2.5, readout_k=9, cls_hidden=512, **kwargs):
        super().__init__(**kwargs)
        self.model.regression_head = nn.Identity()
        self.feat_dim = self.model.d_model
        self._init_cls(label_sigma_m, readout_k, cls_hidden)
        self.save_hyperparameters()

    def _features(self, x) -> torch.Tensor:
        return WiFiTransformerModule.forward(self, x)
