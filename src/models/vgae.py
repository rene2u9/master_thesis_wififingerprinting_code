"""VGAE with GCNConv encoder, loss MSE + kl_weight * KL.
Kipf and Welling, 2016. https://arxiv.org/abs/1611.07308
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    Data = None
    Batch = None

from .base_module import BaseWiFiModule


class GCNVGAEEncoder(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int, dropout: float = 0.3):
        super().__init__()
        if not HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required")
        # the graph transform already adds RSSI-weighted self-loops
        self.gcn_shared = GCNConv(in_channels, hidden_channels, add_self_loops=False)
        self.gcn_mu = GCNConv(hidden_channels, latent_dim, add_self_loops=False)
        self.gcn_logvar = GCNConv(hidden_channels, latent_dim, add_self_loops=False)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.gcn_shared(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        mu = self.gcn_mu(x, edge_index, edge_weight)
        logvar = self.gcn_logvar(x, edge_index, edge_weight)
        return mu, logvar


class ReferenceGCNVGAE(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.3,
        pooling: str = "concat",
        kl_weight: float = 0.1,
        num_aps: int = 256,
        node_proj_dim: int = 4,
    ):
        super().__init__()
        if not HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required")

        self.pooling = pooling
        self.kl_weight = kl_weight
        self.num_aps = num_aps

        self.encoder = GCNVGAEEncoder(in_channels, hidden_channels, latent_dim, dropout)

        if pooling == "concat":
            self.node_proj = nn.Linear(latent_dim, node_proj_dim)
            final_dim = num_aps * node_proj_dim
        else:
            self.node_proj = None
            final_dim = latent_dim

        self.regression_head = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, data: Data, return_variational: bool = False):
        x, edge_index = data.x, data.edge_index
        edge_weight = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight = data.edge_attr.squeeze(-1).clamp(0, 1)

        mu, logvar = self.encoder(x, edge_index, edge_weight)
        z = self.reparameterize(mu, logvar)

        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        if self.pooling == "concat":
            z_proj = self.node_proj(z)
            z_graph = z_proj.view(batch_size, -1)
        else:
            visible_mask = data.x[:, 0] > 0
            if not visible_mask.any():
                visible_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            z_graph = global_mean_pool(z[visible_mask], batch[visible_mask], size=batch_size)

        coordinates = self.regression_head(z_graph)

        if return_variational:
            return coordinates, mu, logvar
        return coordinates


class ReferenceGCNVGAEModule(BaseWiFiModule):

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.3,
        pooling: str = "concat",
        kl_weight: float = 0.1,
        num_aps: int = 256,
        node_proj_dim: int = 4,
        **kwargs
    ):
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
        self.save_hyperparameters()

    def forward(self, x):
        if isinstance(x, (Data, Batch)) or (hasattr(x, 'x') and hasattr(x, 'edge_index')):
            return self.model(x, return_variational=False)
        raise TypeError(f"Expected PyG Data/Batch, got {type(x)}")

    def training_step(self, batch, batch_idx):
        features, targets, metadata = batch
        predictions, mu, logvar = self.model(features, return_variational=True)

        mse_loss = self.criterion(predictions, targets)
        # KL over visible APs only, the ~220 missing-AP nodes would dominate
        kl_per_node = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        node_mask = features.x[:, 0] > 0
        kl_loss = kl_per_node[node_mask].mean() if node_mask.any() else kl_per_node.mean()
        loss = mse_loss + self.kl_weight * kl_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_kl_loss', kl_loss, on_step=False, on_epoch=True)
        if torch.isfinite(predictions).all():
            self.train_metrics.update(self._to_metres(predictions.detach()), self._to_metres(targets))
        return loss

