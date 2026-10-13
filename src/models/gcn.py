"""GNN over the AP graph, adapted from GUFU (Chiu et al., 2024).
https://dl.acm.org/doi/10.1145/3712277

GraphConv instead of SAGEConv because it uses the edge weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GraphConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    Data = None
    Batch = None

from .base_module import BaseWiFiModule


class SAGEEncoder(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        if not HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required")

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class ReferenceSAGE(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "concat",
        num_aps: int = 256,
        node_proj_dim: int = 8,
    ):
        super().__init__()
        if not HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required")

        self.pooling = pooling
        self.num_aps = num_aps

        self.encoder = SAGEEncoder(in_channels, hidden_channels, num_layers, dropout)

        if pooling == "concat":
            self.node_proj = nn.Linear(hidden_channels, node_proj_dim)
            final_dim = num_aps * node_proj_dim
        else:
            self.node_proj = None
            final_dim = hidden_channels

        self.regression_head = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_weight = None
        if getattr(data, 'edge_attr', None) is not None:
            edge_weight = data.edge_attr.squeeze(-1)
        x = self.encoder(x, edge_index, edge_weight)

        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        if self.pooling == "concat":
            x = self.node_proj(x)
            # bipartite graphs have an extra RP node as last row, drop it
            per_graph = x.size(0) // batch_size
            x = x.view(batch_size, per_graph, -1)[:, :self.num_aps, :]
            graph_embedding = x.reshape(batch_size, -1)
        else:
            visible_mask = data.x[:, 0] > 0
            if not visible_mask.any():
                visible_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            graph_embedding = global_mean_pool(x[visible_mask], batch[visible_mask], size=batch_size)

        return self.regression_head(graph_embedding)


class ReferenceSAGEModule(BaseWiFiModule):

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "concat",
        num_aps: int = 256,
        node_proj_dim: int = 8,
        **kwargs
    ):
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
        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, (Data, Batch)) or (hasattr(x, 'x') and hasattr(x, 'edge_index')):
            return self.model(x)
        raise TypeError(f"Expected PyG Data/Batch, got {type(x)}")
