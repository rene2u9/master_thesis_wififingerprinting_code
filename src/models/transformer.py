"""Encoder-only transformer with learned AP embeddings, missing APs masked.
Following HyTra (2023), AaTs (2024), WiFiGPT (2025).
variable_length=True uses only the detected APs as tokens (needs the sparse collate_fn).
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Tuple, Dict, List

from .base_module import BaseWiFiModule


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class SpatialPositionalEncoding(nn.Module):
    """Positional encoding from physical AP coordinates (lat, lon)."""

    def __init__(self, num_aps: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        # Positions buffer: (num_aps, 2) normalised lat/lon
        self.register_buffer(
            'ap_positions',
            torch.zeros(num_aps, 2, dtype=torch.float32),
        )

    def load_positions(self, db_path: str, sorted_aps: List[str]) -> None:
        """Load AP positions from database and store as buffer."""
        import sqlite3
        from ..data import WAREHOUSE_LAT_RANGE, WAREHOUSE_LON_RANGE

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT mac_prefix, latitude, longitude FROM ap_positions"
        ).fetchall()
        conn.close()

        pos = {r[0]: (float(r[1]), float(r[2])) for r in rows}
        centre = (WAREHOUSE_LAT_RANGE / 2, WAREHOUSE_LON_RANGE / 2)

        positions = np.zeros((self.ap_positions.size(0), 2), dtype=np.float32)
        matched = 0
        for i, ap in enumerate(sorted_aps):
            if i >= positions.shape[0]:
                break
            lat, lon = pos.get(ap, centre)
            positions[i] = [lat / WAREHOUSE_LAT_RANGE, lon / WAREHOUSE_LON_RANGE]
            if ap in pos:
                matched += 1
        # Remaining slots (padding) get centre position
        for i in range(len(sorted_aps), positions.shape[0]):
            positions[i] = [0.5, 0.5]

        self.ap_positions.copy_(torch.from_numpy(positions))
        print(f"SpatialPositionalEncoding: {matched}/{len(sorted_aps)} APs matched to positions")

    def forward(self, ap_indices: torch.Tensor) -> torch.Tensor:
        """ap_indices: (B, seq_len) -> (B, seq_len, d_model)"""
        coords = self.ap_positions[ap_indices]  # (B, seq_len, 2)
        return self.dropout(self.pos_proj(coords))


class APEmbedding(nn.Module):
    """Per-AP learnable embedding combined with projected RSSI value."""
    def __init__(
        self,
        num_aps: int,
        d_model: int,
        missing_value: float = -100.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_aps = num_aps
        self.d_model = d_model
        self.missing_value = missing_value

        self.value_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )

        self.ap_embeddings = nn.Embedding(num_aps, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, rssi_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dense mode: full RSSI vector (B, num_aps)."""
        batch_size, num_aps = rssi_vector.shape
        device = rssi_vector.device

        mask = (rssi_vector == self.missing_value)  # True = ignore in attention

        rssi_values = rssi_vector.unsqueeze(-1)
        value_embeddings = self.value_proj(rssi_values)

        ap_indices = torch.arange(num_aps, device=device).unsqueeze(0).expand(batch_size, -1)
        ap_emb = self.ap_embeddings(ap_indices)

        embeddings = value_embeddings + ap_emb

        # masked tokens would still leak through the pre-norm residual
        embeddings = embeddings * (~mask).unsqueeze(-1).float()

        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings, mask

    def forward_sparse(
        self,
        ap_indices: torch.Tensor,
        rssi_values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Variable-length mode: only detected APs.
        ap_indices:   (B, max_len), AP index of each token
        rssi_values:  (B, max_len), normalised RSSI values
        padding_mask: (B, max_len), True = padding position
        """
        val_emb = self.value_proj(rssi_values.unsqueeze(-1))  # (B, max_len, d_model)
        ap_emb = self.ap_embeddings(ap_indices)                # (B, max_len, d_model)
        embeddings = val_emb + ap_emb

        embeddings = embeddings * (~padding_mask).unsqueeze(-1).float()
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class TransformerRegressionHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 512,
        dropout: float = 0.5
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class WiFiTransformer(nn.Module):
    def __init__(
        self,
        num_aps: int = 576,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        missing_value: float = -100.0,
        use_positional_encoding: bool = True,
        masking_strategy: str = "additive",  # "additive" or "boolean"
        pooling: str = "mean",               # "mean", "max", "cls"
        variable_length: bool = False,
        spatial_positional_encoding: bool = False,
    ):
        super().__init__()

        self.num_aps = num_aps
        self.d_model = d_model
        self.missing_value = missing_value
        self.masking_strategy = masking_strategy
        self.pooling = pooling
        self.variable_length = variable_length

        self.ap_embedding = APEmbedding(
            num_aps=num_aps,
            d_model=d_model,
            missing_value=missing_value,
            dropout=dropout
        )

        self.use_positional_encoding = use_positional_encoding and not spatial_positional_encoding
        if self.use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                d_model=d_model,
                max_len=num_aps + 1,
                dropout=dropout
            )

        self.spatial_positional_encoding = spatial_positional_encoding
        if spatial_positional_encoding:
            self.spatial_pos_encoder = SpatialPositionalEncoding(
                num_aps=num_aps,
                d_model=d_model,
                dropout=dropout,
            )

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.regression_head = TransformerRegressionHead(
            d_model=d_model,
            hidden_dim=dim_feedforward // 2,
            dropout=dropout
        )

    def pool_output(
        self,
        encoder_output: torch.Tensor,
        padding_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return encoder_output[:, 0, :]

        elif self.pooling == "mean":
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            sum_output = (encoder_output * valid_mask).sum(dim=1)
            num_valid = valid_mask.sum(dim=1).clamp(min=1)
            return sum_output / num_valid

        elif self.pooling == "max":
            masked_output = encoder_output.masked_fill(
                padding_mask.unsqueeze(-1), float('-inf')
            )
            return masked_output.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def _prepend_cls(self, embeddings, padding_mask, device):
        batch_size = embeddings.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        return embeddings, padding_mask

    def forward_dense(self, rssi_vector: torch.Tensor) -> torch.Tensor:
        """Dense mode: full RSSI vector with masking."""
        embeddings, padding_mask = self.ap_embedding(rssi_vector)

        if self.pooling == "cls":
            embeddings, padding_mask = self._prepend_cls(
                embeddings, padding_mask, rssi_vector.device
            )

        if self.use_positional_encoding:
            embeddings = self.pos_encoder(embeddings)

        encoder_output = self.transformer_encoder(
            embeddings, src_key_padding_mask=padding_mask
        )

        pooled = self.pool_output(encoder_output, padding_mask)
        return self.regression_head(pooled)

    def forward_sparse(
        self,
        ap_indices: torch.Tensor,
        rssi_values: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Variable-length mode: only detected APs."""
        device = ap_indices.device

        embeddings = self.ap_embedding.forward_sparse(
            ap_indices, rssi_values, padding_mask
        )

        if self.spatial_positional_encoding:
            embeddings = embeddings + self.spatial_pos_encoder(ap_indices)

        if self.pooling == "cls":
            embeddings, padding_mask = self._prepend_cls(
                embeddings, padding_mask, device
            )

        encoder_output = self.transformer_encoder(
            embeddings, src_key_padding_mask=padding_mask
        )

        pooled = self.pool_output(encoder_output, padding_mask)
        return self.regression_head(pooled)

    def forward(self, x):
        if isinstance(x, dict):
            return self.forward_sparse(
                x['ap_indices'], x['rssi_values'], x['padding_mask']
            )
        return self.forward_dense(x)


class WiFiTransformerModule(BaseWiFiModule):
    def __init__(
        self,
        num_aps: int = 576,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        missing_value: float = -100.0,
        use_positional_encoding: bool = True,
        masking_strategy: str = "additive",
        pooling: str = "mean",
        variable_length: bool = False,
        spatial_positional_encoding: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = WiFiTransformer(
            num_aps=num_aps,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            missing_value=missing_value,
            use_positional_encoding=use_positional_encoding,
            masking_strategy=masking_strategy,
            pooling=pooling,
            variable_length=variable_length,
            spatial_positional_encoding=spatial_positional_encoding,
        )

        self.save_hyperparameters()

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, dict):
            return self.model(x)
        if isinstance(x, torch.Tensor) and x.dim() in (3, 4):
            x = x.view(x.size(0), -1)
        return self.model(x)

