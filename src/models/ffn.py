"""FFN baseline, following J. Chen's wifi-positioning-dnn."""

import torch
import torch.nn as nn
from collections import OrderedDict

from .base_module import BaseWiFiModule


class ReferenceFFN(BaseWiFiModule):
    def __init__(
        self,
        num_aps: int = 256,
        hidden_layers: list = None,
        dropout: float = 0.25,
        activation: str = 'relu',
        **kwargs
    ):
        super().__init__(**kwargs)

        if hidden_layers is None:
            hidden_layers = [256, 128]

        act_fn = nn.ReLU if activation == 'relu' else nn.Sigmoid

        layers = []
        prev_dim = num_aps
        for i, units in enumerate(hidden_layers):
            layers.append((f'fc{i}', nn.Linear(prev_dim, units)))
            layers.append((f'act{i}', act_fn()))
            layers.append((f'drop{i}', nn.Dropout(dropout)))
            prev_dim = units
        layers.append(('output', nn.Linear(prev_dim, 2)))

        self.network = nn.Sequential(OrderedDict(layers))

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
