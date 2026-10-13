import torch.nn as nn


class MLPHead(nn.Sequential):
    """Linear -> ReLU -> Dropout -> Linear on a pooled vector.

    nn.Sequential subclass so the checkpoint keys stay 0.weight, 3.weight.
    """

    def __init__(self, in_features: int, out_dim: int = 2,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
