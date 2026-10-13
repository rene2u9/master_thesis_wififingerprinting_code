"""Bay classification over train-split centroids: classifier, loss, readout.

Soft targets are softmax(-d^2 / 2 sigma^2) with d in metres.
"""

import torch
import torch.nn as nn


def build_classifier(feat_dim: int, hidden: int, num_classes: int) -> nn.Sequential:
    """LayerNorm -> Linear -> GELU -> Linear.

    Do not reorder, checkpoints use the keys classifier.{0,1,3}.*
    """
    return nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, num_classes),
    )


def topk_soft_argmax(logits: torch.Tensor, centroids: torch.Tensor, k: int) -> torch.Tensor:
    """Softmax-weighted mean of the k best centroids. logits [B, K]."""
    k = min(k, logits.shape[1])
    vals, idx = logits.topk(k, dim=1)
    w = torch.softmax(vals, dim=1)
    return (w.unsqueeze(-1) * centroids[idx]).sum(dim=1)


def soft_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    centroids: torch.Tensor,
    coord_scale: torch.Tensor,
    sigma_m: float,
) -> torch.Tensor:
    """targets [B, 2] normalized, coord_scale [2] in metres per unit."""
    d2 = (((targets.unsqueeze(1) - centroids.unsqueeze(0)) * coord_scale)
          .pow(2).sum(dim=-1))                                     # [B, K]
    soft = torch.softmax(-d2 / (2 * sigma_m ** 2), dim=1)
    return -(soft * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
