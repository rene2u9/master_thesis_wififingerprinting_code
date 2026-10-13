"""Lightning mixin for the bay-classification variants.

Host defines feat_dim and _features(x) and calls _init_cls() in __init__.
The trainer calls set_centroids() before fit.
"""

import torch

from .heads.cls import build_classifier, soft_ce, topk_soft_argmax


class ClsHeadMixin:

    def _init_cls(self, label_sigma_m: float, readout_k: int, cls_hidden: int) -> None:
        self.label_sigma_m = label_sigma_m
        self.readout_k = readout_k
        self.cls_hidden = cls_hidden
        self.classifier = None

    def set_centroids(self, centroids: torch.Tensor) -> None:
        """centroids: [K, 2] unique train coordinates, normalized."""
        self.register_buffer('centroids', centroids.float())
        num_bays = centroids.shape[0]
        self.classifier = build_classifier(self.feat_dim, self.cls_hidden, num_bays)
        print(f"Classification head: {self.feat_dim} -> {self.cls_hidden} -> "
              f"{num_bays} bay centroids (sigma={self.label_sigma_m}m, "
              f"readout top-{self.readout_k})")

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._features(x))

    def _coords(self, logits: torch.Tensor) -> torch.Tensor:
        return topk_soft_argmax(logits, self.centroids, self.readout_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns coordinates so the base val/test steps work as is
        return self._coords(self.forward_logits(x))

    def _soft_ce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return soft_ce(logits, targets, self.centroids,
                       self._coord_scale, self.label_sigma_m)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        features, targets, metadata = batch
        logits = self.forward_logits(features)
        loss = self._soft_ce(logits, targets)
        with torch.no_grad():
            predictions = self._coords(logits)
        self.train_metrics.update(self._to_metres(predictions), self._to_metres(targets))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # trained at full LR in _cls_param_groups, extend for new fresh modules
    _head_param_prefixes = ('classifier',)

    def _cls_param_groups(self):
        """Disc-LR split by name prefix, for hosts without 'backbone' in param names."""
        if self.backbone_lr_mult == 1.0:
            return self.parameters()
        backbone, head = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (head if name.startswith(self._head_param_prefixes) else backbone).append(p)
        print(f"Disc-LR groups: backbone {len(backbone)} params @ "
              f"{self.learning_rate * self.backbone_lr_mult:g}, "
              f"head {len(head)} params @ {self.learning_rate:g} "
              f"(head prefixes: {self._head_param_prefixes})")
        return [
            {'params': backbone, 'lr': self.learning_rate * self.backbone_lr_mult},
            {'params': head, 'lr': self.learning_rate},
        ]
