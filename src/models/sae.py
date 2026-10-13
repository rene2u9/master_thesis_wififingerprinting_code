"""Stacked denoising autoencoder with regression head, pretrained layer-wise.
Kim et al., IEEE Access, 2018. https://ieeexplore.ieee.org/document/8444743
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset

from .base_module import BaseWiFiModule


class _MaskingNoiseDataset(Dataset):
    """Applies fresh masking noise to each sample on every access."""

    def __init__(self, x_clean: np.ndarray, corruption_level: float):
        self.x_clean = torch.tensor(x_clean, dtype=torch.float32)
        self.corruption_level = corruption_level

    def __len__(self):
        return len(self.x_clean)

    def __getitem__(self, idx):
        x = self.x_clean[idx].clone()
        mask = torch.rand_like(x) < self.corruption_level
        x[mask] = 0.0
        return x, self.x_clean[idx]


class ReferenceSAE(BaseWiFiModule):
    """Stacked Denoising Autoencoder. Call pretrain_encoder() before trainer.fit()."""

    def __init__(
        self,
        num_aps: int = 256,
        encoder_layers: list = None,
        regression_layers: list = None,
        dropout: float = 0.25,
        corruption_level: float = 0.1,
        pretrain_epochs: int = 50,
        pretrain_batch_size: int = 64,
        pretrain_lr: float = 0.001,
        freeze_encoder: bool = False,
        encoder_activation: str = "sigmoid",
        **kwargs
    ):
        super().__init__(**kwargs)

        if encoder_layers is None:
            encoder_layers = [256, 128, 64]
        if regression_layers is None:
            regression_layers = [256]

        self.num_aps = num_aps
        self.encoder_layer_sizes = encoder_layers
        self.regression_layer_sizes = regression_layers
        self.corruption_level = corruption_level
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_batch_size = pretrain_batch_size
        self.pretrain_lr = pretrain_lr
        self.freeze_encoder_flag = freeze_encoder
        self.encoder_activation = encoder_activation.lower()

        _act_map = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "tanh": nn.Tanh}
        act_cls = _act_map.get(self.encoder_activation, nn.Sigmoid)

        all_dims = [num_aps] + encoder_layers
        encoder_units = []
        for i in range(len(encoder_layers)):
            encoder_units.append((f'fc{i}', nn.Linear(all_dims[i], all_dims[i + 1])))
            encoder_units.append((f'act{i}', act_cls()))
        self.encoder = nn.Sequential(OrderedDict(encoder_units))

        reg_units = []
        prev_dim = encoder_layers[-1]
        reg_units.append(('bn_in', nn.BatchNorm1d(prev_dim)))
        reg_units.append(('relu_in', nn.ReLU()))
        reg_units.append(('drop_in', nn.Dropout(dropout)))
        for i, units in enumerate(regression_layers):
            reg_units.append((f'fc{i}', nn.Linear(prev_dim, units)))
            reg_units.append((f'bn{i}', nn.BatchNorm1d(units)))
            reg_units.append((f'relu{i}', nn.ReLU()))
            reg_units.append((f'drop{i}', nn.Dropout(dropout)))
            prev_dim = units
        reg_units.append(('output', nn.Linear(prev_dim, 2)))
        self.regression = nn.Sequential(OrderedDict(reg_units))

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.regression(z)

    def _param_groups(self):
        # no 'backbone' in param names, and the encoder diverges above lr ~1e-4
        if self.backbone_lr_mult == 1.0:
            return self.parameters()
        enc, head = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (enc if name.startswith('encoder') else head).append(p)
        return [
            {'params': enc, 'lr': self.learning_rate * self.backbone_lr_mult},
            {'params': head, 'lr': self.learning_rate},
        ]

    def pretrain_encoder(self, train_dataloader: DataLoader):
        """Greedy layer-wise SDAE pretraining, following sdae_pt.py of Kim et al."""
        if self.pretrain_epochs <= 0:
            print("SDAE pretraining skipped (pretrain_epochs=0)")
            return

        print(f"\n{'=' * 60}")
        print(f"SDAE Pretraining: {len(self.encoder_layer_sizes)} layers, "
              f"{self.pretrain_epochs} epochs each, corruption={self.corruption_level}")
        print(f"{'=' * 60}")

        all_features = []
        for batch in train_dataloader:
            features, _, _ = batch
            all_features.append(features)
        x_all = torch.cat(all_features, dim=0).detach().cpu().numpy()

        device = next(self.parameters()).device
        all_dims = [self.num_aps] + self.encoder_layer_sizes
        x = x_all.copy()

        for layer_idx in range(len(self.encoder_layer_sizes)):
            in_dim = all_dims[layer_idx]
            out_dim = all_dims[layer_idx + 1]
            print(f"\nPretraining layer {layer_idx + 1}/{len(self.encoder_layer_sizes)}: "
                  f"{in_dim} -> {out_dim}")

            ae = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Sigmoid(),
                nn.Linear(out_dim, in_dim)
            ).to(device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(ae.parameters(), lr=self.pretrain_lr, momentum=0.9)

            dataset = _MaskingNoiseDataset(x.astype(np.float32), self.corruption_level)
            loader = DataLoader(dataset, batch_size=self.pretrain_batch_size, shuffle=True)

            ae.train()
            for epoch in range(self.pretrain_epochs):
                running_loss = 0.0
                for x_batch, y_batch in loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    optimizer.zero_grad()
                    y_pred = ae(x_batch)
                    loss = criterion(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"  Epoch {epoch + 1}/{self.pretrain_epochs}: "
                          f"loss={running_loss / len(loader):.6f}")

            with torch.no_grad():
                self.encoder[2 * layer_idx].weight.copy_(ae[0].weight)
                self.encoder[2 * layer_idx].bias.copy_(ae[0].bias)

            encoder_only = nn.Sequential(ae[0], ae[1]).to(device)
            encoder_only.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(x.astype(np.float32), device=device)
                x = encoder_only(x_tensor).detach().cpu().numpy()

        if self.freeze_encoder_flag:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("\nEncoder weights frozen for regression training.")

        print(f"\n{'=' * 60}")
        print("SDAE Pretraining complete!")
        print(f"{'=' * 60}\n")
