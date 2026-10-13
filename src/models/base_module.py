"""Base Lightning module: training loop, metrics in metres, optimizer setup."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any

from ..metrics.evaluation import WiFiMetrics, ClassificationMetrics, InferenceTimer
from ..training.optimizers import build_optimizer, build_lr_config


class BaseWiFiModule(pl.LightningModule):

    def __init__(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        optimizer: str = 'adam',
        scheduler: str = 'reduce_on_plateau',
        warmup_epochs: int = 5,
        cosine_t_max: int = None,      # None -> max_epochs
        target_scale: list = None,     # [lat_range, lon_range] in metres
        backbone_lr_mult: float = 1.0,
        cdf_thresholds: list = [1.0, 2.0, 5.0, 10.0],
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.warmup_epochs = warmup_epochs
        self.cosine_t_max = cosine_t_max
        self.backbone_lr_mult = backbone_lr_mult

        self.criterion = nn.MSELoss()

        self.train_metrics = WiFiMetrics(cdf_thresholds=cdf_thresholds)
        self.val_metrics = WiFiMetrics(cdf_thresholds=cdf_thresholds)
        self.test_metrics = WiFiMetrics(cdf_thresholds=cdf_thresholds)

        self.val_classification = ClassificationMetrics(tolerance=5.0)
        self.test_classification = ClassificationMetrics(tolerance=5.0)

        # targets are in [0, 1], metrics are computed in metres
        if target_scale is not None:
            self.register_buffer('_coord_scale', torch.tensor(target_scale, dtype=torch.float32))
        else:
            self.register_buffer('_coord_scale', None)

    def _to_metres(self, coords: torch.Tensor) -> torch.Tensor:
        if self._coord_scale is not None:
            return coords * self._coord_scale
        return coords

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward()")

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        features, targets, metadata = batch
        predictions = self(features)
        loss = self.criterion(predictions, targets)
        self.train_metrics.update(self._to_metres(predictions), self._to_metres(targets))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        features, targets, metadata = batch
        timer = InferenceTimer(device=str(self.device))
        with timer:
            predictions = self(features)
        loss = self.criterion(predictions, targets)
        self.val_metrics.update(self._to_metres(predictions), self._to_metres(targets), timer.get_elapsed())
        self.val_classification.update(self._to_metres(predictions), self._to_metres(targets))
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_test_epoch_start(self):
        # per-sample predictions in metres, saved if logging.save_predictions
        self._test_records = {"id": [], "pred": [], "true": []}

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        features, targets, metadata = batch
        timer = InferenceTimer(device=str(self.device))
        with timer:
            predictions = self(features)
        loss = self.criterion(predictions, targets)
        self.test_metrics.update(self._to_metres(predictions), self._to_metres(targets), timer.get_elapsed())
        self.test_classification.update(self._to_metres(predictions), self._to_metres(targets))
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        if isinstance(metadata, dict) and "id" in metadata:
            ids = metadata["id"]
            self._test_records["id"].extend(
                ids.tolist() if torch.is_tensor(ids) else list(ids))
            self._test_records["pred"].extend(
                self._to_metres(predictions).detach().cpu().tolist())
            self._test_records["true"].extend(
                self._to_metres(targets).detach().cpu().tolist())

        return loss

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f'train_{name}', value)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f'val_{name}', value)
        class_metrics = self.val_classification.compute()
        for name, value in class_metrics.items():
            self.log(f'val_{name}', value)
        self.val_metrics.reset()
        self.val_classification.reset()

    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f'test_{name}', value)
        class_metrics = self.test_classification.compute()
        for name, value in class_metrics.items():
            self.log(f'test_{name}', value)

        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Mean Positioning Error: {metrics.get('mpe', 0):.2f} m")
        print(f"Median Error: {metrics.get('median_error', 0):.2f} m")
        print(f"P80: {metrics.get('p80', 0):.2f} m")
        print(f"P95: {metrics.get('p95', 0):.2f} m")
        print("\nCDF:")
        for threshold in [1.0, 2.0, 5.0, 10.0]:
            pct = metrics.get(f'cdf_{threshold}m', 0)
            print(f"  {threshold}m: {pct:.1f}%")
        print(f"\nAccuracy (exact): {class_metrics.get('accuracy_exact', 0):.1f}%")
        print(f"Accuracy (5m tolerance): {class_metrics.get('accuracy_tolerance', 0):.1f}%")
        if 'mean_inference_time' in metrics:
            print(f"\nMean Inference Time: {metrics['mean_inference_time']*1000:.2f} ms")
        print("="*60 + "\n")

        self.test_metrics.reset()
        self.test_classification.reset()

    def _param_groups(self):
        """Disc-LR: params named '*backbone*' get learning_rate * backbone_lr_mult."""
        if self.backbone_lr_mult == 1.0:
            return self.parameters()
        backbone, head = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (backbone if 'backbone' in name else head).append(p)
        return [
            {'params': backbone, 'lr': self.learning_rate * self.backbone_lr_mult},
            {'params': head, 'lr': self.learning_rate},
        ]

    def configure_optimizers(self) -> Dict:
        optimizer = build_optimizer(
            self._param_groups(), self.optimizer_name,
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        max_epochs = None
        if self.scheduler_name in ('cosine', 'cosine_warmup') and self.cosine_t_max is None:
            max_epochs = self.trainer.max_epochs
        return build_lr_config(
            optimizer, self.scheduler_name,
            warmup_epochs=self.warmup_epochs,
            cosine_t_max=self.cosine_t_max,
            max_epochs=max_epochs,
        )
