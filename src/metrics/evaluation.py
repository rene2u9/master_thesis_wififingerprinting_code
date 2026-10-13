"""Positioning error metrics (MPE, median, percentiles, CDF) and timing."""

import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torchmetrics


class WiFiMetrics(torchmetrics.Metric):
    """Euclidean error statistics in the unit of the targets (metres)."""

    def __init__(self, cdf_thresholds: List[float] = [1.0, 2.0, 5.0, 10.0]):
        super().__init__()
        self.cdf_thresholds = cdf_thresholds
        self.add_state("errors", default=[], dist_reduce_fx="cat")
        self.add_state("inference_times", default=[], dist_reduce_fx="cat")

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inference_time: Optional[float] = None
    ):
        self.errors.append(torch.norm(predictions.float() - targets.float(), dim=1))
        if inference_time is not None:
            self.inference_times.append(torch.tensor([inference_time]))

    def compute(self) -> Dict[str, float]:
        if not self.errors:
            return {}

        errors = torch.cat(self.errors).cpu().numpy()

        metrics = {
            'mpe': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'p50': float(np.percentile(errors, 50)),
            'p80': float(np.percentile(errors, 80)),
            'p90': float(np.percentile(errors, 90)),
            'p95': float(np.percentile(errors, 95)),
            'p99': float(np.percentile(errors, 99)),
        }

        for threshold in self.cdf_thresholds:
            metrics[f'cdf_{threshold}m'] = float((errors <= threshold).mean() * 100)

        if self.inference_times:
            times = torch.cat(self.inference_times).cpu().numpy()
            metrics['mean_inference_time'] = float(np.mean(times))
            metrics['median_inference_time'] = float(np.median(times))
            metrics['std_inference_time'] = float(np.std(times))

        return metrics

    def get_error_distribution(self) -> np.ndarray:
        return torch.cat(self.errors).cpu().numpy()


class ClassificationMetrics(torchmetrics.Metric):
    """Share of predictions within 1 m (exact) and within tolerance."""

    def __init__(self, tolerance: float = 5.0):
        super().__init__()
        self.tolerance = tolerance
        self.add_state("correct_exact", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_tolerance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        errors = torch.norm(predictions.float() - targets.float(), dim=1)
        self.correct_exact += (errors < 1.0).sum()
        self.correct_tolerance += (errors <= self.tolerance).sum()
        self.total += errors.numel()

    def compute(self) -> Dict[str, float]:
        if self.total == 0:
            return {}

        return {
            'accuracy_exact': float(self.correct_exact / self.total * 100),
            'accuracy_tolerance': float(self.correct_tolerance / self.total * 100),
        }


class InferenceTimer:
    """Context manager timing a forward pass, syncs CUDA."""

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.elapsed = None

    def __enter__(self):
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

    def get_elapsed(self) -> float:
        return self.elapsed
