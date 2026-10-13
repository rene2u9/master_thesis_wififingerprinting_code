"""Optimizer and LR-scheduler factories."""

from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)

OPTIMIZERS = ('adam', 'adamw', 'sgd')
SCHEDULERS = ('reduce_on_plateau', 'cosine', 'cosine_warmup', 'step', 'constant', 'none')


def build_optimizer(params, name: str, lr: float, weight_decay: float):
    if name == 'adam':
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer {name!r}; expected one of {OPTIMIZERS}")


def build_lr_config(
    optimizer,
    name: str,
    *,
    warmup_epochs: int = 5,
    cosine_t_max: int = None,
    max_epochs: int = None,
    monitor: str = 'val_mpe',
):
    """Return value for Lightning's configure_optimizers."""
    if name in ('constant', 'none', None):
        # flat LR worked better than warmup+cosine for pretrained backbones
        return optimizer

    if name == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': monitor,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    if name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=cosine_t_max or max_epochs)
    elif name == 'cosine_warmup':
        warmup = max(1, warmup_epochs)
        # cosine_t_max includes the warmup
        total = cosine_t_max or max_epochs
        cosine_len = max(1, total - warmup)
        scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, start_factor=0.01, total_iters=warmup),
                CosineAnnealingLR(optimizer, T_max=cosine_len),
            ],
            milestones=[warmup],
        )
    elif name == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        raise ValueError(f"Unknown scheduler {name!r}; expected one of {SCHEDULERS}")

    return [optimizer], [scheduler]
