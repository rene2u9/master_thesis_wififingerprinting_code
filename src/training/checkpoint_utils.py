"""Checkpoint discovery utilities."""

import re
from pathlib import Path
from typing import Optional


def find_latest_checkpoint(model_name: str, experiment_dir: str = "experiments") -> Optional[str]:
    model_dir = Path(experiment_dir) / model_name
    if not model_dir.exists():
        return None

    latest_ckpt, latest_mtime = None, None
    for run_dir in model_dir.iterdir():
        if not run_dir.is_dir():
            continue
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            for ckpt in ckpt_dir.glob("*.ckpt"):
                mtime = ckpt.stat().st_mtime
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_ckpt = str(ckpt)
    return latest_ckpt


def find_best_checkpoint(model_name: str, experiment_dir: str = "experiments") -> Optional[str]:
    model_dir = Path(experiment_dir) / model_name
    if not model_dir.exists():
        return None

    best_ckpt, best_mpe = None, float('inf')
    for run_dir in model_dir.iterdir():
        if not run_dir.is_dir():
            continue
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            for ckpt in ckpt_dir.glob("*.ckpt"):
                match = re.search(r'val_mpe=([\d.]+)', ckpt.name)
                if match:
                    mpe = float(match.group(1))
                    if mpe < best_mpe:
                        best_mpe = mpe
                        best_ckpt = str(ckpt)
    return best_ckpt


def find_incomplete_runs(experiment_dir: str = "experiments") -> dict[str, str]:
    """Runs with checkpoints but no final_model.ckpt, mapped to their best checkpoint."""
    incomplete = {}
    exp_path = Path(experiment_dir)
    if not exp_path.exists():
        return incomplete

    for model_dir in exp_path.iterdir():
        if not model_dir.is_dir():
            continue
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            has_ckpts = (run_dir / "checkpoints").exists()
            has_final = (run_dir / "final_model.ckpt").exists()
            if has_ckpts and not has_final:
                ckpt_dir = run_dir / "checkpoints"
                best_ckpt, best_mpe = None, float('inf')
                for ckpt in ckpt_dir.glob("*.ckpt"):
                    match = re.search(r'val_mpe=([\d.]+)', ckpt.name)
                    if match:
                        mpe = float(match.group(1))
                        if mpe < best_mpe:
                            best_mpe = mpe
                            best_ckpt = str(ckpt)
                if best_ckpt:
                    incomplete[model_dir.name] = best_ckpt
    return incomplete
