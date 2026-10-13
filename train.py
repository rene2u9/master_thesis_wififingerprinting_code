#!/usr/bin/env python3
"""Train runs from config.yaml.

    python train.py --models dscnn_gaussian vit ffn
    python train.py --skip-completed
"""

import argparse
import sys
import traceback
from pathlib import Path

import yaml

from src.training.trainer import WiFiTrainer
from src.training.checkpoint_utils import find_latest_checkpoint, find_incomplete_runs

_CONFIG_OVERRIDES = [
    ('db_path',        'data',     'db_path'),
    ('max_epochs',     'training', 'max_epochs'),
    ('batch_size',     'data',     'batch_size'),
    ('learning_rate',  'training', 'learning_rate'),
]


def is_completed(run_name: str, experiment_dir: str = "experiments") -> bool:
    run_dir = Path(experiment_dir) / run_name
    if not run_dir.exists():
        return False
    return any((d / "final_model.ckpt").exists() for d in run_dir.iterdir() if d.is_dir())


def train_single(config: dict, run_name: str, experiment_dir: str = "experiments",
                 resume_from: str = None, experiment_name: str = None) -> bool:
    print(f"\n{'='*60}\n  {run_name}\n{'='*60}\n")
    try:
        WiFiTrainer(
            config, run_name,
            experiment_dir=experiment_dir,
            resume_from=resume_from,
            experiment_name=experiment_name,
        ).train()
        return True
    except Exception as e:
        print(f"{run_name} failed: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Train WiFi fingerprinting models')

    parser.add_argument('--models', nargs='+', help='Run names from config.yaml (default: all)')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--experiment-dir', default='experiments')

    parser.add_argument('--db-path', type=str, default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--fast-dev-run', action='store_true',
                        help='3 epochs, 100k samples')

    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path (single model only)')
    parser.add_argument('--resume-latest', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--continue-from-failures', action='store_true',
                        help='Resume all incomplete runs')

    parser.add_argument('--skip-completed', action='store_true')
    parser.add_argument('--experiment-name', type=str, default=None)
    parser.add_argument('--experiment-suffix', type=str, default=None)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    available = list(config.get('runs', {}).keys())
    models = args.models or available

    invalid = [m for m in models if m not in available]
    if invalid:
        parser.error(f"Unknown runs: {invalid}. Available: {available}")

    for attr, section, key in _CONFIG_OVERRIDES:
        val = getattr(args, attr)
        if val is not None:
            config[section][key] = val

    if args.fast_dev_run:
        config['training']['max_epochs'] = 3
        config['data']['batch_size'] = 16
        config['data']['max_samples'] = 100_000
        print("Fast-dev-run: 100k samples, batch_size=16, 3 epochs")

    if args.experiment_suffix:
        config['experiment_suffix'] = args.experiment_suffix

    if args.resume and len(models) == 1:
        train_single(config, models[0], args.experiment_dir,
                     resume_from=args.resume, experiment_name=args.experiment_name)
        return

    if args.resume_latest and len(models) == 1:
        ckpt = find_latest_checkpoint(models[0], args.experiment_dir)
        if not ckpt:
            print(f"Error: No checkpoints found for '{models[0]}'")
            sys.exit(1)
        print(f"Resuming from: {ckpt}")
        train_single(config, models[0], args.experiment_dir, resume_from=ckpt)
        return

    if args.continue_from_failures:
        incomplete = find_incomplete_runs()
        if not incomplete:
            print("No incomplete runs found")
            return
        models = [m for m in models if m in incomplete]

    completed, failed, skipped = [], [], []

    for i, run_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {run_name}")
        if args.skip_completed and is_completed(run_name, args.experiment_dir):
            print(f"Skipping {run_name} (already completed)")
            skipped.append(run_name)
            continue

        resume = None
        if args.continue_from_failures:
            incomplete = find_incomplete_runs()
            resume = incomplete.get(run_name)

        ok = train_single(config, run_name, args.experiment_dir, resume_from=resume)
        (completed if ok else failed).append(run_name)

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    if completed: print(f"Completed: {', '.join(completed)}")
    if failed:    print(f"Failed: {', '.join(failed)}")
    if skipped:   print(f"Skipped: {', '.join(skipped)}")


if __name__ == '__main__':
    main()
