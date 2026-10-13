#!/usr/bin/env python3
"""Constant-prediction baselines (train mean, train median, most frequent bay).

Usage:
    python scripts/centroid_baseline.py --run resnet50_cls100k
    python scripts/centroid_baseline.py --run resnet50_cls100k --split temporal
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import WiFiFingerprintDataModule

CDF_THRESHOLDS = [1.0, 2.0, 5.0, 10.0]


def build_module(cfg, run_cfg, split_override=None):
    d = cfg["data"]
    return WiFiFingerprintDataModule(
        db_path=d["db_path"],
        train_split=d["train_split"],
        val_split=d["val_split"],
        test_split=d["test_split"],
        random_seed=run_cfg.get("random_seed", d["random_seed"]),
        max_samples=run_cfg.get("max_samples", d.get("max_samples")),
        storage_place_filter=d.get("storage_place_filter"),
        deduplicate=run_cfg.get("deduplicate", d.get("deduplicate", False)),
        sampling=run_cfg.get("sampling", d.get("sampling", "random")),
        reps_per_place=run_cfg.get("reps_per_place", d.get("reps_per_place", 10)),
        split_strategy=split_override or run_cfg.get(
            "split_strategy", d.get("split_strategy", "random")),
        max_aps=d["max_aps"],
        missing_value=d["missing_value"],
        ssid_filter=d.get("filter_ssid_prefix"),
        target_scale=tuple(d["target_scale"]) if d.get("target_scale") else None,
        rssi_normalization=d.get("rssi_normalization", "minmax"),
        rssi_min=d.get("rssi_min", -100.0),
        rssi_max=d.get("rssi_max", 0.0),
    )


def report(tag, pred, yte, scale):
    err = ((torch.as_tensor(pred, dtype=torch.float32) - yte) * scale).norm(dim=1)
    cdf = "  ".join(f"{(err < t).float().mean().item() * 100:6.1f}%" for t in CDF_THRESHOLDS)
    print(f"{tag:>14}  {err.mean().item():>8.2f}  {err.median().item():>9.2f}  {cdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", default="resnet50_cls100k",
                    help="run name whose data config defines the regime")
    ap.add_argument("--split", choices=["random", "temporal", "group"], default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run_cfg = cfg.get("runs", {}).get(args.run, {})
    dm = build_module(cfg, run_cfg, args.split)
    ds_tr, ds_te = dm.get_dataset("train"), dm.get_dataset("test")
    ytr = ds_tr._targets
    yte = torch.from_numpy(ds_te._targets)
    scale = torch.tensor(cfg["data"]["target_scale"], dtype=torch.float32)
    n_te = yte.shape[0]

    bays = [";".join(d.split(";")[:3]) for d in ds_tr._storage_descs]
    mode_bay, mode_count = Counter(bays).most_common(1)[0]
    mask = np.array([b == mode_bay for b in bays])
    mode_centroid = ytr[mask].mean(axis=0)

    print(f"\nregime=run:{args.run}  split={dm.split_strategy}  "
          f"train={ytr.shape[0]:,}  test={n_te:,}")
    print(f"mode bay: {mode_bay} ({mode_count} train rows)")
    print(f"{'baseline':>14}  {'MPE(m)':>8}  {'median(m)':>9}  "
          + "  ".join(f"CDF@{int(t)}m" for t in CDF_THRESHOLDS))
    report("global-mean", np.tile(ytr.mean(axis=0), (n_te, 1)), yte, scale)
    report("global-median", np.tile(np.median(ytr, axis=0), (n_te, 1)), yte, scale)
    report("mode-bay", np.tile(mode_centroid, (n_te, 1)), yte, scale)


if __name__ == "__main__":
    main()
