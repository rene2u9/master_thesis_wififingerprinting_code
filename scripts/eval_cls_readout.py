#!/usr/bin/env python3
"""Sweep readout_k of a trained bay-classification checkpoint (k=0: full softmax).

Usage:
    python scripts/eval_cls_readout.py --ckpt experiments/resnet50_cls100k \
        [--run resnet50_cls100k] [--k 1 3 5 9 15 25 0]
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import WiFiFingerprintDataModule
from src.models.cls_variants import ResNetClsWiFiModule
from src.models.heads.cls import topk_soft_argmax

CDF_THRESHOLDS = [1.0, 2.0, 5.0, 10.0]


def resolve_ckpt(path):
    if path.endswith(".ckpt"):
        return path
    ckpts = glob.glob(os.path.join(path, "**", "*.ckpt"), recursive=True)
    scored = [(float(m.group(1)), p) for p in ckpts
              if (m := re.search(r"val_mpe=(\d+(?:\.\d+)?)", os.path.basename(p)))]
    if not scored:
        raise SystemExit(f"no val_mpe checkpoints under {path}")
    return min(scored)[1]


def build_module(cfg, run_cfg):
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
        split_strategy=run_cfg.get("split_strategy", d.get("split_strategy", "random")),
        max_aps=d["max_aps"],
        missing_value=d["missing_value"],
        ssid_filter=d.get("filter_ssid_prefix"),
        target_scale=tuple(d["target_scale"]) if d.get("target_scale") else None,
        rssi_normalization=d.get("rssi_normalization", "minmax"),
        rssi_min=d.get("rssi_min", -100.0),
        rssi_max=d.get("rssi_max", 0.0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help=".ckpt file or experiments dir")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", default="resnet50_cls100k")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 9, 15, 25, 0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run_cfg = cfg.get("runs", {}).get(args.run, {})
    ckpt_path = resolve_ckpt(args.ckpt)
    print(f"checkpoint: {ckpt_path}")

    dm = build_module(cfg, run_cfg)
    ds_te = dm.get_dataset("test")
    h, w, c = run_cfg.get("image_shape", cfg["data"]["image_shape"])
    Xte = torch.from_numpy(ds_te._features).view(-1, h, w, c)
    yte = torch.from_numpy(ds_te._targets)
    scale = torch.tensor(cfg["data"]["target_scale"], dtype=torch.float32)
    print(f"test={Xte.shape[0]:,}  image={h}x{w}x{c}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    num_bays = state["centroids"].shape[0]
    model = ResNetClsWiFiModule(
        variant=run_cfg.get("variant", "resnet50"), pretrained=False,
        image_shape=(h, w, c), target_scale=list(scale.numpy()),
        label_sigma_m=run_cfg.get("label_sigma_m", 2.5),
        readout_k=run_cfg.get("readout_k", 9),
    )
    model.set_centroids(torch.zeros(num_bays, 2))
    model.load_state_dict(state)
    model.eval().to(args.device)

    logits_all = []
    with torch.no_grad():
        for i in range(0, Xte.shape[0], args.batch):
            logits_all.append(model.forward_logits(Xte[i:i + args.batch].to(args.device)).cpu())
    logits = torch.cat(logits_all)
    cents = model.centroids.cpu()

    print(f"\n{'k':>5}  {'MPE(m)':>8}  {'median(m)':>9}  " +
          "  ".join(f"CDF@{int(t)}m" for t in CDF_THRESHOLDS))
    for k in args.k:
        if k == 0:
            w_ = torch.softmax(logits, dim=1)
            pred = w_ @ cents
            tag = "full"
        else:
            pred = topk_soft_argmax(logits, cents, k)
            tag = str(k)
        err = ((pred - yte) * scale).norm(dim=1)
        cdf = "  ".join(f"{(err < t).float().mean().item() * 100:6.1f}%" for t in CDF_THRESHOLDS)
        print(f"{tag:>5}  {err.mean().item():>8.2f}  {err.median().item():>9.2f}  {cdf}")


if __name__ == "__main__":
    main()
