#!/usr/bin/env python3
"""Simulate AP outages at test time (zero visible APs) for WKNN and cls checkpoints.

Usage:
    python scripts/eval_ap_dropout.py --model wknn --run resnet50_cls100k
    python scripts/eval_ap_dropout.py --model resnet50_cls \
        --ckpt experiments/resnet50_cls100k --run resnet50_cls100k
    python scripts/eval_ap_dropout.py --model transformer_cls \
        --ckpt experiments/transformer_cls_100k --run transformer_cls_100k \
        --drop-frac 0.0 0.1 0.2 0.3 0.5 --drop-strongest 1 2
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
from src.models.cls_variants import ResNetClsWiFiModule, TransformerClsWiFiModule
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


def drop_visible_fraction(X: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Zero a random fraction of the visible APs per row."""
    if frac <= 0:
        return X
    rng = np.random.RandomState(seed)
    X = X.copy()
    for i in range(X.shape[0]):
        vis = np.flatnonzero(X[i] > 0)
        n_drop = int(round(frac * len(vis)))
        if n_drop:
            X[i, rng.choice(vis, size=n_drop, replace=False)] = 0.0
    return X


def drop_strongest(X: np.ndarray, k: int) -> np.ndarray:
    """Zero the k strongest APs per row."""
    if k <= 0:
        return X
    X = X.copy()
    for i in range(X.shape[0]):
        vis = np.flatnonzero(X[i] > 0)
        if len(vis):
            top = vis[np.argsort(X[i, vis])[-min(k, len(vis)):]]
            X[i, top] = 0.0
    return X


def to_sparse_batch(X: np.ndarray) -> dict:
    """Same as sparse_transformer_collate_fn, for a raw feature matrix."""
    idxs = [np.flatnonzero(row) for row in X]
    max_len = max(1, max(len(i) for i in idxs))
    b = X.shape[0]
    ap_indices = torch.zeros(b, max_len, dtype=torch.long)
    rssi_values = torch.zeros(b, max_len)
    padding_mask = torch.ones(b, max_len, dtype=torch.bool)
    for i, idx in enumerate(idxs):
        n = len(idx)
        if n:
            ap_indices[i, :n] = torch.from_numpy(idx)
            rssi_values[i, :n] = torch.from_numpy(X[i, idx])
            padding_mask[i, :n] = False
    return {"ap_indices": ap_indices, "rssi_values": rssi_values,
            "padding_mask": padding_mask}


def wknn_predict(Xtr, ytr, Xte, k, device, batch=256):
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    preds = []
    for i in range(0, Xte.shape[0], batch):
        q = torch.from_numpy(Xte[i:i + batch]).to(device)
        d = torch.cdist(q, Xtr_t)
        dk, idx = d.topk(k, largest=False)
        w = 1.0 / (dk + 1e-8)
        w = w / w.sum(dim=1, keepdim=True)
        preds.append((w.unsqueeze(-1) * ytr_t[idx]).sum(dim=1).cpu())
    return torch.cat(preds)


def cls_predict(model, Xte, shape, k, device, batch=512, sparse=False):
    logits_all = []
    with torch.no_grad():
        for i in range(0, Xte.shape[0], batch):
            if sparse:
                x = {kk: v.to(device)
                     for kk, v in to_sparse_batch(Xte[i:i + batch]).items()}
            else:
                x = torch.from_numpy(Xte[i:i + batch])
                x = (x.view(-1, *shape) if shape else x).to(device)
            logits_all.append(model.forward_logits(x).cpu())
    logits = torch.cat(logits_all)
    return topk_soft_argmax(logits, model.centroids.cpu(), k)


def report(tag, pred, yte, scale, n_vis):
    err = ((pred - yte) * scale).norm(dim=1)
    cdf = "  ".join(f"{(err < t).float().mean().item() * 100:6.1f}%" for t in CDF_THRESHOLDS)
    print(f"{tag:>14}  {n_vis:>7.2f}  {err.mean().item():>8.2f}  "
          f"{err.median().item():>9.2f}  {cdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["wknn", "resnet50_cls", "transformer_cls"])
    ap.add_argument("--ckpt", help=".ckpt file or experiments dir (cls models)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", default="resnet50_cls100k",
                    help="run name whose data config defines the regime")
    ap.add_argument("--k", type=int, default=None,
                    help="WKNN k or cls readout k (default: 9 / trained readout_k)")
    ap.add_argument("--drop-frac", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--drop-strongest", type=int, nargs="+", default=[],
                    help="additionally: zero the k strongest APs per scan")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run_cfg = cfg.get("runs", {}).get(args.run, {})
    dm = build_module(cfg, run_cfg)
    ds_te = dm.get_dataset("test")
    Xte, yte = ds_te._features, torch.from_numpy(ds_te._targets)
    scale = torch.tensor(cfg["data"]["target_scale"], dtype=torch.float32)

    if args.model == "wknn":
        ds_tr = dm.get_dataset("train")
        Xtr, ytr = ds_tr._features, ds_tr._targets
        k = args.k or 9
        predict = lambda X: wknn_predict(Xtr, ytr, X, k, args.device)
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt required for cls models")
        ckpt_path = resolve_ckpt(args.ckpt)
        print(f"checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        num_bays = state["centroids"].shape[0]
        sigma = run_cfg.get("label_sigma_m", 2.5)
        k = args.k or run_cfg.get("readout_k", 9)
        if args.model == "resnet50_cls":
            h, w, c = run_cfg.get("image_shape", cfg["data"]["image_shape"])
            model = ResNetClsWiFiModule(
                variant=run_cfg.get("variant", "resnet50"), pretrained=False,
                image_shape=(h, w, c), target_scale=list(scale.numpy()),
                label_sigma_m=sigma, readout_k=k)
            shape = (h, w, c)
        else:
            model = TransformerClsWiFiModule(
                num_aps=cfg["data"]["max_aps"],
                d_model=run_cfg.get("d_model", 128),
                nhead=run_cfg.get("nhead", 8),
                num_encoder_layers=run_cfg.get("num_encoder_layers", 4),
                dim_feedforward=run_cfg.get("dim_feedforward", 512),
                dropout=run_cfg.get("dropout", 0.1),
                activation=run_cfg.get("activation", "relu"),
                missing_value=0.0,  # trainer uses 0.0 after normalisation
                masking_strategy=run_cfg.get("masking_strategy", "additive"),
                pooling=run_cfg.get("pooling", "cls"),
                variable_length=run_cfg.get("variable_length", True),
                spatial_positional_encoding=run_cfg.get(
                    "spatial_positional_encoding", False),
                target_scale=list(scale.numpy()),
                label_sigma_m=sigma, readout_k=k)
            shape = None
        model.set_centroids(torch.zeros(num_bays, 2))
        model.load_state_dict(state)
        model.eval().to(args.device)
        # trained on the sparse path, the dense path would shift the PE
        sparse = args.model == "transformer_cls"
        predict = lambda X: cls_predict(model, X, shape, k, args.device,
                                        sparse=sparse)

    print(f"\nmodel={args.model}  k={k}  test={Xte.shape[0]:,}  "
          f"regime=run:{args.run}")
    print(f"{'condition':>14}  {'#vis':>7}  {'MPE(m)':>8}  {'median(m)':>9}  "
          + "  ".join(f"CDF@{int(t)}m" for t in CDF_THRESHOLDS))
    for frac in args.drop_frac:
        Xd = drop_visible_fraction(Xte, frac, args.seed)
        report(f"drop {int(frac * 100)}%", predict(Xd), yte, scale,
               (Xd > 0).sum(axis=1).mean())
    for ks in args.drop_strongest:
        Xd = drop_strongest(Xte, ks)
        report(f"top-{ks} out", predict(Xd), yte, scale,
               (Xd > 0).sum(axis=1).mean())


if __name__ == "__main__":
    main()
