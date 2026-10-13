"""WKNN baseline on the same data pipeline as the neural models.

Usage:
    python scripts/wknn_baseline.py [--max-samples 100000] [--sampling random|balanced]
                                    [--reps 10] [--k 1 3 5 7 9]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import WiFiFingerprintDataModule

CDF_THRESHOLDS = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]


def build_module(cfg, max_samples, sampling, reps, split="random"):
    d = cfg["data"]
    return WiFiFingerprintDataModule(
        db_path=d["db_path"],
        train_split=d["train_split"],
        val_split=d["val_split"],
        test_split=d["test_split"],
        random_seed=d["random_seed"],
        max_samples=max_samples,
        storage_place_filter=d.get("storage_place_filter"),
        deduplicate=d.get("deduplicate", False),
        sampling=sampling,
        reps_per_place=reps,
        split_strategy=split,
        max_aps=d["max_aps"],
        missing_value=d["missing_value"],
        ssid_filter=d.get("filter_ssid_prefix"),
        target_scale=tuple(d["target_scale"]) if d.get("target_scale") else None,
        rssi_normalization=d.get("rssi_normalization", "minmax"),
        rssi_min=d.get("rssi_min", -100.0),
        rssi_max=d.get("rssi_max", 0.0),
    )


def encode_sae(ckpt, Xtr, Xte):
    """Encode RSSI vectors with a trained SAE encoder."""
    from src.models.sae import ReferenceSAE
    m = ReferenceSAE.load_from_checkpoint(ckpt, map_location="cpu").eval()
    with torch.no_grad():
        Ztr = m.encoder(Xtr)
        Zte = m.encoder(Xte)
    # latent std is ~1e-4 per dim, standardize or distances are noise
    mu, sd = Ztr.mean(0, keepdim=True), Ztr.std(0, keepdim=True) + 1e-8
    Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    print(f"SAE-encoded + standardized: {Xtr.shape[1]}d raw -> {Ztr.shape[1]}d latent")
    return Ztr, Zte


def wknn(Xtr, ytr, Xte, yte, scale, ks, device, batch=512):
    """Return {k: (mean, median, cdf, pred)}, errors in metres."""
    Xtr, ytr, Xte, yte, scale = (t.to(device) for t in (Xtr, ytr, Xte, yte, scale))
    kmax = max(ks)
    all_idx, all_d = [], []
    for i in range(0, Xte.shape[0], batch):
        d = torch.cdist(Xte[i : i + batch], Xtr)
        dk, idx = d.topk(kmax, dim=1, largest=False)
        all_d.append(dk)
        all_idx.append(idx)
    all_d = torch.cat(all_d)
    all_idx = torch.cat(all_idx)

    out = {}
    for k in ks:
        dk = all_d[:, :k]
        idx = all_idx[:, :k]
        w = 1.0 / (dk + 1e-8)
        w = w / w.sum(dim=1, keepdim=True)
        nbr = ytr[idx]
        pred = (w.unsqueeze(-1) * nbr).sum(dim=1)
        err = ((pred - yte) * scale).norm(dim=1)
        cdf = {t: (err < t).float().mean().item() * 100 for t in CDF_THRESHOLDS}
        out[k] = (err.mean().item(), err.median().item(), cdf, (pred * scale).cpu())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-samples", type=int, default=100000)
    ap.add_argument("--sampling", default="random", choices=["random", "balanced"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--split", default="random", choices=["random", "group", "temporal"],
                    help="group = hold out whole bays; temporal = train past, test future")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--encoder-ckpt", default=None,
                    help="SAE checkpoint: WKNN in its denoised latent instead of raw RSSI")
    ap.add_argument("--save-predictions", default=None, metavar="PATH.npz",
                    help="save per-sample test predictions (metres) for offline analysis")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    print(f"WKNN baseline | sampling={args.sampling} max_samples={args.max_samples} "
          f"reps={args.reps} split={args.split} device={args.device}")

    t = time.time()
    dm = build_module(cfg, args.max_samples, args.sampling, args.reps, args.split)
    ds_tr, ds_te = dm.get_dataset("train"), dm.get_dataset("test")
    Xtr = torch.from_numpy(ds_tr._features)
    ytr = torch.from_numpy(ds_tr._targets)
    Xte = torch.from_numpy(ds_te._features)
    yte = torch.from_numpy(ds_te._targets)
    scale = torch.tensor(cfg["data"]["target_scale"], dtype=torch.float32)
    print(f"reference={Xtr.shape[0]:,}  query={Xte.shape[0]:,}  dim={Xtr.shape[1]}  "
          f"(loaded {time.time()-t:.0f}s)")

    if args.encoder_ckpt:
        Xtr, Xte = encode_sae(args.encoder_ckpt, Xtr, Xte)

    res = wknn(Xtr, ytr, Xte, yte, scale, args.k, args.device)
    print(f"\n{'k':>3}  {'MPE(m)':>8}  {'median(m)':>9}  " +
          "  ".join(f'CDF@{int(t)}m' for t in CDF_THRESHOLDS))
    for k in args.k:
        m, md, cdf, _ = res[k]
        print(f"{k:>3}  {m:>8.2f}  {md:>9.2f}  " +
              "  ".join(f'{cdf[t]:6.1f}%' for t in CDF_THRESHOLDS))
    best_k = min(res, key=lambda k: res[k][0])
    print(f"\nbest k={best_k}: MPE {res[best_k][0]:.2f}m / median {res[best_k][1]:.2f}m")

    if args.save_predictions:
        arrays = {
            "id": np.asarray(ds_te._ids, dtype=np.int64),
            "true": (yte * scale).numpy().astype(np.float32),
        }
        for k in args.k:
            arrays[f"pred_k{k}"] = res[k][3].numpy().astype(np.float32)
        arrays["pred"] = res[best_k][3].numpy().astype(np.float32)
        np.savez(args.save_predictions, **arrays)
        print(f"Saved test predictions (metres) to {args.save_predictions} "
              f"(pred = best k={best_k})")


if __name__ == "__main__":
    main()
