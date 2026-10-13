"""Drop train points with the largest leave-one-out WKNN residual, re-evaluate WKNN on test.

Usage: python scripts/wknn_clean.py [--max-samples 100000]
"""

import argparse
import os
import sys
import time

import torch
import yaml

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from wknn_baseline import build_module, wknn  # noqa: E402

FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


def loo_residuals(X, y, scale, k, batch=512):
    """Leave-one-out WKNN residual in metres."""
    N = X.shape[0]
    res = torch.empty(N)
    for i in range(0, N, batch):
        b = min(batch, N - i)
        d = torch.cdist(X[i : i + b], X)
        d[torch.arange(b), torch.arange(i, i + b)] = float("inf")  # exclude self
        dk, idx = d.topk(k, dim=1, largest=False)
        w = 1.0 / (dk + 1e-8)
        w = w / w.sum(1, keepdim=True)
        pred = (w.unsqueeze(-1) * y[idx]).sum(1)
        res[i : i + b] = ((pred - y[i : i + b]) * scale).norm(dim=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-samples", type=int, default=100000)
    ap.add_argument("--sampling", default="random")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--k", type=int, default=9)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    t = time.time()
    dm = build_module(cfg, args.max_samples, args.sampling, args.reps)
    ds_tr, ds_te = dm.get_dataset("train"), dm.get_dataset("test")
    Xtr = torch.from_numpy(ds_tr._features); ytr = torch.from_numpy(ds_tr._targets)
    Xte = torch.from_numpy(ds_te._features); yte = torch.from_numpy(ds_te._targets)
    scale = torch.tensor(cfg["data"]["target_scale"], dtype=torch.float32)
    print(f"train(ref)={Xtr.shape[0]:,}  test={Xte.shape[0]:,}  (loaded {time.time()-t:.0f}s)")

    print("Computing leave-one-out residuals on train ...")
    res = loo_residuals(Xtr, ytr, scale, args.k)
    q = lambda p: res.quantile(p).item()
    print(f"residual (m): median {q(.5):.2f}  p80 {q(.8):.2f}  p90 {q(.9):.2f}  "
          f"p95 {q(.95):.2f}  p99 {q(.99):.2f}  max {res.max():.1f}")
    print(f"  share with residual >10m: {(res>10).float().mean()*100:.1f}%   "
          f">20m: {(res>20).float().mean()*100:.1f}%")

    print(f"\n{'drop%':>6} {'ref_n':>7} {'test_MPE':>9} {'test_med':>9}")
    for frac in FRACTIONS:
        thr = res.quantile(1 - frac) if frac > 0 else float("inf")
        keep = res <= thr
        Xref, yref = Xtr[keep], ytr[keep]
        m, md, _, _ = wknn(Xref, yref, Xte, yte, scale, [args.k], "cpu")[args.k]
        print(f"{frac*100:>5.0f}% {int(keep.sum()):>7,} {m:>9.2f} {md:>9.2f}")


if __name__ == "__main__":
    main()
