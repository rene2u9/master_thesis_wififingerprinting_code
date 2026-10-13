#!/usr/bin/env python3
"""Trajectory experiments of regime R3: per-scan classification and HMM decoding.

Usage:
    python scripts/sub1m/lab.py prep              # build the tensor cache
    python scripts/sub1m/lab.py run --epochs 60   # train, then decode
    python scripts/sub1m/lab.py decode | aisle | reject
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.heads.cls import build_classifier, soft_ce, topk_soft_argmax

DB = "data/wifi_fingerprints_sub1m.db"
CACHE = "data/sub1m_cache.pt"
FULL_DB = "data/wifi_fingerprints_full_ps.db"
FULL_CACHE = "data/sub1m_cache_full.pt"


def _paths(args):
    sfx = "_gcn" if getattr(args, "model", "mlp") == "gcn" else ""
    if getattr(args, "full", False):
        return FULL_CACHE, f"data/sub1m_logits_full{sfx}.pt"
    return CACHE, f"data/sub1m_logits{sfx}.pt"


SSID_PREFIX = os.environ.get("SSID_PREFIX", "")
MAX_APS = 256
RSSI_MIN, RSSI_MAX, MISSING = -90.0, -50.0, -100.0


def _row_query(full):
    # time-ordered so keep-first dedup keeps the earliest submission
    where = "" if full else "AND start_datetime IS NOT NULL AND reporter IS NOT NULL"
    return f"""SELECT reporter, COALESCE(start_datetime, created_at) AS tkey,
                      (start_datetime IS NOT NULL) AS has_ts,
                      storage_place_desc, latitude, longitude, readings
               FROM wifi_fingerprints
               WHERE storage_place_desc IS NOT NULL {where}
               ORDER BY tkey"""


def prep(full=False):
    import hashlib
    from collections import Counter

    db, cache = (FULL_DB, FULL_CACHE) if full else (DB, CACHE)
    c = sqlite3.connect(db)

    def ap_key(bssid):
        return ":".join(bssid.split(":")[:5])

    def scan_hash(readings):
        return hashlib.md5("|".join(
            sorted(f"{x['bssid']}:{x['level']}" for x in readings)).encode()).digest()

    # pass 1: drop repeated scans (keep the earliest), count APs by 5-octet prefix
    seen = set()
    keep_flags = []
    ap_count = Counter()
    total = 0
    for rep, tkey, has_ts, desc, lat, lon, rj in c.execute(_row_query(full)):
        total += 1
        readings = json.loads(rj)
        h = scan_hash(readings)
        if h in seen:
            keep_flags.append(False)
            continue
        seen.add(h)
        keep_flags.append(True)
        for r in readings:
            if r.get("ssid", "").startswith(SSID_PREFIX) and r.get("bssid"):
                ap_count[ap_key(r["bssid"])] += 1
    n = sum(keep_flags)
    print(f"rows: {total:,}, after dedup: {n:,}")
    aps = [a for a, _ in ap_count.most_common(MAX_APS)]
    idx = {a: i for i, a in enumerate(aps)}
    print(f"access points: {len(aps)} of {len(ap_count)}")
    del seen

    # pass 2: fill arrays
    X = np.zeros((n, MAX_APS), dtype=np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    rep_ids = np.zeros(n, dtype=np.int32)
    tstamp = np.zeros(n, dtype=np.float64)
    has_ts_arr = np.zeros(n, dtype=bool)
    bay_key = []
    reps = {}
    i = 0
    for k, (rep, tkey, has_ts, desc, lat, lon, rj) in zip(
            keep_flags, c.execute(_row_query(full))):
        if not k:
            continue
        strongest = {}
        for r in json.loads(rj):
            if not (r.get("ssid", "").startswith(SSID_PREFIX) and r.get("bssid")):
                continue
            j = idx.get(ap_key(r["bssid"]))
            if j is not None and r["level"] > strongest.get(j, -1000):
                strongest[j] = r["level"]
        for j, lv in strongest.items():
            X[i, j] = (max(min(lv, RSSI_MAX), RSSI_MIN) - RSSI_MIN) / (RSSI_MAX - RSSI_MIN)
        y[i] = (lat, lon)
        rep_ids[i] = reps.setdefault(rep or "?", len(reps))
        tstamp[i] = datetime.fromisoformat(tkey).timestamp()
        has_ts_arr[i] = bool(has_ts)
        bay_key.append(";".join(desc.split(";")[:3]))
        i += 1

    torch.save({
        "X": torch.from_numpy(X), "y": torch.from_numpy(y),
        "rep": torch.from_numpy(rep_ids), "t": torch.from_numpy(tstamp),
        "has_ts": torch.from_numpy(has_ts_arr),
        "bay": bay_key, "aps": aps,
    }, cache)
    print(f"cached -> {cache}")


class ClsMLP(nn.Module):
    def __init__(self, in_dim, hidden, centroids, sigma_m, scale):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.register_buffer("centroids", centroids)      # metres
        self.register_buffer("scale", scale)
        self.classifier = build_classifier(hidden, 512, centroids.shape[0])
        self.sigma_m = sigma_m

    def logits(self, x):
        return self.classifier(self.trunk(x))

    def loss(self, x, targets):
        return soft_ce(self.logits(x), targets, self.centroids, self.scale, self.sigma_m)


class ClsGCN(nn.Module):
    def __init__(self, num_aps, centroids, sigma_m, scale,
                 hidden=128, num_layers=3, dropout=0.3, node_proj_dim=8):
        super().__init__()
        from src.models.gcn import SAGEEncoder
        self.encoder = SAGEEncoder(1, hidden, num_layers, dropout)
        self.node_proj = nn.Linear(hidden, node_proj_dim)
        self.num_aps = num_aps
        self.register_buffer("centroids", centroids)
        self.register_buffer("scale", scale)
        self.classifier = build_classifier(
            num_aps * node_proj_dim, 512, centroids.shape[0])
        self.sigma_m = sigma_m

    def logits(self, batch):
        ew = batch.edge_attr.squeeze(-1) if batch.edge_attr is not None else None
        x = self.encoder(batch.x, batch.edge_index, ew)
        x = self.node_proj(x)
        b = int(batch.batch.max()) + 1
        x = x.view(b, self.num_aps, -1).reshape(b, -1)
        return self.classifier(x)

    def loss(self, batch, targets):
        return soft_ce(self.logits(batch), targets, self.centroids, self.scale, self.sigma_m)


class RegMLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.head(self.trunk(x))


class RegGCN(nn.Module):
    def __init__(self, num_aps, hidden=128, num_layers=3, dropout=0.3,
                 node_proj_dim=8):
        super().__init__()
        from src.models.gcn import SAGEEncoder
        self.encoder = SAGEEncoder(1, hidden, num_layers, dropout)
        self.node_proj = nn.Linear(hidden, node_proj_dim)
        self.num_aps = num_aps
        self.head = nn.Sequential(
            nn.Linear(num_aps * node_proj_dim, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, batch):
        ew = batch.edge_attr.squeeze(-1) if batch.edge_attr is not None else None
        x = self.encoder(batch.x, batch.edge_index, ew)
        x = self.node_proj(x)
        b = int(batch.batch.max()) + 1
        x = x.view(b, self.num_aps, -1).reshape(b, -1)
        return self.head(x)


def run_reg(args, X, y, tr, te, dev):
    mu, sd = y[tr].mean(0), y[tr].std(0).clamp(min=1e-6)
    yn = (y - mu) / sd
    torch.manual_seed(args.seed)
    if getattr(args, "model", "mlp") == "gcn":
        from src.data.graph_transforms import RSSIToGraph
        from torch_geometric.loader import DataLoader as GeoLoader
        to_graph = RSSIToGraph(
            num_aps=X.shape[1], missing_value=MISSING, rssi_threshold=-95.0,
            graph_topology="covisible", rssi_normalization="minmax",
            rssi_min=RSSI_MIN, rssi_max=RSSI_MAX,
        )

        def graphs(idx):
            out = []
            for i in idx.tolist():
                g = to_graph(X[i])
                g.y = yn[i].unsqueeze(0)
                out.append(g)
            return out

        print("building graphs ...", flush=True)
        gtr, gte = graphs(tr), graphs(te)
        m = RegGCN(X.shape[1]).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        loader = GeoLoader(gtr, batch_size=args.batch, shuffle=True)
        m.train()
        for ep in range(args.epochs):
            tot, nb = 0.0, 0
            for batch in loader:
                batch = batch.to(dev)
                loss = nn.functional.mse_loss(m(batch), batch.y)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"epoch {ep:3d}  train MSE {tot / nb:.4f}", flush=True)
        m.eval()
        with torch.no_grad():
            pred = torch.cat([m(batch.to(dev)).cpu()
                              for batch in GeoLoader(gte, batch_size=1024)])
    else:
        m = RegMLP(X.shape[1], args.hidden).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        Xtr, ytr = X[tr].to(dev), yn[tr].to(dev)
        m.train()
        steps = max(len(tr) // args.batch, 1)
        for ep in range(args.epochs):
            perm = torch.randperm(len(tr), device=dev)
            tot = 0.0
            for s in range(steps):
                i = perm[s * args.batch:(s + 1) * args.batch]
                loss = nn.functional.mse_loss(m(Xtr[i]), ytr[i])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"epoch {ep:3d}  train MSE {tot / steps:.4f}", flush=True)
        m.eval()
        with torch.no_grad():
            pred = torch.cat([m(X[te][i:i + 2048].to(dev)).cpu()
                              for i in range(0, len(te), 2048)])
    pred = pred * sd + mu
    print("\nper-scan regression")
    report(f"{getattr(args, 'model', 'mlp')}-regression", (pred - y[te]).norm(dim=1))


def _trans_logp(d, dt, args, pair_counts=None):
    """Transition log-probability for distance d (m) over dt (s)."""
    # the broad component keeps fast moves, e.g. on a cart, possible
    sig_l = min(args.base_sigma + args.speed * dt, args.sigma_cap)
    local = -(d ** 2) / (2 * sig_l ** 2)
    broad = -(d ** 2) / (2 * args.sigma_broad ** 2)
    kernel = torch.logaddexp(np.log(1 - args.eps) + local, np.log(args.eps) + broad)
    if pair_counts is None:
        return kernel
    # rows: current candidates, columns: previous candidates
    ker_p = torch.softmax(kernel, dim=0)
    big_p = pair_counts / pair_counts.sum(dim=0, keepdim=True).clamp(min=1e-9)
    lam = torch.where(pair_counts.sum(dim=0) > 0,
                      torch.tensor(args.route_lambda), torch.tensor(0.0))
    return torch.log(lam * big_p + (1 - lam) * ker_p + 1e-12)


def _pair_counts(cand_now, cand_prev, adj):
    """Transition counts between candidate bays, shape [M_now, M_prev]."""
    M_now, M_prev = len(cand_now), len(cand_prev)
    pos = {int(b): j for j, b in enumerate(cand_now)}
    out = torch.zeros(M_now, M_prev)
    for i, b_prev in enumerate(cand_prev.tolist()):
        row = adj.get(b_prev)
        if row:
            for b_now, cnt in row.items():
                j = pos.get(b_now)
                if j is not None:
                    out[j, i] = cnt
    return out


def hmm_decode(logp, cand, cents, ts, args, adj=None):
    """Viterbi decoding over the top-M candidate bays (offline)."""
    T, M = logp.shape
    dp = logp[0].clone()
    bp = torch.zeros(T, M, dtype=torch.long)
    for t in range(1, T):
        dt = max(min(ts[t] - ts[t - 1], 300.0), 1.0)
        d = torch.cdist(cents[cand[t]], cents[cand[t - 1]])
        pc = _pair_counts(cand[t], cand[t - 1], adj) if adj else None
        scores = dp.unsqueeze(0) + _trans_logp(d, dt, args, pc)
        best = scores.max(dim=1)
        dp = best.values + logp[t]
        bp[t] = best.indices
    path = torch.zeros(T, dtype=torch.long)
    path[-1] = dp.argmax()
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return cents[cand[torch.arange(T), path]]


def filter_decode(logp, cand, cents, ts, args, adj=None):
    """Causal forward filter, returns the posterior mean per scan."""
    T, M = logp.shape
    out = torch.zeros(T, 2)
    alpha = logp[0].clone()
    out[0] = (torch.softmax(alpha, 0).unsqueeze(1) * cents[cand[0]]).sum(0)
    for t in range(1, T):
        dt = max(min(ts[t] - ts[t - 1], 300.0), 1.0)
        d = torch.cdist(cents[cand[t]], cents[cand[t - 1]])
        pc = _pair_counts(cand[t], cand[t - 1], adj) if adj else None
        alpha = torch.logsumexp(alpha.unsqueeze(0) + _trans_logp(d, dt, args, pc), dim=1) + logp[t]
        alpha -= alpha.max()
        out[t] = (torch.softmax(alpha, 0).unsqueeze(1) * cents[cand[t]]).sum(0)
    return out


def reject_analysis(args):
    """Error against coverage when rejecting scans by posterior spread."""
    cache_path, logits_path = _paths(args)
    d = torch.load(cache_path, weights_only=False)
    s = torch.load(logits_path, weights_only=False)
    logits, te, cents = s["logits"], s["te"], s["cents"]
    y, rep, ts = d["y"], d["rep"], d["t"]
    yte = y[te]

    logprob = torch.log_softmax(logits, dim=1)
    top = logprob.topk(args.topm, dim=1)
    rep_te, ts_te = rep[te], ts[te]

    errs, confs = [], []
    for r in rep_te.unique():
        mask = (rep_te == r).nonzero().squeeze(1)
        if len(mask) < 3:
            continue
        o = mask[torch.argsort(ts_te[mask])]
        lp, cd, t_ = top.values[o], top.indices[o], ts_te[o].numpy()
        T, M = lp.shape
        alpha = lp[0].clone()
        for t in range(T):
            if t > 0:
                dt = max(min(t_[t] - t_[t - 1], 300.0), 1.0)
                dd = torch.cdist(cents[cd[t]], cents[cd[t - 1]])
                alpha = torch.logsumexp(alpha.unsqueeze(0) + _trans_logp(dd, dt, args), dim=1) + lp[t]
                alpha -= alpha.max()
            w = torch.softmax(alpha, 0)
            pred = (w.unsqueeze(1) * cents[cd[t]]).sum(0)
            spread = torch.sqrt((w * (cents[cd[t]] - pred).pow(2).sum(1)).sum() + 1e-9)
            errs.append(float((pred - yte[o[t]]).norm()))
            confs.append(float(spread))
    errs, confs = np.array(errs), np.array(confs)
    order = np.argsort(confs)
    print(f"\nreject option, n={len(errs):,}")
    print(f"  {'keep':>5}  {'mean':>7}  {'median':>7}  {'p90':>7}  {'CDF@1m':>7}  {'thresh(m)':>9}")
    for frac in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5):
        k = max(int(len(errs) * frac), 1)
        e = errs[order[:k]]
        thr = confs[order[k - 1]]
        print(f"  {int(frac*100):>4}%  {e.mean():>7.2f}  {np.median(e):>7.2f}  "
              f"{np.percentile(e,90):>7.2f}  {100*(e<1).mean():>6.1f}%  {thr:>9.2f}")


def aisle_decode(args):
    """Viterbi over aisles, then weighted readout within the decoded aisle."""
    cache_path, logits_path = _paths(args)
    d = torch.load(cache_path, weights_only=False)
    s = torch.load(logits_path, weights_only=False)
    logits, te, cents = s["logits"], s["te"], s["cents"]
    y, rep, ts, bays = d["y"], d["rep"], d["t"], d["bay"]
    yte = y[te]

    coord2aisle = {}
    for i in range(len(y)):
        coord2aisle[(round(float(y[i, 0]), 3), round(float(y[i, 1]), 3))] = \
            ";".join(bays[i].split(";")[:2])
    aisle_names = sorted({a for a in coord2aisle.values()})
    a_idx = {a: i for i, a in enumerate(aisle_names)}
    A = len(aisle_names)
    cent_aisle = torch.tensor([a_idx[coord2aisle[(round(float(c[0]), 3), round(float(c[1]), 3))]]
                               for c in cents])
    print(f"aisles: {A}")

    n = len(y)
    te_mask = torch.zeros(n, dtype=torch.bool); te_mask[te] = True
    trans = torch.full((A, A), 0.5)
    order = torch.argsort(ts); prev = {}
    for i in order.tolist():
        if te_mask[i]:
            continue
        a = a_idx[";".join(bays[i].split(";")[:2])]
        r = int(rep[i])
        if r in prev and float(ts[i]) - prev[r][1] <= 300.0:
            trans[prev[r][0], a] += 1.0
        prev[r] = (a, float(ts[i]))
    logT = torch.log(trans / trans.sum(dim=1, keepdim=True)).T

    # aisle score: best bay log-probability within the aisle
    logprob = torch.log_softmax(logits, dim=1)
    aisle_lp = torch.full((len(te), A), -60.0)
    aisle_lp.scatter_reduce_(1, cent_aisle.unsqueeze(0).expand(len(te), -1),
                             logprob, reduce="amax")
    probs = torch.softmax(logits, dim=1)

    rep_te, ts_te = rep[te], ts[te]
    errs = []
    for r in rep_te.unique():
        mask = (rep_te == r).nonzero().squeeze(1)
        if len(mask) < 3:
            continue
        o = mask[torch.argsort(ts_te[mask])]
        lp = aisle_lp[o]
        T_ = len(o)
        dp = lp[0].clone(); bp = torch.zeros(T_, A, dtype=torch.long)
        for t in range(1, T_):
            sc = dp.unsqueeze(0) + logT
            best = sc.max(dim=1)
            dp = best.values + lp[t]; bp[t] = best.indices
        path = torch.zeros(T_, dtype=torch.long); path[-1] = dp.argmax()
        for t in range(T_ - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]
        for t in range(T_):
            m_ = cent_aisle == path[t]
            w = probs[o[t]] * m_
            tot = w.sum()
            pred = (w.unsqueeze(1) * cents).sum(0) / tot if tot > 1e-9 \
                else cents[probs[o[t]].argmax()]
            errs.append(float((pred - yte[o[t]]).norm()))
    e = np.array(errs)
    print(f"  aisle HMM                   mean {e.mean():6.2f}  median {np.median(e):6.2f}  "
          f"p90 {np.percentile(e, 90):6.2f}  CDF@1m {100 * (e < 1).mean():5.1f}%  "
          f"@2m {100 * (e < 2).mean():5.1f}%  @5m {100 * (e < 5).mean():5.1f}%")


def report(name, err):
    e = err.numpy()
    print(f"  {name:24s} mean {e.mean():6.2f}  median {np.median(e):6.2f}  "
          f"p90 {np.percentile(e, 90):6.2f}  CDF@1m {100 * (e < 1).mean():5.1f}%  "
          f"@2m {100 * (e < 2).mean():5.1f}%  @5m {100 * (e < 5).mean():5.1f}%")


def run(args):
    cache_path, logits_path = _paths(args)
    d = torch.load(cache_path, weights_only=False)
    X, y, rep, ts = d["X"], d["y"], d["rep"], d["t"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # with --full, only timestamped scans are used for testing
    order = torch.argsort(ts)
    if getattr(args, "full", False):
        ts_rows = d["has_ts"]
        t_ts = ts[ts_rows]
        boundary = t_ts.sort().values[int(len(t_ts) * 0.85)]
        tr = (ts < boundary).nonzero().squeeze(1)
        te = (ts_rows & (ts >= boundary)).nonzero().squeeze(1)
    else:
        n = len(order)
        tr = order[: int(n * 0.85)]
        te = order[int(n * 0.85):]
    print(f"temporal split: train {len(tr):,} test {len(te):,} "
          f"(boundary {datetime.fromtimestamp(float(ts[tr[-1]])):%Y-%m-%d %H:%M})")

    if getattr(args, "head", "cls") == "reg":
        return run_reg(args, X, y, tr, te, dev)

    cents = torch.from_numpy(np.unique(y[tr].numpy(), axis=0))
    print(f"centroid codebook: {cents.shape[0]:,} bays (train split)")
    scale = torch.ones(2)

    torch.manual_seed(args.seed)
    if getattr(args, "model", "mlp") == "gcn":
        from src.data.graph_transforms import RSSIToGraph
        from torch_geometric.loader import DataLoader as GeoLoader
        if args.ap_dropout > 0 or args.jitter > 0:
            raise SystemExit("--model gcn does not support augmentation flags")
        to_graph = RSSIToGraph(
            num_aps=X.shape[1], missing_value=MISSING, rssi_threshold=-95.0,
            graph_topology="covisible", rssi_normalization="minmax",
            rssi_min=RSSI_MIN, rssi_max=RSSI_MAX,
        )

        def graphs(idx):
            out = []
            for i in idx.tolist():
                g = to_graph(X[i])
                g.y = y[i].unsqueeze(0)
                out.append(g)
            return out

        print("building graphs ...", flush=True)
        gtr, gte = graphs(tr), graphs(te)
        m = ClsGCN(X.shape[1], cents, args.sigma, scale).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        loader = GeoLoader(gtr, batch_size=args.batch, shuffle=True)
        m.train()
        for ep in range(args.epochs):
            tot, nb = 0.0, 0
            for batch in loader:
                batch = batch.to(dev)
                loss = m.loss(batch, batch.y)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"epoch {ep:3d}  train soft-CE {tot / nb:.3f}", flush=True)
        m.eval()
        with torch.no_grad():
            logits = torch.cat([m.logits(batch.to(dev)).cpu()
                                for batch in GeoLoader(gte, batch_size=1024)])
    else:
        m = ClsMLP(X.shape[1], args.hidden, cents, args.sigma, scale).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        Xtr, ytr = X[tr].to(dev), y[tr].to(dev)
        m.train()
        steps_per_epoch = max(len(tr) // args.batch, 1)
        for ep in range(args.epochs):
            perm = torch.randperm(len(tr), device=dev)
            tot = 0.0
            for s in range(steps_per_epoch):
                i = perm[s * args.batch:(s + 1) * args.batch]
                xb = Xtr[i]
                # augmentation touches visible APs only, the minmax window is 40 dB
                if args.ap_dropout > 0 or args.jitter > 0:
                    xb = xb.clone()
                    vis = xb > 0
                    if args.ap_dropout > 0:
                        drop = (torch.rand_like(xb) < args.ap_dropout) & vis
                        xb[drop] = 0.0
                    if args.jitter > 0:
                        noise = torch.randn_like(xb) * (args.jitter / 40.0)
                        xb = torch.where(vis, (xb + noise).clamp(0.0, 1.0), xb)
                loss = m.loss(xb, ytr[i])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss)
            if ep % 5 == 0 or ep == args.epochs - 1:
                print(f"epoch {ep:3d}  train soft-CE {tot / steps_per_epoch:.3f}")
        m.eval()
        with torch.no_grad():
            logits = torch.cat([m.logits(X[te][i:i + 2048].to(dev)).cpu()
                                for i in range(0, len(te), 2048)])

    yte = y[te]
    torch.save({"logits": logits, "te": te, "cents": cents}, logits_path)
    print(f"saved test logits to {logits_path}")
    print("\nper-scan readouts")
    for k in (1, 9, 25):
        pred = topk_soft_argmax(logits, cents, k)
        report(f"top-{k} soft-argmax", (pred - yte).norm(dim=1))

    if args.no_decode:
        return
    decode(args)


def build_route_prior(d, te, cents):
    """Bay-to-bay transition counts from the training sequences."""
    y, rep, ts = d["y"], d["rep"], d["t"]
    n = len(y)
    te_mask = torch.zeros(n, dtype=torch.bool)
    te_mask[te] = True
    coord2idx = {(round(float(cx), 3), round(float(cy), 3)): i
                 for i, (cx, cy) in enumerate(cents)}
    bay_idx = torch.full((n,), -1, dtype=torch.long)
    for i in range(n):
        bay_idx[i] = coord2idx.get((round(float(y[i, 0]), 3), round(float(y[i, 1]), 3)), -1)

    from collections import defaultdict
    adj = defaultdict(dict)
    n_edges = 0
    order = torch.argsort(ts)
    prev = {}
    for i in order.tolist():
        if te_mask[i] or bay_idx[i] < 0:
            continue
        r = int(d["rep"][i])
        b = int(bay_idx[i])
        if r in prev:
            pb, pt = prev[r]
            if float(ts[i]) - pt <= 300.0:
                adj[pb][b] = adj[pb].get(b, 0.0) + 1.0
                n_edges += 1
        prev[r] = (b, float(ts[i]))
    print(f"  route prior: {n_edges:,} transitions, "
          f"{sum(len(v) for v in adj.values()):,} distinct bay pairs")
    return dict(adj)


def decode(args):
    cache_path, logits_path = _paths(args)
    d = torch.load(cache_path, weights_only=False)
    saved = torch.load(logits_path, weights_only=False)
    logits, te, cents = saved["logits"], saved["te"], saved["cents"]
    y, rep, ts = d["y"], d["rep"], d["t"]
    yte = y[te]

    route = build_route_prior(d, te, cents) if args.route_prior else None

    print(f"\nsequence decoding: top-{args.topm}, speed {args.speed} m/s")
    logprob = torch.log_softmax(logits, dim=1)
    top = logprob.topk(args.topm, dim=1)
    d_true = (cents[top.indices] - yte.unsqueeze(1)).norm(dim=2).min(dim=1).values
    print(f"  true bay among candidates: {100 * (d_true < 1.0).float().mean():.1f}%")

    rep_te, ts_te = rep[te], ts[te]
    vit_err, fil_err = [], []
    for r in rep_te.unique():
        mask = (rep_te == r).nonzero().squeeze(1)
        if len(mask) < 3:
            continue
        o = mask[torch.argsort(ts_te[mask])]
        lp, cd, t_ = top.values[o], top.indices[o], ts_te[o].numpy()
        vit = hmm_decode(lp, cd, cents, t_, args, route)
        fil = filter_decode(lp, cd, cents, t_, args, route)
        vit_err.append((vit - yte[o]).norm(dim=1))
        fil_err.append((fil - yte[o]).norm(dim=1))
    report("HMM Viterbi (offline)", torch.cat(vit_err))
    report("HMM filter (online)", torch.cat(fil_err))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "run", "decode", "aisle", "reject"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=2.5)
    ap.add_argument("--topm", type=int, default=64)
    ap.add_argument("--speed", type=float, default=0.15,
                    help="speed of the motion prior in m/s")
    ap.add_argument("--base-sigma", type=float, default=3.0)
    ap.add_argument("--sigma-cap", type=float, default=25.0)
    ap.add_argument("--eps", type=float, default=0.05, help="weight of the broad transition component")
    ap.add_argument("--sigma-broad", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", choices=["mlp", "gcn"], default="mlp",
                    help="trunk: MLP or graph network without AP positions")
    ap.add_argument("--head", choices=["cls", "reg"], default="cls",
                    help="classification or coordinate regression")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--route-prior", action="store_true",
                    help="add bay-to-bay transition counts to the motion prior")
    ap.add_argument("--route-lambda", type=float, default=0.5)
    ap.add_argument("--full", action="store_true",
                    help="use the full history instead of the timestamped scans")
    ap.add_argument("--ap-dropout", type=float, default=0.0,
                    help="training: probability of dropping each visible AP")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="training: Gaussian noise on visible readings in dBm")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a.full)
    elif a.cmd == "run":
        run(a)
    elif a.cmd == "aisle":
        aisle_decode(a)
    elif a.cmd == "reject":
        reject_analysis(a)
    else:
        decode(a)
