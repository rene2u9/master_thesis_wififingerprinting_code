"""Fingerprint dataset and train/val/test splitting on the SQLite DB.

Features are precomputed as numpy arrays to keep forked DataLoader workers small.
"""

import sqlite3
import json
import collections
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

from .normalization import normalize_rssi


class WiFiFingerprintDataset(Dataset):
    def __init__(
        self,
        db_path: str,
        indices: Optional[List[int]] = None,
        sorted_aps: Optional[List[str]] = None,
        max_aps: int = 576,
        missing_value: float = -100.0,
        ssid_filter: Optional[str] = None,
        transform=None,
        noise_coefficient: float = 0.0,
        target_scale: Optional[Tuple[float, float]] = None,
        rssi_normalization: str = "minmax",
        rssi_min: float = -100.0,
        rssi_max: float = 0.0,
        ap_permutation_seed: Optional[int] = None,
    ):
        """
        Args:
            indices: DB row ids to load
            sorted_aps: AP prefixes in feature order
            missing_value: RSSI for an AP not seen in the scan (dBm)
            target_scale: (lat_range, lon_range) to scale targets to [0, 1], None = metres
            ap_permutation_seed: fixed shuffle of the AP axis, control run for the image layout
        """
        self.max_aps = max_aps
        self.ap_permutation_seed = ap_permutation_seed
        self.missing_value = missing_value
        self.ssid_filter = ssid_filter
        self.transform = transform
        self.noise_coefficient = noise_coefficient
        self.target_scale = target_scale
        self.rssi_normalization = rssi_normalization
        self.rssi_min = rssi_min
        self.rssi_max = rssi_max

        rows = self._load_rows(db_path, indices)

        if sorted_aps is None:
            self.sorted_aps = self._build_ap_list_from_rows(rows)
        else:
            self.sorted_aps = sorted_aps

        self._precompute(rows)
        del rows

        print(f"Loaded {len(self._features)} fingerprints with {len(self.sorted_aps)} unique APs")

    @staticmethod
    def _load_rows(db_path: str, indices: Optional[List[int]]) -> list:
        conn = sqlite3.connect(db_path)
        sql = (
            "SELECT id, storage_place_desc, latitude, longitude, "
            "readings, created_at FROM wifi_fingerprints"
        )
        if indices is not None and len(indices) > 0:
            batch_size = 900
            rows: list = []
            for i in range(0, len(indices), batch_size):
                batch = indices[i : i + batch_size]
                placeholders = ",".join("?" * len(batch))
                rows.extend(
                    conn.execute(
                        f"{sql} WHERE id IN ({placeholders})", batch
                    ).fetchall()
                )
        else:
            rows = conn.execute(sql).fetchall()
        conn.close()
        return rows

    @staticmethod
    def _get_ap_prefix(bssid: str) -> str:
        """First 5 octets of the BSSID = one physical AP (same as the production predictor)."""
        return ":".join(bssid.split(":")[:5])

    def _process_readings(self, readings: List[Dict]) -> Dict[str, int]:
        """Filter by SSID, group by AP prefix, keep the strongest RSSI."""
        result: Dict[str, int] = {}
        for reading in readings:
            bssid = reading.get("bssid", "")
            ssid = reading.get("ssid", "")
            level = reading.get("level", self.missing_value)

            if not bssid:
                continue
            if self.ssid_filter and not ssid.startswith(self.ssid_filter):
                continue

            prefix = self._get_ap_prefix(bssid)
            if prefix not in result or level > result[prefix]:
                result[prefix] = level

        return result

    def _build_ap_list_from_rows(self, rows: list) -> List[str]:
        prefixes: set = set()
        for row in rows:
            readings = json.loads(row[4])
            prefixes.update(self._process_readings(readings).keys())
        return sorted(prefixes)[: self.max_aps]

    def _create_feature_vector(self, readings: List[Dict]) -> np.ndarray:
        """Absent APs and padding get missing_value (-100 dBm), which normalises to 0."""
        processed = self._process_readings(readings)

        feature_vector = np.array(
            [processed.get(ap, self.missing_value) for ap in self.sorted_aps],
            dtype=np.float32,
        )

        if len(feature_vector) < self.max_aps:
            padding = np.full(
                self.max_aps - len(feature_vector),
                self.missing_value,
                dtype=np.float32,
            )
            feature_vector = np.concatenate([feature_vector, padding])

        feature_vector = normalize_rssi(
            feature_vector,
            method=self.rssi_normalization,
            rssi_min=self.rssi_min,
            rssi_max=self.rssi_max,
            missing_value=self.missing_value,
        )

        return feature_vector[: self.max_aps]

    def _precompute(self, rows: list):
        n = len(rows)
        self._features = np.empty((n, self.max_aps), dtype=np.float32)
        self._targets = np.empty((n, 2), dtype=np.float32)
        self._ids = np.empty(n, dtype=np.int64)
        self._storage_descs: List[str] = []
        self._created_ats: List[str] = []

        lat_s = self.target_scale[0] if self.target_scale else 1.0
        lon_s = self.target_scale[1] if self.target_scale else 1.0

        for i, (id_, desc, lat, lon, readings_json, created_at) in enumerate(rows):
            readings = json.loads(readings_json)
            self._features[i] = self._create_feature_vector(readings)
            self._targets[i] = [lat / lat_s, lon / lon_s]
            self._ids[i] = id_
            self._storage_descs.append(desc)
            self._created_ats.append(created_at)

        if self.ap_permutation_seed is not None:
            perm = np.random.RandomState(self.ap_permutation_seed).permutation(self.max_aps)
            self._features = np.ascontiguousarray(self._features[:, perm])
            print(f"Applied fixed AP permutation (seed={self.ap_permutation_seed})")

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        features = self._features[idx].copy()

        if self.noise_coefficient > 0:
            noise = np.random.normal(0, self.noise_coefficient, features.shape).astype(
                np.float32
            )
            features = np.clip(features + noise, 0.0, 1.0)

        target = self._targets[idx]

        metadata = {
            "id": int(self._ids[idx]),
            "storage_place_desc": self._storage_descs[idx],
            "created_at": self._created_ats[idx],
        }

        features = torch.from_numpy(features)
        target = torch.from_numpy(target.copy())

        if self.transform is not None:
            features = self.transform(features)

        return features, target, metadata

    def get_sorted_aps(self) -> List[str]:
        return self.sorted_aps.copy()


class WiFiFingerprintDataModule:
    """Builds the train/val/test id lists and the AP list."""

    def __init__(
        self,
        db_path: str,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42,
        max_samples: int = None,
        storage_place_filter: Optional[str] = None,
        deduplicate: bool = False,
        sampling: str = "random",
        reps_per_place: int = 10,
        clean_fraction: float = 0.0,
        clean_k: int = 9,
        split_strategy: str = "random",
        **dataset_kwargs,
    ):
        """
        Args:
            storage_place_filter: SQL LIKE pattern, e.g. "GRO-4%"
            split_strategy: "random" (per row), "group" (hold out whole bays) or
                "temporal" (oldest rows train, newest test, avoids near-duplicate leakage)
            **dataset_kwargs: passed on to WiFiFingerprintDataset
        """
        assert (
            abs(train_split + val_split + test_split - 1.0) < 1e-6
        ), "Splits must sum to 1.0"

        self.db_path = db_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.max_samples = max_samples
        self.storage_place_filter = storage_place_filter
        self.deduplicate = deduplicate
        self.sampling = sampling
        self.reps_per_place = reps_per_place
        self.clean_fraction = clean_fraction
        self.clean_k = clean_k
        if split_strategy not in ("random", "group", "temporal"):
            raise ValueError(f"Unknown split_strategy: {split_strategy!r}")
        self.split_strategy = split_strategy
        self.dataset_kwargs = dataset_kwargs

        self._create_splits()

        self.sorted_aps = self._build_ap_list()

        if self.clean_fraction > 0:
            self._clean_train()

        print(
            f"Data splits - Train: {len(self.train_indices)}, "
            f"Val: {len(self.val_indices)}, Test: {len(self.test_indices)}"
        )

    def _create_splits(self):
        conn = sqlite3.connect(self.db_path)
        # (0, 0) = no picking area could be mapped
        where = "WHERE NOT (latitude = 0 AND longitude = 0)"
        params = []
        if self.storage_place_filter:
            where += " AND storage_place_desc LIKE ?"
            params.append(self.storage_place_filter)

        if self.deduplicate:
            rows = self._deduplicated_rows(conn, where, params)
        else:
            rows = conn.execute(
                f"SELECT id, created_at, storage_place_desc FROM wifi_fingerprints {where}",
                params).fetchall()
        conn.close()

        if self.sampling == "balanced":
            all_ids = self._balanced_select(rows, self.reps_per_place)
        else:
            all_ids = sorted(r[0] for r in rows)
            if self.max_samples and len(all_ids) > self.max_samples:
                rng = np.random.RandomState(self.random_seed)
                all_ids = rng.choice(all_ids, self.max_samples, replace=False).tolist()

        n = len(all_ids)
        n_train = int(n * self.train_split)
        n_val = int(n * self.val_split)
        n_test = n - n_train - n_val

        if self.split_strategy == "temporal":
            meta = {id_: (ca, desc) for id_, ca, desc in rows}
            ordered = sorted(all_ids, key=lambda i: (meta[i][0], i))
            self.train_indices = ordered[:n_train]
            self.val_indices = ordered[n_train:n_train + n_val]
            self.test_indices = ordered[n_train + n_val:]
            print(f"Temporal split: train <= {meta[self.train_indices[-1]][0]} "
                  f"< val <= {meta[self.val_indices[-1]][0]} < test")
        elif self.split_strategy == "group":
            meta = {id_: (ca, desc) for id_, ca, desc in rows}
            by_place = collections.defaultdict(list)
            for i in all_ids:
                by_place[";".join(meta[i][1].split(";")[:3])].append(i)
            places = sorted(by_place)
            order = np.random.RandomState(self.random_seed).permutation(len(places))
            train_ids, val_ids, test_ids = [], [], []
            bays = [0, 0, 0]
            for j in order:
                ids = by_place[places[j]]
                if len(test_ids) < n_test:
                    test_ids += ids
                    bays[2] += 1
                elif len(val_ids) < n_val:
                    val_ids += ids
                    bays[1] += 1
                else:
                    train_ids += ids
                    bays[0] += 1
            self.train_indices = train_ids
            self.val_indices = val_ids
            self.test_indices = test_ids
            print(f"Group split (held-out bays): {len(places)} bays -> "
                  f"train/val/test = {bays[0]}/{bays[1]}/{bays[2]} bays, "
                  f"{len(train_ids)}/{len(val_ids)}/{len(test_ids)} rows")
        else:
            generator = torch.Generator().manual_seed(self.random_seed)
            train_split, val_split, test_split = random_split(
                all_ids, [n_train, n_val, n_test], generator=generator
            )
            self.train_indices = list(train_split)
            self.val_indices = list(val_split)
            self.test_indices = list(test_split)

    def _deduplicated_rows(self, conn, where, params):
        """Keep the oldest row per readings hash within the filtered set."""
        try:
            rows = conn.execute(
                f"SELECT w.id, w.created_at, w.storage_place_desc, h.readings_hash "
                f"FROM wifi_fingerprints w JOIN fingerprint_hash h ON h.id = w.id {where}",
                params,
            ).fetchall()
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                "data.deduplicate=true but the fingerprint_hash table is missing. "
                "Run: python scripts/deduplicate_fingerprints.py"
            ) from e

        best = {}
        for id_, created_at, desc, h in rows:
            if h not in best or (created_at, id_) < (best[h][0], best[h][1]):
                best[h] = (created_at, id_, desc)
        kept = [(id_, ca, desc) for ca, id_, desc in best.values()]
        print(f"Deduplication: {len(rows)} -> {len(kept)} fingerprints "
              f"({len(rows) - len(kept)} duplicate scans dropped)")
        return kept

    def _balanced_select(self, rows, n):
        """Keep up to n rows per bay, spread evenly over time."""
        by_place = collections.defaultdict(list)
        for id_, created_at, desc in rows:
            place = ";".join(desc.split(";")[:3])
            by_place[place].append((created_at, id_))

        kept = []
        for place in sorted(by_place):
            items = sorted(by_place[place])
            if len(items) <= n:
                kept.extend(id_ for _, id_ in items)
            else:
                idx = np.unique(np.linspace(0, len(items) - 1, n).round().astype(int))
                kept.extend(items[j][1] for j in idx)
        kept = sorted(kept)
        print(f"Balanced sampling: {len(rows)} -> {len(kept)} fingerprints "
              f"({len(by_place)} coordinates, <= {n} time-stratified reps each)")
        return kept

    def _clean_train(self):
        """Drop the train rows with the largest leave-one-out WKNN residual."""
        ds = self.get_dataset("train")
        X = torch.from_numpy(ds._features)
        y = torch.from_numpy(ds._targets)
        scale = self.dataset_kwargs.get("target_scale") or (1.0, 1.0)
        scale = torch.tensor(scale, dtype=torch.float32)
        k = self.clean_k
        res = torch.empty(X.shape[0])
        for i in range(0, X.shape[0], 512):
            b = min(512, X.shape[0] - i)
            d = torch.cdist(X[i : i + b], X)
            d[torch.arange(b), torch.arange(i, i + b)] = float("inf")
            dk, idx = d.topk(k, dim=1, largest=False)
            w = 1.0 / (dk + 1e-8)
            w = w / w.sum(1, keepdim=True)
            pred = (w.unsqueeze(-1) * y[idx]).sum(1)
            res[i : i + b] = ((pred - y[i : i + b]) * scale).norm(dim=1)
        thr = res.quantile(1 - self.clean_fraction)
        keep_ids = set(ds._ids[(res <= thr).numpy()].tolist())
        before = len(self.train_indices)
        self.train_indices = [idx for idx in self.train_indices if int(idx) in keep_ids]
        print(f"Label cleaning: dropped top {self.clean_fraction*100:.0f}% LOO-residual "
              f"(>{thr:.1f}m) - Train {before} -> {len(self.train_indices)}")

    def _build_ap_list(self) -> List[str]:
        """AP list from the training split only."""
        ssid_filter = self.dataset_kwargs.get("ssid_filter")
        max_aps = self.dataset_kwargs.get("max_aps", 256)

        conn = sqlite3.connect(self.db_path)
        prefixes: set = set()
        batch_size = 900
        for i in range(0, len(self.train_indices), batch_size):
            batch = self.train_indices[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            for (readings_json,) in conn.execute(
                f"SELECT readings FROM wifi_fingerprints WHERE id IN ({placeholders})",
                batch,
            ):
                for r in json.loads(readings_json):
                    bssid = r.get("bssid", "")
                    ssid = r.get("ssid", "")
                    if not bssid:
                        continue
                    if ssid_filter and not ssid.startswith(ssid_filter):
                        continue
                    prefixes.add(WiFiFingerprintDataset._get_ap_prefix(bssid))
        conn.close()

        return sorted(prefixes)[:max_aps]

    def get_dataset(self, split: str) -> WiFiFingerprintDataset:
        if split == "train":
            indices = self.train_indices
        elif split == "val":
            indices = self.val_indices
        elif split == "test":
            indices = self.test_indices
        else:
            raise ValueError(f"Unknown split: {split}")

        kwargs = dict(self.dataset_kwargs)
        if split != "train":
            kwargs["noise_coefficient"] = 0.0

        return WiFiFingerprintDataset(
            db_path=self.db_path,
            indices=indices,
            sorted_aps=self.sorted_aps,
            **kwargs,
        )

    def get_sorted_aps(self) -> List[str]:
        return self.sorted_aps.copy()
