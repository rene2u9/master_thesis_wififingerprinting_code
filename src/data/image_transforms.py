"""RSSI vector -> image transforms for the CNN/ViT/Mamba backbones."""

import torch
import numpy as np
from typing import List, Optional, Tuple


class RSSIToImage:
    """Sequential packing of the AP vector into an h x w image."""

    def __init__(self, image_shape: Tuple[int, int, int] = (24, 24, 1)):
        self.h, self.w, self.c = image_shape

    def __call__(self, rssi_vector: torch.Tensor) -> torch.Tensor:
        rssi = rssi_vector.numpy()

        total_pixels = self.h * self.w
        if len(rssi) < total_pixels:
            pad_val = rssi.mean() if len(rssi) > 0 else 0.0
            rssi = np.concatenate([rssi, np.full(total_pixels - len(rssi), pad_val)])
        else:
            rssi = rssi[:total_pixels]

        image = rssi.reshape(self.h, self.w)

        if self.c == 1:
            image = image[np.newaxis, ...]
        else:
            image = np.stack([image] * self.c, axis=0)

        return torch.from_numpy(image).float()


class RSSIToUpscaledImage:
    """RSSIToImage upscaled to the backbone resolution, no AP positions used.

    Control for the position-based transforms.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (224, 224, 1),
        source_shape: Tuple[int, int] = (16, 16),
        mode: str = "nearest",
    ):
        self.h, self.w, self.c = image_shape
        self.sh, self.sw = source_shape
        self.mode = mode
        self._base = RSSIToImage(image_shape=(self.sh, self.sw, 1))
        print(f"  RSSIToUpscaledImage: {self.sh}x{self.sw} -> {self.h}x{self.w} "
              f"({mode}, no AP positions used)")

    def __call__(self, rssi_vector: torch.Tensor) -> torch.Tensor:
        image = self._base(rssi_vector).unsqueeze(0)
        kwargs = {} if self.mode == "nearest" else {"align_corners": False}
        image = torch.nn.functional.interpolate(
            image, size=(self.h, self.w), mode=self.mode, **kwargs
        )
        image = image.squeeze(0)
        if self.c > 1:
            image = image.repeat(self.c, 1, 1)
        return image.float()


class RSSIToSpatialImageOA:
    """One grid cell per AP near its position, via Hungarian assignment."""

    def __init__(
        self,
        db_path: str,
        sorted_aps: List[str],
        image_shape: Tuple[int, int, int] = (16, 16, 1),
    ):
        from scipy.optimize import linear_sum_assignment
        import sqlite3
        from . import WAREHOUSE_LAT_RANGE, WAREHOUSE_LON_RANGE

        self.h, self.w, _ = image_shape
        n_aps = len(sorted_aps)
        total_pixels = self.h * self.w
        assert n_aps <= total_pixels, (
            f"Grid {self.h}x{self.w}={total_pixels} pixels too small for {n_aps} APs"
        )

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT mac_prefix, latitude, longitude FROM ap_positions"
        ).fetchall()
        conn.close()
        pos_map = {r[0]: (float(r[1]), float(r[2])) for r in rows}

        ideal_positions: List[Optional[Tuple[float, float]]] = []
        for ap in sorted_aps:
            if ap in pos_map:
                lat, lon = pos_map[ap]
                r = lat / WAREHOUSE_LAT_RANGE * (self.h - 1)
                c = lon / WAREHOUSE_LON_RANGE * (self.w - 1)
                ideal_positions.append((
                    max(0.0, min(r, self.h - 1.0)),
                    max(0.0, min(c, self.w - 1.0)),
                ))
            else:
                ideal_positions.append(None)

        all_cells = [(r, c) for r in range(self.h) for c in range(self.w)]

        cost = np.full((n_aps, total_pixels), 0.0, dtype=np.float64)
        centre_r, centre_c = (self.h - 1) / 2.0, (self.w - 1) / 2.0
        for i, pos in enumerate(ideal_positions):
            if pos is not None:
                ir, ic = pos
            else:
                ir, ic = centre_r, centre_c
            for j, (cr, cc) in enumerate(all_cells):
                cost[i, j] = (ir - cr) ** 2 + (ic - cc) ** 2

        row_idx, col_idx = linear_sum_assignment(cost)

        self._grid: List[Tuple[int, int]] = [None] * n_aps  # type: ignore[list-item]
        for ai, ci in zip(row_idx, col_idx):
            self._grid[ai] = all_cells[ci]

        total_disp = sum(
            np.sqrt(cost[ai, ci]) for ai, ci in zip(row_idx, col_idx)
        )
        n_known = sum(1 for p in ideal_positions if p is not None)
        print(f"  RSSIToSpatialImageOA: {n_aps} APs -> {self.h}x{self.w} grid, "
              f"{n_known} with known positions, "
              f"mean displacement: {total_disp / n_aps:.2f} cells")

    def __call__(self, rssi_vector: torch.Tensor) -> torch.Tensor:
        rssi = rssi_vector.numpy()
        image = np.zeros((self.h, self.w), dtype=np.float32)
        for i, (r, c) in enumerate(self._grid):
            if i >= len(rssi):
                break
            image[r, c] = rssi[i]
        return torch.from_numpy(image[np.newaxis, ...]).float()


class RSSIToGaussianSplatImage:
    """Each AP's RSSI as a Gaussian blob at its position, blobs summed."""

    def __init__(
        self,
        db_path: str,
        sorted_aps: List[str],
        image_shape: Tuple[int, int, int] = (64, 64, 1),
        sigma: Optional[float] = None,
        sigma_multiplier: float = 1.0,
        truncate: float = 3.0,
        fill_mode: str = "zero",
    ):
        import sqlite3
        from . import WAREHOUSE_LAT_RANGE, WAREHOUSE_LON_RANGE

        self.h, self.w, self._channels = image_shape

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT mac_prefix, latitude, longitude FROM ap_positions"
        ).fetchall()
        conn.close()
        pos_map = {r[0]: (float(r[1]), float(r[2])) for r in rows}

        self._ap_positions: List[Optional[Tuple[float, float]]] = []
        known_positions = []
        for ap in sorted_aps:
            if ap in pos_map:
                lat, lon = pos_map[ap]
                r = lat / WAREHOUSE_LAT_RANGE * (self.h - 1)
                c = lon / WAREHOUSE_LON_RANGE * (self.w - 1)
                r = max(0.0, min(r, self.h - 1.0))
                c = max(0.0, min(c, self.w - 1.0))
                self._ap_positions.append((r, c))
                known_positions.append((r, c))
            else:
                self._ap_positions.append(None)

        if sigma is not None:
            self._sigma = sigma
        else:
            if len(known_positions) >= 2:
                from scipy.spatial.distance import cdist
                coords = np.array(known_positions)
                dists = cdist(coords, coords)
                np.fill_diagonal(dists, np.inf)
                nn_dists = dists.min(axis=1)
                # half the median nearest-neighbour distance between APs
                self._sigma = max(0.5, float(np.median(nn_dists)) / 2.0)
            else:
                self._sigma = 1.0
        self._sigma *= sigma_multiplier

        self._fill_mode = fill_mode
        self._truncate = truncate

        n_aps = len(sorted_aps)
        row_coords = np.arange(self.h, dtype=np.float32)
        col_coords = np.arange(self.w, dtype=np.float32)
        rr, cc_grid = np.meshgrid(row_coords, col_coords, indexing='ij')

        inv_2sigma2 = 1.0 / (2.0 * self._sigma * self._sigma)
        radius = int(np.ceil(self._truncate * self._sigma))

        kernels = np.zeros((n_aps, self.h, self.w), dtype=np.float32)
        for i, pos in enumerate(self._ap_positions):
            if pos is None:
                continue
            cr, cc_pos = pos
            r_lo = max(0, int(cr) - radius)
            r_hi = min(self.h, int(cr) + radius + 1)
            c_lo = max(0, int(cc_pos) - radius)
            c_hi = min(self.w, int(cc_pos) + radius + 1)
            dr = rr[r_lo:r_hi, c_lo:c_hi] - cr
            dc = cc_grid[r_lo:r_hi, c_lo:c_hi] - cc_pos
            kernels[i, r_lo:r_hi, c_lo:c_hi] = np.exp(-(dr**2 + dc**2) * inv_2sigma2)

        self._kernels = torch.from_numpy(kernels.reshape(n_aps, -1))  # (n_aps, H*W)

        # 3-channel mode: constant AP position channel
        if self._channels == 3:
            ap_ones = torch.zeros(n_aps)
            for i, pos in enumerate(self._ap_positions):
                if pos is not None:
                    ap_ones[i] = 1.0
            ap_pos_image = ap_ones @ self._kernels
            max_val = ap_pos_image.max()
            if max_val > 0:
                ap_pos_image = ap_pos_image / max_val
            self._ap_position_channel = ap_pos_image.reshape(self.h, self.w)

        n_known = sum(1 for p in self._ap_positions if p is not None)
        ch_info = f", channels={self._channels}" if self._channels > 1 else ""
        fill_info = f", fill={self._fill_mode}" if self._fill_mode != "zero" else ""
        print(f"  RSSIToGaussianSplatImage: {len(sorted_aps)} APs -> {self.h}x{self.w} grid, "
              f"{n_known} with positions, sigma={self._sigma:.2f}px{ch_info}{fill_info}")

    def _normalize_and_fill(self, image: torch.Tensor) -> torch.Tensor:
        max_val = image.max()
        if max_val > 0:
            image = image / max_val
        if self._fill_mode == "mean":
            signal_mask = image > 0
            if signal_mask.any():
                mean_val = image[signal_mask].mean()
                image = torch.where(signal_mask, image, mean_val)
        return image

    def __call__(self, rssi_vector: torch.Tensor) -> torch.Tensor:
        # drop padding beyond the real APs
        rssi = rssi_vector[:self._kernels.shape[0]]

        ch_rssi = (rssi @ self._kernels).reshape(self.h, self.w)
        ch_rssi = self._normalize_and_fill(ch_rssi)

        if self._channels == 1:
            return ch_rssi.unsqueeze(0).float()

        ch_ap_pos = self._ap_position_channel

        detection = (rssi > 0).float()
        ch_detect = (detection @ self._kernels).reshape(self.h, self.w)
        max_val = ch_detect.max()
        if max_val > 0:
            ch_detect = ch_detect / max_val

        return torch.stack([ch_rssi, ch_ap_pos, ch_detect], dim=0).float()


class RSSIToBilinearImage:
    """Each AP's RSSI spread bilinearly over the 4 pixels around its position."""

    def __init__(
        self,
        db_path: str,
        sorted_aps: List[str],
        image_shape: Tuple[int, int, int] = (224, 224, 1),
        fill_mode: str = "zero",
    ):
        import sqlite3
        from . import WAREHOUSE_LAT_RANGE, WAREHOUSE_LON_RANGE

        self.h, self.w, _ = image_shape
        self._fill_mode = fill_mode

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT mac_prefix, latitude, longitude FROM ap_positions"
        ).fetchall()
        conn.close()
        pos_map = {r[0]: (float(r[1]), float(r[2])) for r in rows}

        n_aps = len(sorted_aps)
        kernels = np.zeros((n_aps, self.h * self.w), dtype=np.float32)
        n_known = 0

        for i, ap in enumerate(sorted_aps):
            if ap not in pos_map:
                continue
            n_known += 1
            lat, lon = pos_map[ap]
            r = lat / WAREHOUSE_LAT_RANGE * (self.h - 1)
            c = lon / WAREHOUSE_LON_RANGE * (self.w - 1)
            r = max(0.0, min(r, self.h - 1.0))
            c = max(0.0, min(c, self.w - 1.0))

            r0, c0 = int(r), int(c)
            r1 = min(r0 + 1, self.h - 1)
            c1 = min(c0 + 1, self.w - 1)
            dr, dc = r - r0, c - c0

            kernels[i, r0 * self.w + c0] += (1 - dr) * (1 - dc)
            kernels[i, r0 * self.w + c1] += (1 - dr) * dc
            kernels[i, r1 * self.w + c0] += dr * (1 - dc)
            kernels[i, r1 * self.w + c1] += dr * dc

        self._kernels = torch.from_numpy(kernels)

        print(f"  RSSIToBilinearImage: {n_aps} APs -> {self.h}x{self.w} grid, "
              f"{n_known} with positions, fill={fill_mode}")

    def __call__(self, rssi_vector: torch.Tensor) -> torch.Tensor:
        rssi = rssi_vector[:self._kernels.shape[0]]
        image = rssi @ self._kernels
        image = image.reshape(self.h, self.w)

        max_val = image.max()
        if max_val > 0:
            image = image / max_val

        if self._fill_mode == "mean":
            signal_mask = image > 0
            if signal_mask.any():
                mean_val = image[signal_mask].mean()
                image = torch.where(signal_mask, image, mean_val)
        return image.unsqueeze(0).float()


class RSSIToSpatialImage:
    """Each AP's RSSI at its grid cell. Collisions keep the max, unknown APs are skipped."""

    def __init__(
        self,
        db_path: str,
        sorted_aps: List[str],
        image_shape: Tuple[int, int, int] = (16, 16, 1),
    ):
        import sqlite3
        from . import WAREHOUSE_LAT_RANGE, WAREHOUSE_LON_RANGE
        self.h, self.w, _ = image_shape
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT mac_prefix, latitude, longitude FROM ap_positions").fetchall()
        conn.close()
        pos_map = {r[0]: (float(r[1]), float(r[2])) for r in rows}

        self._grid: List[Optional[Tuple[int, int]]] = []
        for ap in sorted_aps:
            if ap in pos_map:
                lat, lon = pos_map[ap]
                row = int(lat / WAREHOUSE_LAT_RANGE * self.h)
                col = int(lon / WAREHOUSE_LON_RANGE * self.w)
                self._grid.append((max(0, min(row, self.h - 1)),
                                   max(0, min(col, self.w - 1))))
            else:
                self._grid.append(None)

    def __call__(self, rssi_vector: torch.Tensor) -> torch.Tensor:
        rssi  = rssi_vector.numpy()
        image = np.zeros((self.h, self.w), dtype=np.float32)
        for i, rc in enumerate(self._grid):
            if i >= len(rssi) or rc is None:
                continue
            r, c = rc
            if rssi[i] > image[r, c]:
                image[r, c] = rssi[i]
        return torch.from_numpy(image[np.newaxis, ...]).float()
