"""RSSI vector -> PyG graph over the APs visible in a scan.

Nodes are APs with [rssi, lat, lon] (or [rssi] without AP positions). Edges connect
co-visible APs, weighted by exp(-dist / sigma) or by mean RSSI without positions.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any

from .normalization import normalize_rssi

try:
    from torch_geometric.data import Data
except ImportError:
    print("Warning: torch_geometric not installed. GCN models will not work.")
    Data = None


class RSSIToGraph:
    """Convert a normalised RSSI vector to a PyG Data object."""

    def __init__(
        self,
        num_aps: int = 256,
        missing_value: float = -100.0,
        rssi_threshold: float = -95.0,
        include_self_loops: bool = True,
        db_path: Optional[str] = None,
        sorted_aps: Optional[List[str]] = None,
        proximity_sigma: float = 25.0,        # metres
        graph_topology: str = "covisible",    # or "bipartite" (RP node <-> APs only)
        rssi_normalization: str = "minmax",
        rssi_min: float = -100.0,
        rssi_max: float = 0.0,
    ):
        self.num_aps             = num_aps
        self.missing_value       = missing_value
        self.rssi_threshold      = rssi_threshold
        self.include_self_loops  = include_self_loops
        self.proximity_sigma     = proximity_sigma
        self.graph_topology      = graph_topology
        self.rssi_normalization  = rssi_normalization
        self.rssi_min            = rssi_min
        self.rssi_max            = rssi_max

        # plain lists so the transform pickles into DataLoader workers
        self._ap_data: Optional[Dict[str, Any]] = None

        if db_path is not None and sorted_aps:
            self._load_positions(db_path, sorted_aps)

        self._lat_t:   Optional[torch.Tensor] = None
        self._lon_t:   Optional[torch.Tensor] = None
        self._dist_t:  Optional[torch.Tensor] = None

    def _load_positions(self, db_path: str, sorted_aps: List[str]) -> None:
        import sqlite3

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT mac_prefix, latitude, longitude FROM ap_positions"
        ).fetchall()
        conn.close()
        pos = {r[0]: (float(r[1]), float(r[2])) for r in rows}

        from . import WAREHOUSE_LAT_RANGE, WAREHOUSE_LON_RANGE
        centre = (WAREHOUSE_LAT_RANGE / 2, WAREHOUSE_LON_RANGE / 2)

        lats_m = [pos.get(ap, centre)[0] for ap in sorted_aps]
        lons_m = [pos.get(ap, centre)[1] for ap in sorted_aps]
        pad    = self.num_aps - len(lats_m)
        lats_m += [centre[0]] * pad
        lons_m += [centre[1]] * pad
        lats_m  = lats_m[:self.num_aps]
        lons_m  = lons_m[:self.num_aps]
        matched = sum(1 for ap in sorted_aps if ap in pos)

        lat_arr = np.array(lats_m, dtype=np.float32)
        lon_arr = np.array(lons_m, dtype=np.float32)
        d_lat   = lat_arr[:, None] - lat_arr[None, :]
        d_lon   = lon_arr[:, None] - lon_arr[None, :]
        dist    = np.sqrt(d_lat**2 + d_lon**2)

        self._ap_data = {
            'lat':      [v / WAREHOUSE_LAT_RANGE for v in lats_m],
            'lon':      [v / WAREHOUSE_LON_RANGE for v in lons_m],
            'dist_mat': dist.flatten().tolist(),
        }
        print(f"RSSIToGraph: {matched}/{len(sorted_aps)} APs matched to positions")

    def __getstate__(self):
        """Tensor cache is rebuilt per worker."""
        state = self.__dict__.copy()
        state['_lat_t']  = None
        state['_lon_t']  = None
        state['_dist_t'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _ensure_tensors(self) -> None:
        if self._ap_data is None or self._lat_t is not None:
            return
        d = self._ap_data
        self._lat_t  = torch.tensor(d['lat'],      dtype=torch.float32)
        self._lon_t  = torch.tensor(d['lon'],      dtype=torch.float32)
        self._dist_t = torch.tensor(
            d['dist_mat'], dtype=torch.float32
        ).reshape(self.num_aps, self.num_aps)

    def __call__(self, rssi_vector: torch.Tensor) -> "Data":
        if Data is None:
            raise ImportError("torch_geometric is required for graph models")

        self._ensure_tensors()

        rssi_vector = rssi_vector.float()
        norm_thresh = float(normalize_rssi(
            self.rssi_threshold,
            method=self.rssi_normalization,
            rssi_min=self.rssi_min,
            rssi_max=self.rssi_max,
            missing_value=self.missing_value,
        ))

        if self._lat_t is not None:
            node_features = torch.stack(
                [rssi_vector, self._lat_t, self._lon_t], dim=-1
            )  # (num_aps, 3)
        else:
            node_features = rssi_vector.unsqueeze(-1)  # (num_aps, 1)

        if self.graph_topology == "bipartite":
            edge_index, edge_attr = self._bipartite_edges(rssi_vector, norm_thresh)
            # virtual RP node at index num_aps
            rp_features = torch.zeros(1, node_features.shape[1])
            node_features = torch.cat([node_features, rp_features], dim=0)
            num_nodes = self.num_aps + 1
        else:
            edge_index, edge_attr = self._covisibility_edges(rssi_vector, norm_thresh)
            num_nodes = self.num_aps

        if self.include_self_loops:
            visible = torch.where(rssi_vector > norm_thresh)[0]
            if visible.numel() > 0:
                sl_ei = torch.stack([visible, visible], dim=0)
                sl_ea = rssi_vector[visible].unsqueeze(-1)
                edge_index = torch.cat([edge_index, sl_ei], dim=1)
                edge_attr  = torch.cat([edge_attr,  sl_ea], dim=0)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
        )

    def _covisibility_edges(
        self, rssi_vector: torch.Tensor, norm_thresh: float
    ):
        """Fully connected edges between co-visible APs."""
        visible = torch.where(rssi_vector > norm_thresh)[0]

        if visible.numel() == 0:
            return torch.zeros(2, 1, dtype=torch.long), torch.zeros(1, 1)

        if visible.numel() == 1:
            ei = torch.stack([visible, visible], dim=0)
            ea = rssi_vector[visible].unsqueeze(-1)
            return ei, ea

        n   = visible.numel()
        idx = torch.arange(n)
        ii, jj = torch.meshgrid(idx, idx, indexing="ij")
        mask = ii != jj
        src  = visible[ii[mask]]
        dst  = visible[jj[mask]]

        if self._dist_t is not None:
            dist = self._dist_t[src, dst]
            ea   = torch.exp(-dist / self.proximity_sigma).unsqueeze(-1)
        else:
            ea = ((rssi_vector[src] + rssi_vector[dst]) / 2.0).unsqueeze(-1)

        return torch.stack([src, dst], dim=0), ea

    def _bipartite_edges(
        self, rssi_vector: torch.Tensor, norm_thresh: float
    ):
        """Virtual RP node <-> each visible AP, weighted by RSSI."""
        visible = torch.where(rssi_vector > norm_thresh)[0]

        if visible.numel() == 0:
            return torch.zeros(2, 1, dtype=torch.long), torch.zeros(1, 1)

        rp_idx = self.num_aps
        rp_nodes = torch.full((visible.numel(),), rp_idx, dtype=torch.long)

        src = torch.cat([rp_nodes, visible])
        dst = torch.cat([visible, rp_nodes])
        weights = rssi_vector[visible].repeat(2).unsqueeze(-1)

        return torch.stack([src, dst], dim=0), weights
