"""RSSI -> [0, 1] normalization, shared by the dataset and the graph transform.

"minmax" clips to [rssi_min, rssi_max], "legacy" is (rssi - missing) / -missing.
"""

import numpy as np


def normalize_rssi(
    values,
    method: str = "minmax",
    rssi_min: float = -100.0,
    rssi_max: float = 0.0,
    missing_value: float = -100.0,
):
    """Map raw RSSI (dBm) to [0, 1]. Accepts a scalar or an array-like."""
    arr = np.asarray(values, dtype=np.float32)

    if method == "legacy":
        return (arr - missing_value) / (-missing_value)

    if method == "minmax":
        span = rssi_max - rssi_min
        if span <= 0:
            raise ValueError(
                f"rssi_max ({rssi_max}) must be greater than rssi_min ({rssi_min})"
            )
        return np.clip((arr - rssi_min) / span, 0.0, 1.0).astype(np.float32)

    raise ValueError(
        f"Unknown rssi_normalization {method!r}; expected 'minmax' or 'legacy'"
    )
