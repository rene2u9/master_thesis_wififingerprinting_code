"""Fingerprint deduplication by reading vector.

Android throttles WiFi scans, so one scan result gets re-submitted under several
storage places. Only the oldest copy carries the right label.
"""

import hashlib
from typing import Iterable, Mapping


def readings_hash(readings: Iterable[Mapping]) -> str:
    """Order-independent MD5 over (bssid, level) pairs.

    Must match the server-side canonicalisation: sorted by bssid, "bssid:level" joined by commas.
    """
    canonical = ",".join(
        f"{r['bssid']}:{r['level']}"
        for r in sorted(readings, key=lambda r: r["bssid"])
    )
    return hashlib.md5(canonical.encode("utf-8")).hexdigest()
