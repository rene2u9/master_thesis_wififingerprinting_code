"""Build the fingerprint_hash side-table used for dedup (data.deduplicate=true).

Usage:
    python scripts/deduplicate_fingerprints.py [--db data/wifi_fingerprints.db]
"""

import argparse
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dedup import readings_hash

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(_DATA, "wifi_fingerprints.db")


def build(db_path: str) -> None:
    conn = sqlite3.connect(db_path)

    print("Computing reading-vector hashes ...")
    t = time.time()
    rows = conn.execute("SELECT id, readings FROM wifi_fingerprints").fetchall()
    hashed = [(id_, readings_hash(json.loads(rj))) for id_, rj in rows]
    print(f"  hashed {len(hashed):,} fingerprints in {time.time() - t:.1f}s")

    conn.execute("DROP TABLE IF EXISTS fingerprint_hash")
    conn.execute(
        """
        CREATE TABLE fingerprint_hash (
            id            INTEGER PRIMARY KEY,
            readings_hash TEXT NOT NULL
        )
        """
    )
    conn.executemany("INSERT INTO fingerprint_hash (id, readings_hash) VALUES (?, ?)", hashed)
    conn.execute("CREATE INDEX idx_readings_hash ON fingerprint_hash(readings_hash)")
    conn.commit()

    for label, where in (("GRO-4x", "WHERE storage_place_desc LIKE 'GRO-4%'"), ("ALL", "")):
        stats = conn.execute(
            f"""
            WITH g AS (
                SELECT h.readings_hash AS hh,
                       COUNT(*) AS n,
                       COUNT(DISTINCT w.storage_place_desc) AS labels
                FROM wifi_fingerprints w
                JOIN fingerprint_hash h ON h.id = w.id
                {where}
                GROUP BY h.readings_hash
            )
            SELECT
                SUM(n)                                  AS total,
                SUM(n) - COUNT(*)                       AS removed,
                SUM(CASE WHEN labels > 1 THEN n - 1 ELSE 0 END) AS mislabeled
            FROM g
            """
        ).fetchone()
        total, removed, mislabeled = stats
        pct = 100 * removed / total if total else 0
        mpct = 100 * mislabeled / total if total else 0
        print(
            f"  {label:6}: {total:,} rows -> {total - removed:,} after dedup "
            f"({removed:,} dropped, {pct:.1f}%; {mislabeled:,} of them mislabeled, {mpct:.1f}%)"
        )

    conn.close()
    print(f"Table fingerprint_hash written to {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite database")
    args = parser.parse_args()
    print(f"Database : {os.path.realpath(args.db)}")
    build(args.db)


if __name__ == "__main__":
    main()
