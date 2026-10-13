#!/usr/bin/env python3
"""Export the production fingerprint DB to a local SQLite, labels from PickingSpots.

Usage: python scripts/export_temporal.py [--full]
"""
import json
import os
import re
import sqlite3
import sys

import psycopg2
import psycopg2.extras

CRED_PATH = os.environ.get(
    "WIFI_CREDENTIALS",
    "credentials.json",
)


def _load_pg():
    with open(CRED_PATH) as f:
        conn_str = json.load(f)["ConnectionStrings"]["psql"]
    kv = dict(
        re.match(r"\s*([^=]+)=(.*)", part).groups()
        for part in conn_str.split(";")
        if "=" in part
    )
    kv = {k.strip().lower(): v.strip() for k, v in kv.items()}
    return dict(
        host=kv["host"],
        dbname=kv["database"],
        user=kv["username"],
        password=kv["password"],
        connect_timeout=10,
    )


PG = _load_pg()
SRC = 'schema_wififingerprints."WifiFingerprints"'
PS = 'schema_wififingerprints."PickingSpots"'
OUT = "data/wifi_fingerprints_sub1m.db"  # timestamped rows only
FULL_OUT = "data/wifi_fingerprints_full_ps.db"
CHUNK = 20000
FULL = "--full" in sys.argv


def main():
    conn = psycopg2.connect(**PG)
    conn.set_session(readonly=True, autocommit=False)  # named cursor needs a txn

    # labels always from PickingSpots: latitude=Y, longitude=X
    ps_cur = conn.cursor()
    ps_cur.execute(f'SELECT "PickingSpotId","X","Y" FROM {PS} '
                   f'WHERE "X" IS NOT NULL AND "Y" IS NOT NULL')
    spots = {pid: (y, x) for pid, x, y in ps_cur.fetchall()}
    print(f"PickingSpots loaded: {len(spots):,} bays")

    out = sqlite3.connect(FULL_OUT if FULL else OUT)
    out.executescript(
        """
        DROP TABLE IF EXISTS wifi_fingerprints;
        CREATE TABLE wifi_fingerprints (
            id INTEGER PRIMARY KEY,
            storage_place_desc TEXT NOT NULL,
            latitude REAL NOT NULL,           -- PickingSpots.Y (ground truth)
            longitude REAL NOT NULL,          -- PickingSpots.X (ground truth)
            stored_latitude REAL,             -- original capture value, audit only
            stored_longitude REAL,
            created_at TEXT NOT NULL,
            readings TEXT NOT NULL,
            reporter TEXT,
            start_datetime TEXT,
            end_datetime TEXT,
            picking_area_latitude REAL,
            picking_area_longitude REAL,
            is_test INTEGER NOT NULL DEFAULT 0
        );
        """
    )

    cur = conn.cursor(name="export_cur")  # server-side, streamed
    cur.itersize = CHUNK
    ts_filter = "" if FULL else 'AND "StartDateTime" IS NOT NULL'
    cur.execute(
        f'''SELECT "Id","StoragePlaceDesc","Latitude","Longitude","CreatedAt",
                   "Readings"::text,"Reporter","StartDateTime","EndDateTime",
                   "PickingAreaLatitude","PickingAreaLongitude"
            FROM {SRC}
            WHERE "Latitude" IS NOT NULL AND "Longitude" IS NOT NULL
              AND "Readings" IS NOT NULL
              {ts_filter}'''
    )

    ins = (
        "INSERT OR REPLACE INTO wifi_fingerprints "
        "(id,storage_place_desc,latitude,longitude,stored_latitude,stored_longitude,"
        "created_at,readings,reporter,"
        "start_datetime,end_datetime,picking_area_latitude,picking_area_longitude,is_test) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )

    total = skipped = 0
    while True:
        rows = cur.fetchmany(CHUNK)
        if not rows:
            break
        batch = []
        for (rid, spot, lat, lon, created, readings, reporter, sdt, edt, pa_lat, pa_lon) in rows:
            pid = ";".join(spot.split(";")[:3]) if spot else None
            if pid not in spots:
                skipped += 1
                continue
            ps_lat, ps_lon = spots[pid]
            batch.append(
                (
                    rid, spot, ps_lat, ps_lon, lat, lon,
                    created.isoformat() if created is not None else "",
                    readings, reporter,
                    sdt.isoformat() if sdt is not None else None,
                    edt.isoformat() if edt is not None else None,
                    pa_lat, pa_lon, 0,
                )
            )
        out.executemany(ins, batch)
        total += len(batch)
        out.commit()
        sys.stdout.write(f"\rexported {total:,} rows (skipped {skipped:,} without PickingSpot)")
        sys.stdout.flush()

    print()
    out.executescript(
        """
        CREATE INDEX idx_spot ON wifi_fingerprints(storage_place_desc);
        CREATE INDEX idx_reporter ON wifi_fingerprints(reporter);
        CREATE INDEX idx_reporter_time ON wifi_fingerprints(reporter, start_datetime);
        """
    )
    out.commit()
    out.close()
    conn.close()
    print(f"done: {total:,} rows -> {FULL_OUT if FULL else OUT}  "
          f"(skipped {skipped:,} without PickingSpots bay)")


if __name__ == "__main__":
    main()
