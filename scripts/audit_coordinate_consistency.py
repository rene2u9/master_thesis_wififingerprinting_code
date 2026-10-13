"""Check stored fingerprint lat/lon against the PickingSpots coordinates.

Usage:  python scripts/audit_coordinate_consistency.py <db_path>
"""
import sqlite3, sys

EPS = 0.01  # metres

conn = sqlite3.connect(sys.argv[1])
spots = {p: (lat, lon) for p, lat, lon in
         conn.execute("SELECT pallet_prefix, latitude, longitude FROM pickspots")}
print(f"PickingSpots: {len(spots)} entries")
rows = conn.execute(
    "SELECT storage_place_desc, latitude, longitude FROM wifi_fingerprints"
).fetchall()
conn.close()

matched = unmatched = 0
deviations = []
for desc, lat, lon in rows:
    spot = spots.get(";".join(desc.split(";")[:3]))
    if spot is None:
        unmatched += 1
        continue
    matched += 1
    d = ((lat - spot[0]) ** 2 + (lon - spot[1]) ** 2) ** 0.5
    if d > EPS:
        deviations.append(d)

deviations.sort()
print(f"rows total: {len(rows)} | with pickspot match: {matched} | without: {unmatched}")
if deviations:
    print(f"deviating (> {EPS} m): {len(deviations)} | "
          f"Median {deviations[len(deviations) // 2]:.2f} m | Max {deviations[-1]:.2f} m")
else:
    print(f"deviating (> {EPS} m): 0 - DB consistent with PickingSpots.")
