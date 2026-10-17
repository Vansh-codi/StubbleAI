
import json
import zipfile
from collections import Counter
from pathlib import Path
from shapely.geometry import shape

archive = Path("data/raw/boundaries/v3/district_nwic_geojson.zip")
member = "district_nwic.GeoJSON"

with zipfile.ZipFile(archive, "r") as zf:
    with zf.open(member) as f:
        data = json.load(f)

features = data.get("features", [])
print("=== DATASET SUMMARY ===")
print("GeoJSON type:", data.get("type"))
print("Feature count:", len(features))
print("Top-level keys:", list(data.keys()))

if not features:
    raise SystemExit("STOP: No features found.")

keys = sorted({
    key
    for feature in features
    for key in feature.get("properties", {})
})

print("\n=== PROPERTY SCHEMA ===")
print("First feature properties:")
print(json.dumps(features[0].get("properties", {}), indent=2, ensure_ascii=False))
print("\nAll property keys:", keys)

print("\n=== GEOMETRY AUDIT ===")
types = Counter()
missing = empty = invalid = 0

for feature in features:
    geom_json = feature.get("geometry")
    if not geom_json:
        missing += 1
        continue
    try:
        geom = shape(geom_json)
        types[geom.geom_type] += 1
        empty += int(geom.is_empty)
        invalid += int(not geom.is_valid)
    except Exception:
        invalid += 1

print("Geometry types:", dict(types))
print("Missing geometries:", missing)
print("Empty geometries:", empty)
print("Invalid/unreadable geometries:", invalid)

print("\n=== PROPERTY VALUE AUDIT ===")
for key in keys:
    values = [
        feature.get("properties", {}).get(key)
        for feature in features
    ]
    non_null = [v for v in values if v is not None and v != ""]
    unique = list(dict.fromkeys(str(v) for v in non_null))
    print(f"{key}: non-null={len(non_null)}, unique={len(unique)}")
    print("  Sample:", unique[:12])

print("\nAudit complete. No data files were modified.")
