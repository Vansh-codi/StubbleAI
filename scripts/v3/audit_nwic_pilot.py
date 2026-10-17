
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from shapely.geometry import shape

archive = Path("data/raw/boundaries/v3/district_nwic_geojson.zip")
member = "district_nwic.GeoJSON"

pilot = {
    "PB": "Punjab",
    "HR": "Haryana",
    "UP": "Uttar Pradesh",
    "BR": "Bihar",
    "MP": "Madhya Pradesh",
    "AS": "Assam",
}

with zipfile.ZipFile(archive, "r") as zf:
    with zf.open(member) as f:
        data = json.load(f)

features = data["features"]

# Find repeated district codes.
code_rows = defaultdict(list)
for feature in features:
    p = feature.get("properties", {})
    code_rows[str(p.get("dtcode", "")).strip()].append(
        (p.get("state_name"), p.get("state"), p.get("district"), p.get("id"))
    )

print("=== DUPLICATE DISTRICT CODES ===")
duplicates = {
    code: rows for code, rows in code_rows.items()
    if code and len(rows) > 1
}
if duplicates:
    for code, rows in duplicates.items():
        print(f"dtcode={code}: {rows}")
else:
    print("None")

print("\n=== PILOT STATE INVENTORY ===")
pilot_features = []

for state_code, state_name in pilot.items():
    rows = [
        feature for feature in features
        if str(feature.get("properties", {}).get("state", "")).strip().upper()
        == state_code
    ]
    pilot_features.extend(rows)

    print(f"\n{state_name} [{state_code}]: {len(rows)} features")
    for feature in sorted(
        rows,
        key=lambda f: str(f.get("properties", {}).get("district", ""))
    ):
        p = feature.get("properties", {})
        geom_json = feature.get("geometry")
        try:
            geom = shape(geom_json) if geom_json else None
            valid = geom.is_valid if geom else False
        except Exception:
            valid = False

        print(
            f"  district={p.get('district')!r}; "
            f"dtcode={p.get('dtcode')!r}; "
            f"id={p.get('id')!r}; valid={valid}"
        )

print("\n=== INVALID GEOMETRIES IN PILOT STATES ===")
invalid_count = 0
for feature in pilot_features:
    p = feature.get("properties", {})
    try:
        geom = shape(feature["geometry"])
        valid = geom.is_valid and not geom.is_empty
    except Exception:
        valid = False

    if not valid:
        invalid_count += 1
        print(
            f"{p.get('state_name')} | {p.get('district')} "
            f"| state={p.get('state')} | dtcode={p.get('dtcode')} "
            f"| id={p.get('id')}"
        )

print("Invalid pilot geometries:", invalid_count)

print("\n=== KEY DISTRICT NAME CHECK ===")
all_names = {
    (
        str(f.get("properties", {}).get("state", "")).strip().upper(),
        str(f.get("properties", {}).get("district", "")).strip().casefold()
    )
    for f in features
}

checks = [
    ("PB", "Malerkotla"),
    ("HR", "Charkhi Dadri"),
    ("HR", "Gurugram"),
    ("HR", "Gurgaon"),
    ("HR", "Nuh"),
    ("HR", "Mewat"),
    ("PB", "Ferozepur"),
    ("PB", "Firozpur"),
    ("PB", "Sri Muktsar Sahib"),
    ("PB", "Muktsar"),
]

for state_code, name in checks:
    found = (state_code, name.casefold()) in all_names
    print(f"{state_code} | {name}: {'FOUND' if found else 'NOT FOUND'}")

print("\nAudit complete. No data files were modified.")
