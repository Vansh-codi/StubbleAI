import geopandas as gpd
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DISTRICT_FILE = BASE_DIR / "districts_punjab_haryana.geojson"
OUTPUT_FILE = BASE_DIR / "district_coordinates.csv"

print("Loading district boundaries...")

districts = gpd.read_file(DISTRICT_FILE)

# Keep Punjab + Haryana
districts["state_clean"] = (
    districts["state"]
    .astype(str)
    .str.strip()
    .str.lower()
)

districts = districts[
    districts["state_clean"].isin(["punjab", "haryana"])
].copy()

# Make sure coordinates are geographic
districts = districts.to_crs("EPSG:4326")

# Representative point is safer than raw centroid
# because centroid can fall outside irregular polygons.
districts["representative_point"] = (
    districts.geometry.representative_point()
)

districts["longitude"] = (
    districts["representative_point"].x
)

districts["latitude"] = (
    districts["representative_point"].y
)

output = districts[
    [
        "name",
        "state",
        "lgd_districtcode",
        "latitude",
        "longitude"
    ]
].copy()

output = output.rename(
    columns={
        "name": "district"
    }
)

output = output.sort_values(
    ["state", "district"]
).reset_index(drop=True)

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDistrict coordinates created successfully.")

print(f"Total districts: {len(output)}")

print("\nDistrict coordinates:")
print(output.to_string(index=False))

print(f"\nSaved to:")
print(OUTPUT_FILE)