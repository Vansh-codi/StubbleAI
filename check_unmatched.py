import pandas as pd
import geopandas as gpd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FIRMS_FILE = BASE_DIR / "firms_punjab_haryana_2023_2025.csv"
DISTRICT_FILE = BASE_DIR / "districts_punjab_haryana.geojson"

print("Loading data...")

fires = pd.read_csv(FIRMS_FILE)

fires["acq_date"] = pd.to_datetime(
    fires["acq_date"],
    errors="coerce"
)

fires = fires[
    fires["acq_date"].dt.month.isin([10, 11])
].copy()

fires_gdf = gpd.GeoDataFrame(
    fires,
    geometry=gpd.points_from_xy(
        fires["longitude"],
        fires["latitude"]
    ),
    crs="EPSG:4326"
)

districts = gpd.read_file(DISTRICT_FILE)

districts["state_clean"] = (
    districts["state"]
    .astype(str)
    .str.strip()
    .str.lower()
)

districts = districts[
    districts["state_clean"].isin(
        ["punjab", "haryana"]
    )
].copy()

districts = districts.to_crs("EPSG:4326")

print("\nChecking bounding boxes...")

print(
    "FIRMS longitude:",
    fires_gdf.geometry.x.min(),
    "to",
    fires_gdf.geometry.x.max()
)

print(
    "FIRMS latitude:",
    fires_gdf.geometry.y.min(),
    "to",
    fires_gdf.geometry.y.max()
)

print("\nDistrict boundary:")
print(
    districts.total_bounds
)

print("\nPerforming spatial join...")

joined = gpd.sjoin(
    fires_gdf,
    districts[["name", "state", "geometry"]],
    how="left",
    predicate="within"
)

unmatched = joined[joined["name"].isna()].copy()

print(
    f"\nTotal points: {len(joined):,}"
)

print(
    f"Matched: {len(joined) - len(unmatched):,}"
)

print(
    f"Unmatched: {len(unmatched):,}"
)

print("\nSample unmatched coordinates:")

print(
    unmatched[
        ["latitude", "longitude", "acq_date"]
    ].head(20).to_string(index=False)
)

print("\nUnmatched coordinate ranges:")

print(
    "Longitude:",
    unmatched["longitude"].min(),
    "to",
    unmatched["longitude"].max()
)

print(
    "Latitude:",
    unmatched["latitude"].min(),
    "to",
    unmatched["latitude"].max()
)

print("\nDone.")