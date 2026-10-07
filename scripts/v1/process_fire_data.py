import pandas as pd
import geopandas as gpd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FIRMS_FILE = BASE_DIR / "firms_punjab_haryana_2023_2025.csv"
DISTRICT_FILE = BASE_DIR / "districts_punjab_haryana.geojson"

OUTPUT_FIRE_COUNTS = BASE_DIR / "district_daily_fire_counts.csv"
OUTPUT_PANEL = BASE_DIR / "district_daily_fire_panel.csv"


# ============================================================
# 1. LOAD FIRMS DATA
# ============================================================

print("\nLoading FIRMS data...")

fires = pd.read_csv(FIRMS_FILE)

print(f"Raw FIRMS rows: {len(fires):,}")


# ============================================================
# 2. CLEAN DATE
# ============================================================

fires["acq_date"] = pd.to_datetime(
    fires["acq_date"],
    errors="coerce"
)

fires = fires.dropna(subset=["acq_date", "latitude", "longitude"])


# Keep ONLY October and November
fires = fires[
    fires["acq_date"].dt.month.isin([10, 11])
].copy()

print(f"Rows after October-November filtering: {len(fires):,}")


# ============================================================
# 3. CONVERT FIRE POINTS TO GEOPANDAS
# ============================================================

print("\nConverting FIRMS points to geographic points...")

fires_gdf = gpd.GeoDataFrame(
    fires,
    geometry=gpd.points_from_xy(
        fires["longitude"],
        fires["latitude"]
    ),
    crs="EPSG:4326"
)


# ============================================================
# 4. LOAD DISTRICT BOUNDARIES
# ============================================================

print("Loading district boundaries...")

districts = gpd.read_file(DISTRICT_FILE)

print(f"District polygons loaded: {len(districts)}")

print("\nBoundary columns:")
print(districts.columns.tolist())


# ============================================================
# 5. KEEP ONLY PUNJAB + HARYANA
# ============================================================

districts["state_clean"] = (
    districts["state"]
    .astype(str)
    .str.strip()
    .str.lower()
)

districts = districts[
    districts["state_clean"].isin(["punjab", "haryana"])
].copy()

print(f"\nPunjab + Haryana districts: {len(districts)}")

print(
    districts[
        ["name", "state"]
    ].sort_values(["state", "name"]).to_string(index=False)
)


# ============================================================
# 6. MAKE CRS MATCH
# ============================================================

if districts.crs != fires_gdf.crs:
    print("\nConverting district CRS to EPSG:4326...")
    districts = districts.to_crs(fires_gdf.crs)


# ============================================================
# 7. SELECT USEFUL DISTRICT COLUMNS
# ============================================================

district_columns = [
    "name",
    "state",
    "lgd_districtcode",
    "geometry"
]

districts_small = districts[
    [c for c in district_columns if c in districts.columns]
].copy()


# ============================================================
# 8. SPATIAL JOIN
# ============================================================

print("\nAssigning FIRMS points to districts...")

joined = gpd.sjoin(
    fires_gdf,
    districts_small,
    how="left",
    predicate="within"
)

print("Spatial join completed.")


# ============================================================
# 9. CHECK UNMATCHED POINTS
# ============================================================

unmatched = joined["name"].isna().sum()

print(f"\nUnmatched FIRMS points: {unmatched:,}")

if unmatched > 0:
    print(
        "WARNING: Some fire points were outside the downloaded "
        "Punjab/Haryana district polygons."
    )


# Keep only successfully assigned districts
joined = joined.dropna(subset=["name"]).copy()


# ============================================================
# 10. CREATE DAILY DISTRICT FIRE COUNTS
# ============================================================

print("\nCreating daily district fire counts...")

joined["date"] = joined["acq_date"].dt.date

daily_counts = (
    joined
    .groupby(
        ["date", "state", "name"],
        as_index=False
    )
    .size()
    .rename(columns={"size": "fire_count"})
)

daily_counts["date"] = pd.to_datetime(daily_counts["date"])


# Rename district column
daily_counts = daily_counts.rename(
    columns={"name": "district"}
)


# Sort
daily_counts = daily_counts.sort_values(
    ["date", "state", "district"]
).reset_index(drop=True)


# ============================================================
# 11. SAVE RAW DAILY COUNTS
# ============================================================

daily_counts.to_csv(
    OUTPUT_FIRE_COUNTS,
    index=False
)

print(
    f"\nSaved daily fire counts to:\n"
    f"{OUTPUT_FIRE_COUNTS}"
)


# ============================================================
# 12. CREATE COMPLETE DISTRICT-DAY PANEL
# ============================================================
#
# This is important.
#
# If a district had ZERO detected fires on a particular day,
# that day may not appear in the grouped data.
#
# For machine learning, we need those days as fire_count = 0.
# ============================================================

print("\nCreating complete district-day panel...")

years = [2023, 2024, 2025]

season_dates = pd.concat(
    [
        pd.DataFrame({
            "date": pd.date_range(
                start=f"{year}-10-01",
                end=f"{year}-11-30",
                freq="D"
            )
        })
        for year in years
    ],
    ignore_index=True
)

district_info = (
    districts_small[
        ["name", "state"]
    ]
    .drop_duplicates()
    .rename(columns={"name": "district"})
)

district_info["key"] = 1

season_dates["key"] = 1

# Cartesian product:
# every district × every date
panel = district_info.merge(
    season_dates,
    on="key"
).drop(columns="key")


# Merge actual fire counts
panel = panel.merge(
    daily_counts,
    on=["date", "state", "district"],
    how="left"
)


# Missing means zero detected fires
panel["fire_count"] = (
    panel["fire_count"]
    .fillna(0)
    .astype(int)
)


# Sort
panel = panel.sort_values(
    ["date", "state", "district"]
).reset_index(drop=True)


# ============================================================
# 13. ADD BASIC TIME FEATURES
# ============================================================

panel["year"] = panel["date"].dt.year
panel["month"] = panel["date"].dt.month
panel["day"] = panel["date"].dt.day

panel["day_of_season"] = (
    panel["date"] -
    pd.to_datetime(
        panel["year"].astype(str) + "-10-01"
    )
).dt.days + 1


# ============================================================
# 14. SAVE COMPLETE PANEL
# ============================================================

panel.to_csv(
    OUTPUT_PANEL,
    index=False
)


# ============================================================
# 15. VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

print(f"\nRaw FIRMS rows:             {len(fires):,}")
print(f"Spatially matched rows:     {len(joined):,}")
print(f"Daily aggregated rows:      {len(daily_counts):,}")
print(f"Final district-day rows:    {len(panel):,}")

print(
    f"\nDate range: "
    f"{panel['date'].min().date()} → "
    f"{panel['date'].max().date()}"
)

print("\nStates:")
print(panel["state"].value_counts())


print("\nFire count statistics:")
print(panel["fire_count"].describe())


print("\nTop 15 district-days:")
print(
    panel[
        [
            "date",
            "state",
            "district",
            "fire_count"
        ]
    ]
    .sort_values(
        "fire_count",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)


print("\nYear-wise fire totals:")

year_totals = (
    panel
    .groupby("year")["fire_count"]
    .sum()
)

print(year_totals)


print("\n" + "=" * 60)
print("PROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"\nOutput 1:\n{OUTPUT_FIRE_COUNTS}"
)

print(
    f"\nOutput 2:\n{OUTPUT_PANEL}"
)