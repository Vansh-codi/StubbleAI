import pandas as pd
import geopandas as gpd
from pathlib import Path


# ============================================================
# STUBBLEAI V2 — FIRE DATA PROCESSING
# ============================================================
# Purpose:
# Convert raw FIRMS detections into a richer district-day dataset.
#
# V1 remains untouched.
#
# V2 adds:
#   - fire_count
#   - FRP sum / mean / max
#   - high-confidence fire count
#   - day/night fire counts
#   - brightness summaries
#
# Current test scope:
# Punjab + Haryana
# October-November
# 2023-2025
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIRMS_FILE = PROJECT_ROOT / "data" / "raw" / "firms" / "firms_punjab_haryana_2023_2025.csv"
DISTRICTS_FILE = PROJECT_ROOT / "data" / "processed" / "v1" / "districts_punjab_haryana.geojson"

OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "v2" / "district_daily_fire_panel_v2.csv"


# ------------------------------------------------------------
# 1. Load raw FIRMS data
# ------------------------------------------------------------

print("\n[1/8] Loading FIRMS data...")

firms = pd.read_csv(FIRMS_FILE)

print(f"Raw FIRMS rows: {len(firms):,}")
print("Columns:")
print(firms.columns.tolist())


# ------------------------------------------------------------
# 2. Inspect important categorical fields
# ------------------------------------------------------------

print("\n[2/8] Inspecting FIRMS fields...")

for column in ["confidence", "daynight", "satellite", "query_source"]:
    if column in firms.columns:
        print(f"\n{column}:")
        print(firms[column].value_counts(dropna=False))


# ------------------------------------------------------------
# 3. Basic cleaning
# ------------------------------------------------------------

print("\n[3/8] Cleaning data...")

firms["acq_date"] = pd.to_datetime(
    firms["acq_date"],
    errors="coerce"
)

firms["frp"] = pd.to_numeric(
    firms["frp"],
    errors="coerce"
)

firms["bright_ti4"] = pd.to_numeric(
    firms["bright_ti4"],
    errors="coerce"
)

firms["bright_ti5"] = pd.to_numeric(
    firms["bright_ti5"],
    errors="coerce"
)

# Keep October and November only
firms = firms[
    firms["acq_date"].dt.month.isin([10, 11])
].copy()

# Keep valid coordinates
firms = firms.dropna(
    subset=["latitude", "longitude", "acq_date"]
).copy()

print(f"Rows after cleaning/filtering: {len(firms):,}")


# ------------------------------------------------------------
# 4. Convert FIRMS points to GeoDataFrame
# ------------------------------------------------------------

print("\n[4/8] Performing district spatial join...")

fire_gdf = gpd.GeoDataFrame(
    firms,
    geometry=gpd.points_from_xy(
        firms["longitude"],
        firms["latitude"]
    ),
    crs="EPSG:4326"
)


districts = gpd.read_file(DISTRICTS_FILE)

# Make sure CRS matches
if districts.crs != fire_gdf.crs:
    districts = districts.to_crs(fire_gdf.crs)


# Keep only Punjab + Haryana
if "state" in districts.columns:
    districts = districts[
        districts["state"].isin(["Punjab", "Haryana"])
    ].copy()

elif "STATE" in districts.columns:
    districts = districts[
        districts["STATE"].isin(["Punjab", "Haryana"])
    ].copy()


# Find district-name column
possible_district_columns = [
    "name",
    "district",
    "DISTRICT",
    "district_name",
    "NAME"
]

district_column = None

for col in possible_district_columns:
    if col in districts.columns:
        district_column = col
        break

if district_column is None:
    raise ValueError(
        "Could not find district name column in GeoJSON."
    )


# Find state column
possible_state_columns = [
    "state",
    "STATE",
    "state_name",
    "ST_NM"
]

state_column = None

for col in possible_state_columns:
    if col in districts.columns:
        state_column = col
        break

if state_column is None:
    raise ValueError(
        "Could not find state column in GeoJSON."
    )


districts = districts[
    [district_column, state_column, "geometry"]
].copy()

districts = districts.rename(
    columns={
        district_column: "district",
        state_column: "state"
    }
)


# Spatial join
joined = gpd.sjoin(
    fire_gdf,
    districts,
    how="inner",
    predicate="within"
)

print(f"Matched FIRMS detections: {len(joined):,}")


# ------------------------------------------------------------
# 5. Normalize categorical fields
# ------------------------------------------------------------

print("\n[5/8] Preparing fire attributes...")

# FIRMS confidence can be:
#   l = low
#   n = nominal
#   h = high
#
# We define high-confidence strictly as "h".
joined["high_confidence"] = (
    joined["confidence"]
    .astype(str)
    .str.lower()
    .eq("h")
)

joined["is_day"] = (
    joined["daynight"]
    .astype(str)
    .str.upper()
    .eq("D")
)

joined["is_night"] = (
    joined["daynight"]
    .astype(str)
    .str.upper()
    .eq("N")
)


# ------------------------------------------------------------
# 6. Aggregate FIRMS detections → district-day
# ------------------------------------------------------------

print("\n[6/8] Creating district-day fire features...")

joined["date"] = joined["acq_date"].dt.normalize()

daily = (
    joined
    .groupby(
        ["state", "district", "date"],
        as_index=False
    )
    .agg(
        fire_count=("frp", "size"),

        frp_sum=("frp", "sum"),
        frp_mean=("frp", "mean"),
        frp_max=("frp", "max"),

        high_confidence_fire_count=(
            "high_confidence",
            "sum"
        ),

        day_fire_count=(
            "is_day",
            "sum"
        ),

        night_fire_count=(
            "is_night",
            "sum"
        ),

        bright_ti4_mean=(
            "bright_ti4",
            "mean"
        ),

        bright_ti4_max=(
            "bright_ti4",
            "max"
        ),

        bright_ti5_mean=(
            "bright_ti5",
            "mean"
        )
    )
)


# ------------------------------------------------------------
# 7. Create complete district × date panel
# ------------------------------------------------------------

print("\n[7/8] Creating complete district-date panel...")

years = [2023, 2024, 2025]

all_dates = pd.concat(
    [
        pd.DataFrame(
            {
                "date": pd.date_range(
                    f"{year}-10-01",
                    f"{year}-11-30",
                    freq="D"
                )
            }
        )
        for year in years
    ],
    ignore_index=True
)

district_list = districts[
    ["state", "district"]
].drop_duplicates()

district_list["key"] = 1
all_dates["key"] = 1

panel = district_list.merge(
    all_dates,
    on="key"
).drop(columns="key")


# Merge observed fire data
panel = panel.merge(
    daily,
    on=["state", "district", "date"],
    how="left"
)


# ------------------------------------------------------------
# 8. Fill no-detection days
# ------------------------------------------------------------

count_columns = [
    "fire_count",
    "high_confidence_fire_count",
    "day_fire_count",
    "night_fire_count"
]

for column in count_columns:
    panel[column] = (
        panel[column]
        .fillna(0)
        .astype(int)
    )


# For FRP / brightness:
# no fire detections means zero fire intensity.
intensity_columns = [
    "frp_sum",
    "frp_mean",
    "frp_max",
    "bright_ti4_mean",
    "bright_ti4_max",
    "bright_ti5_mean"
]

for column in intensity_columns:
    panel[column] = (
        panel[column]
        .fillna(0)
    )


# Calendar features
panel["year"] = panel["date"].dt.year
panel["month"] = panel["date"].dt.month
panel["day"] = panel["date"].dt.day
panel["day_of_season"] = (
    panel["date"].dt.dayofyear
    - panel["date"].dt.year.map(
        lambda y: pd.Timestamp(f"{y}-10-01").dayofyear
    )
    + 1
)


# Sort
panel = panel.sort_values(
    ["state", "district", "date"]
).reset_index(drop=True)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

panel.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# Validation summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("V2 FIRE DATASET CREATED")
print("=" * 60)

print(f"Output: {OUTPUT_FILE}")
print(f"Rows: {len(panel):,}")
print(f"Columns: {len(panel.columns)}")

print("\nColumns:")
for column in panel.columns:
    print(f"  - {column}")


print("\nDate range:")
print(panel["date"].min())
print(panel["date"].max())


print("\nStates:")
print(panel["state"].value_counts())


print("\nDistricts:")
print(panel["district"].nunique())


print("\nFire detections:")
print(f"Total fire detections: {panel['fire_count'].sum():,}")

print(
    f"Total high-confidence detections: "
    f"{panel['high_confidence_fire_count'].sum():,}"
)

print(
    f"Total day detections: "
    f"{panel['day_fire_count'].sum():,}"
)

print(
    f"Total night detections: "
    f"{panel['night_fire_count'].sum():,}"
)


print("\nSample:")
print(panel.head(10).to_string(index=False))


print("\nV2 processing complete.")