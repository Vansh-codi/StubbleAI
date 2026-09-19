import pandas as pd
import geopandas as gpd


# ============================================================
# STUBBLEAI - 2026 LIVE DISTRICT FIRE COUNTS
# ============================================================

FIRMS_FILE = "live_firms_2026_history.csv"
BOUNDARY_FILE = "districts_punjab_haryana.geojson"
OUTPUT_FILE = "live_district_fire_2026.csv"


print("=" * 70)
print("STUBBLEAI - 2026 DISTRICT FIRE COUNTS")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load FIRMS data
# ------------------------------------------------------------

fire = pd.read_csv(FIRMS_FILE)

required_fire_columns = [
    "latitude",
    "longitude",
    "acq_date",
]

missing_fire_columns = [
    col for col in required_fire_columns
    if col not in fire.columns
]

if missing_fire_columns:
    raise ValueError(
        f"FIRMS file is missing required columns: {missing_fire_columns}"
    )

fire["acq_date"] = pd.to_datetime(
    fire["acq_date"],
    errors="coerce"
)

if fire["acq_date"].isna().any():
    raise ValueError(
        "FIRMS data contains invalid acquisition dates."
    )

if not fire["acq_date"].dt.year.eq(2026).all():
    raise ValueError(
        "live_district_fire_2026.py received non-2026 FIRMS records."
    )

if fire[["latitude", "longitude"]].isna().any().any():
    raise ValueError(
        "FIRMS data contains missing latitude/longitude values."
    )

if not fire["latitude"].between(-90, 90).all():
    raise ValueError(
        "FIRMS latitude values are outside valid geographic bounds."
    )

if not fire["longitude"].between(-180, 180).all():
    raise ValueError(
        "FIRMS longitude values are outside valid geographic bounds."
    )

print()
print("FIRMS rows:", len(fire))
print(
    "Date range:",
    fire["acq_date"].min().date(),
    "to",
    fire["acq_date"].max().date()
)


# ------------------------------------------------------------
# 2. Convert FIRMS points to GeoDataFrame
# ------------------------------------------------------------

gdf_fire = gpd.GeoDataFrame(
    fire,
    geometry=gpd.points_from_xy(
        fire["longitude"],
        fire["latitude"]
    ),
    crs="EPSG:4326"
)


# ------------------------------------------------------------
# 3. Load district boundaries
# ------------------------------------------------------------

districts = gpd.read_file(BOUNDARY_FILE)
if districts.empty:
    raise ValueError(
        "District boundary file is empty."
    )

if districts.crs is None:
    raise ValueError(
        "District boundary file has no CRS."
    )

if "state" not in districts.columns:
    raise KeyError(
        "District boundary file is missing the state column."
    )
print()
print("District polygons:", len(districts))
print("Boundary columns:")
print(districts.columns.tolist())


# ------------------------------------------------------------
# 4. Make sure both layers use same CRS
# ------------------------------------------------------------

districts = districts.to_crs("EPSG:4326")


# ------------------------------------------------------------
# 5. Standardize district name
# ------------------------------------------------------------

if "district" in districts.columns:

    district_column = "district"

elif "lgd_districtname" in districts.columns:

    district_column = "lgd_districtname"

elif "name" in districts.columns:

    district_column = "name"

else:

    raise KeyError(
        "Could not find district name column."
    )


districts = districts.rename(
    columns={district_column: "district"}
)
districts["state"] = (
    districts["state"]
    .astype(str)
    .str.strip()
)

districts["district"] = (
    districts["district"]
    .astype(str)
    .str.strip()
)

if districts["district"].eq("").any():
    raise ValueError(
        "District boundary contains blank district names."
    )

districts = districts[
    districts["state"].isin(["Punjab", "Haryana"])
].copy()

if districts.empty:
    raise ValueError(
        "No Punjab/Haryana district polygons found in boundary file."
    )

expected_district_count = 45

if districts["district"].nunique() != expected_district_count:
    duplicate_districts = districts.duplicated(
    subset=["state", "district"],
    keep=False
)

if duplicate_districts.any():
    print()
    print("Duplicate district polygons detected.")
    print(
        districts.loc[
            duplicate_districts,
            ["state", "district"]
        ].to_string(index=False)
    )
    raise ValueError(
        "Boundary file contains duplicate state-district entries."
    )
    raise ValueError(
        f"Expected {expected_district_count} Punjab/Haryana districts, "
        f"found {districts['district'].nunique()}."
    )

# ------------------------------------------------------------
# 6. Keep only required columns
# ------------------------------------------------------------

districts = districts[
    ["state", "district", "geometry"]
].copy()


# ------------------------------------------------------------
# 7. Spatial join
# ------------------------------------------------------------

joined = gpd.sjoin(
    gdf_fire,
    districts,
    how="left",
    predicate="within"
)
duplicate_matches = joined.index.duplicated(keep=False)

if duplicate_matches.any():
    print()
    print("Duplicate spatial matches detected.")
    print(
        joined.loc[
            duplicate_matches,
            ["latitude", "longitude", "acq_date", "state", "district"]
        ].head(20).to_string(index=False)
    )
    raise ValueError(
        "A FIRMS observation matched multiple district polygons."
    )

print()
print("Matched FIRMS points:", joined["district"].notna().sum())
print("Unmatched FIRMS points:", joined["district"].isna().sum())


# ------------------------------------------------------------
# 8. Keep Punjab + Haryana only
# ------------------------------------------------------------

joined = joined[
    joined["state"].isin(
        ["Punjab", "Haryana"]
    )
].copy()


print(
    "Punjab/Haryana FIRMS points:",
    len(joined)
)


# ------------------------------------------------------------
# 9. Normalize district names
# ------------------------------------------------------------

joined["state"] = (
    joined["state"]
    .astype(str)
    .str.strip()
)

joined["district"] = (
    joined["district"]
    .astype(str)
    .str.strip()
    .str.replace(
        "S.A.S Nagar",
        "S A S Nagar",
        regex=False
    )
)


# ------------------------------------------------------------
# 10. Aggregate by district and date
# ------------------------------------------------------------

daily = (
    joined
    .groupby(
        ["acq_date", "state", "district"],
        as_index=False
    )
    .size()
    .rename(
        columns={
            "acq_date": "date",
            "size": "fire_count"
        }
    )
)

daily["date"] = pd.to_datetime(daily["date"])


# ------------------------------------------------------------
# 11. Sort
# ------------------------------------------------------------

daily = daily.sort_values(
    ["date", "state", "district"]
).reset_index(drop=True)
if daily.empty:
    raise ValueError(
        "No district-level fire observations were generated."
    )

if daily["date"].isna().any():
    raise ValueError(
        "District fire output contains missing dates."
    )

if not daily["date"].dt.year.eq(2026).all():
    raise ValueError(
        "District fire output contains non-2026 dates."
    )

if daily["fire_count"].isna().any():
    raise ValueError(
        "District fire output contains missing fire counts."
    )

if (daily["fire_count"] < 0).any():
    raise ValueError(
        "District fire output contains negative fire counts."
    )

duplicate_daily = daily.duplicated(
    subset=["date", "state", "district"],
    keep=False
)

if duplicate_daily.any():
    raise ValueError(
        "Duplicate district-date fire observations detected."
    )

# ------------------------------------------------------------
# 12. Save
# ------------------------------------------------------------

daily.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 13. Summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("2026 DISTRICT FIRE COUNTS COMPLETE")
print("=" * 70)

print("Rows:", len(daily))

print(
    "Date range:",
    daily["date"].min().date(),
    "to",
    daily["date"].max().date()
)

print(
    "Districts detected:",
    daily["district"].nunique()
)

print()
print("State distribution:")
print(daily["state"].value_counts())

print()
print("District distribution:")
print(daily["district"].value_counts())

print()
print("Saved to:")
print(OUTPUT_FILE)

print("=" * 70)