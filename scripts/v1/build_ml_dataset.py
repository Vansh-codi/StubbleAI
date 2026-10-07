import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FIRE_FILE = BASE_DIR / "district_daily_fire_panel.csv"
WEATHER_FILE = BASE_DIR / "district_daily_weather.csv"

OUTPUT_FILE = BASE_DIR / "ml_dataset_base.csv"


# ============================================================
# 1. LOAD
# ============================================================

print("Loading fire dataset...")
fire = pd.read_csv(FIRE_FILE)

print("Loading weather dataset...")
weather = pd.read_csv(WEATHER_FILE)

print(f"Fire rows:    {len(fire):,}")
print(f"Weather rows: {len(weather):,}")


# ============================================================
# 2. DATE TYPES
# ============================================================

fire["date"] = pd.to_datetime(fire["date"])
weather["date"] = pd.to_datetime(weather["date"])


# ============================================================
# 3. CHECK UNIQUENESS
# ============================================================

fire_duplicates = fire.duplicated(
    ["date", "state", "district"]
).sum()

weather_duplicates = weather.duplicated(
    ["date", "state", "district"]
).sum()

print("\nDuplicate district-days:")
print("Fire:", fire_duplicates)
print("Weather:", weather_duplicates)


# ============================================================
# 4. SELECT WEATHER VARIABLES
# ============================================================

weather = weather[
    [
        "date",
        "state",
        "district",
        "T2M",
        "RH2M",
        "WS2M",
        "PRECTOTCORR"
    ]
].copy()


# ============================================================
# 5. MERGE
# ============================================================

print("\nMerging fire + weather...")

df = fire.merge(
    weather,
    on=["date", "state", "district"],
    how="inner",
    validate="one_to_one"
)

print(f"Merged rows: {len(df):,}")


# ============================================================
# 6. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    ["district", "state", "date"]
).reset_index(drop=True)


# ============================================================
# 7. HISTORICAL FIRE LAGS
# ============================================================

print("\nCreating lag features...")

group = df.groupby(
    ["state", "district"],
    sort=False
)

df["fire_lag_1d"] = group["fire_count"].shift(1)
df["fire_lag_3d"] = group["fire_count"].shift(3)
df["fire_lag_7d"] = group["fire_count"].shift(7)


# ============================================================
# 8. HISTORICAL ROLLING FEATURES
# ============================================================
#
# IMPORTANT:
# shift(1) happens BEFORE rolling.
#
# Therefore today's rolling feature only uses
# information available BEFORE today.
# ============================================================

df["fire_mean_3d"] = (
    group["fire_count"]
    .shift(1)
    .groupby(
        [df["state"], df["district"]]
    )
    .rolling(3)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

df["fire_mean_7d"] = (
    group["fire_count"]
    .shift(1)
    .groupby(
        [df["state"], df["district"]]
    )
    .rolling(7)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)


# ============================================================
# 9. NEXT-DAY TARGET
# ============================================================

print("Creating next-day target...")

df["next_day_fire_count"] = (
    group["fire_count"].shift(-1)
)


# ============================================================
# 10. CHECK TEMPORAL LEAKAGE
# ============================================================

print("\nFeature examples:")

print(
    df[
        [
            "date",
            "district",
            "fire_count",
            "fire_lag_1d",
            "fire_lag_3d",
            "fire_lag_7d",
            "fire_mean_3d",
            "fire_mean_7d",
            "next_day_fire_count"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# 11. BASIC DATA VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("ML DATASET VALIDATION")
print("=" * 60)

print("\nRows:", len(df))

print(
    "\nDate range:",
    df["date"].min().date(),
    "→",
    df["date"].max().date()
)

print(
    "\nDistricts:",
    df["district"].nunique()
)

print(
    "\nStates:",
    df["state"].nunique()
)

print("\nMissing values:")

feature_columns = [
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
    "fire_lag_1d",
    "fire_lag_3d",
    "fire_lag_7d",
    "fire_mean_3d",
    "fire_mean_7d",
    "next_day_fire_count"
]

print(
    df[feature_columns].isna().sum()
)


# ============================================================
# 12. YEAR SPLIT CHECK
# ============================================================

print("\nRows by year:")

print(
    df.groupby(
        df["date"].dt.year
    ).size()
)


# ============================================================
# 13. FIRE DISTRIBUTION
# ============================================================

print("\nFire-count percentiles:")

print(
    df["fire_count"].describe(
        percentiles=[
            0.50,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            0.99
        ]
    )
)


# ============================================================
# 14. REMOVE ROWS WITHOUT FULL HISTORY / TARGET
# ============================================================

before = len(df)

df = df.dropna(
    subset=[
        "fire_lag_1d",
        "fire_lag_3d",
        "fire_lag_7d",
        "fire_mean_3d",
        "fire_mean_7d",
        "next_day_fire_count"
    ]
).copy()

print(
    f"\nRemoved {before - len(df):,} "
    "rows without sufficient historical information."
)

print(
    f"Final base ML rows: {len(df):,}"
)


# ============================================================
# 15. SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nBUILD COMPLETE.")