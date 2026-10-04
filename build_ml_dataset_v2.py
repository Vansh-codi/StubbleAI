import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STUBBLEAI V2 - ML DATASET BUILDER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

FIRE_FILE = BASE_DIR / "district_daily_fire_panel_v2.csv"
WEATHER_FILE = BASE_DIR / "district_daily_weather.csv"
OUTPUT_FILE = BASE_DIR / "ml_dataset_v2.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("STUBBLEAI V2 - BUILDING ML DATASET")
print("=" * 70)

print("\n[1/8] Loading fire panel...")

fire = pd.read_csv(FIRE_FILE)
fire["date"] = pd.to_datetime(fire["date"])

print(f"Fire panel shape: {fire.shape}")


print("\n[2/8] Loading weather data...")

weather = pd.read_csv(WEATHER_FILE)
weather["date"] = pd.to_datetime(weather["date"])

print(f"Weather shape: {weather.shape}")


# ============================================================
# 2. VALIDATE INPUTS
# ============================================================

print("\n[3/8] Validating input data...")

required_fire_cols = [
    "state",
    "district",
    "date",
    "fire_count",
    "frp_sum",
    "frp_mean",
    "frp_max",
    "high_confidence_fire_count",
    "day_fire_count",
    "night_fire_count",
]

required_weather_cols = [
    "date",
    "state",
    "district",
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
]

missing_fire = [
    col for col in required_fire_cols
    if col not in fire.columns
]

missing_weather = [
    col for col in required_weather_cols
    if col not in weather.columns
]

if missing_fire:
    raise ValueError(
        f"Missing fire columns: {missing_fire}"
    )

if missing_weather:
    raise ValueError(
        f"Missing weather columns: {missing_weather}"
    )


# ------------------------------------------------------------
# Duplicate checks
# ------------------------------------------------------------

fire_duplicates = fire.duplicated(
    subset=["state", "district", "date"]
).sum()

weather_duplicates = weather.duplicated(
    subset=["state", "district", "date"]
).sum()

print(f"Fire duplicate district-days: {fire_duplicates}")
print(f"Weather duplicate district-days: {weather_duplicates}")

if fire_duplicates > 0:
    raise ValueError(
        "Duplicate district/date rows found in fire panel."
    )

if weather_duplicates > 0:
    raise ValueError(
        "Duplicate district/date rows found in weather data."
    )


# ============================================================
# 3. PREPARE BASE DATA
# ============================================================

print("\n[4/8] Preparing base district-day dataset...")

weather = weather[
    [
        "date",
        "state",
        "district",
        "T2M",
        "RH2M",
        "WS2M",
        "PRECTOTCORR",
    ]
].copy()


# Exact date/district merge.
df = fire.merge(
    weather,
    on=["date", "state", "district"],
    how="inner",
    validate="one_to_one",
)


df = df.sort_values(
    ["state", "district", "date"]
).reset_index(drop=True)


print(f"Merged dataset shape: {df.shape}")

if df.empty:
    raise ValueError("Merged fire/weather dataset is empty.")


# ============================================================
# 4. CREATE TEMPORAL FEATURES
# ============================================================

print("\n[5/8] Creating temporal and historical features...")


# ------------------------------------------------------------
# Calendar features
# ------------------------------------------------------------

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_year"] = df["date"].dt.dayofyear


# Cyclic annual seasonality.
df["sin_year"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.25
)

df["cos_year"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.25
)


# ------------------------------------------------------------
# Date-based helper
# ------------------------------------------------------------

keys = ["state", "district", "date"]

base_cols = [
    "state",
    "district",
    "date",
    "fire_count",
    "frp_sum",
]


base = df[base_cols].copy()


# ============================================================
# 5. HISTORICAL FIRE LAGS
# ============================================================

print("Creating fire and FRP lag features...")


LAGS = [1, 2, 3, 5, 7, 14]

for lag in LAGS:

    lag_data = base[
        ["state", "district", "date", "fire_count", "frp_sum"]
    ].copy()

    lag_data["date"] = (
        lag_data["date"]
        + pd.Timedelta(days=lag)
    )

    lag_data = lag_data.rename(
        columns={
            "fire_count": f"fire_lag_{lag}d",
            "frp_sum": f"frp_lag_{lag}d",
        }
    )

    df = df.merge(
        lag_data[
            [
                "state",
                "district",
                "date",
                f"fire_lag_{lag}d",
                f"frp_lag_{lag}d",
            ]
        ],
        on=["state", "district", "date"],
        how="left",
        validate="one_to_one",
    )


# ============================================================
# 6. HONEST RECENT / SPARSE FEATURES
# ============================================================

print("Creating recent and sparse historical summaries...")


# ------------------------------------------------------------
# True recent 3-day mean
#
# Uses:
# t-1, t-2, t-3
# ------------------------------------------------------------

df["fire_mean_3d"] = df[
    [
        "fire_lag_1d",
        "fire_lag_2d",
        "fire_lag_3d",
    ]
].mean(axis=1)

df["frp_mean_3d"] = df[
    [
        "frp_lag_1d",
        "frp_lag_2d",
        "frp_lag_3d",
    ]
].mean(axis=1)


# ------------------------------------------------------------
# Sparse 7-day summary
#
# Uses:
# t-1, t-2, t-3, t-5, t-7
#
# This is intentionally NOT called "7d mean".
# ------------------------------------------------------------

df["fire_sparse_mean_7d"] = df[
    [
        "fire_lag_1d",
        "fire_lag_2d",
        "fire_lag_3d",
        "fire_lag_5d",
        "fire_lag_7d",
    ]
].mean(axis=1)

df["frp_sparse_mean_7d"] = df[
    [
        "frp_lag_1d",
        "frp_lag_2d",
        "frp_lag_3d",
        "frp_lag_5d",
        "frp_lag_7d",
    ]
].mean(axis=1)


# ------------------------------------------------------------
# Sparse 14-day summary
#
# Uses:
# t-1, t-2, t-3, t-5, t-7, t-14
#
# Again, intentionally NOT called "14d mean".
# ------------------------------------------------------------

df["fire_sparse_mean_14d"] = df[
    [
        "fire_lag_1d",
        "fire_lag_2d",
        "fire_lag_3d",
        "fire_lag_5d",
        "fire_lag_7d",
        "fire_lag_14d",
    ]
].mean(axis=1)

df["frp_sparse_mean_14d"] = df[
    [
        "frp_lag_1d",
        "frp_lag_2d",
        "frp_lag_3d",
        "frp_lag_5d",
        "frp_lag_7d",
        "frp_lag_14d",
    ]
].mean(axis=1)


# ============================================================
# 7. FUTURE TARGETS
# ============================================================

print("Creating future prediction targets...")


HORIZONS = [1, 2, 3, 5, 7]

for horizon in HORIZONS:

    future_data = base[
        ["state", "district", "date", "fire_count", "frp_sum"]
    ].copy()

    # Move future observation backward so it aligns
    # with the prediction date.
    future_data["date"] = (
        future_data["date"]
        - pd.Timedelta(days=horizon)
    )

    future_data = future_data.rename(
        columns={
            "fire_count": f"fire_count_t_plus_{horizon}d",
            "frp_sum": f"frp_sum_t_plus_{horizon}d",
        }
    )

    df = df.merge(
        future_data[
            [
                "state",
                "district",
                "date",
                f"fire_count_t_plus_{horizon}d",
                f"frp_sum_t_plus_{horizon}d",
            ]
        ],
        on=["state", "district", "date"],
        how="left",
        validate="one_to_one",
    )


# ============================================================
# 8. FINAL CLEANING / VALIDATION
# ============================================================

print("\n[6/8] Cleaning and validating final dataset...")


# We need 14 days of historical information
# and 7 days of future information.
#
# Therefore rows at the beginning/end of each
# seasonal block naturally lose required values.

required_history = [
    "fire_lag_1d",
    "fire_lag_2d",
    "fire_lag_3d",
    "fire_lag_5d",
    "fire_lag_7d",
    "fire_lag_14d",
    "frp_lag_1d",
    "frp_lag_2d",
    "frp_lag_3d",
    "frp_lag_5d",
    "frp_lag_7d",
    "frp_lag_14d",
]

required_targets = []

for horizon in HORIZONS:
    required_targets.extend(
        [
            f"fire_count_t_plus_{horizon}d",
            f"frp_sum_t_plus_{horizon}d",
        ]
    )


required_columns = (
    required_history
    + required_targets
)


before_drop = len(df)

df = df.dropna(
    subset=required_columns
).copy()

after_drop = len(df)

print(
    f"Rows before history/target filtering: {before_drop}"
)

print(
    f"Rows after history/target filtering: {after_drop}"
)

print(
    f"Rows removed: {before_drop - after_drop}"
)


# ------------------------------------------------------------
# Final duplicate check
# ------------------------------------------------------------

duplicates_final = df.duplicated(
    subset=["state", "district", "date"]
).sum()

print(
    f"Final duplicate district-days: {duplicates_final}"
)

if duplicates_final > 0:
    raise ValueError(
        "Final dataset contains duplicate district-days."
    )


# ------------------------------------------------------------
# Missing-value check
# ------------------------------------------------------------

missing_total = df.isna().sum().sum()

print(
    f"Total missing values: {missing_total}"
)

if missing_total > 0:
    missing = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print("\nColumns with missing values:")
    print(missing[missing > 0])

    raise ValueError(
        "Final dataset still contains missing values."
    )


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

print("\n[7/8] Organizing final columns...")


feature_columns = [
    "state",
    "district",
    "date",

    # Calendar
    "year",
    "month",
    "day",
    "day_of_year",
    "sin_year",
    "cos_year",

    # Current fire state
    "fire_count",
    "frp_sum",
    "frp_mean",
    "frp_max",
    "high_confidence_fire_count",
    "day_fire_count",
    "night_fire_count",
    "bright_ti4_mean",
    "bright_ti4_max",
    "bright_ti5_mean",

    # Weather
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",

    # Fire history
    "fire_lag_1d",
    "fire_lag_2d",
    "fire_lag_3d",
    "fire_lag_5d",
    "fire_lag_7d",
    "fire_lag_14d",

    # FRP history
    "frp_lag_1d",
    "frp_lag_2d",
    "frp_lag_3d",
    "frp_lag_5d",
    "frp_lag_7d",
    "frp_lag_14d",

    # Historical summaries
    "fire_mean_3d",
    "fire_sparse_mean_7d",
    "fire_sparse_mean_14d",

    "frp_mean_3d",
    "frp_sparse_mean_7d",
    "frp_sparse_mean_14d",
]


target_columns = []

for horizon in HORIZONS:

    target_columns.extend(
        [
            f"fire_count_t_plus_{horizon}d",
            f"frp_sum_t_plus_{horizon}d",
        ]
    )


final_columns = feature_columns + target_columns


missing_final_columns = [
    col
    for col in final_columns
    if col not in df.columns
]

if missing_final_columns:
    raise ValueError(
        "Missing expected final columns: "
        f"{missing_final_columns}"
    )


df = df[final_columns].copy()


# ============================================================
# SAVE
# ============================================================

print("\n[8/8] Saving V2 ML dataset...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V2 ML DATASET COMPLETE")
print("=" * 70)

print(f"Output file: {OUTPUT_FILE}")
print(f"Shape: {df.shape}")

print(
    f"Unique states: {df['state'].nunique()}"
)

print(
    f"Unique districts: {df['district'].nunique()}"
)

print(
    f"Date range: "
    f"{df['date'].min().date()} → "
    f"{df['date'].max().date()}"
)

print(
    f"Missing values: {df.isna().sum().sum()}"
)

print(
    f"Duplicate district-days: "
    f"{df.duplicated(['state', 'district', 'date']).sum()}"
)


print("\nFinal columns:")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")


print("\nSample rows:")

print(
    df.head(5).to_string(index=False)
)

print("\nDataset saved successfully.")