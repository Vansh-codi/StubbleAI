import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STUBBLEAI V2.3 - TREND FEATURE EXPERIMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "v2" / "ml_dataset_v2.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "research"
    / "v2"
    / "v2_3"
    / "ml_dataset_v2_3_trend.csv"
)


print("=" * 70)
print("STUBBLEAI V2.3 - TREND FEATURE EXPERIMENT")
print("=" * 70)


# ============================================================
# 1. LOAD FROZEN V2 DATASET
# ============================================================

print("\n[1/6] Loading frozen V2 ML dataset...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print(f"Input shape: {df.shape}")


# ============================================================
# 2. VALIDATE REQUIRED COLUMNS
# ============================================================

print("\n[2/6] Validating required columns...")

required_columns = [
    "state",
    "district",
    "date",
    "fire_count",
    "frp_sum",
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

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# 3. CREATE TREND / CHANGE FEATURES
# ============================================================

print("\n[3/6] Creating trend and change features...")


# ------------------------------------------------------------
# Fire-count change
#
# Positive value  -> recent activity increased
# Negative value  -> recent activity decreased
# ------------------------------------------------------------

df["fire_change_1d"] = (
    df["fire_lag_1d"]
    - df["fire_lag_2d"]
)

df["fire_change_3d"] = (
    df["fire_lag_1d"]
    - df["fire_lag_3d"]
)

df["fire_change_7d"] = (
    df["fire_lag_1d"]
    - df["fire_lag_7d"]
)


# ------------------------------------------------------------
# FRP change
# ------------------------------------------------------------

df["frp_change_1d"] = (
    df["frp_lag_1d"]
    - df["frp_lag_2d"]
)

df["frp_change_3d"] = (
    df["frp_lag_1d"]
    - df["frp_lag_3d"]
)

df["frp_change_7d"] = (
    df["frp_lag_1d"]
    - df["frp_lag_7d"]
)


# ------------------------------------------------------------
# Relative fire activity
#
# Small epsilon prevents division by zero.
# ------------------------------------------------------------

EPSILON = 1e-6

df["fire_ratio_recent_vs_3d"] = (
    df["fire_lag_1d"]
    / (df["fire_mean_3d"] + EPSILON)
)

df["fire_ratio_recent_vs_7d"] = (
    df["fire_lag_1d"]
    / (df["fire_sparse_mean_7d"] + EPSILON)
)

df["frp_ratio_recent_vs_3d"] = (
    df["frp_lag_1d"]
    / (df["frp_mean_3d"] + EPSILON)
)

df["frp_ratio_recent_vs_7d"] = (
    df["frp_lag_1d"]
    / (df["frp_sparse_mean_7d"] + EPSILON)
)


# ------------------------------------------------------------
# Simple linear trend estimates
#
# These use only historical observations:
#
# fire_trend_3d:
#   slope across t-3, t-2, t-1
#
# fire_trend_7d:
#   slope across available sparse historical points
#
# The same is done for FRP.
# ------------------------------------------------------------

def slope_3(values):
    y = np.asarray(values, dtype=float)

    x = np.arange(len(y), dtype=float)

    if np.all(y == y[0]):
        return 0.0

    return float(
        np.polyfit(x, y, 1)[0]
    )


df["fire_trend_3d"] = df.apply(
    lambda row: slope_3([
        row["fire_lag_3d"],
        row["fire_lag_2d"],
        row["fire_lag_1d"],
    ]),
    axis=1,
)

df["frp_trend_3d"] = df.apply(
    lambda row: slope_3([
        row["frp_lag_3d"],
        row["frp_lag_2d"],
        row["frp_lag_1d"],
    ]),
    axis=1,
)


def slope_5(values):
    y = np.asarray(values, dtype=float)

    x = np.arange(len(y), dtype=float)

    if np.all(y == y[0]):
        return 0.0

    return float(
        np.polyfit(x, y, 1)[0]
    )


df["fire_trend_7d"] = df.apply(
    lambda row: slope_5([
        row["fire_lag_7d"],
        row["fire_lag_5d"],
        row["fire_lag_3d"],
        row["fire_lag_2d"],
        row["fire_lag_1d"],
    ]),
    axis=1,
)

df["frp_trend_7d"] = df.apply(
    lambda row: slope_5([
        row["frp_lag_7d"],
        row["frp_lag_5d"],
        row["frp_lag_3d"],
        row["frp_lag_2d"],
        row["frp_lag_1d"],
    ]),
    axis=1,
)


# ------------------------------------------------------------
# Acceleration / deceleration
#
# Difference between the most recent change and
# the previous short-term change.
# ------------------------------------------------------------

df["fire_acceleration"] = (
    df["fire_change_1d"]
    - (
        df["fire_lag_2d"]
        - df["fire_lag_3d"]
    )
)

df["frp_acceleration"] = (
    df["frp_change_1d"]
    - (
        df["frp_lag_2d"]
        - df["frp_lag_3d"]
    )
)


# ============================================================
# 4. VALIDATION
# ============================================================

print("\n[4/6] Validating experiment dataset...")

trend_features = [
    "fire_change_1d",
    "fire_change_3d",
    "fire_change_7d",
    "frp_change_1d",
    "frp_change_3d",
    "frp_change_7d",
    "fire_ratio_recent_vs_3d",
    "fire_ratio_recent_vs_7d",
    "frp_ratio_recent_vs_3d",
    "frp_ratio_recent_vs_7d",
    "fire_trend_3d",
    "fire_trend_7d",
    "frp_trend_3d",
    "frp_trend_7d",
    "fire_acceleration",
    "frp_acceleration",
]

missing_values = (
    df[trend_features]
    .isna()
    .sum()
    .sum()
)

print(
    f"Trend-feature missing values: "
    f"{missing_values}"
)

if missing_values > 0:
    raise ValueError(
        "Trend features contain missing values."
    )


duplicates = df.duplicated(
    subset=["state", "district", "date"]
).sum()

print(
    f"Duplicate district-days: {duplicates}"
)

if duplicates > 0:
    raise ValueError(
        "Duplicate district-days detected."
    )


# Check that original columns are unchanged.
original = pd.read_csv(INPUT_FILE)

if not df[original.columns].equals(
    original.assign(
        date=pd.to_datetime(original["date"])
    )
):
    raise ValueError(
        "Original V2 columns were unexpectedly modified."
    )


# ============================================================
# 5. SAVE
# ============================================================

print("\n[5/6] Saving V2.3 experiment dataset...")

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 6. SUMMARY
# ============================================================

print("\n[6/6] V2.3 TREND DATASET COMPLETE")
print("=" * 70)

print(f"Output: {OUTPUT_FILE}")
print(f"Shape: {df.shape}")

print(
    f"New trend features: "
    f"{len(trend_features)}"
)

print("\nTrend features:")

for feature in trend_features:
    print(f" - {feature}")

print("\nMissing values:")
print(df[trend_features].isna().sum())

print("\nDataset saved successfully.")