import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "ml_dataset_base.csv"

TRAIN_FILE = BASE_DIR / "ml_train.csv"
TEST_FILE = BASE_DIR / "ml_test.csv"


# ============================================================
# LOAD
# ============================================================

print("Loading base ML dataset...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print(f"Rows loaded: {len(df):,}")


# ============================================================
# TARGET
# ============================================================

# Primary task:
# Predict whether tomorrow has elevated active-fire activity.

df["elevated_next_day"] = (
    df["next_day_fire_count"] > 2
).astype(int)


# ============================================================
# TIME FEATURES
# ============================================================

df["year"] = df["date"].dt.year

# October = 1, November = 2
df["season_month"] = df["date"].dt.month

# Cyclic representation of day of season
import numpy as np

df["season_sin"] = np.sin(
    2 * np.pi * df["day_of_season"] / 61
)

df["season_cos"] = np.cos(
    2 * np.pi * df["day_of_season"] / 61
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",

    "fire_lag_1d",
    "fire_lag_3d",
    "fire_lag_7d",

    "fire_mean_3d",
    "fire_mean_7d",

    "season_sin",
    "season_cos",

    "state",
    "district"
]


TARGET = "elevated_next_day"


# ============================================================
# KEEP ONLY REQUIRED COLUMNS
# ============================================================

model_df = df[
    [
        "date",
        "state",
        "district",
        *FEATURES,
        TARGET,
        "next_day_fire_count"
    ]
].copy()


# ============================================================
# TEMPORAL SPLIT
# ============================================================

train = model_df[
    model_df["date"].dt.year.isin([2023, 2024])
].copy()

test = model_df[
    model_df["date"].dt.year == 2025
].copy()


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 65)
print("FINAL ML DATASET")
print("=" * 65)

print(f"\nTraining rows: {len(train):,}")
print(f"Test rows:     {len(test):,}")

print("\nTraining target distribution:")

print(
    train[TARGET]
    .value_counts()
    .sort_index()
)

print("\nTraining target percentage:")

print(
    (
        train[TARGET]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)

print("\nTest target distribution:")

print(
    test[TARGET]
    .value_counts()
    .sort_index()
)

print("\nTest target percentage:")

print(
    (
        test[TARGET]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)


# ============================================================
# CHECK NO MISSING FEATURES
# ============================================================

feature_check = [
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
    "fire_lag_1d",
    "fire_lag_3d",
    "fire_lag_7d",
    "fire_mean_3d",
    "fire_mean_7d",
    "season_sin",
    "season_cos"
]

print("\nMissing feature values:")

print(
    model_df[feature_check].isna().sum()
)


# ============================================================
# SAVE
# ============================================================

train.to_csv(
    TRAIN_FILE,
    index=False
)

test.to_csv(
    TEST_FILE,
    index=False
)

print("\nSaved:")
print(TRAIN_FILE)
print(TEST_FILE)

print("\nPreparation completed.")