import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FILE = BASE_DIR / "ml_dataset_base.csv"

df = pd.read_csv(FILE)
df["date"] = pd.to_datetime(df["date"])

target = "next_day_fire_count"

train = df[df["date"].dt.year.isin([2023, 2024])].copy()
test = df[df["date"].dt.year == 2025].copy()


# ============================================================
# HELPER
# ============================================================

def show_distribution(data, low, high, name):

    def classify(x):
        if x <= low:
            return "Low"
        elif x <= high:
            return "Medium"
        else:
            return "High"

    classes = data[target].apply(classify)

    counts = classes.value_counts()
    pct = classes.value_counts(normalize=True) * 100

    print(f"\n{name}")
    print(f"Thresholds: Low <= {low}, Medium <= {high}, High > {high}")

    for c in ["Low", "Medium", "High"]:
        print(
            f"{c:7s}: "
            f"{counts.get(c, 0):6,} "
            f"({pct.get(c, 0):6.2f}%)"
        )


# ============================================================
# 1. FIXED CANDIDATE THRESHOLDS
# ============================================================

print("=" * 70)
print("FIXED THRESHOLD ROBUSTNESS")
print("=" * 70)

candidates = [
    (0, 2),
    (0, 5),
    (0, 8),
    (0, 14),
    (1, 14),
    (2, 14),
    (4, 14),
    (0, 20),
    (4, 20),
    (5, 20),
]

for low, high in candidates:

    show_distribution(
        train,
        low,
        high,
        "TRAIN (2023-2024)"
    )

    show_distribution(
        test,
        low,
        high,
        "TEST (2025)"
    )


# ============================================================
# 2. BINARY HIGH-RISK OPTIONS
# ============================================================

print("\n" + "=" * 70)
print("BINARY HIGH-RISK ANALYSIS")
print("=" * 70)

for threshold in [0, 1, 2, 4, 5, 8, 10, 14, 20]:

    train_high = (
        train[target] > threshold
    )

    test_high = (
        test[target] > threshold
    )

    print(
        f"\nThreshold > {threshold}"
    )

    print(
        f"TRAIN high-risk: "
        f"{train_high.sum():,} "
        f"({train_high.mean()*100:.2f}%)"
    )

    print(
        f"TEST high-risk:  "
        f"{test_high.sum():,} "
        f"({test_high.mean()*100:.2f}%)"
    )


# ============================================================
# 3. YEAR-BY-YEAR BINARY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("YEAR-BY-YEAR HIGH-RISK RATES")
print("=" * 70)

for threshold in [2, 5, 10, 14]:

    print(f"\nThreshold > {threshold}")

    for year in [2023, 2024, 2025]:

        subset = df[
            df["date"].dt.year == year
        ]

        high = (
            subset[target] > threshold
        ).mean() * 100

        print(
            f"  {year}: {high:.2f}%"
        )


# ============================================================
# 4. STATE-BY-STATE BINARY RATE
# ============================================================

print("\n" + "=" * 70)
print("STATE-BY-STATE HIGH-RISK RATE")
print("=" * 70)

for threshold in [2, 5, 10, 14]:

    print(f"\nThreshold > {threshold}")

    for state in ["Punjab", "Haryana"]:

        subset = train[
            train["state"] == state
        ]

        high = (
            subset[target] > threshold
        ).mean() * 100

        print(
            f"  {state}: {high:.2f}%"
        )


print("\nAnalysis completed.")