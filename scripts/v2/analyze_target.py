import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ml_dataset_base.csv"

print("Loading ML dataset...")
df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

# ------------------------------------------------------------
# TRAINING PERIOD ONLY
# ------------------------------------------------------------
train = df[df["date"].dt.year.isin([2023, 2024])].copy()

test = df[df["date"].dt.year == 2025].copy()

print("\n" + "=" * 65)
print("DATA SPLIT")
print("=" * 65)

print(f"Training rows (2023-2024): {len(train):,}")
print(f"Test rows (2025):          {len(test):,}")

# ------------------------------------------------------------
# TARGET DISTRIBUTION
# ------------------------------------------------------------

target = "next_day_fire_count"

print("\n" + "=" * 65)
print("NEXT-DAY FIRE COUNT — TRAINING PERIOD")
print("=" * 65)

print(train[target].describe(
    percentiles=[
        0.50,
        0.75,
        0.80,
        0.85,
        0.90,
        0.92,
        0.95,
        0.97,
        0.98,
        0.99
    ]
))

# ------------------------------------------------------------
# CANDIDATE THRESHOLDS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("CANDIDATE CLASS DISTRIBUTIONS")
print("=" * 65)

# Candidate threshold pairs
candidates = [
    (0, 0),
    (0, 2),
    (0, 5),
    (0, 8),
    (2, 8),
    (5, 10),
    (2, 10),
    (5, 20),
    (8, 20),
    (10, 50),
    (5, 50),
]

for low_high in candidates:

    medium_threshold = low_high[0]
    high_threshold = low_high[1]

    if high_threshold <= medium_threshold:
        continue

    def classify(x):
        if x <= medium_threshold:
            return "Low"
        elif x <= high_threshold:
            return "Medium"
        else:
            return "High"

    classes = train[target].apply(classify)

    counts = classes.value_counts()
    percentages = classes.value_counts(
        normalize=True
    ) * 100

    print(
        f"\nThresholds: "
        f"Low <= {medium_threshold}, "
        f"Medium <= {high_threshold}, "
        f"High > {high_threshold}"
    )

    for label in ["Low", "Medium", "High"]:
        print(
            f"  {label:7s}: "
            f"{counts.get(label, 0):6,} "
            f"({percentages.get(label, 0):5.2f}%)"
        )

# ------------------------------------------------------------
# QUANTILE-BASED OPTIONS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("QUANTILE-BASED THRESHOLDS")
print("=" * 65)

for q1, q2 in [
    (0.80, 0.95),
    (0.85, 0.95),
    (0.90, 0.95),
    (0.90, 0.97),
    (0.90, 0.98),
]:

    t1 = train[target].quantile(q1)
    t2 = train[target].quantile(q2)

    print(
        f"\nQuantiles {q1:.0%}/{q2:.0%}: "
        f"{t1:.2f} / {t2:.2f}"
    )

    def qclass(x):
        if x <= t1:
            return "Low"
        elif x <= t2:
            return "Medium"
        else:
            return "High"

    classes = train[target].apply(qclass)

    counts = classes.value_counts()
    percentages = classes.value_counts(
        normalize=True
    ) * 100

    for label in ["Low", "Medium", "High"]:
        print(
            f"  {label:7s}: "
            f"{counts.get(label, 0):6,} "
            f"({percentages.get(label, 0):5.2f}%)"
        )

# ------------------------------------------------------------
# STATE COMPARISON
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("TRAINING TARGET BY STATE")
print("=" * 65)

print(
    train.groupby("state")[target].describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

# ------------------------------------------------------------
# TARGET BY YEAR
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("TARGET BY YEAR")
print("=" * 65)

train["year"] = train["date"].dt.year

print(
    train.groupby("year")[target].describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

# ------------------------------------------------------------
# TOP TARGET VALUES
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("TOP NEXT-DAY FIRE VALUES")
print("=" * 65)

print(
    train[
        [
            "date",
            "state",
            "district",
            target
        ]
    ]
    .sort_values(target, ascending=False)
    .head(20)
    .to_string(index=False)
)

print("\nAnalysis completed.")