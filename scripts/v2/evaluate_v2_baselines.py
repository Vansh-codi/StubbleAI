import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.metrics import mean_absolute_error
# ============================================================
# STUBBLEAI V2 - BASELINE BENCHMARK
# ============================================================
# Baselines:
#   1. Persistencea
#   2. Global climatology
#   3. District climatology
#
# Split:
#   2023 → learn baselines
#   2024 → threshold selection
#   2025 → frozen final test
#
# Target:
#   fire_count_t_plus_1d > 2 → Elevated (1)
#   fire_count_t_plus_1d <= 2 → Normal (0)
#
# Output:
#   v2_baseline_results.csv
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_FILE = PROJECT_ROOT / "data" / "processed" / "v2" / "ml_dataset_v2.csv"
OUTPUT_FILE = BASE_DIR / "v2_baseline_results.csv"
THRESHOLD_FILE = BASE_DIR / "v2_baseline_thresholds.json"

HORIZON = 1
TARGET_COLUMN = f"fire_count_t_plus_{HORIZON}d"
ELEVATED_THRESHOLD = 2
RANDOM_STATE = 42

print("=" * 70)
print("STUBBLEAI V2 - BASELINE BENCHMARK")
print(f"Horizon: +{HORIZON}d")
print(f"Elevated threshold: fire_count > {ELEVATED_THRESHOLD}")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/8] Loading ML dataset...")

if not ML_FILE.exists():
    print(f"Missing: {ML_FILE}")
    sys.exit(1)

df = pd.read_csv(ML_FILE)
df["date"] = pd.to_datetime(df["date"])

print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")


# ============================================================
# 2. SPLIT
# ============================================================

print("\n[2/8] Splitting into train / validation / test...")

train = df[df["date"].dt.year == 2023].copy()
val   = df[df["date"].dt.year == 2024].copy()
test  = df[df["date"].dt.year == 2025].copy()

print(f"Train (2023):      {len(train):,} rows")
print(f"Validation (2024): {len(val):,} rows")
print(f"Test (2025):       {len(test):,} rows")

if train.empty or val.empty or test.empty:
    print("ERROR: One or more splits are empty.")
    sys.exit(1)


# ============================================================
# 3. BUILD LABELS
# ============================================================

print("\n[3/8] Building binary labels...")

def make_labels(split):
    return (split[TARGET_COLUMN] > ELEVATED_THRESHOLD).astype(int)

y_train = make_labels(train)
y_val   = make_labels(val)
y_test  = make_labels(test)

print(f"Train   Elevated: {y_train.sum():,} / {len(y_train):,} ({y_train.mean()*100:.1f}%)")
print(f"Val     Elevated: {y_val.sum():,} / {len(y_val):,} ({y_val.mean()*100:.1f}%)")
print(f"Test    Elevated: {y_test.sum():,} / {len(y_test):,} ({y_test.mean()*100:.1f}%)")


# ============================================================
# 4. EVALUATION HELPER
# ============================================================

def evaluate(y_true, scores, threshold, method, split):
    """
    Apply threshold to scores, compute all metrics.
    Returns a result dict.
    """
    y_pred = (scores >= threshold).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    # ROC-AUC and PR-AUC require score variation
    try:
        roc = roc_auc_score(y_true, scores)
    except ValueError:
        roc = float("nan")

    try:
        pr = average_precision_score(y_true, scores)
    except ValueError:
        pr = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "method":    method,
        "split":     split,
        "horizon":   f"+{HORIZON}d",
        "threshold": round(threshold, 6),
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "roc_auc":   round(roc,  4) if np.isfinite(roc) else None,
        "pr_auc":    round(pr,   4) if np.isfinite(pr)  else None,
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
    }


def select_threshold(y_true, scores, candidates=None):
    """
    Select threshold that maximises Elevated-class F1
    on the provided split (always 2024 validation).
    """
    if candidates is None:
        candidates = np.linspace(0.01, 0.99, 199)

    best_f1 = -1
    best_threshold = 0.5

    for t in candidates:
        y_pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold


# ============================================================
# 5. PERSISTENCE BASELINE
# ============================================================


print("\n[4/8] Building persistence baseline...")

def persistence_scores(split):
    fc = split["fire_count"].to_numpy(dtype=float)
    return (fc > ELEVATED_THRESHOLD).astype(float)

scores_persist_train = persistence_scores(train)
scores_persist_val   = persistence_scores(val)
scores_persist_test  = persistence_scores(test)

# Persistence is already binary: 0.0 or 1.0
thresh_persist = 0.5

persist_mae = mean_absolute_error(
    test[TARGET_COLUMN].to_numpy(),
    test["fire_count"].to_numpy(),
)

print(f"Persistence threshold (fixed): {thresh_persist:.4f}")

# Evaluate on validation and test
persist_val  = evaluate(
    y_val,
    scores_persist_val,
    thresh_persist,
    "Persistence",
    "validation"
)

persist_test = evaluate(
    y_test,
    scores_persist_test,
    thresh_persist,
    "Persistence",
    "test"
)

persist_test["mae"] = round(persist_mae, 4)

# ============================================================
# 6. GLOBAL CLIMATOLOGY BASELINE
# ============================================================

print("\n[5/8] Building global climatology baseline...")

# Learn from 2023 only
global_elevated_rate = y_train.mean()

print(f"2023 global Elevated rate: {global_elevated_rate:.4f}")

# Score = same constant for every row
scores_global_val  = np.full(len(val),  global_elevated_rate)
scores_global_test = np.full(len(test), global_elevated_rate)

# Select threshold on 2024 validation
thresh_global = select_threshold(
    y_val,
    scores_global_val,
)

print(f"Global climatology threshold (val): {thresh_global:.4f}")

global_val  = evaluate(y_val,  scores_global_val,  thresh_global, "Global climatology", "validation")
global_test = evaluate(y_test, scores_global_test, thresh_global, "Global climatology", "test")


# ============================================================
# 7. DISTRICT CLIMATOLOGY BASELINE
# ============================================================

print("\n[6/8] Building district climatology baseline...")

# Learn from 2023 only
district_rates = (
    pd.concat([train, y_train.rename("label")], axis=1)
    .groupby("district")["label"]
    .mean()
)

print(f"District Elevated rates (2023):")
print(district_rates.sort_values(ascending=False).to_string())

def district_scores(split):
    return (
        split["district"]
        .map(district_rates)
        .fillna(global_elevated_rate)  # fallback for unseen districts
        .to_numpy(dtype=float)
    )

scores_district_val  = district_scores(val)
scores_district_test = district_scores(test)

# Select threshold on 2024 validation
thresh_district = select_threshold(
    y_val,
    scores_district_val,
)

print(f"District climatology threshold (val): {thresh_district:.4f}")

district_val  = evaluate(y_val,  scores_district_val,  thresh_district, "District climatology", "validation")
district_test = evaluate(y_test, scores_district_test, thresh_district, "District climatology", "test")


# ============================================================
# 8. SAVE THRESHOLDS
# ============================================================

print("\n[7/8] Saving thresholds...")

thresholds = {
    "horizon": f"+{HORIZON}d",
    "persistence": thresh_persist,
    "global_climatology": thresh_global,
    "district_climatology": thresh_district,
    "persistence_rule": "fire_count > 2 -> Elevated",
    "climatology_thresholds_selected_on": "2024 validation — F1 maximisation",
    "frozen_for": "2025 test",
}

with open(THRESHOLD_FILE, "w") as f:
    json.dump(thresholds, f, indent=4)

print(f"Saved: {THRESHOLD_FILE}")


# ============================================================
# 9. COMPILE RESULTS
# ============================================================

print("\n[8/8] Compiling results...")

all_results = [
    persist_val,
    persist_test,
    global_val,
    global_test,
    district_val,
    district_test,
]

results_df = pd.DataFrame(all_results)

results_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")


# ============================================================
# COMPARISON TABLE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE COMPARISON — +1d Elevated-class F1")
print("=" * 70)

display_cols = [
    "method", "split", "threshold",
    "accuracy", "precision", "recall",
    "f1", "roc_auc", "pr_auc",
]

print(
    results_df[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRICES — TEST SPLIT (2025)")
print("=" * 70)

for result in [persist_test, global_test, district_test]:
    print(f"\n{result['method']}")
    print(f"  TP={result['tp']}  FP={result['fp']}")
    print(f"  FN={result['fn']}  TN={result['tn']}")


# ============================================================
# PERSISTENCE FLOOR REMINDER
# ============================================================

print("\n" + "=" * 70)
print("PERSISTENCE FLOOR (2025 TEST)")
print("=" * 70)

print(
    f"F1       = {persist_test['f1']:.4f}  ← V2 models must exceed this"
)
print(
    f"Accuracy = {persist_test['accuracy']:.4f}"
)
print(
    f"ROC-AUC  = {persist_test['roc_auc']}"
)
print(
    f"PR-AUC   = {persist_test['pr_auc']}"
)

print("\n" + "=" * 70)
print("BASELINE BENCHMARK COMPLETE")
print("=" * 70)
print(
    "\nThresholds frozen. "
    "v2_baseline_results.csv ready for model appending."
)
print(
    "Do NOT open 2025 test results again until "
    "all models are evaluated."
)