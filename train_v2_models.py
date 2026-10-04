import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================================
# STUBBLEAI V2 - MODEL TRAINING + EVALUATION
# ============================================================
# Models:
#   1. Logistic Regression
#   2. Random Forest
#   3. XGBoost
#   4. LightGBM
#
# Split:
#   2023 → train
#   2024 → threshold selection (validation)
#   2025 → frozen final test
#
# Features:
#   Fire history lags + FRP lags + weather +
#   seasonal + district/state encoding
#
# Target:
#   fire_count_t_plus_1d > 2 → Elevated (1)
#
# Output:
#   v2_baseline_results.csv  (appended)
#   v2_model_thresholds.json
# ============================================================

BASE_DIR    = Path(__file__).resolve().parent
ML_FILE     = BASE_DIR / "ml_dataset_v2.csv"
BASELINE_FILE   = BASE_DIR / "v2_baseline_results.csv"
THRESHOLD_FILE  = BASE_DIR / "v2_model_thresholds.json"

HORIZON           = 1
TARGET_COLUMN     = f"fire_count_t_plus_{HORIZON}d"
ELEVATED_THRESHOLD = 2
RANDOM_STATE      = 42

print("=" * 70)
print("STUBBLEAI V2 - MODEL TRAINING + EVALUATION")
print(f"Horizon: +{HORIZON}d")
print(f"Elevated threshold: fire_count > {ELEVATED_THRESHOLD}")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/7] Loading ML dataset...")

if not ML_FILE.exists():
    print(f"Missing: {ML_FILE}")
    sys.exit(1)

df = pd.read_csv(ML_FILE)
df["date"] = pd.to_datetime(df["date"])

print(f"Shape: {df.shape}")


# ============================================================
# 2. FEATURE DEFINITION
# ============================================================

print("\n[2/7] Defining features...")

NUMERIC_FEATURES = [
    # Calendar / seasonality
    "day_of_year",
    "sin_year",
    "cos_year",
    "month",

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

CATEGORICAL_FEATURES = [
    "state",
    "district",
]

print(f"Numeric features:     {len(NUMERIC_FEATURES)}")
print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")


# ============================================================
# 3. SPLIT
# ============================================================

print("\n[3/7] Splitting...")

train = df[df["date"].dt.year == 2023].copy()
val   = df[df["date"].dt.year == 2024].copy()
test  = df[df["date"].dt.year == 2025].copy()

print(f"Train (2023):      {len(train):,}")
print(f"Validation (2024): {len(val):,}")
print(f"Test (2025):       {len(test):,}")


# ============================================================
# 4. ENCODE CATEGORICALS + BUILD FEATURE MATRICES
# ============================================================

print("\n[4/7] Encoding categoricals...")

# One-hot encode state and district.
# fit on train only — no leakage from val/test.
dummies_train = pd.get_dummies(
    train[CATEGORICAL_FEATURES],
    drop_first=False,
)

DUMMY_COLUMNS = dummies_train.columns.tolist()

dummies_val = pd.get_dummies(
    val[CATEGORICAL_FEATURES],
    drop_first=False,
).reindex(columns=DUMMY_COLUMNS, fill_value=0)

dummies_test = pd.get_dummies(
    test[CATEGORICAL_FEATURES],
    drop_first=False,
).reindex(columns=DUMMY_COLUMNS, fill_value=0)

ALL_FEATURES = NUMERIC_FEATURES + DUMMY_COLUMNS

X_train = pd.concat(
    [train[NUMERIC_FEATURES].reset_index(drop=True),
     dummies_train.reset_index(drop=True)],
    axis=1,
).to_numpy(dtype=float)

X_val = pd.concat(
    [val[NUMERIC_FEATURES].reset_index(drop=True),
     dummies_val.reset_index(drop=True)],
    axis=1,
).to_numpy(dtype=float)

X_test = pd.concat(
    [test[NUMERIC_FEATURES].reset_index(drop=True),
     dummies_test.reset_index(drop=True)],
    axis=1,
).to_numpy(dtype=float)

y_train = (train[TARGET_COLUMN] > ELEVATED_THRESHOLD).astype(int).to_numpy()
y_val   = (val[TARGET_COLUMN]   > ELEVATED_THRESHOLD).astype(int).to_numpy()
y_test  = (test[TARGET_COLUMN]  > ELEVATED_THRESHOLD).astype(int).to_numpy()

print(f"Feature matrix shape (train): {X_train.shape}")
print(f"Train Elevated: {y_train.mean()*100:.1f}%")
print(f"Val   Elevated: {y_val.mean()*100:.1f}%")
print(f"Test  Elevated: {y_test.mean()*100:.1f}%")

# Validation
if np.isnan(X_train).any():
    print("ERROR: NaN values in training features.")
    sys.exit(1)


# ============================================================
# 5. HELPERS
# ============================================================

def select_threshold(y_true, probs):
    candidates = np.linspace(0.01, 0.99, 199)
    best_f1, best_t = -1, 0.5
    for t in candidates:
        f1 = f1_score(
            y_true,
            (probs >= t).astype(int),
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def evaluate(y_true, probs, threshold, method, split):
    y_pred = (probs >= threshold).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc = roc_auc_score(y_true, probs)
    except ValueError:
        roc = float("nan")

    try:
        pr = average_precision_score(y_true, probs)
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
        "mae": None,
    }


# ============================================================
# 6. TRAIN AND EVALUATE MODELS
# ============================================================

print("\n[5/7] Training models...")

model_results = []
model_thresholds = {}


# ------------------------------------------------------------
# MODEL DEFINITIONS
# ------------------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )),
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0,
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbosity=-1,
    ),
}




# ------------------------------------------------------------
# TRAIN → VALIDATE → TEST
# ------------------------------------------------------------

for name, model in models.items():

    print(f"\n  ── {name}")

    # Train on 2023
    model.fit(X_train, y_train)

    # Probabilities
    probs_val  = model.predict_proba(X_val)[:, 1]
    probs_test = model.predict_proba(X_test)[:, 1]

    # Threshold selection on 2024 validation
    threshold = select_threshold(y_val, probs_val)
    model_thresholds[name] = threshold

    print(f"  Val threshold (F1-max): {threshold:.4f}")

    # Validation metrics
    result_val = evaluate(
        y_val, probs_val, threshold, name, "validation"
    )
    print(
        f"  Val  → F1={result_val['f1']:.4f}  "
        f"ROC-AUC={result_val['roc_auc']:.4f}"
    )

    # Test metrics — frozen threshold
    result_test = evaluate(
        y_test, probs_test, threshold, name, "test"
    )
    print(
        f"  Test → F1={result_test['f1']:.4f}  "
        f"ROC-AUC={result_test['roc_auc']:.4f}"
    )

    model_results.extend([result_val, result_test])


# ============================================================
# 7. SAVE THRESHOLDS
# ============================================================

print("\n[6/7] Saving model thresholds...")

threshold_record = {
    "horizon": f"+{HORIZON}d",
    "models": model_thresholds,
    "selected_on": "2024 validation — F1 maximisation",
    "frozen_for":  "2025 test",
}

with open(THRESHOLD_FILE, "w") as f:
    json.dump(threshold_record, f, indent=4)

print(f"Saved: {THRESHOLD_FILE}")


# ============================================================
# 8. APPEND TO BASELINE RESULTS
# ============================================================
new_rows = pd.DataFrame(model_results)
if BASELINE_FILE.exists():
    existing = pd.read_csv(BASELINE_FILE)

    # Remove any existing rows for these models and this horizon
    mask = (
        existing["method"].isin(new_rows["method"].unique())
        & existing["horizon"].eq(f"+{HORIZON}d")
    )

    existing = existing[~mask]

    combined = pd.concat(
        [existing, new_rows],
        ignore_index=True,
    )
else:
    combined = new_rows

combined.to_csv(BASELINE_FILE, index=False)
print(f"Saved: {BASELINE_FILE}")


# ============================================================
# FINAL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 70)
print("FULL COMPARISON — +1d TEST (2025)")
print("=" * 70)

test_results = combined[
    combined["split"] == "test"
].copy()

display_cols = [
    "method", "threshold",
    "accuracy", "precision", "recall",
    "f1", "roc_auc", "pr_auc",
]

print(
    test_results[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# CONFUSION MATRICES — TEST
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRICES — TEST (2025)")
print("=" * 70)

for _, row in test_results.iterrows():
    print(f"\n{row['method']}")
    print(f"  TP={row['tp']}  FP={row['fp']}")
    print(f"  FN={row['fn']}  TN={row['tn']}")


# ============================================================
# PERSISTENCE FLOOR REMINDER
# ============================================================

print("\n" + "=" * 70)
print("PERSISTENCE FLOOR")
print("=" * 70)

persist_row = test_results[
    test_results["method"] == "Persistence"
]

if not persist_row.empty:
    p = persist_row.iloc[0]
    print(f"F1       = {p['f1']:.4f}")
    print(f"Accuracy = {p['accuracy']:.4f}")
    print(f"ROC-AUC  = {p['roc_auc']}")

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETE")
print("=" * 70)
print(
    "\nNext: extend the best model(s) to "
    "+2d / +3d / +5d / +7d horizons."
)