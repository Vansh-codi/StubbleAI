import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# STUBBLEAI V2.3 - TREND FEATURE EXPERIMENT
# ============================================================
#
# Purpose:
# Compare the frozen V2 RF feature set against V2.3 RF
# with additional fire/FRP trend features.
#
# IMPORTANT:
# - Does NOT modify frozen V2 benchmark files.
# - 2023 = train
# - 2024 = validation / threshold selection
# - 2025 = untouched final test
# - Same RF configuration as frozen V2.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "research"
    / "v2"
    / "v2_3"
    / "ml_dataset_v2_3_trend.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "research"
    / "v2"
    / "v2_3"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


HORIZONS = [1, 2, 3, 5, 7]

SEED = 42
ELEVATED_THRESHOLD = 2


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

BASE_NUMERIC_FEATURES = [
    "year",
    "month",
    "day",
    "day_of_year",
    "sin_year",
    "cos_year",

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

    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",

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

    "fire_mean_3d",
    "fire_sparse_mean_7d",
    "fire_sparse_mean_14d",

    "frp_mean_3d",
    "frp_sparse_mean_7d",
    "frp_sparse_mean_14d",
]


TREND_FEATURES = [
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


CATEGORICAL_FEATURES = [
    "state",
    "district",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STUBBLEAI V2.3 - TREND FEATURE EXPERIMENT")
print("=" * 70)

print("\n[1/8] Loading V2.3 dataset...")

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print(f"Dataset shape: {df.shape}")


# ============================================================
# VALIDATE
# ============================================================

print("\n[2/8] Validating dataset...")

required_columns = (
    BASE_NUMERIC_FEATURES
    + TREND_FEATURES
    + CATEGORICAL_FEATURES
)

for horizon in HORIZONS:
    required_columns.append(
        f"fire_count_t_plus_{horizon}d"
    )

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

if df.isna().sum().sum() > 0:
    raise ValueError(
        "Dataset contains missing values."
    )

if df.duplicated(
    ["state", "district", "date"]
).sum() > 0:
    raise ValueError(
        "Duplicate district-days found."
    )


# ============================================================
# SPLIT
# ============================================================

print("\n[3/8] Creating temporal splits...")

train = df[df["year"] == 2023].copy()
val = df[df["year"] == 2024].copy()
test = df[df["year"] == 2025].copy()

print(f"Train: {train.shape}")
print(f"Validation: {val.shape}")
print(f"Test: {test.shape}")


# ============================================================
# METRIC HELPERS
# ============================================================

def safe_roc_auc(y_true, probability):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(
        y_true,
        probability
    )


def safe_pr_auc(y_true, probability):
    if len(np.unique(y_true)) < 2:
        return np.nan

    return average_precision_score(
        y_true,
        probability
    )


def evaluate_at_threshold(
    y_true,
    probability,
    threshold
):
    prediction = (
        probability >= threshold
    ).astype(int)

    return {
        "accuracy": accuracy_score(
            y_true,
            prediction
        ),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0
        ),
        "roc_auc": safe_roc_auc(
            y_true,
            probability
        ),
        "pr_auc": safe_pr_auc(
            y_true,
            probability
        ),
    }


def select_threshold(y_true, probability):
    thresholds = np.linspace(
        0.01,
        0.99,
        199
    )

    best_threshold = thresholds[0]
    best_f1 = -1.0

    for threshold in thresholds:
        prediction = (
            probability >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            prediction,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold)


# ============================================================
# TRAIN ONE HORIZON
# ============================================================

def run_horizon(horizon):

    target_column = (
        f"fire_count_t_plus_{horizon}d"
    )

    print("\n" + "-" * 70)
    print(f"HORIZON +{horizon}d")
    print("-" * 70)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y_train = (
        train[target_column]
        > ELEVATED_THRESHOLD
    ).astype(int)

    y_val = (
        val[target_column]
        > ELEVATED_THRESHOLD
    ).astype(int)

    y_test = (
        test[target_column]
        > ELEVATED_THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # Exact pandas dummy encoding used by frozen V2.
    # --------------------------------------------------------

    dummies_train = pd.get_dummies(
        train[CATEGORICAL_FEATURES],
        drop_first=False
    )

    dummy_columns = (
        dummies_train.columns.tolist()
    )

    dummies_val = (
        pd.get_dummies(
            val[CATEGORICAL_FEATURES],
            drop_first=False
        )
        .reindex(
            columns=dummy_columns,
            fill_value=0
        )
    )

    dummies_test = (
        pd.get_dummies(
            test[CATEGORICAL_FEATURES],
            drop_first=False
        )
        .reindex(
            columns=dummy_columns,
            fill_value=0
        )
    )

    # --------------------------------------------------------
    # V2.3 feature set
    # --------------------------------------------------------

    all_numeric_features = (
        BASE_NUMERIC_FEATURES
        + TREND_FEATURES
    )

    X_train = pd.concat(
        [
            train[
                all_numeric_features
            ].reset_index(drop=True),

            dummies_train.reset_index(
                drop=True
            ),
        ],
        axis=1
    ).to_numpy(dtype=float)

    X_val = pd.concat(
        [
            val[
                all_numeric_features
            ].reset_index(drop=True),

            dummies_val.reset_index(
                drop=True
            ),
        ],
        axis=1
    ).to_numpy(dtype=float)

    X_test = pd.concat(
        [
            test[
                all_numeric_features
            ].reset_index(drop=True),

            dummies_test.reset_index(
                drop=True
            ),
        ],
        axis=1
    ).to_numpy(dtype=float)

    print(
        f"Features: "
        f"{len(all_numeric_features)} numeric + "
        f"{len(dummy_columns)} categorical"
    )

    # --------------------------------------------------------
    # Exact frozen V2 RF configuration.
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
    )

    print("Training RF...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation threshold selection
    # --------------------------------------------------------

    val_probability = model.predict_proba(
        X_val
    )[:, 1]

    threshold = select_threshold(
        y_val,
        val_probability
    )

    val_metrics = evaluate_at_threshold(
        y_val,
        val_probability,
        threshold
    )

    print(
        f"Validation threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Validation F1: "
        f"{val_metrics['f1']:.4f}"
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = evaluate_at_threshold(
        y_test,
        test_probability,
        threshold
    )

    print(
        f"2025 F1: "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"2025 Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"2025 Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"2025 Recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"2025 ROC-AUC: "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"2025 PR-AUC: "
        f"{test_metrics['pr_auc']:.4f}"
    )

    # --------------------------------------------------------
    # Save predictions for downstream V2.2-style analysis.
    # --------------------------------------------------------

    prediction_output = test[
        [
            "state",
            "district",
            "date",
            "fire_count",
            "frp_sum",
        ]
    ].copy()

    prediction_output[
        "horizon"
    ] = f"+{horizon}d"

    prediction_output[
        "target_fire_count"
    ] = test[target_column].to_numpy()

    prediction_output[
        "actual_elevated"
    ] = y_test.to_numpy()

    prediction_output[
        "rf_probability"
    ] = test_probability

    prediction_output[
        "rf_prediction"
    ] = (
        test_probability >= threshold
    ).astype(int)

    prediction_output[
        "model"
    ] = "V2.3_RF_TREND"

    output_file = (
        OUTPUT_DIR
        / f"v2_3_trend_predictions_plus{horizon}d.csv"
    )

    prediction_output.to_csv(
        output_file,
        index=False
    )

    return {
        "horizon": f"+{horizon}d",
        "threshold": threshold,
        **test_metrics,
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
        "numeric_features": len(
            all_numeric_features
        ),
        "categorical_features": len(
            dummy_columns
        ),
    }


# ============================================================
# RUN ALL HORIZONS
# ============================================================

print("\n[4/8] Running all horizons...")

results = []

for horizon in HORIZONS:
    results.append(
        run_horizon(horizon)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n[5/8] Saving experiment results...")

results_df = pd.DataFrame(results)

results_file = (
    OUTPUT_DIR
    / "v2_3_trend_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

config = {
    "experiment": "V2.3 trend features",
    "seed": SEED,
    "rf": {
        "n_estimators": 400,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1,
    },
    "train_year": 2023,
    "validation_year": 2024,
    "test_year": 2025,
    "elevated_threshold": 2,
    "horizons": HORIZONS,
    "base_numeric_features": BASE_NUMERIC_FEATURES,
    "trend_features": TREND_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
}

config_file = (
    OUTPUT_DIR
    / "v2_3_trend_config.json"
)

with open(
    config_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        config,
        f,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n[6/8] EXPERIMENT COMPLETE")
print("=" * 70)

print(
    results_df[
        [
            "horizon",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
    ].to_string(index=False)
)

print("\n[7/8] Saved:")
print(results_file)
print(config_file)

print("\n[8/8] IMPORTANT")
print("=" * 70)
print(
    "2025 was used only as the final test set."
)
print(
    "No 2025 threshold tuning was performed."
)
print(
    "Frozen V2 benchmark files were not modified."
)