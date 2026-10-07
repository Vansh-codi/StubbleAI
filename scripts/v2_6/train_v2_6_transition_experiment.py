from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# V2.6 Transition-Aware Fire Forecasting
# ============================================================
#
# Arm A: Frozen V2 RF reconstruction
# Arm B: Transition-aware RF
#
# IMPORTANT:
# - Uses frozen ml_dataset_v2.csv
# - 2023 train / 2024 validation / 2025 test
# - 2025 is untouched
# - Exact V2 feature set
# - No trend/spatial/interaction/class-weight features
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "ml_dataset_v2.csv"

OUT_DIR = ROOT / "research" / "v2" / "v2_6"
RESULTS_DIR = OUT_DIR / "results"
PRED_DIR = OUT_DIR / "predictions"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Frozen V2 feature specification
# ------------------------------------------------------------

NUMERIC_FEATURES = [
    "day_of_year",
    "sin_year",
    "cos_year",
    "month",
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

CATEGORICAL_FEATURES = [
    "state",
    "district",
]

HORIZONS = [1, 2, 3, 5, 7]

RF_PARAMS = dict(
    n_estimators=400,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight=None,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def metric_row(
    y_true,
    probability,
    prediction,
    horizon,
    arm,
    threshold,
):
    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "arm": arm,
        "horizon": horizon,
        "threshold": threshold,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(
            y_true, prediction, zero_division=0
        ),
        "recall": recall_score(
            y_true, prediction, zero_division=0
        ),
        "f1": f1_score(
            y_true, prediction, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def transition_metrics(df, probability, prediction, horizon, arm):
    current_elevated = df["fire_count"].to_numpy() > 2
    actual_elevated = df[f"fire_count_t_plus_{horizon}d"].to_numpy() > 2

    predicted_elevated = prediction.astype(bool)

    rows = []

    for current_name, current_mask in [
        ("N", ~current_elevated),
        ("E", current_elevated),
    ]:
        for future_name, future_mask in [
            ("N", ~actual_elevated),
            ("E", actual_elevated),
        ]:
            mask = current_mask & future_mask

            if mask.sum() == 0:
                continue

            # For a transition class, correctness means the model's
            # binary forecast equals the actual future state.
            correct = (
                predicted_elevated[mask] == actual_elevated[mask]
            )

            rows.append(
                {
                    "arm": arm,
                    "horizon": horizon,
                    "transition": f"{current_name}->{future_name}",
                    "n": int(mask.sum()),
                    "correct": int(correct.sum()),
                    "accuracy": float(correct.mean()),
                }
            )

    # N -> E recall:
    # among actual N->E transitions, how many did the model predict E?
    onset_mask = (~current_elevated) & actual_elevated

    if onset_mask.sum() > 0:
        onset_recall = (
            predicted_elevated[onset_mask].mean()
        )
    else:
        onset_recall = np.nan

    # E -> E continuation recall
    continuation_mask = current_elevated & actual_elevated

    if continuation_mask.sum() > 0:
        continuation_recall = (
            predicted_elevated[continuation_mask].mean()
        )
    else:
        continuation_recall = np.nan

    return rows, {
        "arm": arm,
        "horizon": horizon,
        "n_to_e_recall": onset_recall,
        "e_to_e_recall": continuation_recall,
    }


def select_threshold(y_true, probability):
    thresholds = np.linspace(0.01, 0.99, 199)

    best_threshold = None
    best_f1 = -1.0

    for threshold in thresholds:
        prediction = (probability >= threshold).astype(int)

        score = f1_score(
            y_true,
            prediction,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def build_features(train_df, other_dfs):
    """
    Fit one-hot columns using 2023 training rows only.

    Returns:
        X_train, transformed other DataFrames
    """

    train_cat = pd.get_dummies(
        train_df[CATEGORICAL_FEATURES],
        drop_first=False,
    )

    transformed = []

    for df in other_dfs:
        cat = pd.get_dummies(
            df[CATEGORICAL_FEATURES],
            drop_first=False,
        )

        cat = cat.reindex(
            columns=train_cat.columns,
            fill_value=0,
        )

        numeric = df[NUMERIC_FEATURES].copy()

        transformed.append(
            pd.concat(
                [
                    numeric.reset_index(drop=True),
                    cat.reset_index(drop=True),
                ],
                axis=1,
            )
        )

    train_numeric = train_df[NUMERIC_FEATURES].copy()

    X_train = pd.concat(
        [
            train_numeric.reset_index(drop=True),
            train_cat.reset_index(drop=True),
        ],
        axis=1,
    )

    return X_train, transformed


# ------------------------------------------------------------
# Load and validate dataset
# ------------------------------------------------------------

print("=" * 70)
print("V2.6 TRANSITION-AWARE FIRE FORECASTING")
print("=" * 70)

print("\nLoading:", DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Missing frozen V2 dataset: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

required = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + ["date", "year", "fire_count"]
    + [
        f"fire_count_t_plus_{h}d"
        for h in HORIZONS
    ]
)

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

df["date"] = pd.to_datetime(df["date"])

if df[required].isna().any().any():
    bad = df[required].isna().sum()
    bad = bad[bad > 0]
    raise ValueError(
        f"Missing values detected:\n{bad}"
    )

if df.duplicated(
    subset=["state", "district", "date"]
).any():
    raise ValueError(
        "Duplicate state/district/date rows detected."
    )


# ------------------------------------------------------------
# Temporal split
# ------------------------------------------------------------

train_df = df[df["year"] == 2023].copy()
val_df = df[df["year"] == 2024].copy()
test_df = df[df["year"] == 2025].copy()

print("\nTemporal split:")
print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

if len(train_df) != 1800:
    raise ValueError(
        f"Unexpected 2023 train size: {len(train_df)}"
    )

if len(val_df) != 1800:
    raise ValueError(
        f"Unexpected 2024 validation size: {len(val_df)}"
    )

if len(test_df) != 1800:
    raise ValueError(
        f"Unexpected 2025 test size: {len(test_df)}"
    )


# ------------------------------------------------------------
# Build common V2 design matrix
# ------------------------------------------------------------

print("\nBuilding frozen V2 design matrix...")

X_train, [X_val, X_test] = build_features(
    train_df,
    [val_df, test_df],
)

print("Feature matrix shape:", X_train.shape)

if list(X_train.columns) != list(X_val.columns):
    raise ValueError("Train/validation feature columns differ.")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train/test feature columns differ.")


# ------------------------------------------------------------
# Containers
# ------------------------------------------------------------

all_results = []
all_transition_rows = []
all_transition_summary = []


# ============================================================
# HORIZON LOOP
# ============================================================

for horizon in HORIZONS:

    print("\n" + "=" * 70)
    print(f"HORIZON +{horizon}D")
    print("=" * 70)

    target_col = f"fire_count_t_plus_{horizon}d"

    y_train = (
        train_df[target_col].to_numpy() > 2
    ).astype(int)

    y_val = (
        val_df[target_col].to_numpy() > 2
    ).astype(int)

    y_test = (
        test_df[target_col].to_numpy() > 2
    ).astype(int)

    # ========================================================
    # ARM A â€” FROZEN V2 RF
    # ========================================================

    print("\n[Arm A] Training frozen V2 RF...")

    baseline_model = RandomForestClassifier(
        **RF_PARAMS
    )

    baseline_model.fit(
        X_train,
        y_train,
    )

    val_prob_baseline = baseline_model.predict_proba(
        X_val
    )[:, 1]

    test_prob_baseline = baseline_model.predict_proba(
        X_test
    )[:, 1]

    baseline_threshold, baseline_val_f1 = select_threshold(
        y_val,
        val_prob_baseline,
    )

    val_pred_baseline = (
        val_prob_baseline >= baseline_threshold
    ).astype(int)

    test_pred_baseline = (
        test_prob_baseline >= baseline_threshold
    ).astype(int)

    baseline_result = metric_row(
        y_test,
        test_prob_baseline,
        test_pred_baseline,
        horizon,
        "frozen_v2_rf",
        baseline_threshold,
    )

    baseline_result["validation_f1"] = baseline_val_f1

    all_results.append(baseline_result)

    transition_rows, transition_summary = transition_metrics(
        test_df,
        test_prob_baseline,
        test_pred_baseline,
        horizon,
        "frozen_v2_rf",
    )

    all_transition_rows.extend(transition_rows)
    all_transition_summary.append(transition_summary)

    print(
        f"Frozen V2 threshold: {baseline_threshold:.6f}"
    )
    print(
        f"Frozen V2 test F1: {baseline_result['f1']:.6f}"
    )
    print(
        f"Frozen V2 N->E recall: "
        f"{transition_summary['n_to_e_recall']:.6f}"
    )

    # ========================================================
    # ARM B â€” TRANSITION-AWARE RF
    # ========================================================

    print("\n[Arm B] Training transition-aware RF...")

    train_current_normal = (
        train_df["fire_count"].to_numpy() <= 2
    )

    train_current_elevated = (
        train_df["fire_count"].to_numpy() > 2
    )

    # --------------------------------
    # Onset model: N -> E
    # --------------------------------

    onset_model = RandomForestClassifier(
        **RF_PARAMS
    )

    onset_model.fit(
        X_train.loc[train_current_normal],
        y_train[train_current_normal],
    )

    # --------------------------------
    # Continuation model: E -> E
    # --------------------------------

    continuation_model = RandomForestClassifier(
        **RF_PARAMS
    )

    continuation_model.fit(
        X_train.loc[train_current_elevated],
        y_train[train_current_elevated],
    )

    # Current regime routing
    val_current_normal = (
        val_df["fire_count"].to_numpy() <= 2
    )

    val_current_elevated = (
        val_df["fire_count"].to_numpy() > 2
    )

    test_current_normal = (
        test_df["fire_count"].to_numpy() <= 2
    )

    test_current_elevated = (
        test_df["fire_count"].to_numpy() > 2
    )

    # --------------------------------
    # Validation predictions
    # --------------------------------

    val_prob_transition = np.zeros(len(val_df))

    if val_current_normal.any():
        val_prob_transition[val_current_normal] = (
            onset_model.predict_proba(
                X_val.loc[val_current_normal]
            )[:, 1]
        )

    if val_current_elevated.any():
        val_prob_transition[val_current_elevated] = (
            continuation_model.predict_proba(
                X_val.loc[val_current_elevated]
            )[:, 1]
        )

    # --------------------------------
    # Test predictions
    # --------------------------------

    test_prob_transition = np.zeros(len(test_df))

    if test_current_normal.any():
        test_prob_transition[test_current_normal] = (
            onset_model.predict_proba(
                X_test.loc[test_current_normal]
            )[:, 1]
        )

    if test_current_elevated.any():
        test_prob_transition[test_current_elevated] = (
            continuation_model.predict_proba(
                X_test.loc[test_current_elevated]
            )[:, 1]
        )

    # --------------------------------
    # Threshold selected ONLY on 2024
    # --------------------------------

    transition_threshold, transition_val_f1 = (
        select_threshold(
            y_val,
            val_prob_transition,
        )
    )

    val_pred_transition = (
        val_prob_transition >= transition_threshold
    ).astype(int)

    test_pred_transition = (
        test_prob_transition >= transition_threshold
    ).astype(int)

    transition_result = metric_row(
        y_test,
        test_prob_transition,
        test_pred_transition,
        horizon,
        "transition_rf",
        transition_threshold,
    )

    transition_result["validation_f1"] = (
        transition_val_f1
    )

    all_results.append(transition_result)

    transition_rows, transition_summary = transition_metrics(
        test_df,
        test_prob_transition,
        test_pred_transition,
        horizon,
        "transition_rf",
    )

    all_transition_rows.extend(transition_rows)
    all_transition_summary.append(transition_summary)

    print(
        f"Transition threshold: "
        f"{transition_threshold:.6f}"
    )
    print(
        f"Transition RF test F1: "
        f"{transition_result['f1']:.6f}"
    )
    print(
        f"Transition RF N->E recall: "
        f"{transition_summary['n_to_e_recall']:.6f}"
    )
    print(
        f"Transition RF E->E recall: "
        f"{transition_summary['e_to_e_recall']:.6f}"
    )

    # ========================================================
    # Save prediction file for this horizon
    # ========================================================

    prediction_output = test_df[
        [
            "state",
            "district",
            "date",
            "fire_count",
        ]
        + [target_col]
    ].copy()

    prediction_output["horizon"] = horizon

    prediction_output["actual_elevated"] = y_test

    prediction_output["v2_probability"] = (
        test_prob_baseline
    )

    prediction_output["v2_prediction"] = (
        test_pred_baseline
    )

    prediction_output["transition_probability"] = (
        test_prob_transition
    )

    prediction_output["transition_prediction"] = (
        test_pred_transition
    )

    prediction_output["current_regime"] = np.where(
        test_current_elevated,
        "Elevated",
        "Normal",
    )

    prediction_output["transition_actual"] = np.select(
        [
            (~test_current_elevated) & (~y_test.astype(bool)),
            (~test_current_elevated) & y_test.astype(bool),
            test_current_elevated & (~y_test.astype(bool)),
            test_current_elevated & y_test.astype(bool),
        ],
        [
            "N->N",
            "N->E",
            "E->N",
            "E->E",
        ],
        default="UNKNOWN",
    )

    prediction_output.to_csv(
        PRED_DIR / f"v2_6_predictions_plus{horizon}d.csv",
        index=False,
    )


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(all_results)

transition_df = pd.DataFrame(
    all_transition_rows
)

transition_summary_df = pd.DataFrame(
    all_transition_summary
)

results_df.to_csv(
    RESULTS_DIR / "v2_6_results.csv",
    index=False,
)

transition_df.to_csv(
    RESULTS_DIR / "v2_6_transition_breakdown.csv",
    index=False,
)

transition_summary_df.to_csv(
    RESULTS_DIR / "v2_6_transition_summary.csv",
    index=False,
)


# ============================================================
# Frozen V2 reproduction guard
# ============================================================

expected_f1 = {
    1: 0.7754,
    2: 0.7513,
    3: 0.7305,
    5: 0.7303,
    7: 0.7085,
}

baseline_check = results_df[
    results_df["arm"] == "frozen_v2_rf"
].copy()

baseline_check["expected_f1"] = baseline_check[
    "horizon"
].map(expected_f1)

baseline_check["f1_difference"] = (
    baseline_check["f1"]
    - baseline_check["expected_f1"]
)

print("\n" + "=" * 70)
print("FROZEN V2 REPRODUCTION CHECK")
print("=" * 70)

print(
    baseline_check[
        [
            "horizon",
            "f1",
            "expected_f1",
            "f1_difference",
        ]
    ].to_string(index=False)
)

max_difference = baseline_check[
    "f1_difference"
].abs().max()

if max_difference > 1e-4:
    raise RuntimeError(
        "FROZEN V2 REPRODUCTION CHECK FAILED. "
        f"Maximum F1 difference = {max_difference}"
    )

print("\nPASS: Frozen V2 RF reproduced exactly.")


# ============================================================
# Configuration
# ============================================================

config = {
    "experiment": "V2.6 Transition-Aware Fire Forecasting",
    "dataset": str(DATA_PATH),
    "train_year": 2023,
    "validation_year": 2024,
    "test_year": 2025,
    "horizons": HORIZONS,
    "target_rule": "future fire_count > 2",
    "current_normal_rule": "fire_count <= 2",
    "current_elevated_rule": "fire_count > 2",
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "random_forest": RF_PARAMS,
    "threshold_candidates": 199,
    "threshold_selection": "2024 validation overall Elevated F1",
    "arms": [
        "frozen_v2_rf",
        "transition_rf",
    ],
    "frozen_2025": True,
    "v2_reproduction_max_abs_f1_difference": float(
        max_difference
    ),
}

with open(
    RESULTS_DIR / "v2_6_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("V2.6 COMPLETE")
print("=" * 70)

summary_cols = [
    "arm",
    "horizon",
    "threshold",
    "validation_f1",
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
]

print(
    results_df[
        summary_cols
    ].to_string(index=False)
)

print("\nTransition summary:")
print(
    transition_summary_df.to_string(index=False)
)

print("\nArtifacts written to:")
print(OUT_DIR)

