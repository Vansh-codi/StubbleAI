from pathlib import Path
import json

import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)



# ============================================================
# STUBBLEAI V2.2 - 2025 RF ERROR RECONSTRUCTION
# ============================================================

DATA_PATH = Path("ml_dataset_v2.csv")
OUTPUT_DIR = Path("research/v2/error_analysis")

RANDOM_STATE = 42
ELEVATED_THRESHOLD = 2

HORIZONS = [1, 2, 3, 5, 7]

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


def load_frozen_threshold(horizon):
    """
    Load the exact frozen Random Forest threshold recorded
    in the V2 benchmark results.

    These thresholds were selected on 2024 validation using
    F1 maximisation and frozen before the 2025 test period.
    """

    frozen_thresholds = {
        1: 0.410909,
        2: 0.366364,
        3: 0.297071,
        5: 0.356465,
        7: 0.331717,
    }

    if horizon not in frozen_thresholds:
        raise ValueError(
            f"No frozen RF threshold recorded for +{horizon}d"
        )

    threshold = frozen_thresholds[horizon]

    print(
        f"  Frozen RF threshold: {threshold:.6f}"
    )
    print(
        "  Source: frozen V2 benchmark results"
    )
    print(
        "  Selected on: 2024 validation — F1 maximisation"
    )
    print(
        "  Frozen for: 2025 test"
    )

    return threshold

def build_features(train_df, val_df, test_df):
    """
    Reproduce the exact frozen V2 feature encoding protocol.

    Categorical columns are encoded with pandas get_dummies()
    using the 2023 training set as the reference column set.
    """

    dummies_train = pd.get_dummies(
        train_df[CATEGORICAL_FEATURES],
        drop_first=False,
    )

    dummy_columns = dummies_train.columns.tolist()

    dummies_val = (
        pd.get_dummies(
            val_df[CATEGORICAL_FEATURES],
            drop_first=False,
        )
        .reindex(
            columns=dummy_columns,
            fill_value=0,
        )
    )

    dummies_test = (
        pd.get_dummies(
            test_df[CATEGORICAL_FEATURES],
            drop_first=False,
        )
        .reindex(
            columns=dummy_columns,
            fill_value=0,
        )
    )

    X_train = pd.concat(
        [
            train_df[NUMERIC_FEATURES].reset_index(drop=True),
            dummies_train.reset_index(drop=True),
        ],
        axis=1,
    ).to_numpy(dtype=float)

    X_val = pd.concat(
        [
            val_df[NUMERIC_FEATURES].reset_index(drop=True),
            dummies_val.reset_index(drop=True),
        ],
        axis=1,
    ).to_numpy(dtype=float)

    X_test = pd.concat(
        [
            test_df[NUMERIC_FEATURES].reset_index(drop=True),
            dummies_test.reset_index(drop=True),
        ],
        axis=1,
    ).to_numpy(dtype=float)

    return X_train, X_val, X_test


def make_model():
    return RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def classify_error(actual, predicted):
    if actual == 1 and predicted == 1:
        return "TP"
    if actual == 0 and predicted == 0:
        return "TN"
    if actual == 0 and predicted == 1:
        return "FP"
    return "FN"


def main():
    print("=" * 75)
    print("STUBBLEAI V2.2 - 2025 RF ERROR RECONSTRUCTION")
    print("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Loading V2 ML dataset...")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    print(f"Dataset shape: {df.shape}")
    print(
        f"Date range: "
        f"{df['date'].min().date()} -> {df['date'].max().date()}"
    )

    print("\n[2/6] Creating temporal splits...")

    train = df[
        (df["date"] >= "2023-10-15")
        & (df["date"] <= "2023-11-23")
    ].copy()

    validation = df[
        (df["date"] >= "2024-10-15")
        & (df["date"] <= "2024-11-23")
    ].copy()

    test = df[
        (df["date"] >= "2025-10-15")
        & (df["date"] <= "2025-11-23")
    ].copy()

    print(f"Training 2023:   {len(train):>5}")
    print(f"Validation 2024: {len(validation):>5}")
    print(f"Test 2025:       {len(test):>5}")

    all_horizon_results = []
    all_predictions = []

    print("\n[3/6] Processing horizons...")

    for horizon in HORIZONS:
        print("\n" + "-" * 75)
        print(f"HORIZON: +{horizon}d")
        print("-" * 75)

        target = f"fire_count_t_plus_{horizon}d"

        threshold = load_frozen_threshold(horizon)

        print(f"Frozen RF threshold: {threshold:.6f}")

        X_train, X_val, X_test = build_features(
            train,
            validation,
            test,
        )

        y_train = (
            train[target].values > ELEVATED_THRESHOLD
        ).astype(int)

        y_val = (
            validation[target].values > ELEVATED_THRESHOLD
        ).astype(int)

        y_test = (
            test[target].values > ELEVATED_THRESHOLD
        ).astype(int)

        print("Training RF on 2023...")
        model = make_model()
        model.fit(X_train, y_train)

        val_probability = model.predict_proba(X_val)[:, 1]
        test_probability = model.predict_proba(X_test)[:, 1]

        val_prediction = (
            val_probability >= threshold
        ).astype(int)

        test_prediction = (
            test_probability >= threshold
        ).astype(int)

        print("Validation F1:")
        print(
            f"  {f1_score(y_val, val_prediction, zero_division=0):.4f}"
        )

        print("2025 test metrics:")

        accuracy = accuracy_score(y_test, test_prediction)
        balanced_accuracy = balanced_accuracy_score(
            y_test,
            test_prediction,
        )
        precision = precision_score(
            y_test,
            test_prediction,
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            test_prediction,
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            test_prediction,
            zero_division=0,
        )
        roc_auc = roc_auc_score(y_test, test_probability)
        pr_auc = average_precision_score(
            y_test,
            test_probability,
        )

        print(f"  Accuracy:          {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"  Precision:         {precision:.4f}")
        print(f"  Recall:            {recall:.4f}")
        print(f"  F1:                {f1:.4f}")
        print(f"  ROC-AUC:           {roc_auc:.4f}")
        print(f"  PR-AUC:            {pr_auc:.4f}")

        result = {
            "horizon": f"+{horizon}d",
            "threshold": threshold,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

        all_horizon_results.append(result)

        horizon_predictions = test[
            [
                "state",
                "district",
                "date",
                "fire_count",
                "frp_sum",
            ]
        ].copy()

        horizon_predictions["horizon"] = f"+{horizon}d"
        horizon_predictions["target_fire_count"] = test[target].values
        horizon_predictions["actual_elevated"] = y_test
        horizon_predictions["rf_probability"] = test_probability
        horizon_predictions["rf_prediction"] = test_prediction
        horizon_predictions["persistence_prediction"] = (
            test["fire_count"].values > ELEVATED_THRESHOLD
        ).astype(int)

        horizon_predictions["error_type"] = [
            classify_error(a, p)
            for a, p in zip(y_test, test_prediction)
        ]

        horizon_predictions["persistence_error_type"] = [
            classify_error(a, p)
            for a, p in zip(
                y_test,
                horizon_predictions["persistence_prediction"],
            )
        ]

        all_predictions.append(horizon_predictions)

    print("\n[4/6] Combining prediction-level errors...")

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    results_df = pd.DataFrame(all_horizon_results)

    predictions_path = (
        OUTPUT_DIR / "rf_2025_error_predictions.csv"
    )

    results_path = (
        OUTPUT_DIR / "rf_2025_horizon_metrics.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print(f"Saved: {predictions_path}")
    print(f"Saved: {results_path}")

    print("\n[5/6] Creating error summary...")

    summary_rows = []

    for horizon in HORIZONS:
        subset = predictions_df[
            predictions_df["horizon"] == f"+{horizon}d"
        ]

        counts = subset["error_type"].value_counts()

        tp = int(counts.get("TP", 0))
        tn = int(counts.get("TN", 0))
        fp = int(counts.get("FP", 0))
        fn = int(counts.get("FN", 0))

        summary_rows.append(
            {
                "horizon": f"+{horizon}d",
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "total": len(subset),
                "actual_elevated": int(
                    subset["actual_elevated"].sum()
                ),
                "predicted_elevated": int(
                    subset["rf_prediction"].sum()
                ),
            }
        )

    error_summary = pd.DataFrame(summary_rows)

    error_summary_path = (
        OUTPUT_DIR / "rf_2025_error_summary.csv"
    )

    error_summary.to_csv(
        error_summary_path,
        index=False,
    )

    print(f"Saved: {error_summary_path}")

    print("\n[6/6] Final reconstruction summary")

    print("\n" + error_summary.to_string(index=False))

    print("\n" + "=" * 75)
    print("V2.2 ERROR RECONSTRUCTION COMPLETE")
    print("=" * 75)
    print(
        "\nNo model files, benchmark checkpoints, "
        "or V1 live files were modified."
    )


if __name__ == "__main__":
    main()