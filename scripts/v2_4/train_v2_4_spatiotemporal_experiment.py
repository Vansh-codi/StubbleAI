from pathlib import Path
import json
import sys

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


# ============================================================
# STUBBLEAI V2.4
# SPATIO-TEMPORAL INTERACTION EXPERIMENT
# ============================================================
#
# Arms:
#   A = Frozen V2 baseline
#   B = V2.3 spatial context
#   C = V2.4 spatio-temporal interaction
#
# Protocol:
#   2023 -> training
#   2024 -> validation / threshold selection
#   2025 -> untouched final test
#
# Model:
#   Random Forest
#   n_estimators=400
#   min_samples_leaf=2
#   random_state=42
#   class_weight=None
#
# Target:
#   fire_count_t_plus_Hd > 2 -> Elevated
#
# IMPORTANT:
#   This script does NOT modify frozen V2/V2.3 files.
#   It only creates V2.4 experiment outputs.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

V2_FILE = (
    BASE_DIR
    / "ml_dataset_v2.csv"
)

SPATIAL_FILE = (
    BASE_DIR
    / "research"
    / "v2"
    / "v2_3"
    / "spatial"
    / "ml_dataset_v2_3_spatial.csv"
)

SPATIOTEMPORAL_FILE = (
    BASE_DIR
    / "research"
    / "v2"
    / "v2_4"
    / "ml_dataset_v2_4_spatiotemporal.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "research"
    / "v2"
    / "v2_4"
)

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

RESULTS_FILE = OUTPUT_DIR / "v2_4_spatiotemporal_results.csv"
CONFIG_FILE = OUTPUT_DIR / "v2_4_spatiotemporal_config.json"


HORIZONS = [1, 2, 3, 5, 7]

ELEVATED_THRESHOLD = 2

RANDOM_STATE = 42

N_ESTIMATORS = 400
MIN_SAMPLES_LEAF = 2

THRESHOLD_CANDIDATES = np.linspace(0.01, 0.99, 199)


# ============================================================
# EXACT FROZEN V2 NUMERIC FEATURES
# ============================================================

BASE_NUMERIC_FEATURES = [
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


SPATIAL_FEATURES = [
    "neighbor_fire_current",
    "neighbor_fire_lag_1d",
    "neighbor_fire_lag_3d",
    "neighbor_fire_lag_7d",
    "neighbor_frp_current",
    "neighbor_frp_lag_1d",
    "neighbor_frp_lag_3d",
    "neighbor_frp_lag_7d",
]


SPATIOTEMPORAL_FEATURES = [
    "fire_neighbor_interaction_current",
    "fire_neighbor_interaction_lag_1d",
    "fire_neighbor_interaction_lag_3d",
    "fire_neighbor_interaction_lag_7d",

    "frp_neighbor_interaction_current",
    "frp_neighbor_interaction_lag_1d",
    "frp_neighbor_interaction_lag_3d",
    "frp_neighbor_interaction_lag_7d",

    "neighbor_fire_relative_current",
    "neighbor_fire_relative_lag_1d",
    "neighbor_fire_relative_lag_3d",
    "neighbor_fire_relative_lag_7d",

    "neighbor_frp_relative_current",
    "neighbor_frp_relative_lag_1d",
    "neighbor_frp_relative_lag_3d",
    "neighbor_frp_relative_lag_7d",
]


ARM_CONFIG = {
    "v2": {
        "file": V2_FILE,
        "extra_features": [],
        "label": "V2",
    },
    "spatial": {
        "file": SPATIAL_FILE,
        "extra_features": SPATIAL_FEATURES,
        "label": "V2.3 Spatial",
    },
    "spatiotemporal": {
        "file": SPATIOTEMPORAL_FILE,
        "extra_features": SPATIAL_FEATURES + SPATIOTEMPORAL_FEATURES,
        "label": "V2.4 Spatio-Temporal",
    },
}


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print()
    print("=" * 80)
    print("ERROR")
    print("=" * 80)
    print(message)
    print("=" * 80)
    sys.exit(1)


def load_dataset(path):
    if not path.exists():
        fail(f"Missing dataset:\n{path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    return df


def validate_dataset(df, name):
    required = (
        BASE_NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [
            "date",
            "fire_count",
            "frp_sum",
        ]
        + [
            f"fire_count_t_plus_{h}d"
            for h in HORIZONS
        ]
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        fail(
            f"{name}: missing required columns:\n"
            + "\n".join(missing)
        )

    duplicate_count = df.duplicated(
        subset=["state", "district", "date"]
    ).sum()

    if duplicate_count:
        fail(
            f"{name}: found {duplicate_count} "
            "duplicate state/district/date rows."
        )

    if df[required].isna().any().any():
        missing_counts = (
            df[required]
            .isna()
            .sum()
        )

        missing_counts = missing_counts[
            missing_counts > 0
        ]

        fail(
            f"{name}: missing values detected:\n"
            f"{missing_counts}"
        )

    print(
        f"{name}: {len(df):,} rows x "
        f"{len(df.columns):,} columns"
    )


def make_split(df):
    train = df[
        df["date"].dt.year == 2023
    ].copy()

    val = df[
        df["date"].dt.year == 2024
    ].copy()

    test = df[
        df["date"].dt.year == 2025
    ].copy()

    if len(train) != 1800:
        fail(
            f"Unexpected 2023 train rows: {len(train)}"
        )

    if len(val) != 1800:
        fail(
            f"Unexpected 2024 validation rows: {len(val)}"
        )

    if len(test) != 1800:
        fail(
            f"Unexpected 2025 test rows: {len(test)}"
        )

    return train, val, test


def build_feature_matrices(
    train,
    val,
    test,
    numeric_features,
):
    # Exact V2 categorical encoding:
    # fit dummy columns on training data only.
    dummies_train = pd.get_dummies(
        train[CATEGORICAL_FEATURES],
        drop_first=False,
    )

    dummy_columns = dummies_train.columns.tolist()

    dummies_val = (
        pd.get_dummies(
            val[CATEGORICAL_FEATURES],
            drop_first=False,
        )
        .reindex(
            columns=dummy_columns,
            fill_value=0,
        )
    )

    dummies_test = (
        pd.get_dummies(
            test[CATEGORICAL_FEATURES],
            drop_first=False,
        )
        .reindex(
            columns=dummy_columns,
            fill_value=0,
        )
    )

    X_train = pd.concat(
        [
            train[numeric_features]
            .reset_index(drop=True),
            dummies_train.reset_index(drop=True),
        ],
        axis=1,
    ).to_numpy(dtype=float)

    X_val = pd.concat(
        [
            val[numeric_features]
            .reset_index(drop=True),
            dummies_val.reset_index(drop=True),
        ],
        axis=1,
    ).to_numpy(dtype=float)

    X_test = pd.concat(
        [
            test[numeric_features]
            .reset_index(drop=True),
            dummies_test.reset_index(drop=True),
        ],
        axis=1,
    ).to_numpy(dtype=float)

    if not np.isfinite(X_train).all():
        fail("Non-finite value found in X_train.")

    if not np.isfinite(X_val).all():
        fail("Non-finite value found in X_val.")

    if not np.isfinite(X_test).all():
        fail("Non-finite value found in X_test.")

    return (
        X_train,
        X_val,
        X_test,
        dummy_columns,
    )


def select_threshold(y_true, probabilities):
    best_f1 = -1.0
    best_threshold = 0.5

    for threshold in THRESHOLD_CANDIDATES:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold)


def evaluate(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )
    except ValueError:
        pr_auc = np.nan

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }, predictions


def assert_same_keys(datasets):
    base = datasets["v2"]

    base_keys = base[
        ["state", "district", "date"]
    ].copy()

    base_keys = base_keys.sort_values(
        ["state", "district", "date"]
    ).reset_index(drop=True)

    for name in ["spatial", "spatiotemporal"]:
        current = datasets[name][
            ["state", "district", "date"]
        ].copy()

        current = current.sort_values(
            ["state", "district", "date"]
        ).reset_index(drop=True)

        if not base_keys.equals(current):
            fail(
                f"Dataset key mismatch between V2 "
                f"and {name}."
            )


def assert_same_targets(datasets):
    for horizon in HORIZONS:
        target = f"fire_count_t_plus_{horizon}d"

        base = datasets["v2"][
            ["state", "district", "date", target]
        ].copy()

        base = base.sort_values(
            ["state", "district", "date"]
        ).reset_index(drop=True)

        for name in ["spatial", "spatiotemporal"]:
            current = datasets[name][
                ["state", "district", "date", target]
            ].copy()

            current = current.sort_values(
                ["state", "district", "date"]
            ).reset_index(drop=True)

            if not np.array_equal(
                base[target].to_numpy(),
                current[target].to_numpy(),
            ):
                fail(
                    f"Target mismatch for "
                    f"{name}, horizon +{horizon}d."
                )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STUBBLEAI V2.4")
    print("SPATIO-TEMPORAL INTERACTION EXPERIMENT")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\n[1/7] Loading datasets...")

    datasets = {}

    for arm_name, config in ARM_CONFIG.items():

        df = load_dataset(
            config["file"]
        )

        validate_dataset(
            df,
            config["label"],
        )

        datasets[arm_name] = df

    # --------------------------------------------------------
    # FAIRNESS CHECKS
    # --------------------------------------------------------

    print("\n[2/7] Checking identical keys and targets...")

    assert_same_keys(datasets)
    assert_same_targets(datasets)

    print("Key alignment: PASS")
    print("Target alignment: PASS")

    # --------------------------------------------------------
    # SPLITS
    # --------------------------------------------------------

    print("\n[3/7] Creating temporal splits...")

    splits = {}

    for arm_name, df in datasets.items():

        train, val, test = make_split(df)

        splits[arm_name] = {
            "train": train,
            "val": val,
            "test": test,
        }

        print(
            f"{ARM_CONFIG[arm_name]['label']}: "
            f"train={len(train)}, "
            f"val={len(val)}, "
            f"test={len(test)}"
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\n[4/7] Training 3 arms x 5 horizons...")
    print(
        "This means 15 Random Forest models."
    )

    all_results = []

    # --------------------------------------------------------
    # LOOP ARMS
    # --------------------------------------------------------

    for arm_name, config in ARM_CONFIG.items():

        print()
        print("=" * 80)
        print(
            f"ARM: {config['label']}"
        )
        print("=" * 80)

        numeric_features = (
            BASE_NUMERIC_FEATURES
            + config["extra_features"]
        )

        train = splits[arm_name]["train"]
        val = splits[arm_name]["val"]
        test = splits[arm_name]["test"]

        (
            X_train,
            X_val,
            X_test,
            dummy_columns,
        ) = build_feature_matrices(
            train,
            val,
            test,
            numeric_features,
        )

        print(
            f"Numeric features: "
            f"{len(numeric_features)}"
        )

        print(
            f"Categorical dummy features: "
            f"{len(dummy_columns)}"
        )

        print(
            f"Total model features: "
            f"{X_train.shape[1]}"
        )

        for horizon in HORIZONS:

            target_column = (
                f"fire_count_t_plus_{horizon}d"
            )

            print()
            print(
                f"--- {config['label']} "
                f"+{horizon}d ---"
            )

            y_train = (
                train[target_column]
                .to_numpy()
                > ELEVATED_THRESHOLD
            ).astype(int)

            y_val = (
                val[target_column]
                .to_numpy()
                > ELEVATED_THRESHOLD
            ).astype(int)

            y_test = (
                test[target_column]
                .to_numpy()
                > ELEVATED_THRESHOLD
            ).astype(int)

            print(
                f"Elevated rates: "
                f"train={y_train.mean():.4f}, "
                f"val={y_val.mean():.4f}, "
                f"test={y_test.mean():.4f}"
            )

            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight=None,
            )

            model.fit(
                X_train,
                y_train,
            )

            probabilities_val = (
                model.predict_proba(X_val)[:, 1]
            )

            probabilities_test = (
                model.predict_proba(X_test)[:, 1]
            )

            # Threshold ONLY from 2024.
            threshold = select_threshold(
                y_val,
                probabilities_val,
            )

            val_metrics, _ = evaluate(
                y_val,
                probabilities_val,
                threshold,
            )

            test_metrics, predictions_test = evaluate(
                y_test,
                probabilities_test,
                threshold,
            )

            print(
                f"Threshold: {threshold:.6f}"
            )

            print(
                f"Validation F1: "
                f"{val_metrics['f1']:.4f}"
            )

            print(
                f"2025 Test: "
                f"F1={test_metrics['f1']:.4f} | "
                f"ROC-AUC={test_metrics['roc_auc']:.4f} | "
                f"PR-AUC={test_metrics['pr_auc']:.4f}"
            )

            # ------------------------------------------------
            # SAVE TEST PREDICTIONS
            # ------------------------------------------------

            prediction_frame = pd.DataFrame({
                "state": test["state"].to_numpy(),
                "district": test["district"].to_numpy(),
                "date": test["date"].dt.strftime(
                    "%Y-%m-%d"
                ),
                "fire_count": test[
                    "fire_count"
                ].to_numpy(),
                "frp_sum": test[
                    "frp_sum"
                ].to_numpy(),
                "horizon": f"+{horizon}d",
                "target_fire_count": test[
                    target_column
                ].to_numpy(),
                "actual_elevated": y_test,
                "rf_probability": probabilities_test,
                "rf_prediction": predictions_test,
                "threshold": threshold,
                "error_type": np.where(
                    (predictions_test == 1)
                    & (y_test == 1),
                    "TP",
                    np.where(
                        (predictions_test == 1)
                        & (y_test == 0),
                        "FP",
                        np.where(
                            (predictions_test == 0)
                            & (y_test == 1),
                            "FN",
                            "TN",
                        ),
                    ),
                ),
            })

            prediction_file = (
                PREDICTIONS_DIR
                / f"{arm_name}_predictions_plus{horizon}d.csv"
            )

            prediction_frame.to_csv(
                prediction_file,
                index=False,
            )

            # ------------------------------------------------
            # RESULT ROW
            # ------------------------------------------------

            row = {
                "arm": arm_name,
                "arm_label": config["label"],
                "horizon": f"+{horizon}d",
                "train_rows": len(train),
                "validation_rows": len(val),
                "test_rows": len(test),
                "numeric_features": len(
                    numeric_features
                ),
                "categorical_dummy_features": len(
                    dummy_columns
                ),
                "total_features": X_train.shape[1],
                "threshold": threshold,
                "validation_f1": val_metrics["f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "tn": test_metrics["tn"],
                "fp": test_metrics["fp"],
                "fn": test_metrics["fn"],
                "tp": test_metrics["tp"],
            }

            all_results.append(row)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    print("\n[5/7] Saving experiment results...")

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        ["horizon", "arm"]
    ).reset_index(drop=True)

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # SANITY CHECK V2 ARM
    # --------------------------------------------------------

    print("\n[6/7] Checking frozen V2 reproduction...")

    expected_thresholds = {
        "+1d": 0.410909,
        "+2d": 0.366364,
        "+3d": 0.297071,
        "+5d": 0.356465,
        "+7d": 0.331717,
    }

    expected_f1 = {
        "+1d": 0.7754,
        "+2d": 0.7513,
        "+3d": 0.7305,
        "+5d": 0.7303,
        "+7d": 0.7085,
    }

    v2_results = results_df[
        results_df["arm"] == "v2"
    ]

    reproduction_pass = True

    for _, row in v2_results.iterrows():

        horizon = row["horizon"]

        threshold_ok = np.isclose(
            row["threshold"],
            expected_thresholds[horizon],
            atol=1e-6,
        )

        f1_ok = np.isclose(
            row["test_f1"],
            expected_f1[horizon],
            atol=1e-4,
        )

        print(
            f"{horizon}: "
            f"threshold={'PASS' if threshold_ok else 'FAIL'} "
            f"F1={'PASS' if f1_ok else 'FAIL'}"
        )

        if not threshold_ok or not f1_ok:
            reproduction_pass = False

    if not reproduction_pass:
        fail(
            "FROZEN V2 REPRODUCTION FAILED.\n"
            "Do NOT interpret V2.4 results.\n"
            "The training protocol does not reproduce "
            "the frozen V2 benchmark."
        )

    print(
        "Frozen V2 reproduction: PASS"
    )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config_output = {
        "experiment": "V2.4 Spatio-Temporal Interaction",
        "research_question": (
            "Can interaction between local fire dynamics "
            "and neighboring-district fire activity improve "
            "multi-horizon forecasting?"
        ),
        "arms": {
            "v2": "Frozen V2 baseline",
            "spatial": "V2.3 spatial context",
            "spatiotemporal": (
                "V2.4 spatio-temporal interactions"
            ),
        },
        "horizons": HORIZONS,
        "elevated_target": (
            "fire_count_t_plus_Hd > 2"
        ),
        "train_year": 2023,
        "validation_year": 2024,
        "test_year": 2025,
        "threshold_selection": (
            "2024 validation Elevated-class F1"
        ),
        "threshold_candidates": (
            "np.linspace(0.01, 0.99, 199)"
        ),
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "class_weight": None,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "base_numeric_features": BASE_NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "spatial_features": SPATIAL_FEATURES,
        "spatiotemporal_features": SPATIOTEMPORAL_FEATURES,
        "frozen_test": True,
        "v2_reproduction_pass": True,
        "outputs": {
            "results": str(RESULTS_FILE),
            "predictions": str(PREDICTIONS_DIR),
        },
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config_output,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n[7/7] FINAL V2.4 SUMMARY")
    print("=" * 80)

    summary = results_df[
        [
            "arm_label",
            "horizon",
            "threshold",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc",
        ]
    ].copy()

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 80)
    print("V2.4 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Results: {RESULTS_FILE}")
    print(f"Predictions: {PREDICTIONS_DIR}")
    print(f"Config: {CONFIG_FILE}")
    print()
    print(
        "IMPORTANT: 2025 was used only as frozen test data."
    )
    print(
        "No V2/V2.3 frozen benchmark file was modified."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()