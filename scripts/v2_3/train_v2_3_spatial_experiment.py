import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


ROOT = Path(__file__).resolve().parent

DATASET = (
    ROOT
    / "research"
    / "v2"
    / "v2_3"
    / "spatial"
    / "ml_dataset_v2_3_spatial.csv"
)

OUT_DIR = (
    ROOT
    / "research"
    / "v2"
    / "v2_3"
    / "spatial"
)

RESULTS_FILE = OUT_DIR / "v2_3_spatial_results.csv"
CONFIG_FILE = OUT_DIR / "v2_3_spatial_config.json"


HORIZONS = [1, 2, 3, 5, 7]
TARGET_THRESHOLD = 2
SEED = 42

NUMERIC_FEATURES = [
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
    "day_of_year",
    "sin_year",
    "cos_year",
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

CATEGORICAL_FEATURES = [
    "state",
    "district",
]


def evaluate(y_true, probability, threshold):

    prediction = (probability >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, prediction)
        ),
        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probability
            )
        )
    }


def main():

    print("=" * 70)
    print("V2.3 EXPERIMENT 2 — SPATIAL RANDOM FOREST")
    print("=" * 70)

    df = pd.read_csv(DATASET)

    print("\nDataset:", df.shape)

    df["date"] = pd.to_datetime(df["date"])

    all_results = []
    thresholds = {}
    feature_counts = {}

    for horizon in HORIZONS:

        print("\n" + "-" * 70)
        print(f"HORIZON +{horizon} DAY")
        print("-" * 70)

        target = f"fire_count_t_plus_{horizon}d"

        if target not in df.columns:
            raise ValueError(
                f"Missing target column: {target}"
            )

        data = df.copy()

        data["target"] = (
            data[target] > TARGET_THRESHOLD
        ).astype(int)

        train = data[
            data["year"] == 2023
        ].copy()

        val = data[
            data["year"] == 2024
        ].copy()

        test = data[
            data["year"] == 2025
        ].copy()

        print(
            "Train:",
            train.shape,
            "Validation:",
            val.shape,
            "Test:",
            test.shape
        )

        feature_columns = (
            NUMERIC_FEATURES
            + SPATIAL_FEATURES
        )

        dummies_train = pd.get_dummies(
            train[CATEGORICAL_FEATURES],
            drop_first=False
        )

        dummy_columns = dummies_train.columns.tolist()

        dummies_val = pd.get_dummies(
            val[CATEGORICAL_FEATURES],
            drop_first=False
        ).reindex(
            columns=dummy_columns,
            fill_value=0
        )

        dummies_test = pd.get_dummies(
            test[CATEGORICAL_FEATURES],
            drop_first=False
        ).reindex(
            columns=dummy_columns,
            fill_value=0
        )

        X_train = pd.concat(
            [
                train[feature_columns].reset_index(drop=True),
                dummies_train.reset_index(drop=True)
            ],
            axis=1
        ).to_numpy(dtype=float)

        X_val = pd.concat(
            [
                val[feature_columns].reset_index(drop=True),
                dummies_val.reset_index(drop=True)
            ],
            axis=1
        ).to_numpy(dtype=float)

        X_test = pd.concat(
            [
                test[feature_columns].reset_index(drop=True),
                dummies_test.reset_index(drop=True)
            ],
            axis=1
        ).to_numpy(dtype=float)

        y_train = train["target"].to_numpy()
        y_val = val["target"].to_numpy()
        y_test = test["target"].to_numpy()

        print(
            "Features:",
            len(feature_columns),
            "+",
            len(dummy_columns),
            "categorical dummy columns"
        )

        model = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        val_probability = model.predict_proba(
            X_val
        )[:, 1]

        test_probability = model.predict_proba(
            X_test
        )[:, 1]

        best_threshold = None
        best_f1 = -1

        for threshold in np.linspace(
            0.01,
            0.99,
            199
        ):

            prediction = (
                val_probability >= threshold
            ).astype(int)

            score = f1_score(
                y_val,
                prediction,
                zero_division=0
            )

            if score > best_f1:

                best_f1 = score
                best_threshold = float(
                    threshold
                )

        val_metrics = evaluate(
            y_val,
            val_probability,
            best_threshold
        )

        test_metrics = evaluate(
            y_test,
            test_probability,
            best_threshold
        )

        thresholds[f"+{horizon}d"] = (
            best_threshold
        )

        feature_counts[f"+{horizon}d"] = (
            len(feature_columns)
        )

        prediction_output = test[
            [
                "state",
                "district",
                "date",
                "fire_count",
                "frp_sum",
                target
            ]
        ].copy()

        prediction_output["horizon"] = horizon
        prediction_output["target_fire_count"] = (
            prediction_output[target]
        )
        prediction_output["actual_elevated"] = y_test
        prediction_output["spatial_probability"] = (
            test_probability
        )
        prediction_output["spatial_prediction"] = (
            test_probability >= best_threshold
        ).astype(int)

        prediction_output = prediction_output[
            [
                "state",
                "district",
                "date",
                "fire_count",
                "frp_sum",
                "horizon",
                "target_fire_count",
                "actual_elevated",
                "spatial_probability",
                "spatial_prediction"
            ]
        ]

        prediction_output.to_csv(
            OUT_DIR
            / f"spatial_rf_predictions_plus{horizon}d.csv",
            index=False
        )

        row = {
            "horizon": horizon,
            "threshold": best_threshold,

            "validation_accuracy":
                val_metrics["accuracy"],
            "validation_precision":
                val_metrics["precision"],
            "validation_recall":
                val_metrics["recall"],
            "validation_f1":
                val_metrics["f1"],

            "test_accuracy":
                test_metrics["accuracy"],
            "test_precision":
                test_metrics["precision"],
            "test_recall":
                test_metrics["recall"],
            "test_f1":
                test_metrics["f1"],
            "test_roc_auc":
                test_metrics["roc_auc"],
            "test_pr_auc":
                test_metrics["pr_auc"],

            "n_train": len(train),
            "n_validation": len(val),
            "n_test": len(test),
            "feature_count": len(feature_columns),
            "dummy_feature_count": len(dummy_columns)
        }

        all_results.append(row)

        print(
            f"Validation F1: "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"Selected threshold: "
            f"{best_threshold:.6f}"
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

    results = pd.DataFrame(
        all_results
    )

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    config = {
        "experiment": "V2.3 spatial neighbor context",
        "dataset": str(DATASET),
        "horizons": HORIZONS,
        "target_definition":
            "fire_count_t_plus_H > 2",
        "train_year": 2023,
        "validation_year": 2024,
        "test_year": 2025,
        "threshold_candidates":
            "np.linspace(0.01, 0.99, 199)",
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1
        },
        "spatial_features":
            SPATIAL_FEATURES,
        "thresholds": thresholds,
        "feature_counts": feature_counts,
        "frozen_v2_modified": False
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config,
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("FINAL V2.3 SPATIAL RESULTS")
    print("=" * 70)

    print(
        results[
            [
                "horizon",
                "threshold",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
                "test_pr_auc"
            ]
        ].to_string(index=False)
    )

    print("\nResults:", RESULTS_FILE)
    print("Config:", CONFIG_FILE)

    print("\nFrozen V2 files were not modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()

