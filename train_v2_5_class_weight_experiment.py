from pathlib import Path
import json
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


BASE_DIR = Path(__file__).resolve().parent
V2_FILE = BASE_DIR / "ml_dataset_v2.csv"

OUTPUT_DIR = BASE_DIR / "research" / "v2" / "v2_5"
PRED_DIR = OUTPUT_DIR / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)


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

CATEGORICAL_FEATURES = ["state", "district"]

CLASS_WEIGHTS = {
    "baseline": None,
    "balanced": "balanced",
    "positive_1_5": {0: 1, 1: 1.5},
    "positive_2": {0: 1, 1: 2},
    "positive_3": {0: 1, 1: 3},
}

EXPECTED_V2_THRESHOLDS = {
    1: 0.4109090909090909,
    2: 0.36636363636363634,
    3: 0.29707070707070707,
    5: 0.3564646464646465,
    7: 0.3317171717171717,
}

EXPECTED_V2_F1 = {
    1: 0.7754,
    2: 0.7513,
    3: 0.7305,
    5: 0.7303,
    7: 0.7085,
}


def prepare_features(train_df, other_df=None):
    """Create numeric + categorical feature matrix.

    Dummy columns are learned from the training split only.
    """
    if other_df is None:
        other_df = train_df

    train_part = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    other_part = other_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    train_encoded = pd.get_dummies(
        train_part,
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
    )

    other_encoded = pd.get_dummies(
        other_part,
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
    )

    other_encoded = other_encoded.reindex(
        columns=train_encoded.columns,
        fill_value=0,
    )

    return train_encoded, other_encoded


def select_threshold(y_true, probabilities):
    thresholds = np.linspace(0.01, 0.99, 199)

    best_threshold = None
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def transition_metrics(y_true, predictions):
    current_elevated = None


def main():
    if not V2_FILE.exists():
        raise FileNotFoundError(f"Missing frozen V2 dataset: {V2_FILE}")

    df = pd.read_csv(V2_FILE)

    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    for horizon in HORIZONS:
        required_columns.add(f"fire_count_t_plus_{horizon}d")

    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(
            "Frozen V2 dataset is missing required columns: "
            + ", ".join(missing)
        )

    df["date"] = pd.to_datetime(df["date"])

    train_df = df[df["year"] == 2023].copy()
    validation_df = df[df["year"] == 2024].copy()
    test_df = df[df["year"] == 2025].copy()

    print("=" * 80)
    print("V2.5 COST-SENSITIVE EMERGING-FIRE EXPERIMENT")
    print("=" * 80)

    print(f"Dataset: {V2_FILE}")
    print(
        f"Splits: train={len(train_df)}, "
        f"validation={len(validation_df)}, "
        f"test={len(test_df)}"
    )

    X_train, X_validation = prepare_features(
        train_df,
        validation_df,
    )

    _, X_test = prepare_features(
        train_df,
        test_df,
    )

    print(f"Feature matrix columns: {X_train.shape[1]}")

    all_results = []
    config_results = []

    for arm_name, class_weight in CLASS_WEIGHTS.items():

        print()
        print("-" * 80)
        print(f"ARM: {arm_name}")
        print(f"class_weight = {class_weight}")
        print("-" * 80)

        for horizon in HORIZONS:

            target_column = f"fire_count_t_plus_{horizon}d"

            y_train = (
                train_df[target_column].to_numpy() > 2
            ).astype(int)

            y_validation = (
                validation_df[target_column].to_numpy() > 2
            ).astype(int)

            y_test = (
                test_df[target_column].to_numpy() > 2
            ).astype(int)

            model = RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight=class_weight,
            )

            model.fit(X_train, y_train)

            validation_probability = model.predict_proba(
                X_validation
            )[:, 1]

            threshold, validation_f1 = select_threshold(
                y_validation,
                validation_probability,
            )

            test_probability = model.predict_proba(
                X_test
            )[:, 1]

            metrics = calculate_metrics(
                y_test,
                test_probability,
                threshold,
            )

            predictions = (
                test_probability >= threshold
            ).astype(int)

            result = {
                "arm": arm_name,
                "class_weight": str(class_weight),
                "horizon_days": horizon,
                "threshold": threshold,
                "validation_f1": validation_f1,
                **metrics,
            }

            all_results.append(result)

            prediction_output = pd.DataFrame({
                "state": test_df["state"].to_numpy(),
                "district": test_df["district"].to_numpy(),
                "date": test_df["date"].dt.strftime("%Y-%m-%d"),
                "fire_count": test_df["fire_count"].to_numpy(),
                "frp_sum": test_df["frp_sum"].to_numpy(),
                "horizon": horizon,
                "target_fire_count": test_df[target_column].to_numpy(),
                "actual_elevated": y_test,
                "probability": test_probability,
                "prediction": predictions,
            })

            prediction_file = (
                PRED_DIR
                / f"{arm_name}_predictions_plus{horizon}d.csv"
            )

            prediction_output.to_csv(
                prediction_file,
                index=False,
            )

            print(
                f"+{horizon}d | "
                f"threshold={threshold:.6f} | "
                f"val_F1={validation_f1:.4f} | "
                f"test_F1={metrics['f1']:.4f} | "
                f"emerging recall={metrics['recall']:.4f}"
            )

            if arm_name == "baseline":
                expected_threshold = EXPECTED_V2_THRESHOLDS[horizon]
                expected_f1 = EXPECTED_V2_F1[horizon]

                if abs(threshold - expected_threshold) > 1e-9:
                    raise RuntimeError(
                        f"Frozen V2 threshold reproduction failed "
                        f"for +{horizon}d: "
                        f"{threshold} != {expected_threshold}"
                    )

                if abs(metrics["f1"] - expected_f1) > 0.0002:
                    raise RuntimeError(
                        f"Frozen V2 F1 reproduction failed "
                        f"for +{horizon}d: "
                        f"{metrics['f1']} != {expected_f1}"
                    )

    results_df = pd.DataFrame(all_results)

    results_file = (
        OUTPUT_DIR / "v2_5_class_weight_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    config = {
        "experiment": "V2.5 Cost-Sensitive Emerging-Fire Experiment",
        "dataset": str(V2_FILE),
        "horizons": HORIZONS,
        "target_rule": "fire_count_t_plus_H > 2",
        "train_year": 2023,
        "validation_year": 2024,
        "test_year": 2025,
        "test_tuning": False,
        "model": {
            "type": "RandomForestClassifier",
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        },
        "class_weights": {
            name: str(weight)
            for name, weight in CLASS_WEIGHTS.items()
        },
        "numeric_feature_count": len(NUMERIC_FEATURES),
        "categorical_features": CATEGORICAL_FEATURES,
        "threshold_selection": {
            "range": [0.01, 0.99],
            "count": 199,
            "objective": "validation Elevated-class F1",
        },
        "frozen_v2_reproduction_guard": True,
    }

    config_file = (
        OUTPUT_DIR / "v2_5_class_weight_config.json"
    )

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print()
    print("=" * 80)
    print("FROZEN V2 REPRODUCTION GUARD: PASS")
    print("=" * 80)

    print()
    print("RESULTS")
    print(results_df.to_string(index=False))

    print()
    print(f"Results: {results_file}")
    print(f"Config:  {config_file}")
    print(f"Predictions: {PRED_DIR}")


if __name__ == "__main__":
    main()
