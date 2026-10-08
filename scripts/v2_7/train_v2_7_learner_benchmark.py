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
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "v2" / "ml_dataset_v2.csv"

OUT_DIR = PROJECT_ROOT / "research" / "v2" / "v2_7"
RESULT_DIR = OUT_DIR / "results"
PRED_DIR = OUT_DIR / "predictions"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
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

CATEGORICAL_FEATURES = [
    "state",
    "district",
]

OFFICIAL_RF_F1 = {
    1: 0.7754,
    2: 0.7513,
    3: 0.7305,
    5: 0.7303,
    7: 0.7085,
}


def prepare_design_matrices(train_df, other_df):
    """
    Reproduce frozen V2 encoding exactly:
    - numeric features unchanged
    - state/district one-hot encoded using training categories
    - no drop_first
    """

    train_x = train_df[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    other_x = other_df[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    train_x = pd.get_dummies(
        train_x,
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
    )

    other_x = pd.get_dummies(
        other_x,
        columns=CATEGORICAL_FEATURES,
        drop_first=False,
    )

    other_x = other_x.reindex(
        columns=train_x.columns,
        fill_value=0,
    )

    train_x = train_x.astype(float)
    other_x = other_x.astype(float)

    return train_x, other_x


def target_column(horizon):
    return f"fire_count_t_plus_{horizon}d"


def evaluate_predictions(
    y_true,
    probability,
    threshold,
):
    probability = np.clip(
        np.asarray(probability),
        1e-7,
        1 - 1e-7,
    )

    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(
            y_true,
            prediction,
        ),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probability,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probability,
        ),
        "brier": brier_score_loss(
            y_true,
            probability,
        ),
        "logloss": log_loss(
            y_true,
            probability,
        ),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(y_true, probability):
    thresholds = np.linspace(
        0.01,
        0.99,
        199,
    )

    best_threshold = None
    best_f1 = -1.0

    for threshold in thresholds:

        prediction = (
            probability >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            prediction,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def transition_metrics(
    df,
    prediction_column,
):
    rows = []

    for transition in [
        "N->N",
        "N->E",
        "E->N",
        "E->E",
    ]:

        part = df[
            df["transition_actual"] == transition
        ]

        if len(part) == 0:
            continue

        actual = part["actual_elevated"].astype(int)
        pred = part[prediction_column].astype(int)

        if transition == "N->E":
            metric_name = "n_to_e_recall"
            value = (
                ((pred == 1) & (actual == 1)).sum()
                / max((actual == 1).sum(), 1)
            )
        elif transition == "E->E":
            metric_name = "e_to_e_recall"
            value = (
                ((pred == 1) & (actual == 1)).sum()
                / max((actual == 1).sum(), 1)
            )
        else:
            metric_name = "transition_accuracy"
            value = (pred == actual).mean()

        rows.append(
            {
                "transition": transition,
                "metric": metric_name,
                "value": value,
                "n": len(part),
            }
        )

    return rows


def make_transition_labels(
    df,
    horizon,
):
    current_elevated = (
        df["fire_count"] > 2
    )

    future_elevated = (
        df[
            target_column(horizon)
        ] > 2
    )

    labels = np.select(
        [
            (~current_elevated)
            & (~future_elevated),

            (~current_elevated)
            & future_elevated,

            current_elevated
            & (~future_elevated),

            current_elevated
            & future_elevated,
        ],
        [
            "N->N",
            "N->E",
            "E->N",
            "E->E",
        ],
        default="UNKNOWN",
    )

    return labels


def train_rf(X_train, y_train):
    return RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight=None,
    ).fit(
        X_train,
        y_train,
    )


def train_xgb(X_train, y_train):
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    ).fit(
        X_train,
        y_train,
    )


def train_lgbm(X_train, y_train):
    return LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    ).fit(
        X_train,
        y_train,
    )


print("=" * 80)
print("STUBBLEAI V2.7 LEARNER BENCHMARK")
print("=" * 80)

df = pd.read_csv(DATA_PATH)

print("Dataset:", df.shape)

required_columns = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + ["date", "year", "fire_count"]
)

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df["date"] = pd.to_datetime(df["date"])

train_df = df[
    df["year"] == 2023
].copy()

val_df = df[
    df["year"] == 2024
].copy()

test_df = df[
    df["year"] == 2025
].copy()

print(
    "Split:",
    len(train_df),
    len(val_df),
    len(test_df),
)

# Frozen V2 feature encoding
X_train, X_val = prepare_design_matrices(
    train_df,
    val_df,
)

_, X_test = prepare_design_matrices(
    train_df,
    test_df,
)

print(
    "Design matrix:",
    X_train.shape,
)

# Ensure no leakage through index/order
assert list(X_train.columns) == list(
    X_val.columns
)
assert list(X_train.columns) == list(
    X_test.columns
)

all_results = []
all_predictions = []
all_transition_rows = []

models = [
    ("rf", train_rf),
    ("xgboost", train_xgb),
    ("lightgbm", train_lgbm),
]

for horizon in HORIZONS:

    print("\n" + "=" * 80)
    print(f"HORIZON +{horizon}D")
    print("=" * 80)

    target = target_column(horizon)

    y_train = (
        train_df[target] > 2
    ).astype(int)

    y_val = (
        val_df[target] > 2
    ).astype(int)

    y_test = (
        test_df[target] > 2
    ).astype(int)

    for model_name, trainer in models:

        print(
            f"\nTraining {model_name}..."
        )

        model = trainer(
            X_train,
            y_train,
        )

        val_probability = model.predict_proba(
            X_val
        )[:, 1]

        test_probability = model.predict_proba(
            X_test
        )[:, 1]

        threshold, val_f1 = choose_threshold(
            y_val,
            val_probability,
        )

        val_metrics = evaluate_predictions(
            y_val,
            val_probability,
            threshold,
        )

        test_metrics = evaluate_predictions(
            y_test,
            test_probability,
            threshold,
        )

        # Frozen V2 RF reproduction guard
        if model_name == "rf":

            official = OFFICIAL_RF_F1[horizon]

            if abs(
                test_metrics["f1"] - official
            ) > 1e-4:

                raise RuntimeError(
                    f"FROZEN RF REPRODUCTION FAILED "
                    f"+{horizon}d: "
                    f"got {test_metrics['f1']:.6f}, "
                    f"expected approximately "
                    f"{official:.4f}"
                )

            print(
                "RF reproduction guard: PASS"
            )

        print(
            f"{model_name}: "
            f"threshold={threshold:.6f} "
            f"val_F1={val_metrics['f1']:.6f} "
            f"test_F1={test_metrics['f1']:.6f} "
            f"test_P={test_metrics['precision']:.6f} "
            f"test_R={test_metrics['recall']:.6f} "
            f"ROC={test_metrics['roc_auc']:.6f} "
            f"PR={test_metrics['pr_auc']:.6f}"
        )

        # Build test diagnostic frame
        pred_df = test_df[
            [
                "state",
                "district",
                "date",
                "fire_count",
            ]
        ].copy()

        pred_df["horizon"] = horizon
        pred_df["model"] = model_name
        pred_df["target_fire_count"] = test_df[
            target
        ].values

        pred_df["actual_elevated"] = (
            y_test.values
        )

        pred_df["probability"] = (
            test_probability
        )

        pred_df["prediction"] = (
            test_probability >= threshold
        ).astype(int)

        pred_df["transition_actual"] = (
            make_transition_labels(
                test_df,
                horizon,
            )
        )

        all_predictions.append(
            pred_df
        )

        # Transition diagnostics
        transition_frame = pred_df.copy()

        transition_rows = transition_metrics(
            transition_frame,
            "prediction",
        )

        for row in transition_rows:
            all_transition_rows.append(
                {
                    "horizon": horizon,
                    "model": model_name,
                    **row,
                }
            )

        all_results.append(
            {
                "horizon": horizon,
                "model": model_name,

                "validation_f1":
                    val_metrics["f1"],

                "validation_threshold":
                    threshold,

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

                "test_brier":
                    test_metrics["brier"],

                "test_logloss":
                    test_metrics["logloss"],

                "tn":
                    test_metrics["tn"],

                "fp":
                    test_metrics["fp"],

                "fn":
                    test_metrics["fn"],

                "tp":
                    test_metrics["tp"],
            }
        )


results_df = pd.DataFrame(
    all_results
)

transition_df = pd.DataFrame(
    all_transition_rows
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

results_df.to_csv(
    RESULT_DIR / "v2_7_learner_results.csv",
    index=False,
)

transition_df.to_csv(
    RESULT_DIR
    / "v2_7_transition_diagnostics.csv",
    index=False,
)

predictions_df.to_csv(
    PRED_DIR
    / "v2_7_test_predictions.csv",
    index=False,
)

config = {
    "experiment": "V2.7 learner benchmark",
    "dataset": "ml_dataset_v2.csv",
    "features": "Frozen V2 36 numeric + state/district one-hot",
    "train_year": 2023,
    "validation_year": 2024,
    "test_year": 2025,
    "horizons": HORIZONS,
    "threshold_candidates": 199,
    "threshold_range": [0.01, 0.99],
    "random_state": 42,
    "rf": {
        "n_estimators": 400,
        "min_samples_leaf": 2,
        "class_weight": None,
    },
    "xgboost": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
    },
    "lightgbm": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "2025_tuning": False,
}

with open(
    RESULT_DIR / "v2_7_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )

print("\n" + "=" * 80)
print("V2.7 COMPLETE")
print("=" * 80)

print("\nOverall results:")
print(
    results_df[
        [
            "horizon",
            "model",
            "validation_f1",
            "validation_threshold",
            "test_f1",
            "test_precision",
            "test_recall",
            "test_roc_auc",
            "test_pr_auc",
            "test_brier",
            "test_logloss",
        ]
    ]
    .round(5)
    .to_string(index=False)
)

print("\nTransition diagnostics:")
print(
    transition_df.round(5).to_string(
        index=False
    )
)

print("\nArtifacts:")
print(OUT_DIR)
