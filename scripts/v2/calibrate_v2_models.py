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
    brier_score_loss,
    log_loss,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# STUBBLEAI V2 - TEMPORAL PROBABILITY CALIBRATION
# ============================================================
#
# Purpose:
#   Evaluate whether probability calibration improves the
#   reliability of V2 model probabilities.
#
# Models:
#   1. Logistic Regression
#   2. Random Forest
#   3. XGBoost
#   4. LightGBM
#
# Frozen model-training split:
#   2023 -> model training
#
# Calibration protocol:
#   2024-10-15 -> 2024-11-03
#       calibration fitting
#
#   2024-11-04 -> 2024-11-23
#       calibration selection/check
#
#   2025-10-15 -> 2025-11-23
#       untouched final test
#
# Target:
#   fire_count_t_plus_7d > 2 -> Elevated (1)
#
# IMPORTANT:
#   This script does NOT modify the V2 benchmark files.
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ML_FILE = BASE_DIR / "ml_dataset_v2.csv"

OUTPUT_DIR = BASE_DIR / "research" / "v2" / "calibration"

RESULTS_FILE = OUTPUT_DIR / "calibration_results.csv"
SUMMARY_FILE = OUTPUT_DIR / "CALIBRATION_RESULTS.md"
CONFIG_FILE = OUTPUT_DIR / "calibration_config.json"


HORIZON = 7
TARGET_COLUMN = f"fire_count_t_plus_{HORIZON}d"

ELEVATED_THRESHOLD = 2
RANDOM_STATE = 42

CALIBRATION_START = pd.Timestamp("2024-10-15")
CALIBRATION_FIT_END = pd.Timestamp("2024-11-03")

CALIBRATION_CHECK_START = pd.Timestamp("2024-11-04")
CALIBRATION_CHECK_END = pd.Timestamp("2024-11-23")

TEST_START = pd.Timestamp("2025-10-15")
TEST_END = pd.Timestamp("2025-11-23")


print("=" * 75)
print("STUBBLEAI V2 - TEMPORAL PROBABILITY CALIBRATION")
print("=" * 75)
print(f"Horizon: +{HORIZON}d")
print(f"Target: {TARGET_COLUMN} > {ELEVATED_THRESHOLD}")
print()
print("Temporal protocol:")
print("  2023              -> model training")
print("  2024-10-15/11-03  -> calibration fitting")
print("  2024-11-04/11-23  -> calibration check")
print("  2025-10-15/11-23  -> untouched final test")
print("=" * 75)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/8] Loading ML dataset...")

if not ML_FILE.exists():
    print(f"ERROR: Missing {ML_FILE}")
    sys.exit(1)

df = pd.read_csv(ML_FILE)
df["date"] = pd.to_datetime(df["date"])

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")


# ============================================================
# 2. FEATURE DEFINITION
# ============================================================

print("\n[2/8] Defining V2 features...")

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


# ============================================================
# 3. TEMPORAL DATA SPLITS
# ============================================================

print("\n[3/8] Creating temporal splits...")

train = df[df["date"].dt.year == 2023].copy()

cal_fit = df[
    (df["date"] >= CALIBRATION_START)
    & (df["date"] <= CALIBRATION_FIT_END)
].copy()

cal_check = df[
    (df["date"] >= CALIBRATION_CHECK_START)
    & (df["date"] <= CALIBRATION_CHECK_END)
].copy()

test = df[
    (df["date"] >= TEST_START)
    & (df["date"] <= TEST_END)
].copy()


print(f"Model training 2023:       {len(train):,}")
print(f"Calibration fitting:       {len(cal_fit):,}")
print(f"Calibration check:         {len(cal_check):,}")
print(f"Final test 2025:           {len(test):,}")

print()
print(
    f"Calibration fit dates: "
    f"{cal_fit['date'].min().date()} -> {cal_fit['date'].max().date()}"
)
print(
    f"Calibration check dates: "
    f"{cal_check['date'].min().date()} -> {cal_check['date'].max().date()}"
)
print(
    f"Final test dates: "
    f"{test['date'].min().date()} -> {test['date'].max().date()}"
)


# ============================================================
# 4. BUILD FEATURES
# ============================================================

print("\n[4/8] Encoding categoricals using training data only...")

dummies_train = pd.get_dummies(
    train[CATEGORICAL_FEATURES],
    drop_first=False,
)

DUMMY_COLUMNS = dummies_train.columns.tolist()


def build_matrix(frame):
    dummies = pd.get_dummies(
        frame[CATEGORICAL_FEATURES],
        drop_first=False,
    ).reindex(
        columns=DUMMY_COLUMNS,
        fill_value=0,
    )

    matrix = pd.concat(
        [
            frame[NUMERIC_FEATURES].reset_index(drop=True),
            dummies.reset_index(drop=True),
        ],
        axis=1,
    )

    return matrix.to_numpy(dtype=float)


X_train = build_matrix(train)
X_cal_fit = build_matrix(cal_fit)
X_cal_check = build_matrix(cal_check)
X_test = build_matrix(test)


y_train = (
    train[TARGET_COLUMN] > ELEVATED_THRESHOLD
).astype(int).to_numpy()

y_cal_fit = (
    cal_fit[TARGET_COLUMN] > ELEVATED_THRESHOLD
).astype(int).to_numpy()

y_cal_check = (
    cal_check[TARGET_COLUMN] > ELEVATED_THRESHOLD
).astype(int).to_numpy()

y_test = (
    test[TARGET_COLUMN] > ELEVATED_THRESHOLD
).astype(int).to_numpy()


print(f"Feature matrix shape: {X_train.shape}")

print(
    f"Elevated rate - train:       {y_train.mean() * 100:.2f}%"
)
print(
    f"Elevated rate - cal fit:     {y_cal_fit.mean() * 100:.2f}%"
)
print(
    f"Elevated rate - cal check:   {y_cal_check.mean() * 100:.2f}%"
)
print(
    f"Elevated rate - test:        {y_test.mean() * 100:.2f}%"
)


if np.isnan(X_train).any():
    print("ERROR: NaN values detected in training features.")
    sys.exit(1)


# ============================================================
# 5. CALIBRATION HELPERS
# ============================================================

def fit_sigmoid_calibrator(raw_probs, y_true):
    """
    Platt/sigmoid calibration.

    A logistic regression is fitted on the model's log-odds.
    This learns a monotonic mapping:

        raw probability -> calibrated probability
    """

    eps = 1e-6

    clipped = np.clip(raw_probs, eps, 1 - eps)

    logit = np.log(
        clipped / (1.0 - clipped)
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        random_state=RANDOM_STATE,
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(logit, y_true)

    return calibrator


def apply_sigmoid_calibrator(calibrator, raw_probs):
    eps = 1e-6

    clipped = np.clip(raw_probs, eps, 1 - eps)

    logit = np.log(
        clipped / (1.0 - clipped)
    ).reshape(-1, 1)

    calibrated = calibrator.predict_proba(logit)[:, 1]

    return calibrated


def expected_calibration_error(
    y_true,
    probs,
    n_bins=10,
):
    """
    Expected Calibration Error.

    Equal-width probability bins from 0 to 1.
    """

    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (
                (probs >= bins[i])
                & (probs <= bins[i + 1])
            )
        else:
            mask = (
                (probs >= bins[i])
                & (probs < bins[i + 1])
            )

        if not np.any(mask):
            continue

        bin_accuracy = y_true[mask].mean()
        bin_confidence = probs[mask].mean()

        bin_weight = mask.mean()

        ece += (
            bin_weight
            * abs(bin_accuracy - bin_confidence)
        )

    return float(ece)


def probability_metrics(y_true, probs):
    """
    Metrics focused on probability quality.
    """

    return {
        "brier_score": float(
            brier_score_loss(y_true, probs)
        ),
        "log_loss": float(
            log_loss(
                y_true,
                np.column_stack(
                    [1 - probs, probs]
                ),
                labels=[0, 1],
            )
        ),
        "ece_10": expected_calibration_error(
            y_true,
            probs,
            n_bins=10,
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probs)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probs)
        ),
    }


def decision_metrics(
    y_true,
    probs,
    threshold,
):
    predictions = (
        probs >= threshold
    ).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }


def select_threshold(y_true, probs):
    """
    Same threshold-search principle used by V2 training:
    maximize F1 on the validation/development data.
    """

    candidates = np.linspace(
        0.01,
        0.99,
        199,
    )

    best_f1 = -1.0
    best_threshold = 0.5

    for threshold in candidates:

        predictions = (
            probs >= threshold
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


# ============================================================
# 6. MODEL DEFINITIONS
# ============================================================

print("\n[5/8] Building V2 models...")

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


# ============================================================
# 7. TRAIN -> CALIBRATE -> TEST
# ============================================================

print("\n[6/8] Training and calibrating models...")

all_results = []
calibration_configs = {}

for model_name, model in models.items():

    print()
    print("-" * 75)
    print(f"MODEL: {model_name}")
    print("-" * 75)

    # --------------------------------------------------------
    # Train original V2 model on 2023 only
    # --------------------------------------------------------

    print("Training on 2023...")

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Raw probabilities
    # --------------------------------------------------------

    raw_cal_fit = model.predict_proba(
        X_cal_fit
    )[:, 1]

    raw_cal_check = model.predict_proba(
        X_cal_check
    )[:, 1]

    raw_test = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Baseline raw-probability metrics
    # --------------------------------------------------------

    raw_cal_fit_metrics = probability_metrics(
        y_cal_fit,
        raw_cal_fit,
    )

    raw_cal_check_metrics = probability_metrics(
        y_cal_check,
        raw_cal_check,
    )

    raw_test_metrics = probability_metrics(
        y_test,
        raw_test,
    )

    # --------------------------------------------------------
    # Fit sigmoid calibration ONLY on 2024-A
    # --------------------------------------------------------

    print("Fitting sigmoid calibration on 2024-A...")

    calibrator = fit_sigmoid_calibrator(
        raw_cal_fit,
        y_cal_fit,
    )

    # --------------------------------------------------------
    # Apply calibration
    # --------------------------------------------------------

    calibrated_cal_fit = apply_sigmoid_calibrator(
        calibrator,
        raw_cal_fit,
    )

    calibrated_cal_check = apply_sigmoid_calibrator(
        calibrator,
        raw_cal_check,
    )

    calibrated_test = apply_sigmoid_calibrator(
        calibrator,
        raw_test,
    )

    # --------------------------------------------------------
    # Calibration metrics
    # --------------------------------------------------------

    calibrated_cal_fit_metrics = probability_metrics(
        y_cal_fit,
        calibrated_cal_fit,
    )

    calibrated_cal_check_metrics = probability_metrics(
        y_cal_check,
        calibrated_cal_check,
    )

    calibrated_test_metrics = probability_metrics(
        y_test,
        calibrated_test,
    )

    # --------------------------------------------------------
    # Decision threshold
    #
    # IMPORTANT:
    # Threshold is selected on 2024-B only.
    # 2025 remains untouched until final evaluation.
    # --------------------------------------------------------

    raw_threshold = select_threshold(
        y_cal_check,
        raw_cal_check,
    )

    calibrated_threshold = select_threshold(
        y_cal_check,
        calibrated_cal_check,
    )

    raw_test_decision = decision_metrics(
        y_test,
        raw_test,
        raw_threshold,
    )

    calibrated_test_decision = decision_metrics(
        y_test,
        calibrated_test,
        calibrated_threshold,
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    for split_name, raw_probs, calibrated_probs, y_true in [
        (
            "calibration_fit_2024A",
            raw_cal_fit,
            calibrated_cal_fit,
            y_cal_fit,
        ),
        (
            "calibration_check_2024B",
            raw_cal_check,
            calibrated_cal_check,
            y_cal_check,
        ),
        (
            "test_2025",
            raw_test,
            calibrated_test,
            y_test,
        ),
    ]:

        raw_metrics = probability_metrics(
            y_true,
            raw_probs,
        )

        calibrated_metrics = probability_metrics(
            y_true,
            calibrated_probs,
        )

        for probability_type, metrics in [
            ("raw", raw_metrics),
            ("sigmoid_calibrated", calibrated_metrics),
        ]:

            row = {
                "model": model_name,
                "horizon": f"+{HORIZON}d",
                "split": split_name,
                "probability_type": probability_type,
            }

            row.update(metrics)

            all_results.append(row)

    calibration_configs[model_name] = {
        "method": "sigmoid_platt",
        "calibration_fit_start": str(
            CALIBRATION_START.date()
        ),
        "calibration_fit_end": str(
            CALIBRATION_FIT_END.date()
        ),
        "calibration_check_start": str(
            CALIBRATION_CHECK_START.date()
        ),
        "calibration_check_end": str(
            CALIBRATION_CHECK_END.date()
        ),
        "raw_threshold_selected_on_2024B": raw_threshold,
        "calibrated_threshold_selected_on_2024B":
            calibrated_threshold,
        "calibrator_intercept": float(
            calibrator.intercept_[0]
        ),
        "calibrator_coefficient": float(
            calibrator.coef_[0][0]
        ),
    }

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(
        f"Raw 2025: "
        f"Brier={raw_test_metrics['brier_score']:.4f}, "
        f"LogLoss={raw_test_metrics['log_loss']:.4f}, "
        f"ECE={raw_test_metrics['ece_10']:.4f}, "
        f"ROC-AUC={raw_test_metrics['roc_auc']:.4f}, "
        f"PR-AUC={raw_test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Cal 2025: "
        f"Brier={calibrated_test_metrics['brier_score']:.4f}, "
        f"LogLoss={calibrated_test_metrics['log_loss']:.4f}, "
        f"ECE={calibrated_test_metrics['ece_10']:.4f}, "
        f"ROC-AUC={calibrated_test_metrics['roc_auc']:.4f}, "
        f"PR-AUC={calibrated_test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Raw threshold: {raw_threshold:.6f}"
    )

    print(
        f"Calibrated threshold: "
        f"{calibrated_threshold:.6f}"
    )

    print(
        f"Raw 2025 F1: "
        f"{raw_test_decision['f1']:.4f}"
    )

    print(
        f"Calibrated 2025 F1: "
        f"{calibrated_test_decision['f1']:.4f}"
    )


# ============================================================
# 8. SAVE RESEARCH OUTPUTS
# ============================================================

print("\n[7/8] Saving calibration results...")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    RESULTS_FILE,
    index=False,
)

config = {
    "experiment": "V2 temporal probability calibration",
    "horizon": HORIZON,
    "target_column": TARGET_COLUMN,
    "elevated_threshold": ELEVATED_THRESHOLD,
    "random_state": RANDOM_STATE,
    "model_training": {
        "start": "2023-10-15",
        "end": "2023-11-23",
    },
    "calibration_fit": {
        "start": str(CALIBRATION_START.date()),
        "end": str(CALIBRATION_FIT_END.date()),
    },
    "calibration_check": {
        "start": str(CALIBRATION_CHECK_START.date()),
        "end": str(CALIBRATION_CHECK_END.date()),
    },
    "final_test": {
        "start": str(TEST_START.date()),
        "end": str(TEST_END.date()),
    },
    "calibration_method": "sigmoid_platt",
    "models": list(models.keys()),
    "feature_count_numeric": len(NUMERIC_FEATURES),
    "categorical_features": CATEGORICAL_FEATURES,
    "notes": [
        "Original V2 model hyperparameters are preserved.",
        "Categorical encoding is fit on 2023 training data only.",
        "Calibration is fitted only on 2024-A.",
        "Calibration method is checked on 2024-B.",
        "2025 is reserved as the final untouched test.",
    ],
    "model_calibration_parameters": calibration_configs,
}

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# RESEARCH MARKDOWN SUMMARY
# ============================================================

print("[8/8] Writing research summary...")

summary_lines = []

summary_lines.append(
    "# StubbleAI V2 - Temporal Probability Calibration"
)

summary_lines.append("")

summary_lines.append(
    "## Status"
)

summary_lines.append(
    "Experimental calibration study. "
    "The original V2 benchmark remains unchanged."
)

summary_lines.append("")

summary_lines.append(
    "## Protocol"
)

summary_lines.append(
    "- 2023: model training"
)

summary_lines.append(
    "- 2024-10-15 to 2024-11-03: calibration fitting"
)

summary_lines.append(
    "- 2024-11-04 to 2024-11-23: calibration check and threshold selection"
)

summary_lines.append(
    "- 2025-10-15 to 2025-11-23: untouched final test"
)

summary_lines.append("")

summary_lines.append(
    "## Calibration method"
)

summary_lines.append(
    "Sigmoid / Platt calibration implemented using a logistic "
    "mapping from model log-odds to calibrated probability."
)

summary_lines.append("")

summary_lines.append(
    "## Models"
)

for model_name in models:
    summary_lines.append(
        f"- {model_name}"
    )

summary_lines.append("")

summary_lines.append(
    "## Probability metrics"
)

summary_lines.append(
    "- Brier score: lower is better"
)

summary_lines.append(
    "- Log loss: lower is better"
)

summary_lines.append(
    "- ECE-10: lower is better"
)

summary_lines.append(
    "- ROC-AUC: discrimination"
)

summary_lines.append(
    "- PR-AUC: positive-class ranking quality"
)

summary_lines.append("")

summary_lines.append(
    "## 2025 results"
)

summary_lines.append("")

summary_lines.append(
    "| Model | Probability | Brier | Log Loss | ECE | ROC-AUC | PR-AUC |"
)

summary_lines.append(
    "|---|---|---:|---:|---:|---:|---:|"
)

test_rows = results_df[
    results_df["split"] == "test_2025"
]

for _, row in test_rows.iterrows():

    summary_lines.append(
        f"| {row['model']} "
        f"| {row['probability_type']} "
        f"| {row['brier_score']:.4f} "
        f"| {row['log_loss']:.4f} "
        f"| {row['ece_10']:.4f} "
        f"| {row['roc_auc']:.4f} "
        f"| {row['pr_auc']:.4f} |"
    )

summary_lines.append("")

summary_lines.append(
    "## Interpretation"
)

summary_lines.append(
    "Calibration should be considered useful primarily when "
    "probability-quality metrics improve on the untouched 2025 "
    "test set without materially damaging discrimination."
)

summary_lines.append("")

summary_lines.append(
    "The 2025 test set was not used to fit the calibration "
    "mapping or choose the calibration method."
)

summary_lines.append("")

summary_lines.append(
    "## Limitations"
)

summary_lines.append(
    "- Only Punjab and Haryana are currently covered."
)

summary_lines.append(
    "- The experiment evaluates sigmoid calibration only."
)

summary_lines.append(
    "- ECE depends on the selected binning procedure."
)

summary_lines.append(
    "- Calibration quality may vary across districts and horizons."
)

summary_lines.append(
    "- Future horizons require independent calibration evaluation."
)

SUMMARY_FILE.write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)

print()
print("=" * 75)
print("CALIBRATION EXPERIMENT COMPLETE")
print("=" * 75)

print(f"Results: {RESULTS_FILE}")
print(f"Config:  {CONFIG_FILE}")
print(f"Summary: {SUMMARY_FILE}")

print()
print("IMPORTANT:")
print("2025 was not used for calibration fitting.")
print("Original V2 benchmark files were not modified.")
print("=" * 75)