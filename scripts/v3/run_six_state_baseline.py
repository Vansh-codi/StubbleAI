
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/v3/v3_district_ml_dataset_1d.csv"
OUTPUT_ROOT = ROOT / "data/processed/v3/experiments/six_state_baseline"

TARGET = "target_elevated_t_plus_1d"
FUTURE_COLUMNS = {
    "fire_count_t_plus_1d",
    "frp_sum_t_plus_1d",
    TARGET,
}
KEYS = ["state", "district", "date"]
EXPECTED_STATES = {
    "ASSAM",
    "BIHAR",
    "HARYANA",
    "MADHYA PRADESH",
    "PUNJAB",
    "UTTAR PRADESH",
}
LAGS = [1, 2, 3, 5, 7, 14]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def metrics_for(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "rows": int(len(y_true)),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    # Deterministic grid; ties choose the lowest threshold.
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 197),
                probabilities,
                np.array([0.5]),
            ]
        )
    )

    results = [
        {
            "threshold": float(threshold),
            "f1": float(
                f1_score(
                    y_true,
                    (probabilities >= threshold).astype(int),
                    zero_division=0,
                )
            ),
        }
        for threshold in candidates
    ]
    best = max(results, key=lambda item: (item["f1"], -item["threshold"]))
    return {
        **best,
        "selection_rule": (
            "Maximize 2024 validation F1; choose lowest threshold on ties"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Six-state StubbleAI V3 temporal baseline"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate inputs and configuration without training",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train, lock validation threshold, and evaluate 2025",
    )
    args = parser.parse_args()

    require(args.preflight != args.train,
            "Specify exactly one of --preflight or --train")
    require(DATA_PATH.is_file(), f"Dataset not found: {DATA_PATH}")

    # Load source and verify its identity before any experiment output.
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    require(df[KEYS].notna().all().all(), "Null observation keys found")
    require(not df.duplicated(KEYS).any(), "Duplicate observation keys found")
    require(set(df["state"].dropna().unique()) == EXPECTED_STATES,
            "Unexpected state coverage")
    require(TARGET in df.columns, f"Missing target: {TARGET}")

    lag_columns = [
        f"{feature}_lag_{lag}d"
        for lag in LAGS
        for feature in [
            "fire_count",
            "frp_sum",
            "active_cell_count",
            "max_cell_fire_count",
            "max_cell_frp",
            "max_cell_share",
        ]
    ]
    missing_lags = sorted(set(lag_columns) - set(df.columns))
    require(not missing_lags, f"Missing required lag columns: {missing_lags}")

    # Eligibility is fixed before model fitting.
    eligible = df[
        df[TARGET].notna()
        & df[lag_columns].notna().all(axis=1)
    ].copy()
    eligible[TARGET] = eligible[TARGET].astype(int)
    eligible["year"] = eligible["date"].dt.year

    require(set(eligible["year"].unique()) == {2023, 2024, 2025},
            "Expected eligible observations in 2023, 2024, and 2025")

    # Verify balanced row coverage within each state across years.
    row_counts = eligible.groupby(["state", "year"]).size().unstack()
    require(row_counts.notna().all().all(),
            "At least one state/year has no eligible rows")
    require((row_counts.nunique(axis=1) == 1).all(),
            "Eligible row counts vary by year within a state")

    feature_columns = [
        col for col in df.columns
        if col not in {"date"} | FUTURE_COLUMNS
    ]

    require(len(feature_columns) == 69,
            f"Expected 69 predictors including state/district; "
            f"found {len(feature_columns)}")

    require(not (set(feature_columns) & ({"date"} | FUTURE_COLUMNS)),
            "Leakage: date or future-target column in predictors")

    categorical = ["state", "district"]
    numeric = [col for col in feature_columns if col not in categorical]

    require(set(categorical).issubset(feature_columns),
            "State/district categorical features missing")

    # Preserve undefined zero-fire statistics; impute any NaNs at fit time.
    # SimpleImputer's missing indicators are disabled for fixed feature count.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight=None,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    train = eligible[eligible["year"] == 2023].copy()
    validation = eligible[eligible["year"] == 2024].copy()
    test = eligible[eligible["year"] == 2025].copy()

    split_summary = {}
    for name, frame in [
        ("train_2023", train),
        ("validation_2024", validation),
        ("test_2025", test),
    ]:
        split_summary[name] = {
            "rows": int(len(frame)),
            "positives": int(frame[TARGET].sum()),
            "prevalence": float(frame[TARGET].mean()),
            "by_state": {
                state: {
                    "rows": int(len(group)),
                    "positives": int(group[TARGET].sum()),
                    "prevalence": float(group[TARGET].mean()),
                }
                for state, group in frame.groupby("state")
            },
        }

    print("DATASET:", DATA_PATH)
    print("DATASET SHA256:", sha256_file(DATA_PATH))
    print("DATASET SHAPE:", df.shape)
    print("ELIGIBLE ROWS:", len(eligible))
    print("PREDICTORS:", len(feature_columns))
    print("SKLEARN:", sklearn.__version__)
    print("PYTHON:", platform.python_version())
    print("SPLITS:")
    print(json.dumps(split_summary, indent=2))

    if args.preflight:
        print("\nPREFLIGHT PASS — no model trained; no outputs written.")
        return

    # Confirm the run directory does not exist; never overwrite prior results.
    run_id = datetime.now(timezone.utc).strftime(
        "baseline_%Y%m%dT%H%M%SZ"
    )
    run_dir = OUTPUT_ROOT / "runs" / run_id
    require(not run_dir.exists(), f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    try:
        print("\nTraining on 2023...")
        model.fit(train[feature_columns], train[TARGET])

        print("Selecting threshold on 2024...")
        validation_prob = model.predict_proba(
            validation[feature_columns]
        )[:, 1]
        threshold_record = choose_threshold(
            validation[TARGET].to_numpy(),
            validation_prob,
        )

        # Persist the threshold lock before requesting 2025 predictions.
        threshold_record["locked_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        threshold_record["validation_rows"] = int(len(validation))
        threshold_record["validation_positives"] = int(
            validation[TARGET].sum()
        )
        threshold_record["dataset_sha256"] = sha256_file(DATA_PATH)

        threshold_path = run_dir / "threshold_lock.json"
        threshold_path.write_text(
            json.dumps(threshold_record, indent=2),
            encoding="utf-8",
        )
        print("THRESHOLD LOCKED:", json.dumps(threshold_record))

        print("Evaluating held-out 2025...")
        test_prob = model.predict_proba(test[feature_columns])[:, 1]
        threshold = threshold_record["threshold"]
        test_pred = (test_prob >= threshold).astype(int)

        prediction_frame = test[
            ["state", "district", "date", TARGET]
        ].copy()
        prediction_frame = prediction_frame.rename(
            columns={TARGET: "y_true"}
        )
        prediction_frame["predicted_probability"] = test_prob
        prediction_frame["locked_threshold"] = threshold
        prediction_frame["y_pred"] = test_pred
        prediction_frame["variant"] = "six_state_v3_baseline"

        prediction_frame["date"] = prediction_frame["date"].dt.strftime(
            "%Y-%m-%d"
        )
        prediction_frame.to_csv(
            run_dir / "test_predictions_2025.csv",
            index=False,
        )

        pooled = metrics_for(
            test[TARGET].to_numpy(),
            test_prob,
            threshold,
        )
        pooled["scope"] = "POOLED_SIX_STATE"
        pooled["validation_threshold"] = threshold

        per_state = []
        for state, group in prediction_frame.groupby("state"):
            # Grouping uses saved predictions to ensure identical evaluated rows.
            original = test[
                (test["state"] == state)
            ].copy()
            metrics = metrics_for(
                original[TARGET].to_numpy(),
                original["state"].map(
                    dict(zip(test["state"], test_prob))
                ).to_numpy()
                if False else test_prob[test["state"].eq(state).to_numpy()],
                threshold,
            )
            metrics["state"] = state
            per_state.append(metrics)

        pd.DataFrame([pooled]).to_csv(
            run_dir / "pooled_test_metrics.csv", index=False
        )
        pd.DataFrame(per_state).to_csv(
            run_dir / "per_state_test_metrics.csv", index=False
        )

        manifest = {
            "status": "COMPLETE",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(DATA_PATH.relative_to(ROOT)),
            "dataset_sha256": sha256_file(DATA_PATH),
            "dataset_rows": int(len(df)),
            "eligible_rows": int(len(eligible)),
            "predictor_count": len(feature_columns),
            "predictors": feature_columns,
            "excluded_columns": sorted(set(KEYS) | FUTURE_COLUMNS),
            "states": sorted(EXPECTED_STATES),
            "season_scope": "October-November only",
            "split": split_summary,
            "model": {
                "class": "RandomForestClassifier",
                "n_estimators": 400,
                "min_samples_leaf": 2,
                "random_state": 42,
                "class_weight": None,
                "n_jobs": -1,
            },
            "threshold_lock": threshold_record,
            "metrics": pooled,
            "versions": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "artifacts": [
                "threshold_lock.json",
                "test_predictions_2025.csv",
                "pooled_test_metrics.csv",
                "per_state_test_metrics.csv",
                "manifest.json",
            ],
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        print("\nPOOLED 2025 METRICS:")
        print(json.dumps(pooled, indent=2))
        print("\nPER-STATE 2025 METRICS:")
        print(pd.DataFrame(per_state).to_string(index=False))
        print("\nRUN COMPLETE:", run_dir)

    except Exception as exc:
        failure = {
            "status": "FAILED",
            "run_id": run_id,
            "error": repr(exc),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "failure.json").write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
