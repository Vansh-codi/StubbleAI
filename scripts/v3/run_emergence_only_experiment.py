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
BASELINE_RUN = (
    ROOT / "data/processed/v3/experiments/six_state_baseline/runs/"
    "baseline_20261017T020007Z"
)
OUTPUT_ROOT = ROOT / "data/processed/v3/experiments/emergence_only"

TARGET = "target_elevated_t_plus_1d"
FUTURE_COLUMNS = {
    "fire_count_t_plus_1d",
    "frp_sum_t_plus_1d",
    TARGET,
}
STATES = {
    "ASSAM", "BIHAR", "HARYANA",
    "MADHYA PRADESH", "PUNJAB", "UTTAR PRADESH",
}
LAGS = [1, 2, 3, 5, 7, 14]
KEYS = ["state", "district", "date"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_model(numeric: list[str], categorical: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                numeric,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
            class_weight=None,
            n_jobs=-1,
        )),
    ])


def choose_threshold(y: np.ndarray, probabilities: np.ndarray) -> dict:
    candidates = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 197),
        probabilities,
        np.array([0.5]),
    ]))
    scores = [
        (
            float(f1_score(
                y, probabilities >= threshold, zero_division=0
            )),
            float(threshold),
        )
        for threshold in candidates
    ]
    best_f1, best_threshold = max(scores, key=lambda item: (item[0], -item[1]))
    return {
        "threshold": best_threshold,
        "validation_f1": best_f1,
        "selection_rule": "Maximize 2024 F1; lowest threshold wins ties",
    }


def metric_record(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": float(threshold),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def load_data():
    require(DATA_PATH.is_file(), f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    require(df[KEYS].notna().all().all(), "Null observation keys")
    require(not df.duplicated(KEYS).any(), "Duplicate observation keys")
    require(set(df["state"].dropna().unique()) == STATES, "Unexpected states")
    require(TARGET in df.columns, f"Missing target: {TARGET}")

    lag_columns = [
        f"{feature}_lag_{lag}d"
        for lag in LAGS
        for feature in [
            "fire_count", "frp_sum", "active_cell_count",
            "max_cell_fire_count", "max_cell_frp", "max_cell_share",
        ]
    ]
    require(not (set(lag_columns) - set(df.columns)), "Required lag columns missing")

    eligible = df[
        df[TARGET].notna()
        & df[lag_columns].notna().all(axis=1)
    ].copy()
    eligible[TARGET] = eligible[TARGET].astype(int)
    eligible["year"] = eligible["date"].dt.year

    features = [
        col for col in df.columns
        if col not in {"date"} | FUTURE_COLUMNS
    ]
    require(len(features) == 69, f"Expected 69 features; found {len(features)}")
    require(not (set(features) & ({"date"} | FUTURE_COLUMNS)), "Target leakage")
    require({"state", "district"}.issubset(features), "Categorical features missing")

    normal = eligible[eligible["fire_count"] <= 2].copy()
    normal["state"] = normal["state"].astype(str).str.upper()
    normal["district"] = normal["district"].astype(str).str.upper()

    expected = {2023: (8016, 809, 7207), 2024: (7906, 925, 6981),
                2025: (8267, 829, 7438)}
    actual = {}
    for year, group in normal.groupby("year"):
        actual[int(year)] = (
            len(group), int(group[TARGET].sum()),
            int((group[TARGET] == 0).sum())
        )
    require(actual == expected, f"Emergence eligibility mismatch: {actual}")

    return df, eligible, normal, features


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--train", action="store_true")
    args = parser.parse_args()

    df, eligible, normal, features = load_data()
    numeric = [c for c in features if c not in {"state", "district"}]
    categorical = ["state", "district"]

    require(BASELINE_RUN.is_dir(), f"Expected baseline run missing: {BASELINE_RUN}")
    baseline_predictions_path = BASELINE_RUN / "test_predictions_2025.csv"
    require(baseline_predictions_path.is_file(),
            f"Baseline predictions missing: {baseline_predictions_path}")

    baseline_predictions = pd.read_csv(baseline_predictions_path)
    baseline_predictions["state"] = baseline_predictions["state"].astype(str).str.upper()
    baseline_predictions["district"] = baseline_predictions["district"].astype(str).str.upper()
    baseline_predictions["date"] = pd.to_datetime(baseline_predictions["date"])

    test_normal = normal[normal["year"] == 2025]
    join_keys = ["state", "district", "date"]
    baseline_normal = baseline_predictions.merge(
        test_normal[join_keys], on=join_keys, how="inner", validate="one_to_one"
    )
    require(len(baseline_normal) == len(test_normal),
            "Baseline predictions do not cover every normal-state test row")

    print("DATASET SHA256:", sha256_file(DATA_PATH))
    print("BASELINE ELIGIBLE:", len(eligible))
    print("FEATURE COUNT:", len(features))
    print("EMERGENCE COUNTS:")
    print(normal.groupby(["year", TARGET]).size().unstack(fill_value=0).to_string())
    print("\nTRAINING ROWS BY STATE (2023):")
    train = normal[normal["year"] == 2023]
    print(train.groupby("state")[TARGET].agg(["size", "sum"]).to_string())
    print("\nBASELINE NORMAL-STATE TEST MATCH:", len(baseline_normal))
    print("OUTPUT ROOT:", OUTPUT_ROOT)

    if args.preflight:
        print("\nPREFLIGHT PASS — no model trained and no experiment outputs written.")
        return

    run_id = datetime.now(timezone.utc).strftime("emergence_%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / "runs" / run_id
    require(not run_dir.exists(), f"Refusing to overwrite: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    try:
        train = normal[normal["year"] == 2023].copy()
        validation = normal[normal["year"] == 2024].copy()
        test = normal[normal["year"] == 2025].copy()

        row_counts = train.groupby("state").size()
        state_weights = len(train) / (len(STATES) * row_counts)
        train["sample_weight"] = train["state"].map(state_weights).astype(float)
        weight_audit = train.groupby("state").agg(
            rows=("sample_weight", "size"),
            positives=(TARGET, "sum"),
            total_weight=("sample_weight", "sum"),
            mean_weight=("sample_weight", "mean"),
        )
        require(np.allclose(
            weight_audit["total_weight"].to_numpy(),
            len(train) / len(STATES)
        ), "State-balanced weights failed audit")

        all_predictions = []
        summaries = []
        thresholds = {}

        for variant in ["unweighted", "state_balanced"]:
            model = make_model(numeric, categorical)
            fit_kwargs = {}
            if variant == "state_balanced":
                fit_kwargs["classifier__sample_weight"] = train["sample_weight"].to_numpy()

            print(f"\nTraining variant: {variant}")
            model.fit(train[features], train[TARGET], **fit_kwargs)

            validation_p = model.predict_proba(validation[features])[:, 1]
            threshold_info = choose_threshold(validation[TARGET].to_numpy(), validation_p)
            thresholds[variant] = threshold_info

            test_p = model.predict_proba(test[features])[:, 1]
            test_y = test[TARGET].to_numpy()
            metrics = metric_record(test_y, test_p, threshold_info["threshold"])
            metrics["variant"] = variant
            summaries.append(metrics)

            pred = test[join_keys + [TARGET]].copy()
            pred = pred.rename(columns={TARGET: "y_true"})
            pred["predicted_probability"] = test_p
            pred["locked_threshold"] = threshold_info["threshold"]
            pred["y_pred"] = (test_p >= threshold_info["threshold"]).astype(int)
            pred["variant"] = variant
            all_predictions.append(pred)

            print("THRESHOLD:", threshold_info)
            print("TEST METRICS:", json.dumps(metrics, indent=2))

        # The existing baseline's probabilities are joined by full observation key.
        baseline_full = baseline_predictions.copy()
        baseline_full["date"] = pd.to_datetime(baseline_full["date"])
        baseline_full = baseline_full.merge(
            test[join_keys], on=join_keys, how="inner", validate="one_to_one"
        )
        require(len(baseline_full) == len(test), "Baseline join mismatch on test subset")

        baseline_metrics = metric_record(
            baseline_full["y_true"].to_numpy(),
            baseline_full["predicted_probability"].to_numpy(),
            float(baseline_full["locked_threshold"].iloc[0]),
        )
        baseline_metrics["variant"] = "baseline_on_same_normal_subset"
        summaries.append(baseline_metrics)

        pd.DataFrame(summaries).to_csv(run_dir / "pooled_test_metrics.csv", index=False)
        pd.concat(all_predictions, ignore_index=True).to_csv(
            run_dir / "test_predictions_2025.csv", index=False
        )
        pd.DataFrame([thresholds]).to_json(
            run_dir / "thresholds.json", orient="records", indent=2
        )
        weight_audit.to_csv(run_dir / "state_weight_audit.csv")

        per_state = []
        combined = pd.concat(all_predictions, ignore_index=True)
        for variant, frame in combined.groupby("variant"):
            for state, group in frame.groupby("state"):
                m = metric_record(
                    group["y_true"].to_numpy(),
                    group["predicted_probability"].to_numpy(),
                    float(group["locked_threshold"].iloc[0]),
                )
                m.update({"variant": variant, "state": state})
                per_state.append(m)
        for state, group in baseline_full.groupby("state"):
            m = metric_record(
                group["y_true"].to_numpy(),
                group["predicted_probability"].to_numpy(),
                float(group["locked_threshold"].iloc[0]),
            )
            m.update({
                "variant": "baseline_on_same_normal_subset",
                "state": state,
            })
            per_state.append(m)

        pd.DataFrame(per_state).to_csv(run_dir / "per_state_test_metrics.csv", index=False)

        manifest = {
            "status": "COMPLETE",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(DATA_PATH.relative_to(ROOT)),
            "dataset_sha256": sha256_file(DATA_PATH),
            "baseline_run": str(BASELINE_RUN.relative_to(ROOT)),
            "baseline_test_normal_rows": int(len(test)),
            "features": features,
            "feature_count": len(features),
            "eligibility": {
                str(year): {
                    "rows": int(len(group)),
                    "positives": int(group[TARGET].sum()),
                }
                for year, group in normal.groupby("year")
            },
            "model": {
                "class": "RandomForestClassifier",
                "n_estimators": 400,
                "min_samples_leaf": 2,
                "random_state": 42,
                "class_weight": None,
            },
            "state_weight_formula": "N_train / (number_of_states * state_train_rows)",
            "state_weight_audit": weight_audit.reset_index().to_dict(orient="records"),
            "thresholds": thresholds,
            "versions": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print("\nRUN COMPLETE:", run_dir)

    except Exception as exc:
        (run_dir / "failure.json").write_text(
            json.dumps({
                "status": "FAILED",
                "error": repr(exc),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=2),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()

