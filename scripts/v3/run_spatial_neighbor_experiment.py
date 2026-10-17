from __future__ import annotations

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
SPATIAL_PATH = ROOT / (
    "data/processed/v3/experiments/spatial_neighbors_v3/"
    "district_neighbor_features.csv"
)
OUTPUT_ROOT = ROOT / "data/processed/v3/experiments/spatial_model_comparison"

TARGET = "target_elevated_t_plus_1d"
FUTURE = {"fire_count_t_plus_1d", "frp_sum_t_plus_1d", TARGET}
KEYS = ["state", "district", "date"]
CATEGORICAL = ["state", "district"]
LAGS = [1, 2, 3, 5, 7, 14]

SPATIAL_FEATURES = [
    "neighbor_fire_count_lag_1d_mean",
    "neighbor_fire_count_lag_1d_max",
    "neighbor_fire_count_per_neighbor_mean",
    "neighbor_frp_sum_lag_1d_mean",
    "neighbor_frp_per_neighbor_mean",
    "neighbor_active_cells_lag_1d_mean",
    "neighbor_max_fire_count_lag_1d_max",
    "neighbor_coverage_fraction_mean",
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_threshold(y, p):
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 197), p, [0.5]]))
    scored = [
        (float(f1_score(y, p >= t, zero_division=0)), float(t))
        for t in candidates
    ]
    best_f1, best_t = max(scored, key=lambda z: (z[0], -z[1]))
    return {"threshold": best_t, "f1": best_f1,
            "selection_rule": "Maximize 2024 validation F1; lowest threshold on ties"}


def calculate_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "rows": int(len(y)),
        "positives": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "threshold": float(threshold),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def make_model(numeric):
    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
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


def main():
    require(DATA_PATH.is_file(), f"Missing dataset: {DATA_PATH}")
    require(SPATIAL_PATH.is_file(), f"Missing spatial features: {SPATIAL_PATH}")

    base = pd.read_csv(DATA_PATH, parse_dates=["date"])
    spatial = pd.read_csv(SPATIAL_PATH, parse_dates=["date"])

    # Normalize join-key capitalization in memory; source CSVs remain untouched.
    for frame in (base, spatial):
        frame["state"] = frame["state"].str.strip().str.upper()
        frame["district"] = frame["district"].str.strip().str.upper()

    for name, frame in [("baseline", base), ("spatial", spatial)]:
        require(frame[KEYS].notna().all().all(), f"{name}: null join keys")
        require(not frame.duplicated(KEYS).any(), f"{name}: duplicate keys")

    require(TARGET in base.columns, f"Missing target {TARGET}")
    require(set(SPATIAL_FEATURES).issubset(spatial.columns),
            "Missing required spatial features")

    overlap = set(SPATIAL_FEATURES) & set(base.columns)
    require(not overlap, f"Spatial feature name collision: {sorted(overlap)}")

    # One-to-one join, preserving all baseline rows.
    joined = base.merge(
        spatial[KEYS + SPATIAL_FEATURES],
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    require((joined["_merge"] == "both").all(),
            "Baseline and spatial feature key populations differ")
    joined.drop(columns="_merge", inplace=True)

    lag_columns = [
        f"{feature}_lag_{lag}d"
        for lag in LAGS
        for feature in [
            "fire_count", "frp_sum", "active_cell_count",
            "max_cell_fire_count", "max_cell_frp", "max_cell_share",
        ]
    ]
    require(set(lag_columns).issubset(joined.columns),
            "Missing required baseline lag features")

    eligible = joined[
        joined[TARGET].notna()
        & joined[lag_columns].notna().all(axis=1)
    ].copy()
    eligible[TARGET] = eligible[TARGET].astype(int)
    eligible["year"] = eligible["date"].dt.year

    # Compare on exactly the same rows: spatial features must all be present.
    eligible = eligible[eligible[SPATIAL_FEATURES].notna().all(axis=1)].copy()
    require(set(eligible["year"].unique()) == {2023, 2024, 2025},
            "Expected eligible rows in all three years")

    feature_base = [
        c for c in base.columns if c not in {"date"} | FUTURE
    ]
    feature_spatial = feature_base + SPATIAL_FEATURES
    require(not (set(feature_spatial) & FUTURE),
            "Future target leakage in predictors")

    train = eligible[eligible.year == 2023]
    valid = eligible[eligible.year == 2024]
    test = eligible[eligible.year == 2025]

    print("BASE DATA SHA256:", sha256(DATA_PATH))
    print("SPATIAL DATA SHA256:", sha256(SPATIAL_PATH))
    print("Same-row eligible count:", len(eligible))
    print("Split sizes:", len(train), len(valid), len(test))
    print("Spatial boundary rows excluded:", int(len(joined) - len(eligible)))
    print("Baseline predictors:", len(feature_base))
    print("Spatial predictors:", len(feature_spatial))
    print("Versions:", platform.python_version(), pd.__version__,
          np.__version__, sklearn.__version__)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("spatial_%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / "runs" / run_id
    require(not run_dir.exists(), f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    results = []
    prediction_frames = []

    try:
        for variant, features in [
            ("baseline_common_rows", feature_base),
            ("baseline_plus_spatial_neighbors", feature_spatial),
        ]:
            print(f"\nTraining {variant}...", flush=True)
            model = make_model([c for c in features if c not in CATEGORICAL])
            model.fit(train[features], train[TARGET])

            valid_p = model.predict_proba(valid[features])[:, 1]
            lock = choose_threshold(valid[TARGET].to_numpy(), valid_p)
            test_p = model.predict_proba(test[features])[:, 1]
            metrics = calculate_metrics(
                test[TARGET].to_numpy(), test_p, lock["threshold"]
            )
            metrics["variant"] = variant
            metrics["validation_threshold"] = lock["threshold"]
            metrics["validation_f1"] = lock["f1"]
            results.append(metrics)

            pred = test[KEYS + [TARGET]].copy()
            pred.rename(columns={TARGET: "y_true"}, inplace=True)
            pred["predicted_probability"] = test_p
            pred["locked_threshold"] = lock["threshold"]
            pred["y_pred"] = (test_p >= lock["threshold"]).astype(int)
            pred["variant"] = variant
            prediction_frames.append(pred)

            (run_dir / f"{variant}_threshold_lock.json").write_text(
                json.dumps({
                    **lock,
                    "validation_rows": int(len(valid)),
                    "validation_positives": int(valid[TARGET].sum()),
                    "dataset_sha256": sha256(DATA_PATH),
                    "spatial_sha256": sha256(SPATIAL_PATH),
                }, indent=2), encoding="utf-8"
            )

            del model

        metrics_df = pd.DataFrame(results)
        pred_df = pd.concat(prediction_frames, ignore_index=True)
        pred_df["date"] = pred_df["date"].dt.strftime("%Y-%m-%d")

        metrics_df.to_csv(run_dir / "pooled_test_metrics.csv", index=False)
        pred_df.to_csv(run_dir / "test_predictions_2025.csv", index=False)

        state_results = []
        for variant, group in pred_df.groupby("variant"):
            for state, sg in group.groupby("state"):
                m = calculate_metrics(
                    sg["y_true"].to_numpy(),
                    sg["predicted_probability"].to_numpy(),
                    float(sg["locked_threshold"].iloc[0]),
                )
                m.update({"variant": variant, "state": state})
                state_results.append(m)
        pd.DataFrame(state_results).to_csv(
            run_dir / "per_state_test_metrics.csv", index=False
        )

        manifest = {
            "status": "COMPLETE",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_dataset": str(DATA_PATH.relative_to(ROOT)),
            "baseline_sha256": sha256(DATA_PATH),
            "spatial_dataset": str(SPATIAL_PATH.relative_to(ROOT)),
            "spatial_sha256": sha256(SPATIAL_PATH),
            "common_row_count": int(len(eligible)),
            "split_years": {"train": 2023, "validation": 2024, "test": 2025},
            "season_scope": "October-November only",
            "threshold_rule": "Maximize 2024 validation F1; lowest threshold on ties",
            "model": {
                "class": "RandomForestClassifier", "n_estimators": 400,
                "min_samples_leaf": 2, "random_state": 42,
                "class_weight": None, "n_jobs": -1,
            },
            "baseline_predictors": feature_base,
            "spatial_predictors_added": SPATIAL_FEATURES,
            "metrics": results,
            "versions": {
                "python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            },
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        print("\nPOOLED 2025 RESULTS:")
        print(metrics_df.to_string(index=False))
        print("\nPER-STATE RESULTS:")
        print(pd.DataFrame(state_results).to_string(index=False))
        print("\nRUN COMPLETE:", run_dir)

    except Exception as exc:
        (run_dir / "failure.json").write_text(
            json.dumps({
                "status": "FAILED", "run_id": run_id,
                "error": repr(exc),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=2), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
