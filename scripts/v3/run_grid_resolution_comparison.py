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
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "data/processed/v3/v3_district_ml_dataset_1d.csv"
SPATIAL_005 = ROOT / "data/processed/v3/experiments/spatial_neighbors_v3/district_neighbor_features.csv"
SPATIAL_010 = ROOT / "data/processed/v3/experiments/grid_sensitivity_010/district_neighbor_features_010.csv"
OUTPUT_ROOT = ROOT / "data/processed/v3/experiments/grid_sensitivity_010/resolution_comparison"

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
        raise RuntimeError(message)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize(df):
    df["state"] = df["state"].astype("string").str.strip().str.upper()
    df["district"] = df["district"].astype("string").str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    require(df[KEYS].notna().all().all(), "Null join keys")
    require(not df.duplicated(KEYS).any(), "Duplicate district/date keys")
    return df

def choose_threshold(y, p):
    candidates = np.unique(np.concatenate([np.linspace(0.01, 0.99, 197), p, [0.5]]))
    scores = [
        (float(f1_score(y, p >= t, zero_division=0)), float(t))
        for t in candidates
    ]
    best_f1, best_t = max(scores, key=lambda z: (z[0], -z[1]))
    return {
        "threshold": best_t,
        "f1": best_f1,
        "selection_rule": "Maximize 2024 validation F1; lowest threshold on ties",
    }

def metrics(y, p, threshold):
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

def make_model():
    numeric = [c for c in FEATURES if c not in CATEGORICAL]
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
            n_estimators=400, min_samples_leaf=2,
            random_state=42, class_weight=None, n_jobs=-1,
        )),
    ])

def load_inputs():
    for path in [BASE_PATH, SPATIAL_005, SPATIAL_010]:
        require(path.is_file(), f"Missing input: {path}")

    base = normalize(pd.read_csv(BASE_PATH, parse_dates=["date"]))
    s005 = normalize(pd.read_csv(SPATIAL_005, parse_dates=["date"]))
    s010 = normalize(pd.read_csv(SPATIAL_010, parse_dates=["date"]))

    require(TARGET in base.columns, f"Missing target: {TARGET}")
    for name, spatial in [("0.05", s005), ("0.10", s010)]:
        require(set(SPATIAL_FEATURES).issubset(spatial.columns),
                f"{name} spatial features missing required columns")
        require(not (set(SPATIAL_FEATURES) & set(base.columns)),
                "Spatial feature names collide with baseline columns")

    base_keys = set(map(tuple, base[KEYS].itertuples(index=False, name=None)))
    for name, spatial in [("0.05", s005), ("0.10", s010)]:
        spatial_keys = set(map(tuple, spatial[KEYS].itertuples(index=False, name=None)))
        require(base_keys == spatial_keys, f"{name}: key population differs from baseline")

    joined = base.copy()
    for suffix, spatial in [("005", s005), ("010", s010)]:
        right = spatial[KEYS + SPATIAL_FEATURES].rename(
            columns={c: f"{c}_{suffix}" for c in SPATIAL_FEATURES}
        )
        joined = joined.merge(
            right, on=KEYS, how="left", validate="one_to_one", indicator=f"_merge_{suffix}"
        )
        require((joined[f"_merge_{suffix}"] == "both").all(),
                f"Unmatched {suffix} spatial rows")
        joined.drop(columns=f"_merge_{suffix}", inplace=True)

    lag_columns = [
        f"{feature}_lag_{lag}d"
        for lag in LAGS
        for feature in [
            "fire_count", "frp_sum", "active_cell_count",
            "max_cell_fire_count", "max_cell_frp", "max_cell_share",
        ]
    ]
    require(set(lag_columns).issubset(joined.columns), "Required lag columns missing")
    eligible = joined[
        joined[TARGET].notna()
        & joined[lag_columns].notna().all(axis=1)
    ].copy()

    eligible[TARGET] = eligible[TARGET].astype(int)
    require(set(eligible[TARGET].unique()).issubset({0, 1}), "Invalid target labels")
    eligible["year"] = eligible["date"].dt.year

    all_spatial = [f"{c}_{suffix}" for suffix in ["005", "010"] for c in SPATIAL_FEATURES]
    eligible = eligible[eligible[all_spatial].notna().all(axis=1)].copy()
    require(set(eligible["year"].unique()) == {2023, 2024, 2025},
            "Expected rows in 2023, 2024 and 2025")

    # Verify same eligibility and labels against each spatial dataset separately.
    for suffix in ["005", "010"]:
        cols = [f"{c}_{suffix}" for c in SPATIAL_FEATURES]
        require(eligible[cols].notna().all().all(), f"Null spatial features for {suffix}")

    return base, joined, eligible

def preflight():
    base, joined, eligible = load_inputs()
    print("PREFLIGHT PASS")
    print("Baseline rows:", len(base))
    print("Common eligible rows:", len(eligible))
    print("Split sizes:", *(int((eligible["year"] == y).sum()) for y in [2023, 2024, 2025]))
    print("Eligible positives by year:")
    print(eligible.groupby("year")[TARGET].agg(["count", "sum"]).to_string())
    print("Both feature sets have complete coverage on common rows.")
    print("No models trained and no output files written.")

def run():
    global FEATURES
    base, joined, eligible = load_inputs()
    base_features = [c for c in base.columns if c not in ({"date"} | FUTURE)]
    require(not (set(base_features) & FUTURE), "Future leakage in baseline features")

    train = eligible[eligible["year"] == 2023]
    valid = eligible[eligible["year"] == 2024]
    test = eligible[eligible["year"] == 2025]
    require(len(train) and len(valid) and len(test), "Empty temporal split")

    run_id = datetime.now(timezone.utc).strftime("resolution_%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / "runs" / run_id
    require(not run_dir.exists(), f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    variants = [
        ("baseline", base_features),
        ("spatial_005", base_features + [f"{c}_005" for c in SPATIAL_FEATURES]),
        ("spatial_010", base_features + [f"{c}_010" for c in SPATIAL_FEATURES]),
    ]
    result_rows, prediction_frames, state_rows, locks = [], [], [], {}

    print("BASE SHA256:", sha256(BASE_PATH))
    print("0.05 FEATURE SHA256:", sha256(SPATIAL_005))
    print("0.10 FEATURE SHA256:", sha256(SPATIAL_010))
    print("Common eligible rows:", len(eligible))
    print("Split sizes:", len(train), len(valid), len(test))

    try:
        for variant, features in variants:
            FEATURES = features
            print(f"\nTraining {variant}...", flush=True)
            model = make_model()
            model.fit(train[features], train[TARGET])

            valid_p = model.predict_proba(valid[features])[:, 1]
            lock = choose_threshold(valid[TARGET].to_numpy(), valid_p)
            test_p = model.predict_proba(test[features])[:, 1]
            m = metrics(test[TARGET].to_numpy(), test_p, lock["threshold"])
            m.update({
                "variant": variant,
                "validation_threshold": lock["threshold"],
                "validation_f1": lock["f1"],
                "predictor_count": len(features),
            })
            result_rows.append(m)
            locks[variant] = {
                **lock,
                "validation_rows": int(len(valid)),
                "validation_positives": int(valid[TARGET].sum()),
            }

            pred = test[KEYS + [TARGET]].copy()
            pred.rename(columns={TARGET: "y_true"}, inplace=True)
            pred["predicted_probability"] = test_p
            pred["locked_threshold"] = lock["threshold"]
            pred["y_pred"] = (test_p >= lock["threshold"]).astype(int)
            pred["variant"] = variant
            prediction_frames.append(pred)

            for state, sg in pred.groupby("state"):
                sm = metrics(
                    sg["y_true"].to_numpy(),
                    sg["predicted_probability"].to_numpy(),
                    lock["threshold"],
                )
                sm.update({"variant": variant, "state": state})
                state_rows.append(sm)
            del model

        pd.DataFrame(result_rows).to_csv(run_dir / "pooled_test_metrics.csv", index=False)
        pd.concat(prediction_frames, ignore_index=True).assign(
            date=lambda x: x["date"].dt.strftime("%Y-%m-%d")
        ).to_csv(run_dir / "test_predictions_2025.csv", index=False)
        pd.DataFrame(state_rows).to_csv(run_dir / "per_state_test_metrics.csv", index=False)
        (run_dir / "threshold_locks.json").write_text(json.dumps(locks, indent=2), encoding="utf-8")

        manifest = {
            "status": "COMPLETE",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_dataset": str(BASE_PATH.relative_to(ROOT)),
            "baseline_sha256": sha256(BASE_PATH),
            "spatial_005_dataset": str(SPATIAL_005.relative_to(ROOT)),
            "spatial_005_sha256": sha256(SPATIAL_005),
            "spatial_010_dataset": str(SPATIAL_010.relative_to(ROOT)),
            "spatial_010_sha256": sha256(SPATIAL_010),
            "common_eligible_rows": int(len(eligible)),
            "split_years": {"train": 2023, "validation": 2024, "test": 2025},
            "season_scope": "October-November only",
            "threshold_rule": "Maximize 2024 validation F1; lowest threshold on ties",
            "model": {
                "class": "RandomForestClassifier", "n_estimators": 400,
                "min_samples_leaf": 2, "random_state": 42,
                "class_weight": None, "n_jobs": -1,
            },
            "variants": {
                name: {"predictor_count": len(features), "predictors": features}
                for name, features in variants
            },
            "metrics": result_rows,
            "versions": {
                "python": platform.python_version(), "pandas": pd.__version__,
                "numpy": np.__version__, "scikit_learn": sklearn.__version__,
            },
            "note": "0.10-degree weather is district-level weather repeated across cells; this is not finer-resolution weather.",
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("\nPOOLED 2025 RESULTS")
        print(pd.DataFrame(result_rows).to_string(index=False))
        print("\nPER-STATE RESULTS")
        print(pd.DataFrame(state_rows).to_string(index=False))
        print("\nRUN COMPLETE:", run_dir)
    except Exception as exc:
        (run_dir / "failure.json").write_text(json.dumps({
            "status": "FAILED", "run_id": run_id, "error": repr(exc),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        FEATURES = []
        run()
