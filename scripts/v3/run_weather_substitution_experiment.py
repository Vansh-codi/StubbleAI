from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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
BASE_PATH = ROOT / "data/processed/v3/v3_district_ml_dataset_1d.csv"
VARIANT_PATH = (
    ROOT / "data/processed/v3/experiments/weather_substitution/"
    "v3_district_ml_dataset_1d_v1_weather_verified.csv"
)
PROTOCOL_PATH = ROOT / "research/v3/weather_substitution_experiment_protocol_v3.md"
OUTPUT_ROOT = ROOT / "data/processed/v3/experiments/weather_substitution/runs"

KEYS = ["state", "district", "date"]
TARGET = "target_elevated_t_plus_1d"
FUTURE_COLUMNS = [
    "fire_count_t_plus_1d",
    "frp_sum_t_plus_1d",
    "target_elevated_t_plus_1d",
]
WEATHER = ["T2M", "RH2M", "WS2M", "PRECTOTCORR"]
CATEGORICAL = ["state", "district"]
CALENDAR = [
    "year", "month", "day", "day_of_season",
    "day_of_year", "sin_year", "cos_year",
]
CURRENT_FIRE = [
    "fire_count", "frp_sum", "frp_mean", "frp_max",
    "high_confidence_fire_count", "day_fire_count", "night_fire_count",
    "bright_ti4_mean", "bright_ti4_max", "bright_ti5_mean",
]
GRID = [
    "active_cell_count", "max_cell_fire_count",
    "max_cell_frp", "max_cell_share",
]
LAGS = [
    f"{feature}_lag_{lag}d"
    for feature in [
        "fire_count", "frp_sum", "active_cell_count",
        "max_cell_fire_count", "max_cell_frp", "max_cell_share",
    ]
    for lag in [1, 2, 3, 5, 7, 14]
]
SUMMARIES = [
    "fire_count_mean_3d",
    "fire_count_sparse_mean_7d",
    "fire_count_sparse_mean_14d",
    "frp_sum_mean_3d",
    "frp_sum_sparse_mean_7d",
    "frp_sum_sparse_mean_14d",
]
FEATURES = (
    CATEGORICAL + CALENDAR + CURRENT_FIRE + GRID
    + WEATHER + LAGS + SUMMARIES
)
TOLERANCE = 1e-10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_mask(frame: pd.DataFrame) -> pd.Series:
    # Match the preregistered eligibility rule:
    # target known and all 36 required lag features available.
    mask = frame[TARGET].notna()
    for col in LAGS:
        mask &= frame[col].notna()
    return mask


def compare_numeric(a: pd.Series, b: pd.Series, tolerance: float) -> bool:
    left = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        return False
    valid = np.isfinite(left) & np.isfinite(right)
    return bool(np.all(np.abs(left[valid] - right[valid]) <= tolerance))


def make_pipeline() -> Pipeline:
    prep = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL,
            ),
            ("numeric", "passthrough", [c for c in FEATURES if c not in CATEGORICAL]),
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=42,
        class_weight=None,
    )
    return Pipeline([("preprocess", prep), ("model", model)])


def prepare_inputs():
    if not BASE_PATH.exists():
        raise FileNotFoundError(BASE_PATH)
    if not VARIANT_PATH.exists():
        raise FileNotFoundError(VARIANT_PATH)
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)

    base_all = pd.read_csv(BASE_PATH)
    variant = pd.read_csv(VARIANT_PATH)
    base = base_all[base_all["state"].isin(["PUNJAB", "HARYANA"])].copy()

    for name, frame in [("V3", base), ("V1-weather", variant)]:
        missing = [c for c in KEYS + FEATURES + FUTURE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} has duplicate state-district-date keys")

    if list(base.columns) != list(variant.columns):
        raise ValueError("Dataset columns/order differ")

    merged = base.merge(
        variant,
        on=KEYS,
        how="outer",
        suffixes=("_A", "_B"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"Dataset keys do not match exactly: {counts}")

    for col in FEATURES + FUTURE_COLUMNS:
        if col in WEATHER or col in KEYS:
            continue
        if not compare_numeric(merged[f"{col}_A"], merged[f"{col}_B"], TOLERANCE):
            raise ValueError(f"Non-weather feature/target mismatch: {col}")

    if len(FEATURES) != 69 or len(set(FEATURES)) != 69:
        raise ValueError(f"Expected 69 unique predictors; got {len(FEATURES)}")
    if set(FEATURES) & ({"date"} | set(FUTURE_COLUMNS)):
        raise ValueError("Leakage: keys or future targets in predictors")
    if any(c not in base.columns for c in FEATURES):
        raise ValueError("One or more predictors absent from base dataset")

    eligible_a = eligible_mask(base)
    eligible_b = eligible_mask(variant)
    if not np.array_equal(eligible_a.to_numpy(), eligible_b.to_numpy()):
        raise ValueError("Eligible-row masks differ between variants")

    a = base.loc[eligible_a].copy()
    b = variant.loc[eligible_b].copy()

    # Registered missingness policy: undefined fire-intensity statistics
    # may be missing only on zero-detection days. Preserve NaNs and rows.
    undefined_fire_stats = [
        "frp_mean",
        "frp_max",
        "bright_ti4_mean",
        "bright_ti4_max",
        "bright_ti5_mean",
    ]

    for name, frame in [("V3", a), ("V1-weather", b)]:
        absent = [c for c in undefined_fire_stats if c not in frame.columns]
        if absent:
            raise ValueError(f"{name}: missing audit columns: {absent}")

        for col in undefined_fire_stats:
            bad = frame[col].isna() & frame["fire_count"].ne(0)
            if bad.any():
                raise ValueError(
                    f"{name}: {col} is missing on non-zero-fire rows "
                    f"({int(bad.sum())} rows)"
                )

        other_predictors = [
            c for c in FEATURES
            if c not in undefined_fire_stats
        ]
        unexpected = frame[other_predictors].isna().sum()
        unexpected = unexpected[unexpected > 0]
        if not unexpected.empty:
            raise ValueError(
                f"{name}: unexpected missing predictor values: "
                f"{unexpected.to_dict()}"
            )

    for col in undefined_fire_stats:
        if not np.array_equal(a[col].isna().to_numpy(), b[col].isna().to_numpy()):
            raise ValueError(f"Missingness differs between variants: {col}")

    missing_counts = {
        col: int(a[col].isna().sum())
        for col in undefined_fire_stats
    }
    print("Missingness audit PASS; preserved NaNs:", missing_counts)
    a["date"] = pd.to_datetime(a["date"])
    b["date"] = pd.to_datetime(b["date"])

    expected = {2023: 2070, 2024: 2070, 2025: 2070}
    counts = a.groupby("year").size().to_dict()
    if counts != expected:
        raise ValueError(f"Eligible year counts differ from protocol: {counts}")

    return a, b


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def record_failure(out_dir: Path, manifest: dict, exc: Exception):
    """Persist failure status and re-raise the original exception upstream."""
    manifest["status"] = "FAILED"
    manifest["error"] = f"{type(exc).__name__}: {exc}"
    manifest["failure_recorded_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out_dir / "run_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    if args.preflight == args.train:
        parser.error("Specify exactly one of --preflight or --train")

    a, b = prepare_inputs()
    print("Eligible rows:", len(a))
    print("Features:", len(FEATURES))
    print("Years:", a.groupby("year").size().to_dict())
    print("Positive counts:", a.groupby("year")[TARGET].sum().to_dict())
    print("Non-weather predictors and targets match within", TOLERANCE)

    if args.preflight:
        print("PREFLIGHT PASS — no model trained; 2025 metrics not computed.")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("weather_substitution_%Y%m%dT%H%M%SZ")
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=False, exist_ok=False)

    script_path = Path(__file__).resolve()
    manifest = {
        "status": "RUNNING",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "script_sha256": sha256(script_path),
        "base_dataset_sha256": sha256(BASE_PATH),
        "variant_dataset_sha256": sha256(VARIANT_PATH),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "target": TARGET,
        "predictors": FEATURES,
        "predictor_count": len(FEATURES),
        "numeric_tolerance": TOLERANCE,
        "model": {
            "class": "RandomForestClassifier",
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "random_state": 42,
            "class_weight": None,
        },
        "splits": {"train": 2023, "validation": 2024, "test": 2025},
        "test_results_computed": False,
        "error": None,
        "failure_recorded_utc": None,
    }
    write_json(out_dir / "run_manifest.json", manifest)

    # Stage 1: Fit on 2023 and select thresholds on 2024 only.
    validation_results = []
    for name, frame in [("A_v3_weather", a), ("B_v1_weather", b)]:
        train = frame[frame["year"] == 2023]
        valid = frame[frame["year"] == 2024]

        model = make_pipeline()
        model.fit(train[FEATURES], train[TARGET].astype(int))

        valid_y = valid[TARGET].astype(int).to_numpy()
        valid_prob = model.predict_proba(valid[FEATURES])[:, 1]
        candidates = np.unique(np.concatenate(([0.0, 1.0], valid_prob)))
        scores = [
            (f1_score(valid_y, valid_prob >= threshold, zero_division=0), threshold)
            for threshold in candidates
        ]
        best_f1 = max(score for score, _ in scores)
        threshold = min(t for score, t in scores if score == best_f1)

        validation_results.append({
            "variant": name,
            "threshold": float(threshold),
            "validation_f1": float(best_f1),
            "train_rows": len(train),
            "train_positive": int(train[TARGET].sum()),
            "validation_rows": len(valid),
            "validation_positive": int(valid[TARGET].sum()),
        })

    expected_variants = {"A_v3_weather", "B_v1_weather"}
    if {r["variant"] for r in validation_results} != expected_variants:
        raise RuntimeError("Threshold selection did not produce both variants.")

    # Stage 2: Persist both thresholds once, before reading test predictions.
    threshold_payload = {
        "thresholds": validation_results,
        "threshold_selection": "maximize validation F1; lowest threshold breaks ties",
        "test_results_computed": False,
        "error": None,
        "failure_recorded_utc": None,
        "protocol_sha256": manifest["protocol_sha256"],
        "script_sha256": manifest["script_sha256"],
    }
    threshold_path = out_dir / "locked_thresholds.json"
    write_json(threshold_path, threshold_payload)
    locked_hash = sha256(threshold_path)

    # Verify persisted lock matches the in-memory thresholds.
    persisted = json.loads(threshold_path.read_text(encoding="utf-8"))
    if persisted != threshold_payload or sha256(threshold_path) != locked_hash:
        raise RuntimeError("Threshold lock verification failed.")

    manifest["validation_thresholds"] = validation_results
    manifest["threshold_lock_sha256"] = locked_hash
    manifest["missingness_policy"] = {
        "preserve_nan": True,
        "drop_rows_for_fire_stat_missingness": False,
        "undefined_fire_stats": [
            "frp_mean", "frp_max", "bright_ti4_mean",
            "bright_ti4_max", "bright_ti5_mean",
        ],
        "audit_passed": True,
    }
    write_json(out_dir / "run_manifest.json", manifest)

    # Stage 3: Evaluate on 2025 only after the lock has been written and verified.
    threshold_map = {r["variant"]: r["threshold"] for r in validation_results}
    test_results = []
    for test_name, test_frame, fit_frame in [
        ("A_v3_weather", a[a["year"] == 2025], a[a["year"] == 2023]),
        ("B_v1_weather", b[b["year"] == 2025], b[b["year"] == 2023]),
    ]:
        if sha256(threshold_path) != locked_hash:
            raise RuntimeError("Threshold lock changed before test evaluation.")

        fitted = make_pipeline()
        fitted.fit(fit_frame[FEATURES], fit_frame[TARGET].astype(int))

        test_y = test_frame[TARGET].astype(int).to_numpy()
        test_prob = fitted.predict_proba(test_frame[FEATURES])[:, 1]
        threshold = threshold_map[test_name]
        pred = test_prob >= threshold
        tn, fp, fn, tp = confusion_matrix(test_y, pred, labels=[0, 1]).ravel()

        row = {
            "variant": test_name,
            "test_rows": len(test_frame),
            "test_positive": int(test_y.sum()),
            "positive_prevalence": float(test_y.mean()),
            "threshold": float(threshold),
            "f1": float(f1_score(test_y, pred, zero_division=0)),
            "precision": float(precision_score(test_y, pred, zero_division=0)),
            "recall": float(recall_score(test_y, pred, zero_division=0)),
            "pr_auc": float(average_precision_score(test_y, test_prob)),
            "roc_auc": float(roc_auc_score(test_y, test_prob)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
        pd.DataFrame([row]).to_csv(
            out_dir / f"{test_name}_test_metrics.csv", index=False
        )
        test_results.append(row)

    manifest["test_results"] = test_results
    manifest["status"] = "COMPLETE"
    manifest["test_results_computed"] = True
    write_json(out_dir / "run_manifest.json", manifest)
    print("Experiment complete:", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import sys
        import traceback

        traceback.print_exc()
        # Record the failure in the newest run manifest, if one was created.
        root = Path(__file__).resolve().parents[2]
        runs_root = root / "data/processed/v3/experiments/weather_substitution/runs"
        candidates = sorted(
            runs_root.glob("weather_substitution_*/run_manifest.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            manifest_path = candidates[0]
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") == "RUNNING":
                    record_failure(manifest_path.parent, manifest, exc)
            except Exception:
                traceback.print_exc()
        sys.exit(1)









