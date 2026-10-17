from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "data/processed/v3/experiments/grid_sensitivity_010/resolution_comparison/runs/resolution_20261017T034323Z"
PREDICTIONS = RUN_DIR / "test_predictions_2025.csv"
OUTPUT_ROOT = ROOT / "data/processed/v3/experiments/grid_sensitivity_010/robust_validation"
OUTPUT_DIR = OUTPUT_ROOT / "paired_bootstrap_20261017"

B = 10_000
SEED = 20261017
ALPHA = 0.05
KEYS = ["state", "district", "date"]
VARIANTS = ["baseline", "spatial_005", "spatial_010"]
COMPARISONS = [
    ("baseline", "spatial_005"),
    ("baseline", "spatial_010"),
    ("spatial_005", "spatial_010"),
]
METRICS = ["f1", "pr_auc"]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight():
    require(PREDICTIONS.is_file(), f"Missing predictions: {PREDICTIONS}")
    require(not OUTPUT_DIR.exists(), f"Refusing to overwrite: {OUTPUT_DIR}")
    df = pd.read_csv(PREDICTIONS)
    needed = set(KEYS + [
        "y_true", "predicted_probability", "y_pred",
        "locked_threshold", "variant",
    ])
    require(needed.issubset(df.columns),
            f"Prediction columns missing: {sorted(needed - set(df.columns))}")
    require(set(df["variant"].unique()) == set(VARIANTS),
            "Prediction variants do not match expected variants")
    for variant in VARIANTS:
        part = df[df["variant"] == variant]
        require(not part.duplicated(KEYS).any(), f"{variant}: duplicate test keys")
        require(part["y_true"].notna().all(), f"{variant}: missing labels")
        require(part["predicted_probability"].notna().all(),
                f"{variant}: missing probabilities")
        require(part["y_pred"].notna().all(), f"{variant}: missing predictions")
        require(part["locked_threshold"].notna().all(),
                f"{variant}: missing locked threshold")
        require(part["predicted_probability"].between(0, 1).all(),
                f"{variant}: probabilities outside [0,1]")
        require(part["y_true"].isin([0, 1]).all(), f"{variant}: invalid labels")
        require(part["y_pred"].isin([0, 1]).all(), f"{variant}: invalid predictions")

    print("PREFLIGHT PASS")
    print("Prediction file:", PREDICTIONS)
    print("Rows per variant:", len(df) // len(VARIANTS))
    print("Bootstrap replicates:", B)
    print("Seed:", SEED)
    print("Output directory:", OUTPUT_DIR)
    print("No bootstrap run performed and no outputs written.")


def prepare_data():
    df = pd.read_csv(PREDICTIONS)
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    arrays = {}

    for variant in VARIANTS:
        part = df[df["variant"] == variant].sort_values(KEYS).reset_index(drop=True)
        keys = part[KEYS]
        require(not keys.duplicated().any(), f"{variant}: duplicate row keys")
        if variant == VARIANTS[0]:
            reference_keys = keys
            reference_y = part["y_true"].to_numpy(dtype=np.int8)
        else:
            require(keys.equals(reference_keys),
                    f"{variant}: row keys do not match baseline in order")
            require(np.array_equal(part["y_true"].to_numpy(dtype=np.int8), reference_y),
                    f"{variant}: target labels differ from baseline")

        expected_pred = (
            part["predicted_probability"].to_numpy()
            >= part["locked_threshold"].to_numpy()
        ).astype(np.int8)
        require(np.array_equal(expected_pred, part["y_pred"].to_numpy(dtype=np.int8)),
                f"{variant}: y_pred inconsistent with locked threshold")

        arrays[variant] = {
            "prob": part["predicted_probability"].to_numpy(dtype=np.float64),
            "pred": part["y_pred"].to_numpy(dtype=np.int8),
            "y": part["y_true"].to_numpy(dtype=np.int8),
        }

    require(len(reference_y) == 11408, f"Unexpected 2025 test rows: {len(reference_y)}")
    require(int(reference_y.sum()) == 3181, "Unexpected 2025 positive count")
    require(df["state"].nunique() == 6, "Expected six states")

    return df[df["variant"] == "baseline"].sort_values(KEYS).reset_index(drop=True), arrays


def point_metrics(y, model):
    return {
        "f1": float(f1_score(y, model["pred"], zero_division=0)),
        "pr_auc": float(average_precision_score(y, model["prob"])),
    }


def prepare_weighted_ap(y, prob):
    # Sort once. Bootstrap resampling is represented by integer row weights.
    order = np.argsort(-prob, kind="mergesort")
    sorted_prob = prob[order]
    starts = np.r_[0, np.flatnonzero(sorted_prob[1:] != sorted_prob[:-1]) + 1]
    return order, starts


def weighted_ap(y, row_weights, order, starts):
    weighted_positive = row_weights * y
    sorted_pos = weighted_positive[order]
    sorted_weight = row_weights[order]
    group_pos = np.add.reduceat(sorted_pos, starts)
    group_weight = np.add.reduceat(sorted_weight, starts)
    total_pos = group_pos.sum()
    if total_pos <= 0:
        return np.nan
    tp = np.cumsum(group_pos)
    total = np.cumsum(group_weight)
    precision = tp / np.maximum(total, 1)
    return float(np.sum((group_pos / total_pos) * precision))


def weighted_f1(y, pred, weights):
    tp = float(np.sum(weights * y * pred))
    fp = float(np.sum(weights * (1 - y) * pred))
    fn = float(np.sum(weights * y * (1 - pred)))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp / denom)


def bootstrap_group(label, indices, arrays, rng):
    y = arrays["baseline"]["y"][indices]
    n = len(indices)
    require(n > 1, f"{label}: insufficient rows for bootstrap")

    local = {}
    for variant in VARIANTS:
        local[variant] = {
            "y": arrays[variant]["y"][indices],
            "pred": arrays[variant]["pred"][indices],
            "prob": arrays[variant]["prob"][indices],
        }
        require(np.array_equal(local[variant]["y"], y),
                f"{label}: target mismatch for {variant}")

    points = {v: point_metrics(y, local[v]) for v in VARIANTS}
    ap_structures = {
        v: prepare_weighted_ap(y, local[v]["prob"]) for v in VARIANTS
    }

    boot = {
        (ref, cand, metric): np.empty(B, dtype=np.float64)
        for ref, cand in COMPARISONS for metric in METRICS
    }

    for b in range(B):
        # One paired resample shared by every model in this group.
        sampled = rng.integers(0, n, size=n)
        weights = np.bincount(sampled, minlength=n).astype(np.float64)
        boot_scores = {}

        for variant in VARIANTS:
            m = local[variant]
            f1 = weighted_f1(m["y"], m["pred"], weights)
            order, starts = ap_structures[variant]
            ap = weighted_ap(m["y"], weights, order, starts)
            boot_scores[variant] = {"f1": f1, "pr_auc": ap}

        for ref, cand in COMPARISONS:
            for metric in METRICS:
                boot[(ref, cand, metric)][b] = (
                    boot_scores[cand][metric] - boot_scores[ref][metric]
                )

        if (b + 1) % 2000 == 0:
            print(f"  {label}: {b + 1:,}/{B:,} replicates", flush=True)

    results = []
    for ref, cand in COMPARISONS:
        for metric in METRICS:
            dist = boot[(ref, cand, metric)]
            dist = dist[np.isfinite(dist)]
            require(len(dist) >= int(B * 0.99),
                    f"{label}: too few valid replicates for {metric}")
            delta = points[cand][metric] - points[ref][metric]
            lower, upper = np.quantile(dist, [ALPHA / 2, 1 - ALPHA / 2])
            results.append({
                "group": label,
                "reference": ref,
                "candidate": cand,
                "metric": metric,
                "reference_point_estimate": points[ref][metric],
                "candidate_point_estimate": points[cand][metric],
                "difference_candidate_minus_reference": delta,
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "ci_method": "paired row bootstrap percentile",
                "valid_replicates": int(len(dist)),
                "bootstrap_replicates_requested": B,
                "seed": SEED,
                "rows": n,
                "positives": int(y.sum()),
            })
    return results


def run():
    base, arrays = prepare_data()
    rng = np.random.default_rng(SEED)

    groups = [("POOLED", np.arange(len(base)))]
    for state in ["ASSAM", "BIHAR", "HARYANA", "MADHYA PRADESH", "PUNJAB", "UTTAR PRADESH"]:
        idx = np.flatnonzero(base["state"].astype(str).str.upper().to_numpy() == state)
        require(len(idx) > 0, f"No test rows found for state {state}")
        groups.append((state, idx))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    all_results = []
    for label, idx in groups:
        print(f"\nBootstrapping {label}: {len(idx):,} rows", flush=True)
        all_results.extend(bootstrap_group(label, idx, arrays, rng))

    result_df = pd.DataFrame(all_results)
    pooled = result_df[result_df["group"] == "POOLED"].copy()
    state_results = result_df[result_df["group"] != "POOLED"].copy()

    pooled.to_csv(OUTPUT_DIR / "pooled_paired_bootstrap_cis.csv", index=False)
    state_results.to_csv(OUTPUT_DIR / "per_state_paired_bootstrap_cis.csv", index=False)

    manifest = {
        "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_file": str(PREDICTIONS.relative_to(ROOT)),
        "prediction_sha256": sha256(PREDICTIONS),
        "source_model_run": RUN_DIR.name,
        "bootstrap_replicates": B,
        "seed": SEED,
        "confidence_level": 0.95,
        "interval_method": "Percentile interval from paired row-level bootstrap",
        "resampling_unit": "2025 district-day test row; same sampled row indices for every variant",
        "metrics": METRICS,
        "comparisons": [
            {"reference": r, "candidate": c, "difference": "candidate minus reference"}
            for r, c in COMPARISONS
        ],
        "groups": [name for name, _ in groups],
        "eligible_rows_pooled": int(len(base)),
        "positive_labels_pooled": int(arrays["baseline"]["y"].sum()),
        "outputs": [
            "pooled_paired_bootstrap_cis.csv",
            "per_state_paired_bootstrap_cis.csv",
        ],
        "interpretation_note": (
            "Intervals crossing zero indicate uncertainty about the direction "
            "of the paired metric difference; they do not establish equivalence."
        ),
    }
    (OUTPUT_DIR / "bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nPOOLED PAIRED BOOTSTRAP RESULTS")
    print(pooled[[
        "reference", "candidate", "metric",
        "difference_candidate_minus_reference", "ci95_lower", "ci95_upper",
    ]].to_string(index=False))
    print("\nASSAM AND BIHAR CIs")
    focus = state_results[state_results["group"].isin(["ASSAM", "BIHAR"])]
    print(focus[[
        "group", "reference", "candidate", "metric",
        "difference_candidate_minus_reference", "ci95_lower", "ci95_upper",
    ]].to_string(index=False))
    print("\nBOOTSTRAP COMPLETE:", OUTPUT_DIR)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        run()
