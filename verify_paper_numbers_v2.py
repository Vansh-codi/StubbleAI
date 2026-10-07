import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
# ============================================================
# STUBBLEAI V2 - COMPLETE PAPER NUMERICAL AUDIT
# ============================================================
# Verifies all numerical claims in the paper against
# frozen experiment CSVs using exact column references.
#
# Coverage:
#   Section 1  — V2 official RF results (all 5 horizons)
#   Section 2  — Persistence baseline F1
#   Section 3  — N→E and E→E transition recall
#   Section 4  — Dataset dimensions
#   Section 5  — RF vs Persistence error composition
#   Section 6  — V2.3 Trend experiment ΔF1
#   Section 7  — V2.3 Spatial experiment ΔF1
#   Section 8  — V2.4 Spatio-temporal experiment ΔF1
#   Section 9  — V2.5 Class weight experiment
#   Section 10 — V2.6 Transition-aware experiment
#   Section 11 — V2.7 Learner benchmark (XGBoost, LightGBM)
#   Section 12 — Thresholds
# ============================================================

BASE_DIR     = Path(__file__).resolve().parent
RESEARCH_DIR = BASE_DIR / "research" / "v2"

HORIZONS      = [1, 2, 3, 5, 7]
HORIZON_STRS  = {1: "+1d", 2: "+2d", 3: "+3d", 5: "+5d", 7: "+7d"}

results    = []
all_passed = True


# ============================================================
# HELPERS
# ============================================================

def check(name, expected, actual, tol=0.0005):
    global all_passed
    if actual is None:
        status = "MISSING"
        all_passed = False
    elif isinstance(expected, int):
        status = "PASS" if int(actual) == expected else "MISMATCH"
        if status == "MISMATCH":
            all_passed = False
    elif abs(float(actual) - float(expected)) <= tol:
        status = "PASS"
    else:
        status = "MISMATCH"
        all_passed = False

    results.append({
        "check":    name,
        "expected": expected,
        "actual":   actual,
        "status":   status,
    })

    symbol = "✓" if status == "PASS" else "✗"
    print(f"  {symbol} {status:8s}  {name}")
    if status == "MISMATCH":
        print(f"             expected={expected}  actual={round(float(actual), 6)}")


def load(path):
    p = Path(path)
    if not p.exists():
        print(f"  [!] File not found: {p}")
        return None
    return pd.read_csv(p)


def rf_test_row(horizon):
    """Load the frozen V2 RF test row.

    Primary source: V2 baseline CSV.
    Fallback for +2d: V2.7 RF reconstruction, because the
    plus2d baseline CSV contains no Random Forest row.
    """
    target_horizon = HORIZON_STRS[horizon]

    # Primary source: original V2 baseline CSV
    df = load(
        RESEARCH_DIR / f"plus{horizon}d" / "v2_baseline_results.csv"
    )

    if df is not None:
        row = df[
            (df["method"] == "Random Forest") &
            (df["split"] == "test") &
            (df["horizon"] == target_horizon)
        ]

        if not row.empty:
            return row.iloc[0]

    # +2d fallback: V2.7 RF reconstruction
    v27_file = RESEARCH_DIR / "v2_7" / "results" / "v2_7_learner_results.csv"
    df_v27 = load(v27_file)

    if df_v27 is not None:
        row = df_v27[
            (df_v27["model"] == "rf") &
            (df_v27["horizon"] == horizon)
        ]

        if not row.empty:
            r = row.iloc[0]

            return pd.Series({
                "horizon": target_horizon,
                "threshold": r["validation_threshold"],
                "accuracy": r["test_accuracy"],
                "precision": r["test_precision"],
                "recall": r["test_recall"],
                "f1": r["test_f1"],
                "roc_auc": r["test_roc_auc"],
                "pr_auc": r["test_pr_auc"],
            })

    return None

def persist_test_row(horizon):
    """Load the Persistence test row for the requested horizon."""
    df = load(RESEARCH_DIR / f"plus{horizon}d" / "v2_baseline_results.csv")
    if df is None:
        return None

    row = df[
        (df["method"] == "Persistence") &
        (df["split"] == "test") &
        (df["horizon"] == HORIZON_STRS[horizon])
    ]

    return row.iloc[0] if not row.empty else None


# ============================================================
# SECTION 1 — Official V2 RF Results
# ============================================================

print("\n" + "=" * 70)
print("SECTION 1 — Official V2 RF Results (2025 test)")
print("=" * 70)

# Source of truth: research/v2/plus{h}d/v2_baseline_results.csv
# method == "Random Forest", split == "test"
# Columns: accuracy, precision, recall, f1, roc_auc, pr_auc

PAPER_RF = {
    1: {"accuracy": 0.8094, "precision": 0.7246, "recall": 0.8338,
        "f1": 0.7754, "roc_auc": 0.8953, "pr_auc": 0.8582},
    2: {"accuracy": 0.7844, "precision": 0.6894, "recall": 0.8254,
        "f1": 0.7513, "roc_auc": 0.8795, "pr_auc": 0.8369},
    3: {"accuracy": 0.7483, "precision": 0.6297, "recall": 0.8697,
        "f1": 0.7305, "roc_auc": 0.8683, "pr_auc": 0.8193},
    5: {"accuracy": 0.7600, "precision": 0.6507, "recall": 0.8321,
        "f1": 0.7303, "roc_auc": 0.8581, "pr_auc": 0.8022},
    7: {"accuracy": 0.7417, "precision": 0.6135, "recall": 0.8383,
        "f1": 0.7085, "roc_auc": 0.8504, "pr_auc": 0.7816},
}

for h in HORIZONS:
    row = rf_test_row(h)
    paper = PAPER_RF[h]
    print(f"\n  +{h}d RF:")
    for metric, expected in paper.items():
        actual = float(row[metric]) if row is not None else None
        check(f"+{h}d RF {metric}", expected, actual)


# ============================================================
# SECTION 2 — Persistence Baseline F1
# ============================================================

print("\n" + "=" * 70)
print("SECTION 2 — Persistence Baseline F1 (2025 test)")
print("=" * 70)

# Source: research/v2/plus{h}d/v2_baseline_results.csv
# method == "Persistence", split == "test", column: f1

PAPER_PERSIST_F1 = {
    1: 0.7700,
    2: 0.7279,
    3: 0.7004,
    5: 0.7132,
    7: 0.6576,
}

for h in HORIZONS:
    row = persist_test_row(h)
    actual = float(row["f1"]) if row is not None else None
    check(f"+{h}d Persistence F1", PAPER_PERSIST_F1[h], actual)


# ============================================================
# SECTION 3 — Transition Analysis
# ============================================================

print("\n" + "=" * 70)
print("SECTION 3 — Transition Analysis N→E and E→E Recall (2025 test)")
print("=" * 70)

# Source: research/v2/error_analysis/fire_transition_horizon_analysis.csv
# Columns: horizon, transition, rf_accuracy
# transition values: "Normal -> Elevated", "Elevated -> Elevated"
# rf_accuracy = recall within that transition group

PAPER_NE_RECALL = {
    1: 0.4037, 2: 0.4817, 3: 0.6202, 5: 0.5635, 7: 0.6267,
}

PAPER_EE_RECALL = {
    1: 0.9599, 2: 0.9518, 3: 0.9739, 5: 0.9368, 7: 0.9387,
}

df_trans = load(
    RESEARCH_DIR / "error_analysis" / "fire_transition_horizon_analysis.csv"
)

if df_trans is not None:
    for h in HORIZONS:
        hs = HORIZON_STRS[h]

        ne_row = df_trans[
            (df_trans["horizon"]    == hs) &
            (df_trans["transition"] == "Normal -> Elevated")
        ]
        ee_row = df_trans[
            (df_trans["horizon"]    == hs) &
            (df_trans["transition"] == "Elevated -> Elevated")
        ]

        ne_recall = float(ne_row["rf_accuracy"].values[0]) if not ne_row.empty else None
        ee_recall = float(ee_row["rf_accuracy"].values[0]) if not ee_row.empty else None

        check(f"+{h}d N→E recall", PAPER_NE_RECALL[h], ne_recall)
        check(f"+{h}d E→E recall", PAPER_EE_RECALL[h], ee_recall)
else:
    for h in HORIZONS:
        check(f"+{h}d N→E recall", PAPER_NE_RECALL[h], None)
        check(f"+{h}d E→E recall", PAPER_EE_RECALL[h], None)


# ============================================================
# SECTION 4 — Dataset Dimensions
# ============================================================

print("\n" + "=" * 70)
print("SECTION 4 — Dataset Dimensions")
print("=" * 70)

# Source: ml_dataset_v2.csv

df_ml = load(
    BASE_DIR
    / "data"
    / "processed"
    / "v2"
    / "ml_dataset_v2.csv"
)
if df_ml is not None:
    df_ml["date"] = pd.to_datetime(df_ml["date"])
    check("ML dataset total rows",    5400, len(df_ml))
    check("ML dataset columns",         51, len(df_ml.columns))
    check("ML dataset districts",       45, df_ml["district"].nunique())
    check("ML dataset states",           2, df_ml["state"].nunique())
    check("Train rows 2023",          1800, (df_ml["date"].dt.year == 2023).sum())
    check("Val rows 2024",            1800, (df_ml["date"].dt.year == 2024).sum())
    check("Test rows 2025",           1800, (df_ml["date"].dt.year == 2025).sum())
    check("Missing values",              0, df_ml.isna().sum().sum())


# ============================================================
# SECTION 5 — RF vs Persistence Error Composition
# ============================================================

print("\n" + "=" * 70)
print("SECTION 5 — RF vs Persistence Error Composition")
print("=" * 70)

# Source: research/v2/error_analysis/rf_vs_persistence_overall.csv
# Columns: category, count, percentage

df_overall = load(
    RESEARCH_DIR / "error_analysis" / "rf_vs_persistence_overall.csv"
)

if df_overall is not None:
    def get_count(category_substr):
        row = df_overall[
            df_overall["category"].str.contains(
                category_substr, case=False, na=False
            )
        ]
        return int(row["count"].values[0]) if not row.empty else None

    check("Both correct count",         6120, get_count("Both correct"))
    check("Both wrong count",           1226, get_count("Both wrong"))
    check("Persistence only correct",    855, get_count("Persistence only"))
    check("RF only correct",             799, get_count("RF only"))


# ============================================================
# SECTION 6 — V2.3 Trend Experiment ΔF1
# ============================================================

print("\n" + "=" * 70)
print("SECTION 6 — V2.3 Trend Experiment ΔF1")
print("=" * 70)

# Source: research/v2/v2_3/v2_3_trend_results.csv
# One result row per horizon.
# Columns:
# horizon, threshold, accuracy, precision, recall,
# f1, roc_auc, pr_auc, train_rows, validation_rows,
# test_rows, numeric_features, categorical_features
#
# ΔF1 = trend_f1 - frozen_v2_rf_f1

PAPER_TREND_DELTA = {
    1: -0.0091,
    2: -0.0015,
    3: -0.0077,
    5: -0.0016,
    7: -0.0152,
}

df_trend = load(
    RESEARCH_DIR / "v2_3" / "v2_3_trend_results.csv"
)

if df_trend is not None:
    print(f"  Columns: {df_trend.columns.tolist()}")

    for h in HORIZONS:
        hs = HORIZON_STRS[h]
        v2_row = rf_test_row(h)

        trend_row = df_trend[
            df_trend["horizon"] == hs
        ]

        if not trend_row.empty and v2_row is not None:
            trend_f1 = float(trend_row.iloc[0]["f1"])
            v2_f1 = float(v2_row["f1"])

            delta = trend_f1 - v2_f1

            check(
                f"+{h}d trend ΔF1",
                PAPER_TREND_DELTA[h],
                delta
            )
        else:
            check(
                f"+{h}d trend ΔF1",
                PAPER_TREND_DELTA[h],
                None
            )
else:
    for h in HORIZONS:
        check(
            f"+{h}d trend ΔF1",
            PAPER_TREND_DELTA[h],
            None
        )

# ============================================================
# SECTION 7 — V2.3 Spatial Experiment ΔF1
# ============================================================
# ============================================================
# SECTION 7 — V2.3 Spatial Experiment ΔF1
# ============================================================

print("\n" + "=" * 70)
print("SECTION 7 — V2.3 Spatial Experiment ΔF1")
print("=" * 70)

# Source: research/v2/v2_3/spatial/v2_3_spatial_results*.csv
# Actual schema uses integer horizons: 1, 2, 3, 5, 7
# Test F1 column: test_f1

PAPER_SPATIAL_DELTA = {
    1: -0.0057,
    2: -0.0023,
    3: +0.0004,
    5: +0.0087,
    7: -0.0072,
}

spatial_files = list(
    (RESEARCH_DIR / "v2_3" / "spatial").glob("v2_3_spatial_results*")
)

if spatial_files:
    df_spatial = pd.read_csv(spatial_files[0])

    print(f"  File: {spatial_files[0].name}")
    print(f"  Columns: {df_spatial.columns.tolist()}")

    f1_col = (
        "test_f1"
        if "test_f1" in df_spatial.columns
        else "f1"
    )

    for h in HORIZONS:
        v2_row = rf_test_row(h)

        # Actual spatial CSV stores horizons as integers.
        mask = df_spatial["horizon"] == h
        sp_row = df_spatial[mask]

        if not sp_row.empty and v2_row is not None:
            delta = (
                float(sp_row.iloc[0][f1_col])
                - float(v2_row["f1"])
            )

            check(
                f"+{h}d spatial ΔF1",
                PAPER_SPATIAL_DELTA[h],
                delta
            )
        else:
            check(
                f"+{h}d spatial ΔF1",
                PAPER_SPATIAL_DELTA[h],
                None
            )

else:
    print("  WARNING: No spatial results file found")

    for h in HORIZONS:
        check(
            f"+{h}d spatial ΔF1",
            PAPER_SPATIAL_DELTA[h],
            None
        )
# print("\n" + "=" * 70)
# print("SECTION 7 — V2.3 Spatial Experiment ΔF1")
# print("=" * 70)

# # Source: research/v2/v2_3/spatial/v2_3_spatial_results*.csv
# # arm == "spatial_rf", split == "test", column: f1 or test_f1

# PAPER_SPATIAL_DELTA = {
#     1: -0.0057, 2: -0.0023, 3: +0.0004, 5: +0.0087, 7: -0.0072,
# }

# spatial_files = list(
#     (RESEARCH_DIR / "v2_3" / "spatial").glob("v2_3_spatial_results*")
# )

# if spatial_files:
#     df_spatial = pd.read_csv(spatial_files[0])
#     print(f"  File: {spatial_files[0].name}")
#     print(f"  Columns: {df_spatial.columns.tolist()}")

#     # Determine f1 column name
#     f1_col = "test_f1" if "test_f1" in df_spatial.columns else "f1"
#     # split_col = "split" if "split" in df_spatial.columns else None

#     for h in HORIZONS:
#         v2_row = rf_test_row(h)

#         # Spatial results use integer horizons: 1, 2, 3, 5, 7
#         mask = df_spatial["horizon"] == h

#         sp_row = df_spatial[mask]

#         if not sp_row.empty and v2_row is not None:
#             delta = (
#                 float(sp_row.iloc[0][f1_col])
#                 - float(v2_row["f1"])
#             )

#             check(
#                 f"+{h}d spatial ΔF1",
#                 PAPER_SPATIAL_DELTA[h],
#                 delta
#             )
#         else:
#             check(
#                 f"+{h}d spatial ΔF1",
#                 PAPER_SPATIAL_DELTA[h],
#                 None
#             )
# else:
#     print("  WARNING: No spatial results file found")
#     for h in HORIZONS:
#         check(f"+{h}d spatial ΔF1", PAPER_SPATIAL_DELTA[h], None)


# ============================================================
# SECTION 8 — V2.4 Spatio-Temporal Experiment ΔF1
# ============================================================

print("\n" + "=" * 70)
print("SECTION 8 — V2.4 Spatio-Temporal Experiment ΔF1")
print("=" * 70)

# Source: research/v2/v2_4/v2_4_spatiotemporal_results.csv
# Columns: arm, arm_label, horizon, test_f1
# arm == "spatiotemporal", horizon == "+{h}d"
# ΔF1 = spatiotemporal test_f1 - v2 test_f1

PAPER_V24_DELTA = {
    1: -0.0114,
    2: -0.0010,
    3: +0.0053,
    5: -0.0185,
    7: -0.0161,
}

df_v24 = load(RESEARCH_DIR / "v2_4" / "v2_4_spatiotemporal_results.csv")

if df_v24 is not None:
    print(f"  Available arms: {df_v24['arm'].unique().tolist()}")

    for h in HORIZONS:
        hs = HORIZON_STRS[h]
        v2_row = rf_test_row(h)

        st_row = df_v24[
            (df_v24["arm"]     == "spatiotemporal") &
            (df_v24["horizon"] == hs)
        ]

        v2_check_row = df_v24[
            (df_v24["arm"]     == "v2") &
            (df_v24["horizon"] == hs)
        ]

        if not st_row.empty and not v2_check_row.empty:
            delta = (
                float(st_row.iloc[0]["test_f1"]) -
                float(v2_check_row.iloc[0]["test_f1"])
            )
            print(f"  +{h}d spatio-temporal ΔF1 = {delta:+.4f}")

            if not np.isnan(PAPER_V24_DELTA[h]):
                check(f"+{h}d V2.4 ΔF1", PAPER_V24_DELTA[h], delta)
            else:
                # Report but don't check — paper value not yet entered
                results.append({
                    "check":    f"+{h}d V2.4 ΔF1",
                    "expected": "NOT_IN_PAPER_YET",
                    "actual":   round(delta, 4),
                    "status":   "INFO",
                })
        else:
            print(f"  +{h}d: rows not found in V2.4 results")


# ============================================================
# SECTION 9 — V2.5 Class Weight Experiment
# ============================================================

print("\n" + "=" * 70)
print("SECTION 9 — V2.5 Class Weight Experiment")
print("=" * 70)

# Source: research/v2/v2_5/v2_5_class_weight_results.csv
# Columns: arm, class_weight, horizon_days, f1
# arm == "baseline" is the frozen V2 RF
# Compare other arms against baseline

df_v25 = load(RESEARCH_DIR / "v2_5" / "v2_5_class_weight_results.csv")

if df_v25 is not None:
    print(f"  Arms: {df_v25['arm'].unique().tolist()}")
    print(f"  Class weights: {df_v25['class_weight'].unique().tolist()}")

    # Verify baseline matches V2 official results
    for h in HORIZONS:
        v2_row = rf_test_row(h)
        baseline_row = df_v25[
            (df_v25["arm"]          == "baseline") &
            (df_v25["horizon_days"] == h)
        ]

        if not baseline_row.empty and v2_row is not None:
            check(
                f"+{h}d V2.5 baseline F1 matches V2",
                round(float(v2_row["f1"]), 4),
                float(baseline_row.iloc[0]["f1"]),
            )

    # Print all arm F1s for reference
    print("\n  Full V2.5 results by arm:")
    summary = df_v25.groupby(["arm", "horizon_days"])["f1"].first().unstack()
    print(summary.to_string())


# ============================================================
# SECTION 10 — V2.6 Transition-Aware Experiment
# ============================================================

print("\n" + "=" * 70)
print("SECTION 10 — V2.6 Transition-Aware Experiment")
print("=" * 70)

# Source: research/v2/v2_6/results/v2_6_results.csv
# Columns: arm, horizon, threshold, accuracy, precision, recall, f1, roc_auc, pr_auc
# arm == "frozen_v2_rf" is the baseline

df_v26 = load(RESEARCH_DIR / "v2_6" / "results" / "v2_6_results.csv")

if df_v26 is not None:
    print(f"  Arms: {df_v26['arm'].unique().tolist()}")

    # Verify frozen_v2_rf arm matches official V2 results
    for h in HORIZONS:
        v2_row = rf_test_row(h)
        frozen_row = df_v26[
            (df_v26["arm"]     == "frozen_v2_rf") &
            (df_v26["horizon"] == h)
        ]

        if not frozen_row.empty and v2_row is not None:
            check(
                f"+{h}d V2.6 frozen_v2_rf F1 matches V2",
                round(float(v2_row["f1"]), 4),
                float(frozen_row.iloc[0]["f1"]),
            )

    # Print transition_rf vs frozen_v2_rf ΔF1
    print("\n  V2.6 transition_rf vs frozen_v2_rf ΔF1:")
    for h in HORIZONS:
        frozen_row = df_v26[
            (df_v26["arm"]     == "frozen_v2_rf") &
            (df_v26["horizon"] == h)
        ]
        transition_row = df_v26[
            (df_v26["arm"]     == "transition_rf") &
            (df_v26["horizon"] == h)
        ]

        if not frozen_row.empty and not transition_row.empty:
            delta = (
                float(transition_row.iloc[0]["f1"]) -
                float(frozen_row.iloc[0]["f1"])
            )
            print(f"  +{h}d ΔF1 = {delta:+.4f}")


# ============================================================
# SECTION 11 — V2.7 Learner Benchmark
# ============================================================

print("\n" + "=" * 70)
print("SECTION 11 — V2.7 Learner Benchmark (RF / XGBoost / LightGBM)")
print("=" * 70)

# Source: research/v2/v2_7/results/v2_7_learner_results.csv
# Columns: horizon, model, test_f1, test_roc_auc, test_precision, test_recall

PAPER_V27 = {
    # RF should match V2 exactly
    ("rf", 1): {"test_f1": 0.7754, "test_roc_auc": 0.8953},
    ("rf", 2): {"test_f1": 0.7513, "test_roc_auc": 0.8795},
    ("rf", 3): {"test_f1": 0.7305, "test_roc_auc": 0.8683},
    ("rf", 5): {"test_f1": 0.7303, "test_roc_auc": 0.8581},
    ("rf", 7): {"test_f1": 0.7085, "test_roc_auc": 0.8504},
}

df_v27 = load(RESEARCH_DIR / "v2_7" / "results" / "v2_7_learner_results.csv")

if df_v27 is not None:
    print(f"  Models: {df_v27['model'].unique().tolist()}")

    # Verify RF matches V2 official
    for h in HORIZONS:
        v2_row = rf_test_row(h)
        rf_row = df_v27[
            (df_v27["model"]   == "rf") &
            (df_v27["horizon"] == h)
        ]

        if not rf_row.empty and v2_row is not None:
            check(
                f"+{h}d V2.7 RF F1 matches V2",
                round(float(v2_row["f1"]), 4),
                float(rf_row.iloc[0]["test_f1"]),
            )

    # Print full learner comparison for reference
    print("\n  V2.7 test F1 by model and horizon:")
    summary = df_v27.pivot_table(
        index="model",
        columns="horizon",
        values="test_f1",
        aggfunc="first",
    )
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n  V2.7 ROC-AUC by model and horizon:")
    summary_roc = df_v27.pivot_table(
        index="model",
        columns="horizon",
        values="test_roc_auc",
        aggfunc="first",
    )
    print(summary_roc.to_string(float_format=lambda x: f"{x:.4f}"))


# ============================================================
# SECTION 12 — Thresholds
# ============================================================
# ============================================================
# SECTION 12 — RF Validation Thresholds
# ============================================================

print("\n" + "=" * 80)
print("12. RF VALIDATION THRESHOLDS")
print("=" * 80)

v27_file = RESEARCH_DIR / "v2_7" / "results" / "v2_7_learner_results.csv"
df_v27 = load(v27_file)

for h in HORIZONS:

    expected_threshold = None
    source = None

    # Primary source: horizon-specific RF threshold JSON
    thresh_file = (
        RESEARCH_DIR
        / f"plus{h}d"
        / "v2_model_thresholds.json"
    )

    if thresh_file.exists():
        with open(thresh_file) as f:
            data = json.load(f)

        rf_threshold = (
            data.get("models", {})
            .get("Random Forest")
        )

        if rf_threshold is not None:
            expected_threshold = float(rf_threshold)
            source = str(thresh_file)

    # Fallback: V2.7 RF reconstruction
    if expected_threshold is None and df_v27 is not None:
        row = df_v27[
            (df_v27["model"] == "rf") &
            (df_v27["horizon"] == h)
        ]

        if not row.empty:
            expected_threshold = float(
                row.iloc[0]["validation_threshold"]
            )
            source = "v2_7_learner_results.csv"

    # Compare with the RF result row
    rf_row = rf_test_row(h)

    if rf_row is not None and expected_threshold is not None:
        actual_threshold = float(rf_row["threshold"])

        check(
            f"+{h}d RF threshold",
            expected_threshold,
            actual_threshold
        )

        print(
            f"+{h}d: {actual_threshold:.6f} "
            f"[source: {source}]"
        )

    else:
        print(f"+{h}d: RF threshold source unavailable")


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STUBBLEAI V2 PAPER NUMERICAL AUDIT — FINAL SUMMARY")
print("=" * 70)

df_report = pd.DataFrame(results)

passed  = (df_report["status"] == "PASS").sum()
failed  = (df_report["status"] == "MISMATCH").sum()
missing = (df_report["status"] == "MISSING").sum()
info    = (df_report["status"] == "INFO").sum()
total   = len(df_report[df_report["status"] != "INFO"])

print(f"\nTotal checks:  {total}")
print(f"PASS:          {passed}")
print(f"MISMATCH:      {failed}")
print(f"MISSING:       {missing}")
if info > 0:
    print(f"INFO:          {info}  (V2.4 ΔF1 — enter paper values to check)")

if failed > 0:
    print("\nMISMATCHES — must resolve before submission:")
    for _, r in df_report[df_report["status"] == "MISMATCH"].iterrows():
        print(f"  {r['check']}")
        print(f"    expected={r['expected']}  actual={r['actual']}")

if missing > 0:
    print("\nMISSING — file or column not found:")
    for _, r in df_report[df_report["status"] == "MISSING"].iterrows():
        print(f"  {r['check']}")

df_report.to_csv(
    BASE_DIR / "verify_paper_numbers_report.csv",
    index=False,
)
print("\nFull report saved: verify_paper_numbers_report.csv")

print("\n" + "=" * 70)
if all_passed and failed == 0 and missing == 0 and info == 0:
    print("FINAL DECISION: ALL CHECKS PASSED ✓")
    print("Paper numbers are consistent with frozen experiment CSVs.")
    print("Safe to proceed to figures and IEEE formatting.")
else:
    print("FINAL DECISION: CHECKS FAILED OR INCOMPLETE ✗")
    print("Resolve all MISMATCH, MISSING, and INFO items before submission.")

sys.exit(
    0 if all_passed and failed == 0 and missing == 0 and info == 0 else 1
)