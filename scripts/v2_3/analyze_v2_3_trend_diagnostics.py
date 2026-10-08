import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score


# ============================================================
# STUBBLEAI V2.3 - TREND FEATURE DIAGNOSTICS
# ============================================================
#
# Compare:
#   Frozen V2 RF
#       vs
#   V2.3 RF + trend features
#
# Focus:
#   1. Overall horizon performance
#   2. Fire transitions
#   3. Emerging N -> E events
#   4. State
#   5. Temporal season
#
# This is diagnostic only.
# No frozen V2 files are modified.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent

V2_PREDICTIONS = (
    BASE_DIR
    / "research"
    / "v2"
    / "error_analysis"
    / "rf_2025_error_predictions.csv"
)

V23_DIR = (
    BASE_DIR
    / "research"
    / "v2"
    / "v2_3"
)

V23_OUTPUT = (
    V23_DIR
    / "v2_3_trend_diagnostic_summary.csv"
)


print("=" * 70)
print("STUBBLEAI V2.3 - TREND FEATURE DIAGNOSTICS")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/7] Loading predictions...")

v2 = pd.read_csv(V2_PREDICTIONS)

v23_files = sorted(
    V23_DIR.glob(
        "v2_3_trend_predictions_plus*d.csv"
    )
)

if len(v23_files) != 5:
    raise ValueError(
        f"Expected 5 V2.3 prediction files, "
        f"found {len(v23_files)}."
    )

v23 = pd.concat(
    [
        pd.read_csv(file)
        for file in v23_files
    ],
    ignore_index=True
)

v2["date"] = pd.to_datetime(v2["date"])
v23["date"] = pd.to_datetime(v23["date"])

print(f"V2 rows:  {len(v2)}")
print(f"V2.3 rows: {len(v23)}")


# ============================================================
# 2. NORMALIZE COLUMN NAMES
# ============================================================

print("\n[2/7] Normalizing prediction columns...")

v2 = v2.rename(
    columns={
        "rf_probability": "v2_probability",
        "rf_prediction": "v2_prediction",
    }
)

v23 = v23.rename(
    columns={
        "rf_probability": "v23_probability",
        "rf_prediction": "v23_prediction",
    }
)

required_v2 = [
    "state",
    "district",
    "date",
    "fire_count",
    "target_fire_count",
    "actual_elevated",
    "horizon",
    "v2_probability",
    "v2_prediction",
]

required_v23 = [
    "state",
    "district",
    "date",
    "fire_count",
    "target_fire_count",
    "actual_elevated",
    "horizon",
    "v23_probability",
    "v23_prediction",
]

for col in required_v2:
    if col not in v2.columns:
        raise ValueError(
            f"Missing V2 column: {col}"
        )

for col in required_v23:
    if col not in v23.columns:
        raise ValueError(
            f"Missing V2.3 column: {col}"
        )


# ============================================================
# 3. MERGE V2 AND V2.3
# ============================================================

print("\n[3/7] Aligning V2 and V2.3 predictions...")

keys = [
    "state",
    "district",
    "date",
    "horizon",
]

merged = v2[
    keys
    + [
        "fire_count",
        "target_fire_count",
        "actual_elevated",
        "v2_probability",
        "v2_prediction",
    ]
].merge(
    v23[
        keys
        + [
            "v23_probability",
            "v23_prediction",
        ]
    ],
    on=keys,
    how="inner",
    validate="one_to_one",
)

print(f"Aligned rows: {len(merged)}")

if len(merged) != 9000:
    raise ValueError(
        "Expected exactly 9000 aligned predictions."
    )


# ============================================================
# 4. CREATE TRANSITION / SEASON LABELS
# ============================================================

print("\n[4/7] Creating diagnostic groups...")

current_elevated = (
    merged["fire_count"] > 2
).astype(int)

merged["transition"] = (
    current_elevated.astype(str)
    + "->"
    + merged["actual_elevated"].astype(str)
)

transition_map = {
    "0->0": "N->N",
    "0->1": "N->E",
    "1->0": "E->N",
    "1->1": "E->E",
}

merged["transition"] = (
    merged["transition"]
    .map(transition_map)
)

merged["season_phase"] = np.select(
    [
        merged["date"].dt.month.eq(10),
        (
            merged["date"].dt.month.eq(11)
            & merged["date"].dt.day.le(15)
        ),
        (
            merged["date"].dt.month.eq(11)
            & merged["date"].dt.day.gt(15)
        ),
    ],
    [
        "Early",
        "Middle",
        "Late",
    ],
    default="Other",
)


# ============================================================
# 5. GROUP METRICS
# ============================================================

def metrics_for_group(group):

    y = group["actual_elevated"]

    result = {
        "n": len(group),

        "v2_precision": precision_score(
            y,
            group["v2_prediction"],
            zero_division=0,
        ),

        "v2_recall": recall_score(
            y,
            group["v2_prediction"],
            zero_division=0,
        ),

        "v2_f1": f1_score(
            y,
            group["v2_prediction"],
            zero_division=0,
        ),

        "v23_precision": precision_score(
            y,
            group["v23_prediction"],
            zero_division=0,
        ),

        "v23_recall": recall_score(
            y,
            group["v23_prediction"],
            zero_division=0,
        ),

        "v23_f1": f1_score(
            y,
            group["v23_prediction"],
            zero_division=0,
        ),
    }

    result["delta_precision"] = (
        result["v23_precision"]
        - result["v2_precision"]
    )

    result["delta_recall"] = (
        result["v23_recall"]
        - result["v2_recall"]
    )

    result["delta_f1"] = (
        result["v23_f1"]
        - result["v2_f1"]
    )

    return pd.Series(result)


# ============================================================
# 6. RUN DIAGNOSTICS
# ============================================================

print("\n[5/7] Calculating diagnostics...")

outputs = []


# Horizon
for horizon, group in merged.groupby(
    "horizon",
    sort=True
):
    row = metrics_for_group(group)

    row["analysis"] = "horizon"
    row["group"] = horizon

    outputs.append(row)


# Transition
for transition, group in merged.groupby(
    "transition",
    sort=True
):
    row = metrics_for_group(group)

    row["analysis"] = "transition"
    row["group"] = transition

    outputs.append(row)


# State
for state, group in merged.groupby(
    "state",
    sort=True
):
    row = metrics_for_group(group)

    row["analysis"] = "state"
    row["group"] = state

    outputs.append(row)


# Season
for season, group in merged.groupby(
    "season_phase",
    sort=True
):
    row = metrics_for_group(group)

    row["analysis"] = "season"
    row["group"] = season

    outputs.append(row)


# State × season
for (
    state,
    season,
), group in merged.groupby(
    ["state", "season_phase"],
    sort=True
):
    row = metrics_for_group(group)

    row["analysis"] = "state_x_season"
    row["group"] = (
        f"{state} | {season}"
    )

    outputs.append(row)


results = pd.DataFrame(outputs)

results = results[
    [
        "analysis",
        "group",
        "n",
        "v2_precision",
        "v23_precision",
        "delta_precision",
        "v2_recall",
        "v23_recall",
        "delta_recall",
        "v2_f1",
        "v23_f1",
        "delta_f1",
    ]
]


# ============================================================
# 7. SAVE
# ============================================================

print("\n[6/7] Saving diagnostic results...")

results.to_csv(
    V23_OUTPUT,
    index=False
)

print(
    f"Saved: {V23_OUTPUT}"
)

print("\n[7/7] Diagnostic summary")
print("=" * 70)

print("\nTRANSITIONS")
print(
    results[
        results["analysis"]
        == "transition"
    ].to_string(index=False)
)

print("\nSTATE")
print(
    results[
        results["analysis"]
        == "state"
    ].to_string(index=False)
)

print("\nSEASON")
print(
    results[
        results["analysis"]
        == "season"
    ].to_string(index=False)
)

print("\nHORIZON")
print(
    results[
        results["analysis"]
        == "horizon"
    ].to_string(index=False)
)

print("\nSTATE × SEASON")
print(
    results[
        results["analysis"]
        == "state_x_season"
    ].to_string(index=False)
)

print("\nDiagnostic analysis complete.")