
import numpy as np
from pathlib import Path

import pandas as pd
# ============================================================
# STUBBLEAI - PREDICTION TRACKER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PREDICTION_FILE = BASE_DIR / "stubbleai_2026_predictions.csv"
TRACKER_FILE = BASE_DIR / "prediction_tracker.csv"
ACTUAL_FIRE_FILE = BASE_DIR / "live_district_fire_2026.csv"


print("=" * 70)
print("STUBBLEAI - PREDICTION TRACKER")
print("=" * 70)


# ------------------------------------------------------------
# 1. Check prediction file
# ------------------------------------------------------------

if not PREDICTION_FILE.exists():

    print()
    print("No prediction file found yet.")
    print(
        "Run the live prediction pipeline first. "
        "The tracker cannot verify predictions until "
        "a prediction file exists."
    )

    raise SystemExit(0)


# ------------------------------------------------------------
# 2. Load predictions
# ------------------------------------------------------------

pred = pd.read_csv(
    PREDICTION_FILE
)
required_prediction_columns = [
    "prediction_date",
    "state",
    "district",
    "risk_probability",
    "risk_label",
    "fire_lag_1d",
    "fire_lag_3d",
    "fire_lag_7d",
    "fire_mean_3d",
    "fire_mean_7d",
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
]

missing_prediction_columns = [
    col
    for col in required_prediction_columns
    if col not in pred.columns
]

if missing_prediction_columns:
    raise ValueError(
        "Prediction file is missing required columns: "
        f"{missing_prediction_columns}"
    )
pred["prediction_date"] = pd.to_datetime(
    pred["prediction_date"],
    errors="coerce"
)

if pred["prediction_date"].isna().any():
    raise ValueError(
        "Invalid prediction dates found."
    )
if not pred["prediction_date"].dt.year.eq(2026).all():
    raise ValueError(
        "Prediction data contains dates outside 2026."
    )
pred["state"] = (
    pred["state"]
    .astype(str)
    .str.strip()
)

pred["district"] = (
    pred["district"]
    .astype(str)
    .str.strip()
)
pred["risk_probability"] = pd.to_numeric(
    pred["risk_probability"],
    errors="coerce"
)

if pred["risk_probability"].isna().any():
    raise ValueError(
        "Invalid risk probabilities found."
    )

if (
    (pred["risk_probability"] < 0)
    | (pred["risk_probability"] > 1)
).any():
    raise ValueError(
        "Risk probabilities must be between 0 and 1."
    )
valid_labels = {"Normal", "Elevated"}

invalid_labels = set(
    pred["risk_label"].dropna().unique()
) - valid_labels

if invalid_labels:
    raise ValueError(
        f"Invalid prediction labels found: {invalid_labels}"
    )
print()
print("Prediction rows:", len(pred))

if pred.empty:
    print()
    print("Prediction file contains no prediction rows.")
    raise SystemExit(1)
duplicate_predictions = pred.duplicated(
    subset=[
        "prediction_date",
        "state",
        "district",
    ],
    keep=False,
)

if duplicate_predictions.any():
    print()
    print(
        "Duplicate district-date prediction "
        "observations detected."
    )

    print(
        pred.loc[
            duplicate_predictions,
            [
                "prediction_date",
                "state",
                "district",
                "risk_probability",
                "risk_label",
            ],
        ]
        .head(20)
        .to_string(index=False)
    )

    raise SystemExit(1)
# ------------------------------------------------------------
# 3. Load actual FIRMS district observations
# ------------------------------------------------------------

if ACTUAL_FIRE_FILE.exists():

    actual = pd.read_csv(
        ACTUAL_FIRE_FILE
    )

    required_actual_columns = [
        "date",
        "state",
        "district",
        "fire_count",
    ]

    missing_actual_columns = [
        col
        for col in required_actual_columns
        if col not in actual.columns
    ]

    if missing_actual_columns:
        raise ValueError(
            "Actual FIRMS file is missing required columns: "
            f"{missing_actual_columns}"
        )

    actual["date"] = pd.to_datetime(
        actual["date"],
        errors="coerce"
    )

    if actual["date"].isna().any():
        raise ValueError(
            "Invalid dates found in actual FIRMS data."
        )

    actual["state"] = (
        actual["state"]
        .astype(str)
        .str.strip()
    )

    actual["district"] = (
        actual["district"]
        .astype(str)
        .str.strip()
    )

    actual["fire_count"] = pd.to_numeric(
        actual["fire_count"],
        errors="coerce"
    )

    if actual["fire_count"].isna().any():
        raise ValueError(
            "Invalid fire_count values found."
        )

    if (actual["fire_count"] < 0).any():
        raise ValueError(
            "Negative fire_count values found."
        )
    if not actual.empty:
        if not actual["date"].dt.year.eq(2026).all():
            raise ValueError(
               "Actual FIRMS data contains dates outside 2026."
             )

    

else:

    actual = pd.DataFrame(
        columns=[
            "date",
            "state",
            "district",
            "fire_count"
        ]
    )
duplicate_actual = actual.duplicated(
    subset=[
        "date",
        "state",
        "district"
    ],
    keep=False
)

if duplicate_actual.any():
    print()
    print(
        "Duplicate district-date actual FIRMS "
        "observations detected."
    )

    print(
        actual.loc[
            duplicate_actual,
            [
                "date",
                "state",
                "district",
                "fire_count"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    raise SystemExit(1)

# ------------------------------------------------------------
# 4. Merge predictions with actual observations
# ------------------------------------------------------------

merged = pred.merge(
    actual[
        [
            "date",
            "state",
            "district",
            "fire_count"
        ]
    ],
    left_on=[
        "prediction_date",
        "state",
        "district"
    ],
    right_on=[
        "date",
        "state",
        "district"
    ],
    how="left"
)
if len(merged) != len(pred):
    raise ValueError(
        "Prediction tracker merge changed the number "
        "of prediction rows."
    )

# ------------------------------------------------------------
# 5. Actual class
# ------------------------------------------------------------

merged["actual_risk"] = merged[
    "fire_count"
].apply(
    lambda x:
    "Elevated"
    if pd.notna(x) and x > 2
    else (
        "Normal"
        if pd.notna(x)
        else pd.NA
    )
)


# ------------------------------------------------------------
# 6. Verification status
# ------------------------------------------------------------

merged["verification_status"] = "Pending"

verified_mask = merged["actual_risk"].notna()

merged.loc[
    verified_mask,
    "verification_status"
] = np.where(
    merged.loc[
        verified_mask,
        "risk_label"
    ]
    ==
    merged.loc[
        verified_mask,
        "actual_risk"
    ],
    "Correct",
    "Incorrect"
)

if not merged["verification_status"].isin(
    ["Pending", "Correct", "Incorrect"]
).all():
    raise ValueError(
        "Invalid verification status generated."
    )
    

# ------------------------------------------------------------
# 7. Error type
# ------------------------------------------------------------

def get_error_type(row):

    if row["verification_status"] == "Pending":
        return "Pending"

    predicted = row["risk_label"]
    actual = row["actual_risk"]

    if predicted == "Elevated" and actual == "Elevated":
        return "True Positive"

    if predicted == "Normal" and actual == "Normal":
        return "True Negative"

    if predicted == "Elevated" and actual == "Normal":
        return "False Positive"

    if predicted == "Normal" and actual == "Elevated":
        return "False Negative"

    return "Unknown"


merged["error_type"] = merged.apply(
    get_error_type,
    axis=1
)


# ------------------------------------------------------------
# 8. Clean output
# ------------------------------------------------------------

output_columns = [
    "prediction_date",
    "state",
    "district",
    "risk_probability",
    "risk_label",
    "fire_lag_1d",
    "fire_lag_3d",
    "fire_lag_7d",
    "fire_mean_3d",
    "fire_mean_7d",
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
    "fire_count",
    "actual_risk",
    "verification_status",
    "error_type",
]


tracker = merged[
    output_columns
].copy()


tracker = tracker.sort_values(
    [
        "prediction_date",
        "risk_probability"
    ],
    ascending=[
        True,
        False
    ]
).reset_index(drop=True)


# ------------------------------------------------------------
# 9. Save tracker
# ------------------------------------------------------------

TEMP_TRACKER_FILE = TRACKER_FILE.with_suffix(
    ".tmp.csv"
)

tracker.to_csv(
    TEMP_TRACKER_FILE,
    index=False
)

TEMP_TRACKER_FILE.replace(
    TRACKER_FILE
)


# ------------------------------------------------------------
# 10. Summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("PREDICTION TRACKER UPDATED")
print("=" * 70)

print(
    "Total predictions:",
    len(tracker)
)

print()
print("Verification status:")

print(
    tracker[
        "verification_status"
    ].value_counts()
)


verified = tracker[
    tracker["verification_status"] != "Pending"
].copy()
verification_coverage = (
    len(verified) / len(tracker)
    if len(tracker) > 0
    else 0
)

print()
print(
    f"Verification coverage: "
    f"{verification_coverage:.2%}"
)

if len(verified) > 0:
    

    print()
    print("Verified predictions:", len(verified))

    print()
    print("Error types:")

    print(
        verified[
            "error_type"
        ].value_counts()
    )

    accuracy = (
        verified["verification_status"]
        .eq("Correct")
        .mean()
    )

    print()
    print(
    f"Live verified accuracy "
    f"(verified cases only): "
    f"{accuracy:.4f}"
)
else:

    print()
    print(
        "No predictions have actual FIRMS "
        "verification yet."
    )


print()
print("Saved to:")
print(TRACKER_FILE)

print("=" * 70)