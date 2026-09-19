import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ============================================================
# STUBBLEAI - 2026 LIVE RISK PREDICTION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "model" / "stubbleai_deployment_model.pkl"
CONFIG_FILE = BASE_DIR / "model" / "stubbleai_deployment_config.json"

FIRE_FEATURE_FILE = BASE_DIR / "live_2026_fire_features.csv"
WEATHER_FILE = BASE_DIR / "live_weather_2026.csv"

# Historical engineered dataset used only for cold-start priors
HISTORICAL_FEATURE_FILE = BASE_DIR / "ml_dataset_base.csv"

OUTPUT_FILE = BASE_DIR / "stubbleai_2026_predictions.csv"
HISTORY_FILE = BASE_DIR / "stubbleai_2026_prediction_history.csv"



print("=" * 70)
print("STUBBLEAI - 2026 LIVE RISK PREDICTION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load deployment model
# ------------------------------------------------------------

print()
print("Loading deployment model...")
if not MODEL_FILE.exists():
    print(f"Model file not found: {MODEL_FILE}")
    raise SystemExit(1)

if not CONFIG_FILE.exists():
    print(f"Model configuration file not found: {CONFIG_FILE}")
    raise SystemExit(1)
model = joblib.load(MODEL_FILE)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

threshold = float(
    config.get("threshold", 0.40)
)
if not 0 < threshold < 1:
    raise ValueError(
        f"Invalid model threshold: {threshold}"
    )

print("Model loaded.")
print("Threshold:", threshold)


# ------------------------------------------------------------
# 2. Load fire features
# ------------------------------------------------------------

if not FIRE_FEATURE_FILE.exists():
    print(f"Fire feature file not found: {FIRE_FEATURE_FILE}")
    raise SystemExit(1)

fire = pd.read_csv(FIRE_FEATURE_FILE)

required_fire_columns = [
    "date",
    "state",
    "district",
    "fire_count",
    "fire_lag_1d",
    "fire_lag_3d",
    "fire_lag_7d",
    "fire_mean_3d",
    "fire_mean_7d",
]

missing_fire_columns = [
    col for col in required_fire_columns
    if col not in fire.columns
]

if missing_fire_columns:
    raise ValueError(
        f"Missing required fire-feature columns: {missing_fire_columns}"
    )

fire["date"] = pd.to_datetime(
    fire["date"],
    errors="coerce"
)

if fire["date"].isna().any():
    raise ValueError("Invalid dates found in fire feature data.")

fire["district"] = (
    fire["district"]
    .astype(str)
    .str.strip()
)

fire["state"] = (
    fire["state"]
    .astype(str)
    .str.strip()
)



# ------------------------------------------------------------
# 3. Keep 2026 only
# ------------------------------------------------------------

fire = fire[
    fire["date"].dt.year == 2026
].copy()

duplicate_fire_rows = fire.duplicated(
    subset=["date", "state", "district"],
    keep=False
)

if duplicate_fire_rows.any():
    print()
    print("Duplicate district-date fire feature rows detected.")
    print(
        fire.loc[
            duplicate_fire_rows,
            ["date", "state", "district"]
        ]
        .head(20)
        .to_string(index=False)
    )
    raise SystemExit(1)
if fire.empty:
    print()
    print("No 2026 FIRMS data available.")
    raise SystemExit(1)


# ------------------------------------------------------------
# 4. Load weather
# ------------------------------------------------------------

if not WEATHER_FILE.exists():
    print(f"Weather file not found: {WEATHER_FILE}")
    raise SystemExit(1)

weather = pd.read_csv(WEATHER_FILE)

required_weather_columns = [
    "date",
    "state",
    "district",
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
]

missing_weather_columns = [
    col for col in required_weather_columns
    if col not in weather.columns
]

if missing_weather_columns:
    raise ValueError(
        f"Missing required weather columns: {missing_weather_columns}"
    )

weather["date"] = pd.to_datetime(
    weather["date"],
    errors="coerce"
)

if weather["date"].isna().any():
    raise ValueError("Invalid dates found in weather data.")
if not weather["date"].dt.year.eq(2026).all():
    raise ValueError(
        "Weather data contains non-2026 records."
    )

weather["district"] = (
    weather["district"]
    .astype(str)
    .str.strip()
)

weather["state"] = (
    weather["state"]
    .astype(str)
    .str.strip()
)


duplicate_weather_rows = weather.duplicated(
    subset=["date", "state", "district"],
    keep=False
)

if duplicate_weather_rows.any():
    print()
    print("Duplicate district-date weather rows detected.")
    print(
        weather.loc[
            duplicate_weather_rows,
            ["date", "state", "district"]
        ]
        .head(20)
        .to_string(index=False)
    )
    raise SystemExit(1)
# ------------------------------------------------------------
# 5. Determine latest observed FIRMS date
# ------------------------------------------------------------

latest_fire_date = fire["date"].max()

print()
print(
    "Latest actual FIRMS date:",
    latest_fire_date.date()
)


# ------------------------------------------------------------
# 6. Prediction date = next day
# ------------------------------------------------------------

prediction_date = (
    latest_fire_date
    + pd.Timedelta(days=1)
)
in_training_season = prediction_date.month in [10, 11]

if not in_training_season:
    print()
    print("=" * 70)
    print("OUTSIDE HISTORICAL TRAINING SEASON")
    print("=" * 70)

    print(
        f"Prediction date {prediction_date.date()} "
        "is outside the October-November training season."
    )

    print(
        "Generating an operational inference using "
        "current 2026 fire and weather inputs."
    )

    print(
        "This date is outside the model's historical "
        "seasonal validation scope."
    )

print(
    "Prediction date:",
    prediction_date.date()
)

print(
    "Prediction date:",
    prediction_date.date()
)


# ------------------------------------------------------------
# 7. Select weather for prediction date
# ------------------------------------------------------------

weather_prediction = weather[
    weather["date"] == prediction_date
].copy()

print(
    "Weather rows for prediction date:",
    len(weather_prediction)
)
if len(weather_prediction) != 45:
    raise ValueError(
        f"Expected weather data for 45 districts on "
        f"{prediction_date.date()}, found "
        f"{len(weather_prediction)}."
    )


# ------------------------------------------------------------
# 8. Select fire features from latest observed date
# ------------------------------------------------------------

fire_prediction = fire[
    fire["date"] == latest_fire_date
].copy()

print(
    "Fire-feature rows:",
    len(fire_prediction)
)
if len(fire_prediction) != 45:
    raise ValueError(
        f"Expected fire features for 45 districts on "
        f"{latest_fire_date.date()}, found "
        f"{len(fire_prediction)}."
    )

# ------------------------------------------------------------
# 9. Required fire features
# ------------------------------------------------------------

fire_features = [
    "fire_lag_1d",
    "fire_lag_3d",
    "fire_lag_7d",
    "fire_mean_3d",
    "fire_mean_7d",
]


# ------------------------------------------------------------
# 10. Check whether 7-day live history exists
# ------------------------------------------------------------
missing_fire_features = fire_prediction[
    fire_prediction[fire_features].isna().any(axis=1)
].copy()

if not missing_fire_features.empty:
    print()
    print(
        f"{len(missing_fire_features)} districts have missing "
        "fire-history features."
    )
missing_7day = fire_prediction[
    fire_prediction[
        [
            "fire_lag_7d",
            "fire_mean_7d"
        ]
    ].isna().any(axis=1)
].copy()


# Default prediction mode
prediction_mode = "Live-history"


# ------------------------------------------------------------
# 11. Cold-start fallback
# ------------------------------------------------------------

if len(missing_7day) > 0:

    print()
    print("=" * 70)
    print("7-DAY LIVE HISTORY INCOMPLETE")
    print("=" * 70)

    print(
        f"{len(missing_7day)} districts do not yet have "
        "complete 7-day FIRMS history."
    )

    if not in_training_season:
        print()
        print(
            "Prediction date is outside the historical "
            "October-November training season."
        )

        print(
            "Historical seasonal priors will not be used "
            "for this prediction."
        )

        print(
            "A complete genuine 2026 FIRMS history is required."
        )

        print("=" * 70)

        raise SystemExit(1)

    print()
    print(
        "Using historical district + day-of-season "
        "priors for missing 7-day features."
    )

    # --------------------------------------------------------
    # Load historical engineered dataset
    # --------------------------------------------------------

    historical = pd.read_csv(
        HISTORICAL_FEATURE_FILE
    )

    historical["date"] = pd.to_datetime(
        historical["date"]
    )

    historical["district"] = (
        historical["district"]
        .astype(str)
        .str.strip()
    )

    historical["state"] = (
        historical["state"]
        .astype(str)
        .str.strip()
    )

    # Calculate day of season:
    # Oct 1 = 1
    # Nov 30 = 61
    historical["day_of_season"] = (
    historical["date"].dt.dayofyear
    - pd.to_datetime(
        historical["date"].dt.year.astype(str) + "-10-01"
    ).dt.dayofyear
    + 1
)
    # The prediction date's day of season
    prediction_day_of_season = (
        prediction_date.dayofyear
        - pd.Timestamp(
            year=prediction_date.year,
            month=10,
            day=1
        ).dayofyear
        + 1
    )

    # Keep only historical October-November records
    historical = historical[
        historical["date"].dt.month.isin([10, 11])
    ].copy()

    # --------------------------------------------------------
    # Build historical priors
    # --------------------------------------------------------

    prior_columns = [
        "state",
        "district",
        "day_of_season",
        "fire_lag_7d",
        "fire_mean_7d",
    ]

    historical_priors = (
        historical[
            prior_columns
        ]
        .groupby(
            [
                "state",
                "district",
                "day_of_season"
            ],
            as_index=False
        )[
            [
                "fire_lag_7d",
                "fire_mean_7d"
            ]
        ]
        .mean()
    )

    # --------------------------------------------------------
    # Merge exact district + seasonal position
    # --------------------------------------------------------

    prior_for_prediction = historical_priors[
        historical_priors["day_of_season"]
        == prediction_day_of_season
    ].copy()

    fire_prediction = fire_prediction.merge(
        prior_for_prediction[
            [
                "state",
                "district",
                "fire_lag_7d",
                "fire_mean_7d"
            ]
        ],
        on=[
            "state",
            "district"
        ],
        how="left",
        suffixes=(
            "",
            "_historical"
        )
    )

    # Fill only the missing live values
    fire_prediction["fire_lag_7d"] = (
        fire_prediction["fire_lag_7d"]
        .fillna(
            fire_prediction["fire_lag_7d_historical"]
        )
    )

    fire_prediction["fire_mean_7d"] = (
        fire_prediction["fire_mean_7d"]
        .fillna(
            fire_prediction["fire_mean_7d_historical"]
        )
    )

    # Remove temporary columns
    fire_prediction = fire_prediction.drop(
        columns=[
            "fire_lag_7d_historical",
            "fire_mean_7d_historical"
        ],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Check whether all priors were available
    # --------------------------------------------------------

    still_missing = fire_prediction[
        [
            "fire_lag_7d",
            "fire_mean_7d"
        ]
    ].isna().any(axis=1)

    if still_missing.any():

        print()
        print(
            "Some districts do not have a matching "
            "historical seasonal prior."
        )

        print(
            "Prediction cannot be generated safely "
            "for those districts."
        )

        fire_prediction = fire_prediction[
            ~still_missing
        ].copy()

    prediction_mode = "Warm-start"
remaining_missing_fire = fire_prediction[
    fire_features
].isna().any(axis=1)

if remaining_missing_fire.any():
    print()
    print(
        "Prediction cannot be generated safely because "
        "required fire-history features are still missing."
    )

    print(
        fire_prediction.loc[
            remaining_missing_fire,
            ["state", "district"] + fire_features
        ].to_string(index=False)
    )

    raise SystemExit(1)

# ------------------------------------------------------------
# 12. Merge fire + weather
# ------------------------------------------------------------

prediction = fire_prediction.merge(
    weather_prediction,
    on=[
        "state",
        "district"
    ],
    how="inner",
    suffixes=(
        "_fire",
        "_weather"
    )
)
expected_districts = set(
    zip(
        fire_prediction["state"],
        fire_prediction["district"]
    )
)

actual_districts = set(
    zip(
        prediction["state"],
        prediction["district"]
    )
)

missing_districts = expected_districts - actual_districts

if missing_districts:
    print()
    print(
        f"WARNING: {len(missing_districts)} districts "
        "were lost during fire + weather merge."
    )

    for state, district in sorted(missing_districts):
        print(f"  - {district}, {state}")

    raise SystemExit(1)

# ------------------------------------------------------------
# 13. Check required weather
# ------------------------------------------------------------

weather_features = [
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
]


if prediction.empty:
    print()
    print(
        "No districts available after merging fire "
        "and weather data."
    )
    raise SystemExit(1)


if prediction[weather_features].isna().any().any():

    raise ValueError(
        "Missing weather values found."
    )
numeric_input_features = (
    weather_features +
    fire_features
)

for col in numeric_input_features:
    prediction[col] = pd.to_numeric(
        prediction[col],
        errors="coerce"
    )

if prediction[numeric_input_features].isna().any().any():
    raise ValueError(
        "Non-numeric or missing values found in model input features."
    )

if not np.isfinite(
    prediction[numeric_input_features].to_numpy(dtype=float)
).all():
    raise ValueError(
        "Non-finite values found in model input features."
    )

# ------------------------------------------------------------
# 14. Seasonal features
# ------------------------------------------------------------

prediction["day_of_season"] = (
    prediction["date_fire"].dt.dayofyear
    - pd.to_datetime(
        prediction["date_fire"].dt.year.astype(str) + "-10-01"
    ).dt.dayofyear
    + 1
)

prediction["season_sin"] = np.sin(
    2 * np.pi *
    prediction["day_of_season"] / 61
)

prediction["season_cos"] = np.cos(
    2 * np.pi *
    prediction["day_of_season"] / 61
)


# ------------------------------------------------------------
# 15. Model feature columns
# ------------------------------------------------------------

numeric_features = [
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
    "fire_lag_1d",
    "fire_lag_3d",
    "fire_lag_7d",
    "fire_mean_3d",
    "fire_mean_7d",
    "season_sin",
    "season_cos",
]

categorical_features = [
    "state",
    "district",
]

feature_columns = (
    numeric_features +
    categorical_features
)


# ------------------------------------------------------------
# 16. Generate probabilities
# ------------------------------------------------------------
missing_features = [
    col for col in feature_columns
    if col not in prediction.columns
]

if missing_features:
    raise ValueError(
        f"Missing required model features: {missing_features}"
    )
X = prediction[
    feature_columns
].copy()

probabilities = model.predict_proba(X)[:, 1]

if not np.isfinite(probabilities).all():
    raise ValueError(
        "Model produced non-finite risk probabilities."
    )

if ((probabilities < 0) | (probabilities > 1)).any():
    raise ValueError(
        "Model produced invalid risk probabilities."
    )


# ------------------------------------------------------------
# 17. Apply frozen threshold
# ------------------------------------------------------------

prediction["risk_probability"] = probabilities

prediction["risk_label"] = np.where(
    probabilities >= threshold,
    "Elevated",
    "Normal"
)


# ------------------------------------------------------------
# 18. Prediction metadata
# ------------------------------------------------------------

prediction["prediction_date"] = prediction_date

prediction["prediction_mode"] = prediction_mode

prediction["model_type"] = (
    "Deployment Random Forest"
)

prediction["data_note"] = np.where(
    prediction_mode == "Warm-start",
    "Historical seasonal prior used for missing "
    "7-day fire-history features.",
    "Current 2026 FIRMS history used for fire-history features."
)


# ------------------------------------------------------------
# 19. Output columns
# ------------------------------------------------------------

output = prediction[
    [
        "prediction_date",
        "state",
        "district",
        "risk_probability",
        "risk_label",
        "prediction_mode",
        "fire_lag_1d",
        "fire_lag_3d",
        "fire_lag_7d",
        "fire_mean_3d",
        "fire_mean_7d",
        "T2M",
        "RH2M",
        "WS2M",
        "PRECTOTCORR",
        "data_note",
    ]
].copy()


output = output.sort_values(
    "risk_probability",
    ascending=False
).reset_index(drop=True)


# ------------------------------------------------------------
# 20. Save
# ------------------------------------------------------------
if output.empty:
    print("No predictions generated.")
    raise SystemExit(1)

duplicate_predictions = output.duplicated(
    subset=["prediction_date", "state", "district"],
    keep=False
)

if duplicate_predictions.any():
    raise ValueError(
        "Duplicate district prediction rows detected."
    )

expected_prediction_count = len(fire_prediction)

if len(output) != expected_prediction_count:
    raise ValueError(
        "Prediction count does not match validated fire-feature rows."
    )
# TEMP_OUTPUT_FILE = OUTPUT_FILE.with_suffix(".tmp.csv")

# output.to_csv(
#     TEMP_OUTPUT_FILE,
#     index=False
# )

# TEMP_OUTPUT_FILE.replace(OUTPUT_FILE)
TEMP_OUTPUT_FILE = OUTPUT_FILE.with_suffix(".tmp.csv")
TEMP_HISTORY_FILE = HISTORY_FILE.with_suffix(".tmp.csv")

# ------------------------------------------------------------
# Save latest prediction
# ------------------------------------------------------------

output.to_csv(
    TEMP_OUTPUT_FILE,
    index=False
)

TEMP_OUTPUT_FILE.replace(OUTPUT_FILE)

# ------------------------------------------------------------
# Update prediction history
# ------------------------------------------------------------

if HISTORY_FILE.exists():

    history = pd.read_csv(
        HISTORY_FILE
    )

    history["prediction_date"] = pd.to_datetime(
        history["prediction_date"],
        errors="coerce"
    )

    if history["prediction_date"].isna().any():
        raise ValueError(
            "Prediction history contains invalid dates."
        )

    # Remove only the exact district/date records
    # being regenerated.
    current_keys = output[
        [
            "prediction_date",
            "state",
            "district"
        ]
    ].drop_duplicates()

    history = history.merge(
        current_keys.assign(
            _replace_current=True
        ),
        on=[
            "prediction_date",
            "state",
            "district"
        ],
        how="left"
    )

    history = history[
        history["_replace_current"].isna()
    ].drop(
        columns=["_replace_current"]
    )

else:
    history = pd.DataFrame(
        columns=output.columns
    )

# Add the current prediction
history = pd.concat(
    [history, output],
    ignore_index=True
)

# Validate dates
history["prediction_date"] = pd.to_datetime(
    history["prediction_date"],
    errors="coerce"
)

if history["prediction_date"].isna().any():
    raise ValueError(
        "Prediction history contains invalid dates."
    )

# Validate duplicate district/date records
duplicate_history = history.duplicated(
    subset=[
        "prediction_date",
        "state",
        "district"
    ],
    keep=False
)

if duplicate_history.any():
    raise ValueError(
        "Duplicate district prediction rows "
        "detected in prediction history."
    )

# Sort newest prediction dates first
history = history.sort_values(
    [
        "prediction_date",
        "risk_probability"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)

# Write BOTH temporary files only after validation
output.to_csv(
    TEMP_OUTPUT_FILE,
    index=False
)

history.to_csv(
    TEMP_HISTORY_FILE,
    index=False
)

# Replace final files
TEMP_OUTPUT_FILE.replace(
    OUTPUT_FILE
)

TEMP_HISTORY_FILE.replace(
    HISTORY_FILE
)
# ------------------------------------------------------------
# 21. Summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)

print(
    "Prediction date:",
    prediction_date.date()
)

print(
    "Districts:",
    len(output)
)

print(
    "Prediction mode:",
    prediction_mode
)

print()

print("Risk distribution:")

print(
    output["risk_label"]
    .value_counts()
)

print()

print("Highest-risk districts:")

print(
    output[
        [
            "district",
            "state",
            "risk_probability",
            "risk_label",
            "prediction_mode"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print()

print("Saved to:")
print(OUTPUT_FILE)

print("=" * 70)