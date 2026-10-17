# V3 Weather-Source Substitution Experiment Protocol

Status: PRE-REGISTERED BEFORE MODEL TRAINING
Purpose: Measure sensitivity of the V3 Punjab-Haryana district-level
next-day elevated-fire classifier to weather-source substitution.

## Research question
With rows, labels, fire features, grid-derived features, temporal split,
and model configuration held fixed, does replacing V3 weather with V1
weather materially change predictive performance?

## Population
- Source population: V3 seasonal district ML dataset.
- States: Punjab and Haryana.
- Eligible rows: target known and all required lag features available.
- Expected eligible rows: 6,210; 2,070 per year for 2023, 2024, and 2025.
- Both variants must use exactly the same state-district-date keys.

## Variants
- A: original V3 weather.
- B: V1 weather substituted into the same V3 rows.
- Weather variables: T2M, RH2M, WS2M, PRECTOTCORR.
- All non-weather predictors and target labels must match exactly.

## Target
target_elevated_t_plus_1d, defined by next-day fire_count > 2.
Exclude fire_count_t_plus_1d, frp_sum_t_plus_1d, and all target-derived
columns from the predictor matrix.

## Predictors
Use the same feature set in both variants:
- Calendar features: year, month, day, day_of_season, day_of_year,
  sin_year, cos_year.
- Current-day district fire/FRP features.
- Current-day grid-derived features.
- Four weather variables.
- All existing 1, 2, 3, 5, and 14 day fire/FRP/grid lag features.
- Existing historical summary features.
- State and district as categorical predictors, one-hot encoded with
  unknown categories ignored.

## Model
RandomForestClassifier:
- n_estimators=400
- min_samples_leaf=2
- random_state=42
- class_weight=None
- other parameters left at documented scikit-learn defaults.

Fit preprocessing and model on 2023 only. Do not fit preprocessing on
validation or test data.

## Temporal protocol
- Train: 2023.
- Validation: 2024; choose a threshold maximizing F1 for each variant.
- Test: 2025; evaluate only after code, features, metrics, and thresholds
  selection procedure are fixed.
- Do not use 2025 results to revise features, thresholds, or model settings.
- Any changes motivated by inspected test results are exploratory and must
  be labeled non-confirmatory.

## Metrics
Primary: 2025 F1 at the threshold selected on 2024.
Secondary: precision, recall, PR-AUC, ROC-AUC, confusion matrix, selected
threshold, and positive prevalence.
Report results for both variants and their paired differences.
Record the exact row counts and class counts for every split.

## Interpretation
This experiment isolates weather-source sensitivity within this V3
feature/model pipeline. It is not, by itself, proof of V3 superiority
over frozen V2 or proof of improved forecasting generally.
Weather values are district representative-point estimates, copied across
cells mapped to each district; no within-district weather variation is
represented.

## Artifact discipline
- Preserve original V3 and frozen V2 data/scripts.
- Read verified V1-weather experiment input from:
  data/processed/v3/experiments/weather_substitution/
  v3_district_ml_dataset_1d_v1_weather_verified.csv
- Do not overwrite existing artifacts.
- Save model outputs and audit files only in a new experiment-specific path.
