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
- All existing 1, 2, 3, 5, 7, and 14 day fire/FRP/grid lag features.
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


## Amendment
- Added the existing 7-day lag features to the predictor specification before model training.
- The original protocol file is preserved.

## Pre-training amendment: zero-detection FRP/brightness statistics

- Audit finding: all 2,198 eligible rows with missing values in
  `frp_mean`, `frp_max`, `bright_ti4_mean`, `bright_ti4_max`, and
  `bright_ti5_mean` have `fire_count == 0` in both weather variants.
- These statistics are undefined when no fire detections are recorded.
  Preserve their missing values; do not replace them with zero and do
  not drop these rows.
- The registered eligibility rule remains: target known and all 36
  required lag features available.
- A synthetic test confirmed native NaN support for
  `RandomForestClassifier` under scikit-learn 1.6.1. This verifies
  estimator compatibility, not general data quality or predictive validity.
- Before training, verify that missing values in these five columns occur
  only on zero-fire rows, and that all other predictors are complete.
- Apply identical data handling to both variants. Record this amendment,
  input hashes, script hash, software versions, and missingness audit
  in the run manifest.
- This amendment was recorded before model training and before computing
  2025 test metrics. No test results informed this decision.


## Six-State Scope, Sparsity, and Mandatory Reporting Amendment

### 1. Confirmed temporal scope

The current six-state district-day dataset covers **October and November only**, from 2023-10-01 through 2025-11-30. This is a fixed seasonal study, not a year-round fire-forecasting evaluation. Results must be interpreted within this temporal scope.

The experiment must preserve the chronological split:
- **2023:** training.
- **2024:** validation and threshold selection.
- **2025:** held-out testing after thresholds are locked.

Do not describe this experiment as year-round validation or as proof of generalization to unseen states. A later seasonal-scope experiment must evaluate broader calendar coverage or explicitly defined state-specific fire seasons before claims of India-wide, year-round readiness.

### 2. Descriptive state-level fire-rate audit

The existing six-state dataset contains 45,384 district-days. The following rates are descriptive audits of the current October-November dataset, not model-performance metrics.

| State | District-days | Current-day elevated days (`fire_count > 2`) | Current-day elevated rate | Next-day target rate |
|---|---:|---:|---:|---:|
| Assam | 6,405 | 344 | 5.37% | 5.46% |
| Bihar | 6,954 | 337 | 4.85% | 4.93% |
| Haryana | 4,026 | 1,231 | 30.58% | 30.88% |
| Madhya Pradesh | 10,065 | 3,241 | 32.20% | 32.73% |
| Punjab | 4,209 | 2,131 | 50.63% | 51.18% |
| Uttar Pradesh | 13,725 | 4,004 | 29.17% | 29.64% |

The current-day elevated rate and next-day target rate are distinct quantities and must not be treated as interchangeable. The target for forecasting remains `target_elevated_t_plus_1d`, defined using the next day's `fire_count > 2` rule.

Assam and Bihar have markedly lower positive rates than the other four states. This creates a substantial class-prevalence imbalance across the pooled dataset. It does **not**, by itself, establish why a model may perform poorly in either state.

### 3. Pre-registered research questions and interpretation

The six-state experiment must distinguish three questions:

1. **Pooled six-state forecasting:** How well does a model trained on the pooled six-state dataset forecast next-day elevated fire within the same six-state population and the specified October-November temporal window?
2. **Transfer to unseen states:** Does performance generalize to states excluded from training? The pooled six-state temporal experiment does not answer this question because all six states contribute training data. A separate state-held-out design is required to make a transfer-learning claim.
3. **Bihar/Assam underperformance diagnosis:** If performance is weak in Bihar or Assam, is this consistent with a genuine predictive/model limitation, or is it associated with the small number of positive training examples available for those states in the current seasonal window?

The third question is a diagnostic distinction, not a pre-established conclusion. Report each state's training and validation sample counts, positive counts, and positive prevalence alongside its test metrics. Do not attribute weak performance solely to model failure or sparsity without supporting evidence.

Because a pooled model optimizes across all included rows, differences in state sample sizes and positive prevalence may affect what it learns. State-level results must therefore be examined alongside pooled results. Do not silently rebalance classes or alter model settings for one state in the primary comparison. Any reweighting, resampling, or state-specific model should be a separately specified experiment.

### 4. Comparison validity and locked experiment design

The two weather variants must be compared on identical eligible state/district/date keys, identical target labels, and identical non-weather predictors. The weather source for each variant must be explicitly documented for every included state. Do not label NASA POWER data as V1 weather unless its provenance establishes that it is the actual V1 weather source.

Before training, verify and report:
- State, district, and date coverage for each variant.
- Duplicate keys, missing weather values, and documented source fill values.
- Identical eligible rows and target labels across variants.
- Training, validation, and test counts and positive prevalence, both overall and by state.
- The exact predictor list and exclusion of future-target columns and leakage-prone fields.
- Model parameters, threshold-selection rule, software versions, protocol hash, and script hash.

Preserve missing values where the protocol specifies that they remain missing; do not silently fill zero-fire-day feature NaNs with zero or drop eligible rows to make the comparison easier.

Use the protocol's specified model and threshold-selection procedure. Lock each variant's threshold using 2024 validation data before evaluating 2025. Do not tune thresholds on 2025 test results.

### 5. Mandatory row-level prediction export

Save a row-level prediction file for **each weather variant** for the held-out 2025 test set. Each row must identify the exact evaluated observation and include, at minimum:

- `state`
- `district`
- `date`
- True next-day target label (`target_elevated_t_plus_1d`)
- Predicted probability for the positive class
- The variant's locked threshold
- Predicted binary class using that locked threshold
- Weather variant identifier
- `assignment_method`, when available from the grid-to-district mapping

Retain enough metadata to associate each prediction file with its run manifest, protocol version, model configuration, and threshold-lock record. Confirm that both variants contain the same evaluated keys and true labels before comparing results. Do not overwrite or alter the completed Punjab-Haryana pilot outputs.

### 6. Mandatory evaluation outputs

The experiment report must include both pooled and per-state evaluation on the **2025 held-out test set**.

**Required pooled metrics for each weather variant:**
- F1, precision, recall, PR-AUC, and ROC-AUC.
- Confusion matrix (TN, FP, FN, TP).
- Test row count, positive count, and positive prevalence.
- Locked threshold and the validation metric used to select it.

**Required per-state metrics for each weather variant:**
- F1, precision, recall, PR-AUC, and ROC-AUC for each of Assam, Bihar, Haryana, Madhya Pradesh, Punjab, and Uttar Pradesh.
- Confusion matrix (TN, FP, FN, TP) for each state.
- Test row count, positive count, and positive prevalence for each state.

The per-state breakdown is a mandatory report table, not an optional follow-up analysis. If a metric is undefined for a subset, report it as undefined with the reason; do not substitute zero or omit the state silently.

Include a comparison table showing the difference between weather variants for pooled and per-state metrics. Interpret the results in light of each state's prevalence and available training positives. Aggregate scores must not replace state-level reporting.

### 7. Interpretation limits and next milestone

The current experiment evaluates weather substitution within an October-November, six-state temporal split. It does not establish year-round performance, unseen-state transfer, or readiness for India-wide deployment.

After this fixed-scope experiment is complete, the next research milestone is a separately documented seasonal-scope study using broader calendar coverage or justified state-specific fire seasons. That study should test whether additional seasonal observations provide more positive training examples for low-prevalence states, particularly Assam and Bihar, before drawing conclusions about their long-term forecastability.
