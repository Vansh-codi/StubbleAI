# V2.6 Transition-Aware Fire Forecasting

## Status
Research experiment — not an official V2 replacement.

## Research Question

Can explicitly modeling fire-state transitions improve Normal-to-Elevated
(N?E) detection while preserving Elevated-to-Elevated (E?E) continuation
performance and overall multi-horizon forecasting quality?

## Motivation

V2.2 error analysis identified Normal-to-Elevated event onset as a major
weakness of the frozen V2 Random Forest baseline. V2.3 trend features,
V2.3 spatial context, V2.4 spatio-temporal interactions, and V2.5
cost-sensitive learning were subsequently investigated. None produced
sufficient evidence to replace the frozen V2 baseline.

V2.6 therefore changes the modeling formulation rather than introducing
another broad feature-engineering batch.

## Frozen Reference

The official V2 Random Forest remains the control model.

Dataset:
    ml_dataset_v2.csv

Temporal split:
    2023 = training
    2024 = validation / threshold selection
    2025 = untouched final test

Horizons:
    +1, +2, +3, +5, +7 days

Target:
    Elevated if future fire_count > 2
    Normal otherwise

V2 numeric features:
    The exact frozen V2 feature set.

Categorical features:
    state
    district

Random Forest:
    n_estimators = 400
    min_samples_leaf = 2
    random_state = 42
    n_jobs = -1
    class_weight = None

Threshold selection:
    Candidate thresholds = np.linspace(0.01, 0.99, 199)
    Select using 2024 validation Elevated-class F1 only.

The 2025 test period must not be used for model fitting, threshold
selection, feature selection, or hyperparameter tuning.

## Transition Definition

Current Normal:
    fire_count_t <= 2

Current Elevated:
    fire_count_t > 2

Future Elevated:
    fire_count_t+H > 2

Transition classes:

    N?N = current Normal, future Normal
    N?E = current Normal, future Elevated
    E?N = current Elevated, future Normal
    E?E = current Elevated, future Elevated

Primary weakness under investigation:
    N?E

Important continuation regime:
    E?E

## Experiment Arms

### Arm A — Frozen V2 RF

One Random Forest is trained using the original V2 formulation:

    all training rows
        ?
    future Elevated?
        ?
    binary prediction

This arm is the control and must reproduce the frozen V2 results.

### Arm B — Transition-Aware RF

Two Random Forest models are trained independently for each forecast
horizon.

Onset model:
    training rows where current fire_count <= 2
    target = future Elevated

Continuation model:
    training rows where current fire_count > 2
    target = future Elevated

For validation and test prediction:

    if current fire_count <= 2:
        use onset model
    else:
        use continuation model

The resulting probabilities are assembled into one complete validation
and test prediction set.

A single final threshold per horizon is selected from the assembled
2024 validation predictions using the same 199-candidate F1 procedure
as the frozen V2 benchmark.

### Arm C — Transition-Aware Boosting

Where the existing environment supports the required implementation,
evaluate transition-aware gradient boosting models using the same
transition decomposition and temporal protocol.

Candidate model families:
    XGBoost
    LightGBM

These are challengers, not automatic replacements.

## Feature Isolation

The first V2.6 experiment uses the exact frozen V2 feature set.

It does not introduce:

    V2.3 trend features
    V2.3 spatial features
    V2.4 interaction features
    V2.5 class weighting

This isolates the effect of changing the modeling formulation.

## Primary Evaluation Metrics

For every horizon:

    Accuracy
    Precision
    Recall
    F1
    ROC-AUC
    PR-AUC

Transition diagnostics:

    N?E recall
    E?E recall
    N?N accuracy
    E?N behavior

Robustness diagnostics:

    Punjab vs Haryana
    Early vs Middle vs Late season
    Current fire regime
    Confusion matrix

## Acceptance Criteria

A transition-aware model will only be considered a candidate improvement
if it demonstrates a meaningful improvement in N?E detection while
maintaining overall forecasting quality.

Required considerations:

1. N?E recall improves across multiple horizons rather than only one.
2. Overall F1 remains at least competitive with frozen V2.
3. No major degradation in E?E continuation recall.
4. Precision and PR-AUC remain acceptable.
5. Performance does not collapse for either Punjab or Haryana.
6. Performance remains reasonably stable across the early, middle, and
   late portions of the evaluation season.
7. No 2025 tuning is performed.

A model that improves N?E recall only by predicting Elevated too often
will not be considered successful.

## Scientific Interpretation

Possible outcomes:

### Outcome A — Transition model clearly improves
The transition-aware model becomes a candidate forecasting architecture
for further validation.

### Outcome B — Onset improves but continuation degrades
Investigate a hybrid architecture, but do not automatically replace V2.

### Outcome C — Overall performance does not improve
Reject the transition formulation and retain frozen V2.

### Outcome D — No model resolves N?E robustly
Treat the limitation as potentially information/data-related rather than
continuing uncontrolled model complexity. This supports moving toward
richer national data sources such as crop/land context, higher-resolution
spatial information, and additional atmospheric variables.

## Deep Learning Follow-Up

Deep learning is intentionally outside the first V2.6 transition
experiment.

If transition-aware traditional models show useful signal, a controlled
GRU/TCN experiment may subsequently be performed using comparable temporal
information.

The current dataset is relatively small for deep learning, so neural
models will remain challengers rather than automatic replacements.

## Reproducibility Rules

All new V2.6 artifacts must remain under:

    research/v2/v2_6/

Existing V1 and frozen V2 artifacts must not be overwritten.

Do not stage V1 live CSV files.

Do not modify frozen V2 benchmark/calibration checkpoints.

All final test metrics must be generated from the untouched 2025 test
period.

## Decision Principle

The goal is not to maximize a single metric.

The goal is to determine whether explicit transition modeling provides a
robust, reproducible improvement in emerging-fire detection without
sacrificing continuation forecasting or overall model reliability.
