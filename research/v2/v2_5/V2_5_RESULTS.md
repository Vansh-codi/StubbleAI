# V2.5 Cost-Sensitive Emerging-Fire Experiment

## Status

REJECTED as an official V2 replacement.

## Official Baseline

Frozen V2 multi-horizon Random Forest.

## Research Question

Can cost-sensitive learning improve emerging-fire detection without
substantially degrading continuation detection or overall forecasting
performance?

## Protocol

The experiment changed only Random Forest class weighting.

- Training: 2023
- Validation: 2024
- Frozen test: 2025
- Horizons: +1d, +2d, +3d, +5d, +7d
- Random Forest: 400 trees
- Minimum leaf size: 2
- Random seed: 42
- Threshold candidates: 199 values from 0.01 to 0.99
- Threshold selected using 2024 Elevated-class F1
- No 2025 tuning
- Frozen V2 features were used unchanged

## Experimental Arms

1. Frozen V2 baseline: class_weight=None
2. Balanced: class_weight="balanced"
3. Positive 1.5x: {0:1, 1:1.5}
4. Positive 2x: {0:1, 1:2}
5. Positive 3x: {0:1, 1:3}

## Frozen V2 Reproduction

The baseline arm reproduced the official V2 thresholds and test F1
values for all five horizons.

Therefore the V2.5 comparison is anchored to the frozen V2 protocol.

## Main Results

Frozen V2 average F1 across horizons: approximately 0.7392.

Balanced average F1: approximately 0.7396.
Positive 1.5x average F1: approximately 0.7357.
Positive 2x average F1: approximately 0.7337.
Positive 3x average F1: approximately 0.7357.

No weighted arm provided a meaningful robust improvement in average F1.

## Emerging-Fire Detection

Cost-sensitive learning produced strong horizon-dependent recall
changes.

Balanced weighting improved emerging recall at +5d and +7d but reduced
it at +1d and +3d.

Positive 1.5x improved emerging recall at +2d, +5d and +7d but reduced
it at +1d and +3d.

Positive 2x substantially improved +7d emerging recall but reduced
performance at shorter horizons.

Positive 3x produced very large emerging-recall gains at +5d and +7d,
but these gains were accompanied by substantial continuation and F1
degradation.

## Continuation Detection

Continuation detection is a critical constraint because the V2 model
already performs strongly when elevated fire activity persists.

Increasing positive-class weight caused unacceptable continuation
degradation at some horizons.

Examples:

- Balanced weighting reduced +3d E->E recall by approximately 3.0
  percentage points.
- Positive 2x reduced +3d E->E recall by approximately 5.6 percentage
  points.
- Positive 3x reduced +3d E->E recall by approximately 8.2 percentage
  points.

Therefore improved emerging recall cannot be accepted if it is achieved
by substantially weakening continuation detection.

## Scientific Interpretation

The experiment indicates that emerging-fire detection is not simply a
class-imbalance problem.

Increasing the positive-class training weight changes the
precision-recall tradeoff and can increase detection of future elevated
events at longer horizons. However, these gains are inconsistent across
horizons and can substantially damage continuation performance.

This suggests that the remaining emerging-event weakness is more likely
related to temporal transition structure and event onset dynamics than
to simple class weighting.

## Decision

V2.5 is REJECTED as an official V2 replacement.

The frozen V2 model remains the official research baseline.

No class-weighted arm satisfied the complete acceptance criteria.

## Acceptance Criteria Review

The experiment required:

1. Average F1 at least equal to frozen V2.
2. Emerging recall improvement at at least 3 of 5 horizons.
3. No horizon F1 decrease greater than 0.010.
4. No E->E recall decrease greater than 2 percentage points.
5. Robustness across states, seasons and regimes.
6. Reasonable false-positive behavior.
7. No 2025-derived tuning.

No weighted arm satisfied all requirements.

## What V2.5 Taught Us

The main lesson is that simply assigning greater training cost to the
Elevated class does not robustly solve Normal-to-Elevated transition
detection.

The next research direction should therefore investigate temporal
transition or event-onset modeling rather than continuing to increase
class weights.

## Reproducibility

Artifacts are stored under:

research/v2/v2_5/

Including:

- V2_5_RESEARCH_DESIGN.md
- v2_5_class_weight_config.json
- v2_5_class_weight_results.csv
- 25 prediction files
- transition diagnostics
- emerging-event diagnostics
- continuation diagnostics
- state diagnostics
- season diagnostics
- fire-regime diagnostics
- acceptance summary

The official V2 model and frozen benchmark artifacts were not modified.
