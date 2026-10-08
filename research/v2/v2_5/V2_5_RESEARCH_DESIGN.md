# V2.5 Cost-Sensitive Emerging-Fire Experiment

## Status
Planned

## Research Question

Can cost-sensitive learning improve emerging-fire detection without
substantially degrading continuation detection or overall forecasting
performance?

## Motivation

V2.2 error analysis showed that the major weakness is detection of
Normal-to-Elevated transitions. V2.3 trend features were rejected, and
V2.4 spatio-temporal interactions were rejected as an official replacement.
Therefore V2.5 isolates the training objective rather than introducing
another feature family.

## Experimental Arms

A. Frozen V2 baseline
- class_weight=None

B. Balanced
- class_weight="balanced"

C. Mild positive-class weighting
- class_weight={0: 1, 1: 1.5}

D. Moderate positive-class weighting
- class_weight={0: 1, 1: 2}

E. Strong positive-class weighting
- class_weight={0: 1, 1: 3}

## Frozen Protocol

Features:
- Frozen V2 feature set

Temporal split:
- 2023 training
- 2024 validation
- 2025 untouched test

Model:
- RandomForestClassifier
- n_estimators=400
- min_samples_leaf=2
- random_state=42
- n_jobs=-1

Target:
- fire_count_t_plus_H > 2

Horizons:
- +1d
- +2d
- +3d
- +5d
- +7d

Threshold selection:
- np.linspace(0.01, 0.99, 199)
- maximize Elevated-class F1 on 2024 validation only

No 2025 tuning.

## Evaluation

Overall:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC

Transition:
- N->E recall
- E->E recall
- N->N recall
- E->N recall

Robustness:
- Punjab vs Haryana
- Early vs Middle vs Late season
- Current-fire regimes

## Acceptance Criteria

A candidate must satisfy all of the following:

1. Average F1 across five horizons is at least the frozen V2 average,
   with +0.005 preferred.
2. Emerging-event recall improves at at least 3 of 5 horizons.
3. No horizon loses more than 0.010 F1.
4. E->E recall does not decline by more than 2 percentage points at
   any horizon.
5. Improvements are not confined to one state, season, or regime.
6. False-positive growth remains operationally reasonable.
7. No test-period threshold, class weight, or model selection decision
   is derived from 2025 results.

## Decision Policy

If no candidate satisfies the acceptance criteria, V2.5 will be rejected
as an official replacement and the frozen V2 model will remain the
official baseline.

If a candidate satisfies the criteria, it remains a research candidate
until reproducibility and robustness review are completed.

## Scientific Constraint

V2.5 changes only the training class weighting. Feature engineering,
temporal splitting, target definition, model family, random seed,
tree count, leaf size, and threshold-selection protocol remain fixed.

## Reproducibility

All scripts, configurations, predictions, metrics, and diagnostics will
be stored under research/v2/v2_5/.
