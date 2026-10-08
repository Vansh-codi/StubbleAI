# V2.6 Transition-Aware Forecasting — Diagnostic Results

## Research Question

Can explicitly modeling fire-state transitions improve Normal?Elevated (N?E) detection while preserving continuation detection and overall forecasting performance?

## Experimental Setup

V2.6 compared the frozen V2 Random Forest against a transition-aware Random Forest.

The transition-aware model used separate models for:

- Normal ? Elevated onset
- Elevated ? Elevated continuation

The final prediction was assembled from the appropriate transition model.

The temporal protocol remained unchanged:

- 2023: training
- 2024: validation and threshold selection
- 2025: untouched final test

No 2025 threshold tuning was performed.

## Overall Results

| Horizon | V2 F1 | Transition F1 | Delta |
|---|---:|---:|---:|
| +1d | 0.7754 | 0.7737 | -0.0016 |
| +2d | 0.7513 | 0.7446 | -0.0067 |
| +3d | 0.7305 | 0.7363 | +0.0058 |
| +5d | 0.7303 | 0.7197 | -0.0107 |
| +7d | 0.7085 | 0.6833 | -0.0252 |

The transition-aware model therefore did not provide a consistent overall F1 improvement.

## Emerging-Fire Detection

Normal?Elevated recall:

| Horizon | V2 | Transition RF | Delta |
|---|---:|---:|---:|
| +1d | 0.4037 | 0.4037 | 0.0000 |
| +2d | 0.4817 | 0.4921 | +0.0105 |
| +3d | 0.6202 | 0.4327 | -0.1875 |
| +5d | 0.5635 | 0.6193 | +0.0558 |
| +7d | 0.6267 | 0.7281 | +0.1014 |

The transition formulation therefore produced useful onset-recall gains at +5d and +7d, but these gains were not consistent across horizons.

## Continuation Detection

Elevated?Elevated recall remained strong for most horizons.

The transition model improved continuation recall at +1d, +2d, +5d and +7d, but degraded it substantially at +3d.

This indicates that the transition formulation did not provide a uniformly superior continuation model.

## False Positives

The major limitation was increased false-positive activity, especially at longer horizons.

Normal?Normal false-positive rates:

| Horizon | V2 | Transition RF |
|---|---:|---:|
| +1d | 0.1051 | 0.1127 |
| +2d | 0.1232 | 0.1411 |
| +3d | 0.2043 | 0.0902 |
| +5d | 0.1815 | 0.2187 |
| +7d | 0.2168 | 0.3183 |

The +7d transition model increased the N?N false-positive rate by approximately 10.15 percentage points.

This explains why its higher emerging-fire recall did not translate into better overall F1.

## Probability Quality

The transition model did not consistently improve probability quality.

Brier scores:

| Horizon | V2 | Transition RF |
|---|---:|---:|
| +1d | 0.12725 | 0.12643 |
| +2d | 0.13650 | 0.13817 |
| +3d | 0.14294 | 0.14437 |
| +5d | 0.14842 | 0.14865 |
| +7d | 0.15075 | 0.15303 |

Only +1d showed a small Brier improvement.

Therefore, the transition model should not be described as a general probability-calibration improvement.

## Interpretation

The experiment provides evidence that explicitly modeling fire-state transitions can increase sensitivity to emerging events at longer forecast horizons.

However, the additional sensitivity is accompanied by increased false alarms and lower precision.

The results therefore suggest that the emerging-fire limitation is not solved simply by restructuring the classification task into onset and continuation models.

Together with the V2.5 cost-sensitive experiment, the evidence suggests that the limitation is more likely related to the temporal/event-onset information available in the current dataset than to simple class imbalance.

## Decision

**V2.6 — CONDITIONALLY INFORMATIVE, NOT ACCEPTED AS THE OFFICIAL V2 MODEL.**

The frozen V2 Random Forest remains the official benchmark.

The transition-aware model is retained as a research result and challenger architecture.

It must not replace the official V2 model based on the current evidence.

## Next Experiment

The next experiment should test whether the remaining limitation is primarily caused by the Random Forest learner rather than the available feature information.

V2.7 will therefore benchmark stronger tree-based learners using the exact frozen V2 feature set and identical temporal evaluation protocol.

Candidate learners:

- Random Forest — frozen control
- XGBoost
- LightGBM
- HistGradientBoosting, if justified by implementation availability

The experiment will use:

- 2023 training
- 2024 validation
- 2025 untouched test
- identical five forecast horizons
- threshold selection on 2024 only
- persistence baseline
- overall classification metrics
- ROC-AUC and PR-AUC
- N?E recall
- E?E recall
- state-level robustness
- seasonal robustness

No 2025 tuning will be permitted.

## Reproducibility

V2.6 prediction files, diagnostics, configuration, and scripts are retained under:

`research/v2/v2_6/`

The official V2 model and frozen benchmark artifacts remain unchanged.
