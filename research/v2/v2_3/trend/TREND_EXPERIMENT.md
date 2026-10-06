# V2.3 Trend Feature Experiment

## 1. Purpose

This experiment evaluates whether short-term temporal trend and acceleration features improve the frozen V2 Random Forest forecasting baseline for multi-horizon fire-occurrence prediction.

The experiment was designed as a controlled feature-family study. The frozen V2 dataset, temporal split, target definition, categorical encoding, Random Forest configuration, threshold-selection procedure, and 2025 test protocol were retained.

## 2. Research Status

**Decision: REJECTED AS A FEATURE FAMILY FOR THE CURRENT V2 MODEL**

The added trend features did not improve the frozen V2 baseline consistently across the five forecast horizons. F1, ROC-AUC, and PR-AUC were generally lower, with the largest degradation at +7 days.

The experiment is retained as a negative research result and is not promoted into the official V2 model.

## 3. Experimental Protocol

Input:

`ml_dataset_v2.csv`

Rows:

5,400

Original V2 features:

51 columns

Added trend features:

16

Resulting experimental dataset:

67 columns

Temporal split:

- 2023: training
- 2024: validation and threshold selection
- 2025: untouched final test

Target:

`fire_count_t_plus_H > 2`

Random Forest:

- 400 trees
- `min_samples_leaf=2`
- `random_state=42`
- no class weighting

Validation threshold:

Candidate thresholds from 0.01 to 0.99 in increments of approximately 0.00495, selecting the threshold that maximized validation F1.

No 2025 test tuning or retraining was performed.

## 4. Added Features

The experiment added:

- `fire_change_1d`
- `fire_change_3d`
- `fire_change_7d`
- `frp_change_1d`
- `frp_change_3d`
- `frp_change_7d`
- `fire_ratio_recent_vs_3d`
- `fire_ratio_recent_vs_7d`
- `frp_ratio_recent_vs_3d`
- `frp_ratio_recent_vs_7d`
- `fire_trend_3d`
- `fire_trend_7d`
- `frp_trend_3d`
- `frp_trend_7d`
- `fire_acceleration`
- `frp_acceleration`

The original V2 columns were preserved unchanged.

## 5. Results

| Horizon | Frozen V2 F1 | Trend F1 | Delta |
|---|---:|---:|---:|
| +1d | 0.7754 | 0.7663 | -0.0091 |
| +2d | 0.7513 | 0.7498 | -0.0015 |
| +3d | 0.7305 | 0.7228 | -0.0077 |
| +5d | 0.7303 | 0.7287 | -0.0016 |
| +7d | 0.7085 | 0.6933 | -0.0152 |

ROC-AUC and PR-AUC also declined across the horizons relative to the frozen V2 baseline.

Trend experiment 2025 ROC-AUC:

| Horizon | ROC-AUC |
|---|---:|
| +1d | 0.8855 |
| +2d | 0.8735 |
| +3d | 0.8618 |
| +5d | 0.8559 |
| +7d | 0.8448 |

Trend experiment 2025 PR-AUC:

| Horizon | PR-AUC |
|---|---:|
| +1d | 0.8454 |
| +2d | 0.8282 |
| +3d | 0.8110 |
| +5d | 0.8032 |
| +7d | 0.7705 |

## 6. Diagnostic Findings

The additional trend features did not provide a consistent improvement in the difficult forecasting regimes.

Important diagnostic changes included:

- E?E continuation recall decreased from approximately 0.9526 to 0.9265.
- N?E emerging-event recall decreased from approximately 0.5472 to 0.4959.
- Haryana recall decreased from approximately 0.6472 to 0.5958.
- Late-season recall decreased from approximately 0.5967 to 0.5279.

These results suggest that the trend variables did not provide sufficiently independent predictive information for the current Random Forest configuration.

## 7. Interpretation

The negative result does not establish that temporal trends are useless for fire forecasting.

It establishes only that this particular family of engineered trend, ratio, slope, and acceleration features did not improve the current V2 Random Forest under the frozen evaluation protocol.

Possible explanations include:

1. Existing fire lags and rolling features already capture much of the available short-term temporal information.
2. Trend variables may be highly correlated with existing fire-history features.
3. Simple handcrafted trends may not adequately represent nonlinear fire dynamics.
4. The effect of temporal trends may depend on spatial context or fire regime.
5. The current feature formulation may not be appropriate for longer forecast horizons.

## 8. Research Integrity

The frozen V2 dataset was not modified.

The official V2 benchmark files were not overwritten.

The 2025 test period remained untouched for model selection and tuning.

The experiment therefore remains a valid negative feature-ablation result.

## 9. Future Research

Temporal dynamics should not be permanently discarded.

Future experiments may investigate:

- spatially conditioned temporal trends
- horizon-specific temporal features
- interaction between trend and current fire regime
- nonlinear sequence models
- carefully controlled temporal feature selection
- interactions between neighboring fire activity and local temporal change

Any future experiment must use a new research checkpoint and preserve the frozen V2 benchmark.

## 10. Reproducibility

Primary scripts:

- `build_v2_3_trend_features.py`
- `train_v2_3_trend_experiment.py`
- `analyze_v2_3_trend_diagnostics.py`

Experimental artifacts are stored under:

`research/v2/v2_3/`

## 11. Final Decision

**REJECTED for promotion into the official V2 model.**

The experiment is retained because negative results are part of the research record and prevent repeated testing of the same ineffective feature family.
