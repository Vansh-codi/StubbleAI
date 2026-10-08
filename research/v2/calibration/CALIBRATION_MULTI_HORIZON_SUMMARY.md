# StubbleAI V2 - Multi-Horizon Probability Calibration Study

## Status

Completed multi-horizon temporal probability calibration experiment for StubbleAI V2.

Horizons evaluated:

- +1 day
- +2 days
- +3 days
- +5 days
- +7 days

Models evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Calibration method:

- Sigmoid / Platt scaling

Primary purpose:

Evaluate whether post-hoc probability calibration improves probability reliability for multi-horizon fire-occurrence forecasting without contaminating the final temporal test period.

---

## 1. Temporal Protocol

The same temporal protocol was used for every horizon.

2023-10-15 to 2023-11-23
Model training

2024-10-15 to 2024-11-03
Calibration fitting

2024-11-04 to 2024-11-23
Calibration check and threshold selection

2025-10-15 to 2025-11-23
Untouched final test

Each split contains 1,800 district-day observations.

The 2025 period was never used for calibration fitting or calibration-method selection.

---

## 2. Evaluation Metrics

Probability quality:

- Brier Score - lower is better
- Log Loss - lower is better
- ECE-10 - lower is better

Discrimination:

- ROC-AUC
- PR-AUC

Decision performance:

- Accuracy
- Precision
- Recall
- F1

Sigmoid calibration is monotonic, so ROC-AUC and PR-AUC are expected to remain effectively unchanged.

---

## 3. Cross-Horizon Probability Quality

Average values across the five evaluated horizons:

| Model | Brier Raw | Brier Calibrated | LogLoss Raw | LogLoss Calibrated | ECE Raw | ECE Calibrated |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.1834 | 0.2052 | 0.5429 | 0.5967 | 0.1356 | 0.1867 |
| Random Forest | 0.1412 | 0.1420 | 0.4335 | 0.4378 | 0.0275 | 0.0371 |
| XGBoost | 0.1648 | 0.1533 | 0.5514 | 0.4683 | 0.1035 | 0.0423 |
| LightGBM | 0.1702 | 0.1541 | 0.5916 | 0.4690 | 0.1176 | 0.0440 |

---

## 4. Random Forest

Random Forest produced the strongest probability reliability among the evaluated models in raw form.

Raw ECE across horizons:

| Horizon | Raw ECE |
|---|---:|
| +1d | 0.0320 |
| +2d | 0.0187 |
| +3d | 0.0243 |
| +5d | 0.0278 |
| +7d | 0.0349 |

Sigmoid calibration did not improve RF probability quality.

Average:

- Brier: 0.1412 -> 0.1420
- LogLoss: 0.4335 -> 0.4378
- ECE: 0.0275 -> 0.0371

Therefore, the tested sigmoid calibration should not be applied to the primary Random Forest probabilities based on this experiment.

---

## 5. XGBoost

XGBoost showed systematic probability miscalibration in raw form.

Raw to calibrated ECE:

| Horizon | Raw | Calibrated |
|---|---:|---:|
| +1d | 0.0933 | 0.0620 |
| +2d | 0.1041 | 0.0568 |
| +3d | 0.1139 | 0.0375 |
| +5d | 0.1033 | 0.0272 |
| +7d | 0.1031 | 0.0281 |

Average:

- Brier: 0.1648 -> 0.1533
- LogLoss: 0.5514 -> 0.4683
- ECE: 0.1035 -> 0.0423

The results support sigmoid calibration for XGBoost when probability reliability is required.

---

## 6. LightGBM

LightGBM also showed systematic raw probability miscalibration.

Raw to calibrated ECE:

| Horizon | Raw | Calibrated |
|---|---:|---:|
| +1d | 0.1005 | 0.0656 |
| +2d | 0.1241 | 0.0423 |
| +3d | 0.1279 | 0.0362 |
| +5d | 0.1194 | 0.0361 |
| +7d | 0.1163 | 0.0399 |

Average:

- Brier: 0.1702 -> 0.1541
- LogLoss: 0.5916 -> 0.4690
- ECE: 0.1176 -> 0.0440

The results support sigmoid calibration for LightGBM when probability reliability is required.

---

## 7. Logistic Regression

Logistic Regression did not show a consistent benefit from the tested sigmoid calibration.

Average:

- Brier: 0.1834 -> 0.2052
- LogLoss: 0.5429 -> 0.5967
- ECE: 0.1356 -> 0.1867

All three probability-quality metrics deteriorated on average.

Therefore, there is no evidence from this experiment to justify adding the tested sigmoid calibration to Logistic Regression.

---

## 8. Discrimination

Calibration did not materially change ROC-AUC or PR-AUC.

This is expected because sigmoid calibration is monotonic and therefore preserves the ordering of model scores.

The calibration experiment should therefore be interpreted primarily as a probability-quality experiment rather than a model-discrimination experiment.

---

## 9. Decision Metrics

Calibration did not consistently improve F1.

This is expected.

Calibration changes the mapping from model score to probability; it does not directly optimize classification F1.

Threshold selection therefore remains a separate decision layer.

The frozen V2 benchmark thresholds must not be overwritten by these calibration experiments.

---

## 10. Research Decision

### Primary forecasting model

Random Forest remains the preferred primary V2 model.

Reasons:

1. Strong predictive performance in the frozen V2 benchmark.
2. Strong ROC-AUC and PR-AUC.
3. Raw probabilities are already comparatively well calibrated.
4. Additional sigmoid calibration consistently fails to improve its probability metrics.

### Secondary calibrated models

XGBoost and LightGBM remain useful alternative models.

When probability reliability is required:

- XGBoost + sigmoid calibration is supported by the experiment.
- LightGBM + sigmoid calibration is supported by the experiment.

### Logistic Regression

Useful as a baseline model, but the tested sigmoid calibration does not provide an out-of-time probability-quality benefit.

---

## 11. Important Interpretation

Calibration results do not establish causal relationships between fire activity and air pollution.

FIRMS active-fire detections are satellite observations and should not automatically be interpreted as confirmed stubble-burning events.

Forecast probabilities should be communicated as model estimates rather than certainties.

District-level aggregation can hide sub-district variation.

The system should not be used as the sole basis for enforcement or penalties.

---

## 12. Data Leakage Protection

The final 2025 test period remained untouched during:

- model training
- calibration fitting
- calibration method selection
- calibration threshold selection

Calibration fitting was performed only on the designated 2024-A period.

Calibration checking and threshold selection used the designated 2024-B period.

This maintains temporal separation between model development and final evaluation.

---

## 13. Reproducibility

Random state:

42

Feature encoding:

- categorical encoding fit using training data only
- state and district encoded from the 2023 training period

Calibration:

sigmoid / Platt scaling

The calibration experiments are stored separately from the frozen V2 benchmark checkpoints.

---

## 14. Experiment Directories

research/v2/calibration/

- calibration_results.csv
- calibration_config.json
- CALIBRATION_RESULTS.md
- plus1d/
- plus2d/
- plus3d/
- plus5d/

The original +7d calibration experiment is stored directly in the calibration root directory rather than a plus7d subdirectory. It must remain there unchanged.

The individual horizon experiments must remain reproducible and should not be overwritten by later experiments.

---

## 15. Final Research Conclusion

Across +1, +2, +3, +5 and +7 day forecasting horizons, Random Forest produced comparatively reliable raw probabilities and did not benefit from the tested sigmoid calibration.

XGBoost and LightGBM showed substantially better probability reliability after sigmoid calibration, with consistent reductions in Brier Score, Log Loss and ECE across the evaluated horizons.

Logistic Regression did not benefit from the tested calibration procedure.

Therefore, StubbleAI V2 should retain Random Forest as the primary forecasting model while preserving calibrated XGBoost and LightGBM as experimentally supported alternatives for probability-sensitive comparisons.

This checkpoint does not modify the original V2 predictive benchmark results.
