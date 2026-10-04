# StubbleAI V2 - Temporal Probability Calibration

## Status
Experimental calibration study. The original V2 benchmark remains unchanged.

## Protocol
- 2023: model training
- 2024-10-15 to 2024-11-03: calibration fitting
- 2024-11-04 to 2024-11-23: calibration check and threshold selection
- 2025-10-15 to 2025-11-23: untouched final test

## Calibration method
Sigmoid / Platt calibration implemented using a logistic mapping from model log-odds to calibrated probability.

## Models
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## Probability metrics
- Brier score: lower is better
- Log loss: lower is better
- ECE-10: lower is better
- ROC-AUC: discrimination
- PR-AUC: positive-class ranking quality

## 2025 results

| Model | Probability | Brier | Log Loss | ECE | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | raw | 0.2132 | 0.6256 | 0.1972 | 0.8322 | 0.7391 |
| Logistic Regression | sigmoid_calibrated | 0.2221 | 0.6408 | 0.2074 | 0.8322 | 0.7391 |
| Random Forest | raw | 0.1508 | 0.4621 | 0.0349 | 0.8504 | 0.7816 |
| Random Forest | sigmoid_calibrated | 0.1508 | 0.4668 | 0.0361 | 0.8504 | 0.7816 |
| XGBoost | raw | 0.1703 | 0.5575 | 0.1031 | 0.8384 | 0.7589 |
| XGBoost | sigmoid_calibrated | 0.1573 | 0.4785 | 0.0281 | 0.8384 | 0.7589 |
| LightGBM | raw | 0.1749 | 0.5763 | 0.1163 | 0.8363 | 0.7584 |
| LightGBM | sigmoid_calibrated | 0.1589 | 0.4798 | 0.0399 | 0.8363 | 0.7584 |

## Interpretation
Calibration should be considered useful primarily when probability-quality metrics improve on the untouched 2025 test set without materially damaging discrimination.

The 2025 test set was not used to fit the calibration mapping or choose the calibration method.

## Limitations
- Only Punjab and Haryana are currently covered.
- The experiment evaluates sigmoid calibration only.
- ECE depends on the selected binning procedure.
- Calibration quality may vary across districts and horizons.
- Future horizons require independent calibration evaluation.