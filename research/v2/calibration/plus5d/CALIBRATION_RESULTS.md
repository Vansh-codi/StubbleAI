# StubbleAI V2 - +5d Temporal Probability Calibration

## Protocol
- 2023: model training
- 2024-10-15 to 2024-11-03: calibration fitting
- 2024-11-04 to 2024-11-23: calibration check
- 2025-10-15 to 2025-11-23: untouched final test

## Calibration method
Sigmoid / Platt calibration using a logistic mapping from model log-odds to calibrated probability.

## 2025 Test Results

| Model | Probability | Brier | Log Loss | ECE | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | raw | 0.1741 | 0.5194 | 0.1144 | 0.8415 | 0.7658 |
| Logistic Regression | sigmoid_calibrated | 0.1808 | 0.5338 | 0.1329 | 0.8415 | 0.7658 |
| Random Forest | raw | 0.1484 | 0.4537 | 0.0278 | 0.8581 | 0.8022 |
| Random Forest | sigmoid_calibrated | 0.1498 | 0.4623 | 0.0432 | 0.8581 | 0.8022 |
| XGBoost | raw | 0.1670 | 0.5610 | 0.1033 | 0.8427 | 0.7848 |
| XGBoost | sigmoid_calibrated | 0.1554 | 0.4758 | 0.0272 | 0.8427 | 0.7848 |
| LightGBM | raw | 0.1724 | 0.6080 | 0.1194 | 0.8434 | 0.7894 |
| LightGBM | sigmoid_calibrated | 0.1551 | 0.4730 | 0.0361 | 0.8434 | 0.7894 |

## Interpretation
Lower Brier, Log Loss, and ECE indicate improved probability reliability. ROC-AUC and PR-AUC assess discrimination and ranking quality.

2025 was not used for calibration fitting.
The original V2 benchmark files were not modified.