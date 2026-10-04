# StubbleAI V2 - +2d Temporal Probability Calibration

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
| Logistic Regression | raw | 0.1830 | 0.5410 | 0.1314 | 0.8414 | 0.7691 |
| Logistic Regression | sigmoid_calibrated | 0.2306 | 0.6715 | 0.2396 | 0.8414 | 0.7691 |
| Random Forest | raw | 0.1365 | 0.4198 | 0.0187 | 0.8795 | 0.8369 |
| Random Forest | sigmoid_calibrated | 0.1381 | 0.4263 | 0.0362 | 0.8795 | 0.8369 |
| XGBoost | raw | 0.1639 | 0.5472 | 0.1041 | 0.8507 | 0.8089 |
| XGBoost | sigmoid_calibrated | 0.1553 | 0.4722 | 0.0568 | 0.8507 | 0.8089 |
| LightGBM | raw | 0.1739 | 0.6009 | 0.1241 | 0.8405 | 0.7970 |
| LightGBM | sigmoid_calibrated | 0.1577 | 0.4769 | 0.0423 | 0.8405 | 0.7970 |

## Interpretation
Lower Brier, Log Loss, and ECE indicate improved probability reliability. ROC-AUC and PR-AUC assess discrimination and ranking quality.

2025 was not used for calibration fitting.
The original V2 benchmark files were not modified.