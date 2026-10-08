# StubbleAI V2 - +1d Temporal Probability Calibration

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
| Logistic Regression | raw | 0.1719 | 0.5107 | 0.1247 | 0.8563 | 0.7809 |
| Logistic Regression | sigmoid_calibrated | 0.2019 | 0.5832 | 0.1986 | 0.8563 | 0.7809 |
| Random Forest | raw | 0.1273 | 0.3960 | 0.0320 | 0.8953 | 0.8582 |
| Random Forest | sigmoid_calibrated | 0.1277 | 0.3965 | 0.0336 | 0.8953 | 0.8582 |
| XGBoost | raw | 0.1485 | 0.4946 | 0.0933 | 0.8770 | 0.8414 |
| XGBoost | sigmoid_calibrated | 0.1405 | 0.4340 | 0.0620 | 0.8770 | 0.8414 |
| LightGBM | raw | 0.1509 | 0.5280 | 0.1005 | 0.8763 | 0.8428 |
| LightGBM | sigmoid_calibrated | 0.1404 | 0.4336 | 0.0656 | 0.8763 | 0.8428 |

## Interpretation
Lower Brier, Log Loss, and ECE indicate improved probability reliability. ROC-AUC and PR-AUC assess discrimination and ranking quality.

2025 was not used for calibration fitting.
The original V2 benchmark files were not modified.