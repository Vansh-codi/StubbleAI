# StubbleAI V2 - +3d Temporal Probability Calibration

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
| Logistic Regression | raw | 0.1748 | 0.5180 | 0.1103 | 0.8407 | 0.7582 |
| Logistic Regression | sigmoid_calibrated | 0.1907 | 0.5540 | 0.1548 | 0.8407 | 0.7582 |
| Random Forest | raw | 0.1429 | 0.4359 | 0.0243 | 0.8683 | 0.8193 |
| Random Forest | sigmoid_calibrated | 0.1434 | 0.4373 | 0.0366 | 0.8683 | 0.8193 |
| XGBoost | raw | 0.1743 | 0.5968 | 0.1139 | 0.8398 | 0.7860 |
| XGBoost | sigmoid_calibrated | 0.1581 | 0.4811 | 0.0375 | 0.8398 | 0.7860 |
| LightGBM | raw | 0.1790 | 0.6446 | 0.1279 | 0.8396 | 0.7831 |
| LightGBM | sigmoid_calibrated | 0.1586 | 0.4817 | 0.0362 | 0.8396 | 0.7831 |

## Interpretation
Lower Brier, Log Loss, and ECE indicate improved probability reliability. ROC-AUC and PR-AUC assess discrimination and ranking quality.

2025 was not used for calibration fitting.
The original V2 benchmark files were not modified.