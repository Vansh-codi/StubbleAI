# StubbleAI V2 — +1 Day Benchmark

## Dataset

- Dataset: `ml_dataset_v2.csv`
- Geography: Punjab + Haryana
- Districts: 45
- Training period: 2023
- Validation period: 2024
- Frozen test period: 2025

## Target

`fire_count_t_plus_1d > 2` → Elevated (1)

## Evaluation Protocol

- 2023 → training
- 2024 → validation
- 2025 → frozen final test
- Classification threshold selected on 2024 validation by maximizing Elevated-class F1.
- Selected threshold was frozen before evaluating the 2025 test set.
- Persistence remains the primary naive temporal baseline.

## +1 Day Test Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0.8178 | 0.7668 | 0.7732 | 0.7700 | 0.8100 | 0.6823 |
| Random Forest | 0.8094 | 0.7246 | 0.8338 | **0.7754** | **0.8953** | **0.8582** |
| XGBoost | 0.7772 | 0.6859 | 0.8028 | 0.7398 | 0.8770 | 0.8414 |
| LightGBM | 0.7900 | 0.7145 | 0.7789 | 0.7453 | 0.8763 | 0.8428 |
| Logistic Regression | 0.6639 | 0.5426 | 0.9408 | 0.6883 | 0.8563 | 0.7809 |

## Selected Thresholds

- Logistic Regression: `0.3119`
- Random Forest: `0.4109`
- XGBoost: `0.3119`
- LightGBM: `0.3763`

## Random Forest Confusion Matrix

- TN = 865
- FP = 225
- FN = 118
- TP = 592

## Persistence Confusion Matrix

- TN = 923
- FP = 167
- FN = 161
- TP = 549

## Main Finding

Random Forest achieved the highest F1-score among the evaluated machine-learning models and narrowly exceeded the persistence baseline:

- Random Forest F1 = `0.7754`
- Persistence F1 = `0.7700`
- Absolute F1 improvement = `+0.0054`

Random Forest also achieved substantially higher probability-ranking performance:

- ROC-AUC = `0.8953` vs `0.8100`
- PR-AUC = `0.8582` vs `0.6823`

The improvement in thresholded F1 is modest, so Random Forest is considered the current +1-day candidate rather than a conclusively superior model.

## Models Evaluated

1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM
5. Persistence baseline
6. Global climatology baseline
7. District climatology baseline

## Research Status

This result is a frozen +1-day benchmark.

No test-set threshold tuning or hyperparameter selection was performed based on the 2025 test results.

The next experiment is multi-horizon evaluation at:

- +2 days
- +3 days
- +5 days
- +7 days

The same evaluation methodology should be retained for these experiments.