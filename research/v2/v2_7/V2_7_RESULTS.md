# V2.7 Learner Benchmark — Results

## Research Question

Given the exact frozen V2 feature set and temporal evaluation protocol, can stronger gradient-boosted tree learners outperform the frozen Random Forest?

## Experimental Arms

- Random Forest — frozen V2 control
- XGBoost 3.4.1 — challenger
- LightGBM 4.7.0 — challenger

All models used the same V2 dataset, features, temporal split, horizons, and validation-only threshold selection.

## Protocol

- Dataset: `ml_dataset_v2.csv`
- Training: 2023
- Validation: 2024
- Final test: 2025
- Horizons: +1d, +2d, +3d, +5d, +7d
- Threshold candidates: 0.01–0.99, 199 values
- Threshold selected using 2024 Elevated-class F1
- 2025 remained untouched for model/threshold selection
- V2 Random Forest reproduction guard passed at every horizon

## Model Configuration

### Random Forest

- 400 trees
- minimum leaf size 2
- random state 42
- no class weighting

### XGBoost

- 400 estimators
- max depth 6
- learning rate 0.05
- min child weight 1
- subsample 0.9
- column subsampling 0.9
- histogram tree method
- random state 42

### LightGBM

- 400 estimators
- learning rate 0.05
- 31 leaves
- minimum child samples 20
- subsample 0.9
- column subsampling 0.9
- random state 42

## 2025 Test Results

| Horizon | RF F1 | XGBoost F1 | LightGBM F1 |
|---|---:|---:|---:|
| +1d | 0.7754 | 0.7436 | 0.7409 |
| +2d | 0.7513 | 0.7131 | 0.7076 |
| +3d | 0.7305 | 0.7092 | 0.7083 |
| +5d | 0.7303 | 0.7152 | 0.6951 |
| +7d | 0.7085 | 0.6961 | 0.6863 |

Random Forest achieved the highest F1 at every forecast horizon.

Approximate mean F1 across the five horizons:

- Random Forest: 0.7392
- XGBoost: 0.7154
- LightGBM: 0.7076

## Ranking Performance

Random Forest also achieved the highest ROC-AUC and PR-AUC at every horizon.

This indicates that the Random Forest advantage was not caused only by threshold selection. Its probability ranking was also stronger under this evaluation.

## Probability Quality

Random Forest achieved lower Brier score and LogLoss than both challengers at every horizon.

Therefore the challenger models did not demonstrate a probability-quality advantage.

## Emerging-Fire Detection

The gradient-boosted models showed an important contrasting behavior.

Normal?Elevated recall:

| Horizon | RF | XGBoost | LightGBM |
|---|---:|---:|---:|
| +1d | 0.4037 | 0.4410 | 0.5031 |
| +2d | 0.4817 | 0.5445 | 0.5445 |
| +3d | 0.6202 | 0.6587 | 0.6106 |
| +5d | 0.5635 | 0.5533 | 0.6802 |
| +7d | 0.6267 | 0.6544 | 0.7235 |

The challengers frequently achieved higher emerging-fire recall, especially at longer horizons.

However, the higher sensitivity was accompanied by reduced precision and increased false-positive activity, so the models did not improve overall forecasting performance.

## Interpretation

The benchmark provides evidence that changing the tree-learning algorithm alone does not solve the emerging-fire forecasting limitation.

XGBoost and LightGBM can increase sensitivity to Normal?Elevated transitions, but the additional detections are accompanied by enough false positives to reduce overall F1.

Random Forest provides the strongest balance of:

- overall F1
- precision/recall
- ROC-AUC
- PR-AUC
- Brier score
- LogLoss

under the frozen V2 evaluation protocol.

## Decision

**V2.7 — REJECTED AS A LEARNER REPLACEMENT.**

Random Forest remains the official V2 forecasting learner.

XGBoost and LightGBM are retained as documented challenger experiments.

No challenger model replaces the frozen V2 model.

## Combined Research Interpretation

V2.5 showed that cost-sensitive learning does not robustly solve emerging-fire detection.

V2.6 showed that explicit transition modeling can improve longer-horizon onset recall but increases false alarms and does not consistently improve overall F1.

V2.7 shows that changing the learner to gradient boosting can also increase onset sensitivity without improving the complete forecasting task.

Together, these experiments suggest that the remaining limitation is unlikely to be solved simply by changing classifier type, class weighting, or classification formulation.

The current Punjab/Haryana dataset likely has an information/forecastability limitation for reliable multi-day fire-onset prediction.

This does not establish that additional data will necessarily solve the limitation, but it provides sufficient evidence to justify prioritizing richer spatio-temporal data over continued unconstrained model shopping.

## Next Stage

The next major stage should focus on India-wide data architecture and richer predictive context.

Priority areas include:

- national FIRMS ingestion
- spatial grid representation
- State ? District ? Grid hierarchy
- weather integration
- crop/land context
- spatial neighborhood context
- improved temporal coverage
- verification infrastructure

Air-quality observations, emissions estimation, and atmospheric transport should remain separate scientific layers rather than being treated as direct outputs of the fire classifier.

Deep temporal models should be evaluated only after a substantially richer national spatio-temporal dataset is available.

## Reproducibility

All V2.7 results are stored under:

`research/v2/v2_7/`

The official V2 model and frozen benchmark artifacts remain unchanged.
