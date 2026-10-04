# StubbleAI V2 — Multi-Horizon Benchmark Summary

## Status

**Research checkpoint:** Frozen
**Geography:** Punjab + Haryana
**Districts:** 45
**Historical period:** October-November 2023, 2024, 2025
**Evaluation year:** 2025
**Forecast horizons:** +1, +2, +3, +5, +7 days

This document summarizes the frozen V2 benchmark checkpoints. It does not represent a new training or evaluation run.

---

## 1. Evaluation Design

The V2 forecasting dataset uses a strict temporal split:

- **2023:** training
- **2024:** validation and threshold selection
- **2025:** final held-out test

No test-set tuning was used for model selection or threshold selection.

The main classification target is:

    fire_count at t + horizon > 2 → Elevated
    fire_count at t + horizon <= 2 → Normal

## 2. Baselines

### Persistence

The persistence baseline uses a fixed rule:

    fire_count today > 2 → Elevated
    fire_count today <= 2 → Normal

The persistence threshold was not tuned separately for each horizon.

### Global Climatology

Uses the overall 2023 Elevated rate as the forecast score.

### District Climatology

Uses the district-specific 2023 Elevated rate, with the global climatology used as fallback.

For climatology models, the classification threshold was selected using the 2024 validation period and frozen before 2025 testing.

## 3. Models

The following supervised models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Random Forest uses the same reproducible configuration across the evaluated horizons.

## 4. Frozen 2025 Test Results

| Horizon | Method | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|
| +1d | Persistence | 0.7700 | 0.8100 | 0.6823 |
| +1d | Global climatology | 0.5657 | 0.5000 | 0.3944 |
| +1d | District climatology | 0.7090 | 0.7922 | 0.6330 |
| +1d | Logistic Regression | 0.6883 | 0.8563 | 0.7809 |
| +1d | Random Forest | **0.7754** | **0.8953** | **0.8582** |
| +1d | XGBoost | 0.7398 | 0.8770 | 0.8414 |
| +1d | LightGBM | 0.7453 | 0.8763 | 0.8428 |
| +2d | Persistence | 0.7279 | 0.7751 | 0.6360 |
| +2d | Global climatology | 0.5657 | 0.5000 | 0.3944 |
| +2d | District climatology | 0.7092 | 0.7933 | 0.6419 |
| +2d | Logistic Regression | 0.6680 | 0.8414 | 0.7691 |
| +2d | Random Forest | **0.7513** | **0.8795** | **0.8369** |
| +2d | XGBoost | 0.7139 | 0.8507 | 0.8089 |
| +2d | LightGBM | 0.6951 | 0.8405 | 0.7970 |
| +3d | Persistence | 0.7004 | 0.7531 | 0.6062 |
| +3d | Global climatology | 0.5634 | 0.5000 | 0.3922 |
| +3d | District climatology | 0.7023 | 0.7892 | 0.6357 |
| +3d | Logistic Regression | 0.6397 | 0.8407 | 0.7582 |
| +3d | Random Forest | **0.7305** | **0.8683** | **0.8193** |
| +3d | XGBoost | 0.7077 | 0.8398 | 0.7860 |
| +3d | LightGBM | 0.7116 | 0.8396 | 0.7831 |
| +5d | Persistence | 0.7132 | 0.7642 | 0.6181 |
| +5d | Global climatology | 0.5617 | 0.5000 | 0.3906 |
| +5d | District climatology | 0.6950 | 0.7832 | 0.6347 |
| +5d | Logistic Regression | 0.6374 | 0.8415 | 0.7658 |
| +5d | Random Forest | **0.7303** | **0.8581** | **0.8022** |
| +5d | XGBoost | 0.7008 | 0.8427 | 0.7848 |
| +5d | LightGBM | 0.7029 | 0.8434 | 0.7894 |
| +7d | Persistence | 0.6576 | 0.7240 | 0.5533 |
| +7d | Global climatology | 0.5449 | 0.5000 | 0.3744 |
| +7d | District climatology | 0.6769 | 0.7777 | 0.6220 |
| +7d | Logistic Regression | 0.6228 | 0.8322 | 0.7391 |
| +7d | Random Forest | **0.7085** | **0.8504** | **0.7816** |
| +7d | XGBoost | 0.6929 | 0.8384 | 0.7589 |
| +7d | LightGBM | 0.6879 | 0.8363 | 0.7584 |

## 5. Random Forest vs Persistence

| Horizon | Persistence F1 | Random Forest F1 | Absolute F1 Gain |
|---|---:|---:|---:|
| +1d | 0.7700 | 0.7754 | +0.0054 |
| +2d | 0.7279 | 0.7513 | +0.0234 |
| +3d | 0.7004 | 0.7305 | +0.0301 |
| +5d | 0.7132 | 0.7303 | +0.0171 |
| +7d | 0.6576 | 0.7085 | +0.0509 |

Random Forest outperformed persistence on F1 at every evaluated horizon.

The largest absolute improvement occurred at the +7 day horizon (+0.0509 F1).

## 6. Main Findings

### Finding 1 — Multi-horizon forecasting is feasible

The Random Forest model maintains F1 above 0.70 across all evaluated horizons, including the +7 day forecast.

### Finding 2 — Persistence is a strong short-term baseline

At +1 day, persistence achieves an F1 of 0.7700, demonstrating that recent fire activity is highly informative for near-term forecasting.

Therefore, any operational model must be compared against persistence rather than only against generic statistical baselines.

### Finding 3 — Random Forest consistently improves on persistence

Random Forest exceeds persistence at every evaluated horizon.

### Finding 4 — Forecast discrimination remains strong

Random Forest ROC-AUC remains between 0.8504 and 0.8953 across the five horizons.

PR-AUC remains between 0.7816 and 0.8582.

### Finding 5 — Geographic climatology is a meaningful benchmark

District climatology consistently outperforms global climatology and provides a stronger non-ML comparison.

## 7. Model Selection Result

Based on the frozen 2025 test F1 results:

**Random Forest is the strongest model among the evaluated supervised models for all five horizons.**

The current benchmark therefore supports Random Forest as the V2 reference forecasting model.

This does not imply that Random Forest is universally optimal. Further research should evaluate calibration, uncertainty, spatial generalization, temporal robustness, and additional forecasting approaches.

## 8. Important Interpretation

The V2 target represents an elevated active-fire count threshold.

A FIRMS active-fire detection is not automatically a confirmed crop-residue or stubble-burning event.

Therefore, model predictions should be described as:

> **Predicted elevated fire activity**

rather than automatically as:

> **Predicted stubble burning**

Causal claims about air-quality impact also require atmospheric transport and pollution modelling.

## 9. Limitations

1. The current V2 geography covers Punjab and Haryana rather than all of India.
2. District aggregation can hide substantial sub-district spatial variation.
3. FIRMS detections represent satellite-observed active fires and are not equivalent to confirmed crop-residue burning.
4. The current classification formulation uses an elevated-count threshold rather than directly classifying confirmed residue-burning events.
5. Historical weather used for model development and operational forecast weather may originate from different data-generation processes and should be audited before production deployment.
6. The current benchmark evaluates classification performance; calibration and uncertainty quantification remain future work.
7. AQI and emissions impacts are not inferred directly from the fire prediction benchmark.
8. The 2025 test period is a single held-out year and should not be interpreted as proof of nationwide generalization.

## 10. Reproducibility

The frozen horizon checkpoints are stored separately:

    research/v2/plus1d/
    research/v2/plus2d/
    research/v2/plus3d/
    research/v2/plus5d/
    research/v2/plus7d/

Each checkpoint contains the corresponding benchmark results and threshold information.

These checkpoints should be treated as immutable research records.

Future experiments should create new experiment directories rather than modifying the frozen checkpoints.

## 11. Research Status

### Completed

- V2 fire-data processing
- District-day fire panel
- Historical weather integration
- Multi-horizon target construction
- Dataset leakage audit
- Temporal train/validation/test design
- Persistence baseline
- Global climatology baseline
- District climatology baseline
- Logistic Regression benchmark
- Random Forest benchmark
- XGBoost benchmark
- LightGBM benchmark
- +1, +2, +3, +5, +7 day evaluation
- Frozen research checkpoints

### Current reference model

**Random Forest**

### Next research priorities

1. Model calibration and reliability
2. Error analysis by district and horizon
3. Feature importance and SHAP analysis
4. Spatial robustness analysis
5. Forecast probability calibration
6. Occurrence + intensity modelling
7. India-wide geographic expansion
8. AQ/emissions integration
9. Production forecasting architecture
10. Public/research UI integration

## 12. Checkpoint Policy

The benchmark results in this document represent a frozen research state.

Do not overwrite the existing horizon checkpoint files when experimenting with new models, features, thresholds, or datasets.

New experiments should receive separate versioned directories.

**StubbleAI V2 benchmark state: FROZEN FOR THIS EXPERIMENT SET**
