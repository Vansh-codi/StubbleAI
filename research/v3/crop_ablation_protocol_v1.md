# StubbleAI V3 — Six-State Crop Ablation Protocol v1

Status: Protocol draft; not yet executed
Scope: StubbleAI V3 project only; not Paper 1
Geography: Assam, Bihar, Haryana, Madhya Pradesh, Punjab, Uttar Pradesh
Target: target_elevated_t_plus_1d
Temporal split: 2023 train, 2024 validation, 2025 held-out test

## Research question
Do district-level crop statistics add useful next-day fire-risk
predictive information beyond the existing crop-free V3 baseline?

## Source artifacts
- Daily dataset: data/processed/v3/v3_district_ml_dataset_1d.csv
- Original crop records: data/processed/v3/crop_district_features_2022_2023.csv
- Wide crop features: data/processed/v3/crop_district_features_wide_2022_2023.csv
- Existing six-state baseline run:
  data/processed/v3/experiments/six_state_baseline/runs/baseline_20261017T020007Z

## Required design
1. Preserve the original dataset and completed baseline run.
2. Aggregate crop records to one row per state-district before joining.
3. Verify that the join does not change daily row counts or duplicate keys.
4. Do not encode an absent crop record as zero production or zero area.
5. Treat missing crop values explicitly. Fit any imputation using training data only
   and consider missingness indicators.
6. Investigate when the 2022-2023 crop statistics became available. Document the
   temporal-availability limitation before interpreting the experiment as a
   historically deployable forecast.
7. Compare crop-free and crop-aware models on identical rows, labels, and temporal
   splits. Report any excluded population and the reason.
8. Train on 2023; select thresholds and preprocessing using 2024 only; evaluate
   the locked models on 2025 once.
9. Report pooled and per-state sample counts, prevalence, confusion matrices,
   F1, precision, recall, PR-AUC, and ROC-AUC where appropriate.
10. Preserve null or negative findings. Do not change live V1 or frozen V2.

## Decision gate
Do not begin model training until the crop-feature join, missing-data policy,
feature timing, and matched evaluation population are documented and checked.

## Publication boundary
This is six-state V3 project evidence. Do not merge these results into Paper 1,
which remains restricted to Punjab and Haryana and the established V2 study.

## Status
Draft created before experiment execution. No crop-ablation results are claimed.
