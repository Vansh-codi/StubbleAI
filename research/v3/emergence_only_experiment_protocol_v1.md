# StubbleAI V3 — Emergence-Only Experiment Protocol V1

## 1. Status

Protocol draft for review. No model training is authorized by this document alone.
The six-state baseline and spatial-neighbor experiment remain unchanged.

## 2. Research question

Does a classifier trained specifically on currently normal district-days improve
next-day fire-emergence prediction compared with the existing all-state V3
baseline evaluated on the same normal-state test observations?

## 3. Prediction task

- Observation unit: state-district-day.
- Current normal state: `fire_count <= 2`.
- Positive outcome: `target_elevated_t_plus_1d == 1` (N -> E).
- Negative outcome: `target_elevated_t_plus_1d == 0` (N -> N).
- Current elevated observations (E -> E and E -> N) are excluded from this
  specialist model's training, validation, and test populations.
- The specialist is complementary to the all-state baseline, not a replacement.

## 4. Dataset and eligibility

Input:
`data/processed/v3/v3_district_ml_dataset_1d.csv`

Use the baseline's existing eligibility requirements:
- Target is non-null.
- All 36 required lag features are non-null.
- Current state satisfies `fire_count <= 2`.
- State, district, and date keys are valid and unique.

The exact baseline feature list and preprocessing must be reused. Exclude `date`,
future-target columns, and any other fields that would leak future information.
Verify the actual feature list against the frozen baseline script before training.

Expected counts from the transition audit, pending exact eligibility verification:

| Split | N -> E | N -> N | Total |
|---|---:|---:|---:|
| Train, 2023 | 809 | 7,207 | 8,016 |
| Validation, 2024 | 925 | 6,981 | 7,906 |
| Test, 2025 | 829 | 7,438 | 8,267 |

If the counts differ after applying the baseline's exact eligibility rules, stop
and investigate before training. Do not force the expected counts.

## 5. Temporal evaluation

- Train: 2023.
- Validation: 2024; used for threshold selection only.
- Test: 2025; held out until all model and threshold choices are locked.
- October-November seasonal scope is retained.

## 6. Model variants

### Variant A — Unweighted specialist

Use the baseline Random Forest configuration:
- `n_estimators=400`
- `min_samples_leaf=2`
- `random_state=42`
- `class_weight=None`
- `n_jobs=-1`

Reuse the baseline preprocessing, including median imputation for numeric
features and most-frequent imputation plus one-hot encoding for state and district.

### Variant B — State-balanced specialist

Use the same features, preprocessing, model settings, and temporal split.
Apply sample weights during training so each of the six states contributes equal
total training weight. Define and audit the weight formula before fitting.
Do not change the validation or test populations.

This is an ablation, not an assumed improvement. Report each state's training
rows and positive examples so sparse-state results are interpreted cautiously.

## 7. Threshold selection

Choose a separate threshold for each variant using 2024 validation F1.
Use the baseline's deterministic candidate grid and tie-breaking rule:
maximize F1 and choose the lowest threshold on ties.

Lock each threshold before evaluating 2025. Do not tune using test metrics.

## 8. Comparison design

Compare both specialist variants against the existing baseline predictions
restricted to the exact same 2025 normal-state rows.

Report:
- Rows, positives, prevalence, F1, precision, recall, PR-AUC and ROC-AUC.
- Confusion matrix counts.
- Per-state metrics and emergence recall.
- False positives per state.
- Differences against the all-state baseline on the same subset.

Do not compare the specialist's normal-state-only metrics directly with the
baseline's full-population metrics as if they were the same task.

## 9. Integrity and artifacts

- Do not modify V1, frozen V2, the baseline run, or the spatial-neighbor run.
- Do not modify raw source data or the input ML dataset.
- Create a fresh, unique run directory for each experiment.
- Save the protocol version, dataset SHA256, eligibility counts, feature list,
  model configuration, software versions, validation thresholds, test predictions,
  pooled metrics, per-state metrics, and run manifest.
- Refuse to overwrite existing outputs.
- Preserve failures and their diagnostic logs.

## 10. Interpretation limits

State balancing may help smaller states but is not guaranteed to improve results.
Report pooled and per-state results together; pooled scores alone may conceal
state-level weaknesses. The specialist does not predict E -> E or E -> N.

## 11. Decision gate

Do not train until:
1. Exact eligible N-state counts are independently reproduced.
2. The feature list is verified against the baseline.
3. State-balanced sample weights are defined and audited.
4. The output directory and comparison population are specified.
5. This protocol has been reviewed and accepted.

Only then implement a separate experiment script and run preflight checks.
