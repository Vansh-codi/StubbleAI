# StubbleAI V3 — Emergence-Only Experiment Results

## Run identification

- Run: `emergence_20261017T030501Z`
- Dataset SHA256: `0d0871a8d5e5db3d960269573d849fbcd3bcfd15577d16e8aef3b6f8eae31bd3`
- Temporal split: 2023 train, 2024 validation, 2025 held-out test
- Test population: 8,267 currently normal-state observations
- Test emergence positives: 829
- Test prevalence: 10.03%
- Variants: unweighted emergence-only and state-balanced emergence-only
- Comparator: existing all-state baseline predictions restricted to the same test rows

## Pooled held-out results

| Metric | Baseline on normal subset | Unweighted specialist | State-balanced specialist |
|---|---:|---:|---:|
| F1 | 0.3574 | 0.4000 | 0.3972 |
| Precision | 0.3873 | 0.3246 | 0.3271 |
| Recall | 0.3317 | 0.5211 | 0.5054 |
| PR-AUC | 0.3314 | 0.3302 | 0.3289 |
| ROC-AUC | 0.8139 | 0.8116 | 0.8107 |
| False positives | 435 | 899 | 862 |
| False negatives | 554 | 397 | 410 |
| True positives | 275 | 432 | 419 |
| True negatives | 7,003 | 6,539 | 6,576 |

## Per-state F1

| State | Baseline | Unweighted | State-balanced |
|---|---:|---:|---:|
| Assam | 0.2195 | 0.3065 | 0.3036 |
| Bihar | 0.1573 | 0.2553 | 0.2556 |
| Haryana | 0.3169 | 0.3820 | 0.3802 |
| Madhya Pradesh | 0.3151 | 0.3624 | 0.3521 |
| Punjab | 0.4121 | 0.4460 | 0.4388 |
| Uttar Pradesh | 0.4148 | 0.4452 | 0.4449 |

## Interpretation

1. The unweighted specialist has the highest pooled F1 and recall at the thresholds selected on 2024 validation.
2. The specialist increases recall substantially relative to the baseline on the same normal-state rows, but it also produces more false positives.
3. The baseline has slightly higher pooled PR-AUC and ROC-AUC. The specialist's F1 improvement therefore reflects a different operating-point tradeoff and does not establish better overall probability ranking.
4. State balancing reduces pooled false positives from 899 to 862 compared with the unweighted specialist, but also reduces true positives from 432 to 419.
5. Unweighted F1 is higher in Assam, Haryana, Madhya Pradesh and Punjab. Bihar slightly favors state balancing; Uttar Pradesh is nearly tied.
6. These are results from one held-out year. Do not claim robust generalization or production readiness based on this experiment alone.

## Decision

- Retain the unweighted specialist as the leading research candidate.
- Preserve the state-balanced variant as an ablation.
- Do not replace the all-state baseline or change the live V1 pipeline.
- Before adoption, verify the saved per-state counts, add an explicit operational cost analysis for false alarms versus missed emergence, and assess robustness with additional temporal validation if suitable data becomes available.
- Preserve the run artifacts and do not retrain or overwrite this completed run.
