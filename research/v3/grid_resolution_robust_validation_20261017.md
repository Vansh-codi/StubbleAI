# StubbleAI V3 — Grid-Resolution Robustness Validation

**Report date:** 17 October 2026  
**Status:** Completed  
**Experiment:** Paired row-level bootstrap comparison of district-level baseline, 0.05° spatial-neighbor model, and 0.10° spatial-neighbor model

## 1. Executive summary

The paired bootstrap found **no demonstrated pooled performance advantage** for either spatial-neighbor variant over the district-level baseline, and no clear difference between the 0.05° and 0.10° variants. All six pooled 95% confidence intervals include zero.

This is a non-rejection result, **not proof of equivalence**. The intervals remain compatible with modest improvements or degradations. The appropriate decision is to retain the district-level baseline as the reference model and preserve both spatial variants as documented research ablations.

The 0.10° grid remains computationally attractive: it represents 8,368 cells compared with 31,658 at 0.05°, approximately **73.6% fewer cells**, while pooled predictive metrics remain similar. This supports an efficiency observation, not a claim of better predictive performance.

## 2. Research question

Do spatial-neighbor features and a finer or coarser grid representation provide a reliably better next-day elevated-fire prediction than the district-level baseline across the six-state V3 pilot?

The compared variants are:

- **Baseline:** district-level feature representation.
- **Spatial 0.05°:** spatial-neighbor representation at 0.05° grid resolution.
- **Spatial 0.10°:** spatial-neighbor representation at 0.10° grid resolution.

The 0.05° and 0.10° variants are retained as ablations; neither is adopted as the replacement reference model on the evidence from this experiment.

## 3. Data and evaluation design

- **Geographic scope:** Assam, Bihar, Haryana, Madhya Pradesh, Punjab, and Uttar Pradesh.
- **Test period:** 2025.
- **Pooled test rows:** 11,408 district-day rows.
- **Positive test labels:** 3,181.
- **Bootstrap replicates:** 10,000.
- **Random seed:** `20261017`.
- **Confidence level:** 95%.
- **Interval method:** percentile interval from a paired row-level bootstrap.
- **Resampling unit:** 2025 district-day test row.
- **Metrics:** F1 and PR-AUC.
- **Threshold handling:** each model's previously selected threshold was held fixed during bootstrap resampling.

The same sampled row indices were used for every model in each bootstrap replicate. This preserves pairing between models and makes the uncertainty calculation focus on differences evaluated on matched test observations. The bootstrap did not retrain models or reselect thresholds.

### Source artifacts

- Prediction file: `data/processed/v3/experiments/grid_sensitivity_010/resolution_comparison/runs/resolution_20261017T034323Z/test_predictions_2025.csv`
- Prediction SHA-256: `81f41ac6f48ec18f9d63e6de7a47364a3e595e5e10282b71354bc5d86621328c`
- Bootstrap output directory: `data/processed/v3/experiments/grid_sensitivity_010/robust_validation/paired_bootstrap_20261017`
- Pooled intervals: `pooled_paired_bootstrap_cis.csv`
- State intervals: `per_state_paired_bootstrap_cis.csv`
- Run manifest: `bootstrap_manifest.json`

## 4. Pooled test-set point estimates

The observed pooled 2025 metrics were:

| Variant | F1 | Precision | Recall | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.747349 | 0.721134 | 0.775542 | 0.819809 | 0.911299 |
| Spatial 0.05° | 0.744036 | 0.728890 | 0.759824 | 0.819084 | 0.911140 |
| Spatial 0.10° | 0.744214 | 0.717201 | 0.773342 | 0.819815 | 0.911030 |

These are point estimates at the locked thresholds. They should be read alongside the paired intervals below rather than treated as conclusive rankings.

## 5. Pooled paired-bootstrap confidence intervals

Differences are calculated as **candidate minus reference**. A positive difference favors the candidate for the metric shown; a negative difference favors the reference.

| Reference | Candidate | Metric | Difference | 95% CI lower | 95% CI upper |
|---|---|---|---:|---:|---:|
| Baseline | Spatial 0.05° | F1 | -0.003314 | -0.007709 | 0.001236 |
| Baseline | Spatial 0.05° | PR-AUC | -0.000724 | -0.002327 | 0.000845 |
| Baseline | Spatial 0.10° | F1 | -0.003135 | -0.007756 | 0.001547 |
| Baseline | Spatial 0.10° | PR-AUC | +0.000006 | -0.001543 | 0.001577 |
| Spatial 0.05° | Spatial 0.10° | F1 | +0.000178 | -0.003973 | 0.004468 |
| Spatial 0.05° | Spatial 0.10° | PR-AUC | +0.000730 | -0.000884 | 0.002426 |

**All six pooled intervals include zero.** Therefore, these results do not distinguish the pooled F1 or PR-AUC of the variants with the selected evaluation procedure.

### Non-equivalence caveat

**An interval containing zero does not prove that two models are equivalent.** For example, the baseline-versus-0.05° F1 difference is -0.003314 with a 95% interval from -0.007709 to +0.001236. This interval is compatible with a modest disadvantage, a small advantage, or a difference near zero. Equivalence would require a prespecified practically acceptable equivalence margin and an appropriate equivalence-testing procedure; neither is established by this bootstrap alone.

The present result is best described as **no clear difference demonstrated by the available test sample**, rather than “the models are the same.”

## 6. State-level point estimates and uncertainty

The table below records the state-level point estimates from the saved bootstrap table. The confidence intervals for every state comparison are retained in `per_state_paired_bootstrap_cis.csv`; the full output should be consulted for the exact bounds.

| State | Comparison | F1 difference | PR-AUC difference |
|---|---|---:|---:|
| Assam | Baseline → Spatial 0.05° | +0.006750 | -0.012928 |
| Assam | Baseline → Spatial 0.10° | -0.003721 | -0.004114 |
| Assam | Spatial 0.05° → Spatial 0.10° | -0.010471 | +0.008814 |
| Bihar | Baseline → Spatial 0.05° | +0.000764 | -0.003132 |
| Bihar | Baseline → Spatial 0.10° | +0.002165 | -0.004749 |
| Bihar | Spatial 0.05° → Spatial 0.10° | +0.001401 | -0.001616 |
| Haryana | Baseline → Spatial 0.05° | +0.000170 | +0.007342 |
| Haryana | Baseline → Spatial 0.10° | -0.001802 | +0.000993 |
| Haryana | Spatial 0.05° → Spatial 0.10° | -0.001971 | -0.006348 |
| Madhya Pradesh | Baseline → Spatial 0.05° | +0.001595 | +0.002113 |
| Madhya Pradesh | Baseline → Spatial 0.10° | -0.003604 | +0.000043 |
| Madhya Pradesh | Spatial 0.05° → Spatial 0.10° | -0.005199 | -0.002070 |
| Punjab | Baseline → Spatial 0.05° | +0.004434 | -0.001278 |
| Punjab | Baseline → Spatial 0.10° | +0.002764 | +0.002965 |
| Punjab | Spatial 0.05° → Spatial 0.10° | -0.001670 | +0.004243 |
| Uttar Pradesh | Baseline → Spatial 0.05° | -0.012477 | -0.003780 |
| Uttar Pradesh | Baseline → Spatial 0.10° | -0.006212 | -0.002002 |
| Uttar Pradesh | Spatial 0.05° → Spatial 0.10° | +0.006265 | +0.001778 |

State-level point estimates are mixed. For example, the 0.05° variant has a higher F1 point estimate in Assam, Haryana, Madhya Pradesh, and Punjab, while Uttar Pradesh has lower F1 point estimates for both spatial variants than the baseline. These patterns are descriptive and should not be presented as confirmed state-specific gains without considering the corresponding confidence intervals and multiplicity.

Assam and Bihar have relatively small numbers of positive test labels (104 and 142, respectively, in the full resolution-comparison test set), so their state-level intervals are wider than those for larger or more positive-rich groups. This uncertainty should remain visible in interpretation.

## 7. Multiple comparisons

The report examines:

- 6 pooled intervals: 3 model comparisons × 2 metrics.
- 36 state-level intervals: 3 model comparisons × 2 metrics × 6 states.
- **42 intervals in total.**

Intervals are not adjusted for multiple comparisons. As a result, isolated state-level intervals that appear favorable should not be selected and interpreted independently as confirmatory evidence. The pooled intervals all include zero, and the overall pattern is consistent with no demonstrated pooled advantage. State-level results are exploratory.

## 8. Effect-size interpretation and statistical power

The pooled F1 intervals have widths of approximately 0.0089–0.0093, with the exact width varying by comparison. These widths show that the observed differences are small relative to the uncertainty under the chosen row-bootstrap procedure.

They are **not, by themselves, a formal minimum detectable effect or statistical power calculation**. A formal power assessment would need a defined target effect, a power criterion, and a suitable data-generating or resampling design. Consequently, this analysis does not establish a universal F1 improvement threshold below which effects cannot be detected.

A larger and more diverse future test set, including additional years as data become available, may help resolve whether small persistent effects exist. More observations alone do not guarantee resolution, particularly if temporal or spatial dependence is substantial.

## 9. Bootstrap limitations

1. **Row-level resampling:** the resampling unit is a district-day row. Rows from the same district and nearby dates may be dependent; neighboring districts may also exhibit spatial dependence. The intervals therefore do not fully account for spatial or temporal dependence.
2. **Fixed fitted models:** the bootstrap evaluates saved predictions; it does not retrain the models in each replicate.
3. **Fixed decision thresholds:** thresholds selected before test evaluation remain locked. The intervals do not capture uncertainty caused by reselecting thresholds.
4. **No equivalence test:** the analysis tests uncertainty in observed differences; it does not define or test a practical equivalence margin.
5. **Multiple comparisons:** the 42 intervals are unadjusted.
6. **Future-year generalization:** the 2025 test set does not establish performance in later years or under changing fire, weather, or reporting conditions.
7. **Grid-weather resolution:** weather is inherited from district-day data and repeated across cells. The 0.10° grid does not create finer-scale meteorological information.

A future block-bootstrap sensitivity analysis, with blocks defined to respect temporal or district-level dependence, would be a useful additional robustness check.

## 10. Computational implications

The 0.05° representation contains 31,658 cells, compared with 8,368 cells at 0.10°. The coarser grid therefore uses approximately 73.6% fewer cells.

The pooled F1 point estimates are close: 0.744036 at 0.05° and 0.744214 at 0.10°. Their paired F1 difference is +0.000178 for 0.10° minus 0.05°, with a 95% CI of [-0.003973, +0.004468]. The results support reporting 0.10° as a potentially more efficient representation for further evaluation, but not as a statistically established predictive improvement.

## 11. Decision and conclusion

**Decision:** retain the district-level baseline as the V3 reference model. Keep both spatial-neighbor variants as documented ablations. Do not claim that spatial features or either grid resolution improves pooled predictive performance based on this experiment.

The principal research finding is:

> In the six-state 2025 evaluation, paired row-level bootstrap intervals did not demonstrate a pooled F1 or PR-AUC advantage for either spatial-neighbor variant over the district-level baseline, nor a clear difference between the 0.05° and 0.10° variants. This is not evidence of equivalence; small effects in either direction remain compatible with the intervals. The 0.10° representation substantially reduces the number of grid cells, making computational efficiency a relevant consideration for future experiments.

This is a useful null result: it prevents a more complex spatial representation from being adopted solely on the basis of small point-estimate fluctuations and provides a clear reference for subsequent V3 experiments.

## 12. Reproducibility record

| Item | Value |
|---|---|
| Bootstrap replicates | 10,000 |
| Seed | `20261017` |
| Confidence level | 95% |
| Interval method | Paired row-level percentile bootstrap |
| Resampling unit | 2025 district-day test row |
| Pooled rows | 11,408 |
| Pooled positive labels | 3,181 |
| Metrics | F1, PR-AUC |
| Comparisons | Baseline vs 0.05°, baseline vs 0.10°, 0.05° vs 0.10° |
| Source prediction hash | `81f41ac6f48ec18f9d63e6de7a47364a3e595e5e10282b71354bc5d86621328c` |

**Source of truth:** the saved CSVs and `bootstrap_manifest.json` in `data/processed/v3/experiments/grid_sensitivity_010/robust_validation/paired_bootstrap_20261017`. If a rounded value in this narrative differs from the stored output, use the stored output as the authoritative record.
