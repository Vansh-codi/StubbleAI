# StubbleAI V3 — Research Evidence Ledger

**Purpose:** Central index of V3 research evidence, decisions, limitations, and outstanding validation.
**Status:** Living research record; update only when a new experiment or audit has been completed and its artifacts verified.
**Scope:** Six-state pilot — Assam, Bihar, Haryana, Madhya Pradesh, Punjab, and Uttar Pradesh.

## 1. Research question and scope

V3 investigates whether a grid-first spatial representation and expanded district-day features can address limitations observed in the frozen Punjab–Haryana V2 baseline.

The current six-state dataset covers October and November only, across 2023–2025. It is a seasonal study, not year-round validation. It does not establish generalization to unseen states or production readiness.

Temporal evaluation:
- Training year: 2023
- Validation year: 2024
- Held-out test year: 2025

Thresholds must be selected using validation data and locked before test evaluation.

## 2. Evidence index

The documents below remain the detailed records of their respective work. This ledger does not replace their protocols, manifests, prediction exports, or run artifacts.

| Area | Authoritative record | Status / interpretation |
|---|---|---|
| District feature dataset | `v3_district_feature_dataset_spec.md` | Feature definitions and intended dataset construction |
| Dataset QA | `v3_district_feature_dataset_audit.md` | Read-only dataset audit; not a model evaluation |
| Stage 4 panel QA | `stage4_panel_build_qa.md` | Panel construction and quality checks |
| Grid boundary assignment | `grid_resolution_boundary_assignment_audit.md` | Spatial assignment audit |
| V2 reconstruction comparison | `v2_reconstruction_grid_assignment_comparison.md` | Comparison completed with a documented key-universe limitation |
| Grid robustness | `grid_resolution_robust_validation_20261017.md` | Completed paired bootstrap comparison |
| Emergence experiment protocol | `emergence_only_experiment_protocol_v1.md` | Predefined experiment design |
| Emergence experiment results | `emergence_only_experiment_results_20261017.md` | Completed held-out evaluation |
| Weather protocol history | `weather_substitution_experiment_protocol.md`, `weather_substitution_experiment_protocol_v2.md`, `weather_substitution_experiment_protocol_v3.md` | Preserve as protocol history; do not assume each file represents a separate completed experiment |
| Six-state weather protocol | `weather_substitution_six_state_protocol_v1.md` | Six-state scope and reporting requirements |

For exact figures, run metadata, hashes, and artifact locations, consult the referenced record and its saved outputs. Do not substitute rounded values in this ledger for underlying artifacts.

## 3. Six-state baseline

Reference run:
`data/processed/v3/experiments/six_state_baseline/runs/baseline_20261017T020007Z`

Reported held-out 2025 pooled metrics:
- F1: 0.747349
- Precision: 0.721134
- Recall: 0.775542
- PR-AUC: 0.819809
- ROC-AUC: 0.911299

Reported evaluation population: 11,408 district-day rows, including 3,181 positive labels.

Reported configuration: Random Forest, 400 trees, minimum leaf size 2, random state 42, 69 features. The decision threshold was selected on 2024 validation data.

No fitted model was saved in this run. Treat the saved run report, prediction artifacts, configuration, and manifest as the sources of truth. Do not claim a deployable model artifact exists unless it is separately verified.

## 4. Spatial-neighbor and grid-resolution experiment

Detailed record:
`grid_resolution_robust_validation_20261017.md`

Reported pooled 2025 point estimates:

| Variant | F1 | Precision | Recall | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| District baseline | 0.747349 | 0.721134 | 0.775542 | 0.819809 | 0.911299 |
| Spatial neighbors, 0.05° | 0.744036 | 0.728890 | 0.759824 | 0.819084 | 0.911140 |
| Spatial neighbors, 0.10° | 0.744214 | 0.717201 | 0.773342 | 0.819815 | 0.911030 |

Robustness procedure:
- 10,000 paired row-level bootstrap replicates
- Random seed `20261017`
- Resampling unit: held-out 2025 district-day row
- F1 and PR-AUC comparisons, with previously selected thresholds held fixed
- Saved outputs and manifest under `data/processed/v3/experiments/grid_sensitivity_010/robust_validation/paired_bootstrap_20261017`

All six pooled confidence intervals for the three pairwise comparisons on F1 and PR-AUC include zero.

**Decision:** retain the district-level baseline as the V3 reference. Keep both spatial-neighbor variants as documented ablations. The experiment does not demonstrate a pooled predictive advantage for either spatial variant. An interval containing zero does not establish equivalence.

The 0.10° representation uses substantially fewer cells than the 0.05° representation (8,368 versus 31,658 in the recorded benchmark, approximately 73.6% fewer). This is an efficiency consideration, not evidence of superior predictive performance.

The bootstrap evaluates fixed saved predictions and thresholds. It does not include model-retraining or threshold-selection uncertainty; row-level resampling may also fail to capture district/time dependence fully.

## 5. Emergence-only experiment

Detailed record:
`emergence_only_experiment_results_20261017.md`

Run:
`emergence_20261017T030501Z`

Dataset SHA-256:
`0d0871a8d5e5db3d960269573d849fbcd3bcfd15577d16e8aef3b6f8eae31bd3`

The experiment evaluates 8,267 held-out 2025 observations that were currently in a normal state; 829 were emergence positives (10.03% prevalence). The all-state baseline comparator is restricted to these same rows.

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

**Decision:** retain the unweighted specialist as the leading research candidate; preserve the state-balanced specialist as an ablation. Do not replace the all-state baseline or change live V1 on the basis of this experiment.

Interpretation:
- The unweighted specialist has higher F1 and recall at its validation-selected threshold.
- It also produces substantially more false positives and has slightly lower PR-AUC and ROC-AUC.
- The result reflects an operating-point trade-off; it does not establish better overall probability ranking.
- The comparison is restricted to normal-state test rows, not the entire test population.
- Before adoption, verify saved per-state counts, quantify operational costs of false alarms versus missed emergence, and evaluate additional temporal data if available.

**Artifact protection:** preserve this completed run and its outputs. Do not retrain into its run directory, overwrite predictions, or silently replace its configuration or results.

## 6. Weather-source and coordinate-sensitivity evidence

Protocol history is indexed in Section 2. The earlier weather-source pilot was limited to Punjab and Haryana; its comparison also involved differences in coordinate/source methodology. It must not be presented as a six-state proof that weather-source choice is immaterial.

### Centroid-coordinate sensitivity pilot

Alternative coordinates:
`data/processed/v3/experiments/weather_substitution/district_coordinates_v3_centroid_sensitivity.csv`

Alternative weather:
`data/processed/v3/experiments/weather_substitution/district_daily_weather_v3_centroid_sensitivity.csv`

Audit:
`data/processed/v3/experiments/weather_substitution/district_daily_weather_v3_centroid_sensitivity_audit.txt`

The pilot covered 248 districts and 792 dates per district (196,416 district-day records). The audit reported one-to-one record matching and identical values for all four weather variables between the original and alternative-coordinate weather tables.

**Decision:** treat this as a no-effective-input-variation result. Since the compared weather features did not change, a model comparison would not measure the effect of changed weather inputs. Do not claim this proves broad weather-source robustness or proves the exact internal spatial-cell assignment for every coordinate.

Preserve the alternative-coordinate file, downloaded weather, audit, checkpoints, and script as a separate sensitivity experiment. Do not overwrite the original weather table.

## 7. Spatial and administrative data limitations

The V3 pilot uses six selected states and an administrative district-day representation. District assignment and grid-resolution experiments have their own audit records.

The V2 reconstruction/grid comparison documents a historical key-universe discrepancy, including 202 missing keys in one direct-versus-grid comparison. Do not describe that comparison as a complete row-for-row reconciliation until the discrepancy is resolved.

Keep the distinction between:
- administrative boundary assignment,
- grid representation,
- model predictive performance,
- computational efficiency.

Evidence for one does not automatically establish another.

## 8. Crop-data research direction

Crop features are a proposed future research direction, not an already validated model improvement.

Classify candidate data sources by purpose:

- **Annual/district crop statistics:** useful for descriptive context and dashboard summaries; do not automatically include as predictive features.
- **Spatial cropland fraction or crop-type maps:** candidate features only after coverage, provenance, accuracy, licensing, and controlled ablation checks.
- **Harvest timing and crop phenology:** potentially useful for temporal forecasting, but require source validation and local seasonal relevance.

Before downloading or integrating crop data, create and review:
`research/v3/crop_data_source_evaluation.md`

That evaluation should compare six-state coverage, spatial and temporal resolution, crop/calendar granularity, missingness, accuracy, provenance, licensing, and redistribution terms. Annual crop statistics, crop maps, and crop calendars are not interchangeable.

No crop-data source should be described as validated or integrated until the evidence and implementation are independently recorded.

## 9. Reporting rules

1. Keep the V1 live pipeline operational and unchanged while V3 research proceeds.
2. Keep frozen V2 artifacts and completed experiment outputs immutable.
3. Do not tune against the 2025 held-out test set.
4. Report pooled and per-state metrics, sample counts, positive counts, and prevalence.
5. Preserve the dataset, code, configuration, threshold, protocol, and run hashes where available.
6. Distinguish point estimates from uncertainty and avoid claiming equivalence from a nonsignificant or zero-containing interval.
7. Do not claim year-round performance, unseen-state generalization, causal crop effects, or production readiness from the current evidence.
8. Do not treat a proposed feature or planned experiment as a completed result.
9. Record null results and failed/no-op experiments, not only positive findings.
10. When a new run is performed, use a new unique run directory and preserve its full manifest and prediction outputs.

## 10. Outstanding work and decision gates

- [ ] Verify all six-state per-state counts, prevalence, and confusion matrices against saved run outputs.
- [ ] Complete an operational cost analysis for emergence false alarms versus missed events.
- [ ] Evaluate additional temporal validation when suitable verified data becomes available.
- [ ] Resolve or explicitly bound the historical key-universe discrepancy in the V2 reconstruction comparison.
- [ ] Create `crop_data_source_evaluation.md` before downloading or integrating crop datasets.
- [ ] Expand temporal coverage or define justified state-specific fire seasons before year-round claims.
- [ ] Build and evaluate any crop-feature ablation under a new, documented experiment protocol.
- [ ] Verify a saved, reproducible V3 model artifact and inference pipeline before planning a production replacement.
- [ ] Evaluate the six-state model and dashboard separately from the existing V1 deployment.

## 11. Update policy

For each new experiment, add:
- research question and pre-registered protocol,
- unique run ID and timestamps,
- source dataset and SHA-256,
- code/script and configuration hashes,
- train/validation/test split and exact evaluated population,
- threshold-selection rule and locked threshold,
- pooled and per-state results,
- uncertainty analysis and limitations,
- artifact paths and reproduction command,
- decision and reason for accepting, rejecting, or deferring the result.

Do not edit historical results to make them agree with later experiments. Correct errors through a dated addendum or a new version while preserving the original record.


## Publication Scope and Project Boundary

- **One paper only:** Paper 1 covers the Punjab and Haryana study and its established V2 research design, experiments, and audited findings.
- **V3 is project and deployment work:** the six-state expansion (Punjab, Haryana, Uttar Pradesh, Bihar, Madhya Pradesh, and Assam), spatial experiments, and model-development work are not a second paper.
- **Crop work is for the project:** crop-source evaluation, crop-feature experiments, and crop-aware deployment improvements must be documented here as project evidence; proposed improvements must not be described as validated until tested.
- Keep V3 metrics and artifacts separate from Paper 1's reported results. Do not silently merge the six-state pilot or future crop findings into the paper's experiments.
- Current six-state evidence covers October–November 2023–2025 only; it does not establish year-round performance or production readiness.

## 9. Alternative-coordinate weather sensitivity audit

### Original weather download quality

Audit artifact:
`data/processed/v3/experiments/weather_substitution/district_daily_weather_v3_centroid_sensitivity_audit.txt`

The six-state weather collection covers 248 districts and 792 dates per district, from 2023-10-01 through 2025-11-30. The audit reports 196,416 rows, zero duplicate state-district-date keys, and zero missing values in T2M, RH2M, WS2M, and PRECTOTCORR. State row counts are Assam 27,720; Bihar 30,096; Haryana 17,424; Madhya Pradesh 43,560; Punjab 18,216; and Uttar Pradesh 59,400.

**Quality caveat:** the audit explicitly warns that NASA POWER may use documented fill values. A zero null count does not establish that all parameter values are valid; inspect fill-value conventions and ranges before modelling.

### Alternative-coordinate comparison

The sensitivity experiment compared the original V3 district coordinates against centroids computed from the official district polygons for all 248 districts. The alternative coordinate file and downloaded weather are stored separately under `data/processed/v3/experiments/weather_substitution/`; the resumable downloader is under `scripts/v3/`.

The recorded comparison found one-to-one matching of 196,416 district-date records and identical values for all four weather variables (T2M, RH2M, WS2M, PRECTOTCORR). Thus, this coordinate change produced no observed weather-input variation in the tested period.

This is a null sensitivity result, not proof that different weather sources are interchangeable, that coordinates never affect retrieved values, or that all coordinate pairs necessarily share an internal NASA POWER grid cell. Do not claim predictive equivalence or rerun model comparisons on identical inputs as if this were a distinct weather treatment.
