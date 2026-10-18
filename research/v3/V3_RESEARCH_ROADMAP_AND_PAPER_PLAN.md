# StubbleAI V3 — Research Roadmap and Paper Plan

**Document status:** Planning record
**Research status:** V3 research in progress
**Last updated:** 18 October 2026
**Purpose:** Preserve research decisions, future work, paper workflow, and deployment criteria before implementation begins.

## 1. Project direction

StubbleAI V3 is intended to become a replacement forecasting system, not merely a dashboard extension or crop-information add-on.

The intended scope is an expanded, six-state fire-forecasting system with a stronger spatial foundation, validated temporal coverage, and experimentally justified additional features, including crop-related features where reliable data supports them.

### Geographic pilot

- Assam
- Bihar
- Haryana
- Madhya Pradesh
- Punjab
- Uttar Pradesh

### Non-negotiable guardrail

**Expanding geographic coverage alone is not enough to establish a better forecasting system.**

V3 must demonstrate reproducible performance, appropriate temporal coverage, reliable data provenance, and acceptable operational trade-offs before replacing the existing production system.

## 2. Existing system preservation

### V1 — operational system

- The existing V1 pipeline remains operational.
- Continue preserving its daily predictions, historical outputs, and existing workflow.
- Do not change the live V1 model or pipeline merely because V3 research is progressing.
- Record prospective predictions separately from retrospective experimental results.

### V2 — frozen research baseline

- Preserve the frozen V2 baseline and its completed research artifacts.
- The existing paper is a reference for methodology, structure, and reporting discipline.
- Do not overwrite frozen V2 runs, predictions, configurations, or paper artifacts.

### V3 — research and eventual replacement

V3 research, dashboard development, and production deployment are separate milestones.

A completed experiment does not automatically authorize deployment. A dashboard displaying V3 predictions does not establish that the model is ready to replace V1.

## 3. Current evidence and completed work

The detailed evidence ledger is:

`research/v3/V3_RESEARCH_EVIDENCE_LEDGER.md`

Existing experiment records include:

- `research/v3/v3_district_feature_dataset_spec.md`
- `research/v3/v3_district_feature_dataset_audit.md`
- `research/v3/stage4_panel_build_qa.md`
- `research/v3/grid_resolution_boundary_assignment_audit.md`
- `research/v3/grid_resolution_robust_validation_20261017.md`
- `research/v3/emergence_only_experiment_protocol_v1.md`
- `research/v3/emergence_only_experiment_results_20261017.md`
- `research/v3/v2_reconstruction_grid_assignment_comparison.md`
- `research/v3/weather_substitution_six_state_protocol_v1.md`

Weather protocol history is also preserved in the three earlier protocol files. These files should be treated as version history, not as proof that each version corresponds to a separate completed experiment.

### Established findings

1. The district-level six-state baseline remains the reference model.
2. The evaluated spatial-neighbor variants have not demonstrated a statistically clear pooled predictive advantage over that baseline.
3. The 0.10-degree grid uses substantially fewer cells than the 0.05-degree grid, but computational efficiency is not proof of predictive superiority.
4. The emergence-only specialist increases recall and F1 at its selected operating point but produces more false alarms and has slightly lower PR-AUC than the baseline on the same normal-state test rows.
5. The alternative-coordinate weather sensitivity pilot produced identical values for all four compared weather variables. It therefore provides no effective weather-input variation and cannot establish broad weather-source robustness.
6. The current six-state dataset covers October and November in 2023–2025. It does not establish year-round performance.
7. The V2 reconstruction/grid comparison retains a documented historical key-universe discrepancy that must not be described as fully reconciled until resolved.

The source reports and saved artifacts remain authoritative for exact metrics, hashes, configurations, and run details.

## 4. Research sequence

The planned order is:

1. Finish and verify the research evidence record.
2. Scope and execute temporal expansion.
3. Evaluate crop-data sources.
4. Pre-register and execute crop-aware ablations.
5. Validate the replacement candidate and proceed through deployment gates.

The research paper is a parallel track throughout these stages.

This document records the intended sequence. It does not mark future work as completed or authorize new experiments.

## 5. Step 1 — Close the current validation record

Before beginning temporal expansion:

- Verify the evidence ledger against the saved reports and run artifacts.
- Confirm run IDs, dataset hashes, configuration hashes, and prediction artifact paths where available.
- Preserve completed runs without retraining into their directories or overwriting their outputs.
- Reconcile reported metrics with saved prediction files and manifests.
- Record pooled and per-state sample counts, positive counts, prevalence, and confusion matrices.
- Preserve null, inconclusive, and negative results alongside positive findings.
- Record known limitations and unresolved discrepancies explicitly.

### Single-held-out-year limitation

The current six-state evaluation uses 2025 as the only held-out test year.

The planned mitigation is a second prospective evaluation using 2026 observations, once sufficient outcomes have been verified. The evaluation must not be presented as completed before those outcomes are available and audited.

## 6. Step 2 — Temporal expansion and data acquisition

### Proposed direction

The preferred direction is to evaluate a full calendar year for all six states rather than assuming that one shared October–November window represents every state's fire regime.

This is a proposed research decision to formalize before ingestion. The acquisition scope, data availability, compute requirements, and protocol must be confirmed before implementation.

### Required scoping record

Before downloading new data, document:

- FIRMS source products, satellite sources, date range, and acquisition procedure.
- Historical data availability and known observation limitations.
- Weather source, variables, temporal alignment, and missing-data handling.
- District and grid assignment methodology.
- Fire aggregation, duplicate handling, and target construction.
- Feature availability dates and leakage controls.
- Dataset versioning, hashes, manifests, and reproducibility requirements.
- Storage, runtime, and compute estimates.
- Data quality checks and criteria for accepting the expanded panel.

### Evaluation design

Retain a chronological evaluation design. The existing 2023 training, 2024 validation, and 2025 held-out test split must not be silently changed to make the expanded experiment appear more successful.

The expanded study must define how new years are used, preserve a genuinely held-out evaluation period, and prevent test-set information from influencing feature selection or threshold tuning.

### Seasonal interpretation

Full-calendar-year collection is intended to reduce the risk of selecting an inappropriate season by assumption. It does not imply that every state has the same seasonality, that all observations are equally informative, or that geographic coverage alone establishes generalization.

Report state-level temporal coverage, positive-event counts, prevalence, and missingness.

## 7. Prospective 2026 evaluation

The existing V1 live pipeline has been operating during 2026. Its predictions may provide a prospective evaluation opportunity when their corresponding outcomes become verifiable.

This is a planned validation pathway, not yet an established second held-out-year result.

### Required procedure

1. Preserve predictions as they were issued, including prediction timestamps, model version, feature/data version, threshold, and configuration hash.
2. Define the prediction horizon and eligible district-day population in advance.
3. Match predictions to subsequently observed and quality-checked FIRMS outcomes using the predefined target definition.
4. Record incomplete observation periods and any limitations in outcome verification.
5. Evaluate pooled and per-state performance, including positive-event counts and prevalence.
6. Report F1, precision, recall, PR-AUC, ROC-AUC where appropriate, confusion matrices, false alarms, and missed events.
7. Keep this prospective evaluation separate from retrospective model-development results.

### Duration and sufficiency

Do not automatically treat a few weeks or months as a meaningful second test year.

Define adequacy using temporal coverage, state-specific event counts, prevalence, uncertainty, and representation of relevant fire seasons. Sparse positive events may make estimates unstable even when the overall row count is large.

Use the accumulated evidence to determine whether conclusions are sufficiently supported. Do not invent a fixed duration or claim prospective validation is complete prematurely.

## 8. Step 3 — Crop-data source evaluation

Create and review:

`research/v3/crop_data_source_evaluation.md`

This source-evaluation document must exist before crop data is downloaded or integrated into the model.

### Candidate source categories

**A. Annual or district-level crop statistics**

Potential uses include descriptive context, crop area and production summaries, and dashboard information. Such data should not automatically become a predictive feature merely because it is available.

**B. Spatial cropland fraction or crop-type maps**

Potential model features require evidence of spatial coverage, accuracy, provenance, licensing, redistribution permissions, and compatibility with the six-state study.

**C. Harvest timing and crop phenology**

Potentially useful for temporal fire forecasting, but source availability, calendar detail, validation quality, and state-specific relevance must be verified.

### Required comparison criteria

- Coverage across all six states.
- Spatial resolution and administrative compatibility.
- Temporal resolution and historical coverage.
- Crop-type specificity.
- Harvest or phenological timing detail.
- Missingness and data-quality characteristics.
- Validation accuracy and known uncertainty.
- Source provenance and reproducibility.
- License, attribution, and redistribution terms.
- Suitability for research use and eventual deployment.

Annual statistics, crop-type maps, and harvest calendars are distinct evidence types and must not be treated as interchangeable.

## 9. Step 4 — Crop-aware ablation

Crop features become model inputs only after source evaluation and feature-timing checks.

Before training, create a separate experiment protocol defining:

- Research question and hypothesis.
- Eligible dataset and source hashes.
- Feature definitions and availability dates.
- Missing-data and missing-coverage handling.
- Leakage checks.
- Identical eligible rows and temporal splits for compared models.
- Baseline and crop-aware configurations.
- Threshold selection on validation data only.
- Pooled and per-state metrics.
- Uncertainty analysis and practical acceptance criteria.
- Artifact paths, manifests, and reproduction instructions.

The crop-aware model must be compared with an appropriate crop-free reference using the same evaluated population wherever possible.

A performance difference must not be attributed to crop features if it is explained by different rows, altered labels, changed splits, or threshold tuning on the test set.

If the crop feature fails to improve the predefined evaluation criteria, preserve the result and do not force the feature into the production model.

## 10. Step 5 — V3 replacement and deployment gates

V3 is intended to replace the old forecasting system eventually. That replacement is conditional on evidence, not on completing the dashboard.

Before production transition, require:

- A reproducible, saved model artifact.
- Verified feature preprocessing and inference code.
- Matching training and inference schemas.
- Documented model, dataset, and configuration versions.
- Reproducible pooled and per-state evaluation.
- Appropriate temporal validation and documented uncertainty.
- False-alarm versus missed-event analysis.
- Data-quality and missing-input handling.
- Monitoring, logging, and failure recovery.
- A fallback to the existing operational system.
- Deployment and end-to-end tests.
- A controlled transition plan with rollback criteria.

Do not change the live V1 pipeline until the replacement criteria are met and the transition is explicitly approved.

## 11. Research paper — parallel workstream

The existing V2 paper PDF serves as the structural and methodological reference for the new research manuscript.

Create the following structure:

`research/v3/paper/`

Suggested contents:

- `V3_PAPER_DRAFT.md` — evolving manuscript draft.
- `V3_RESULTS_SOURCE_OF_TRUTH.md` — metric tables and links to authoritative artifacts.
- `V3_NUMERICAL_AUDIT_CHECKLIST.md` — verification checklist before formatting.
- `README.md` — paper scope, status, and reproduction references.

These are planned files; create them when the paper workstream begins.

### Writing discipline

Update the draft as each research stage completes rather than waiting until all experiments are finished.

Every reported metric must be traceable to a saved result. When a metric changes because an error is corrected, record the correction and preserve the historical record instead of silently rewriting prior findings.

The paper should include:
- Six-state study area and data provenance.
- Temporal coverage and the reason for the selected evaluation design.
- Spatial representation and boundary/grid methodology.
- Baseline, spatial-neighbor, emergence-only, grid-resolution, and relevant weather-sensitivity findings.
- Negative and inconclusive results.
- Crop and expanded-temporal experiments only after they are completed.
- Limitations, intended use, and reproducibility information.

### Numerical audit before formatting

Follow the discipline used in the existing V2 paper workflow:

1. Reconcile manuscript metrics against authoritative result files.
2. Check tables, figures, abstract, discussion, and conclusion for consistency.
3. Verify denominators, positive counts, confusion matrices, and rounded values.
4. Confirm experiment IDs, hashes, and test-period descriptions.
5. Verify that claims do not exceed the evidence.
6. Record audit results before beginning final IEEE formatting.

Do not begin final formatting while the numerical audit has unresolved mismatches.

### V2 versus V3 paper scope

The working recommendation is a standalone V3 paper that cites the V2 work as prior research. V3 has a different geographic scope, research question, and experimental record.

**Decision still to confirm:** whether the V2 manuscript has been submitted or remains pre-submission. Do not revise or reclassify a submitted paper as an unpublished draft. Confirm its status before deciding the final publication relationship.

## 12. Evidence and artifact protection rules

- Never overwrite completed experiment outputs.
- Use a new unique run directory for each new experiment.
- Preserve original protocol versions and add amendments rather than erasing history.
- Keep model predictions separate from verified outcomes.
- Record null results and failed or no-op experiments.
- Store source data, configuration, script, protocol, and output hashes where available.
- Distinguish a planned experiment from an executed experiment.
- Distinguish an executed experiment from an independently verified result.
- Distinguish a verified result from evidence sufficient for deployment.
- Keep V1 live operations separate from V3 research and prospective evaluation.
- Do not commit unrelated files accidentally; review Git status before staging.

## 13. Current roadmap status

| Workstream | Status |
|---|---|
| V3 evidence ledger | Created; verification against artifacts remains necessary |
| Temporal expansion protocol | Planned; acquisition scope requires formalization |
| Expanded FIRMS and weather ingestion | Not started under this roadmap |
| Crop-source evaluation | Planned |
| Crop-aware ablation | Blocked on source evaluation and protocol |
| Prospective 2026 evaluation | Outcomes must accumulate and be verified |
| V3 paper structure | Planned parallel workstream |
| Numerical audit of V3 manuscript | To be performed after the relevant manuscript/results are assembled |
| V3 production replacement | Blocked on validation, reproducibility, and deployment gates |

## 14. Final guardrail

The objective is not merely to cover more states, collect more features, or produce a more complex model.

The objective is to determine whether a reproducible forecasting system provides useful, reliable, and operationally acceptable predictions under a defensible evaluation protocol.

**Geographic expansion alone is not sufficient evidence of improved forecasting.**


## Publication Scope Decision

- Publication strategy: **one research paper only**.
- Paper 1 is exclusively the Punjab and Haryana study, using its established V2 research design, experiments, and audited findings.
- V3 six-state expansion (Punjab, Haryana, Uttar Pradesh, Bihar, Madhya Pradesh, and Assam) is for StubbleAI project development and deployment, not a second paper.
- Crop-data evaluation, crop-aware experiments, and crop-related findings are part of the six-state project and deployment workstream, not a separate manuscript.
- Keep V3 experiments and artifacts documented independently. Do not mix their results into Paper 1's reported experiments or claims; use them only for appropriately supported context or future-work discussion.
- Preserve the existing Paper 1 manuscript and frozen V2 evidence unless a deliberate, evidence-supported correction is required.
