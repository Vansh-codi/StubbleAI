# StubbleAI V3: District-Day Feature Dataset Specification

**Status:** Design specification — initial V3 district ML dataset built and audited
**Date:** 16 October 2026
**Experiment:** V3.1 — +1-day district-level fire forecasting

## 1. Research Objective

Test whether grid-derived spatial information improves district-level next-day
fire forecasting while preserving the frozen V2 prediction unit, target
definition, and temporal evaluation protocol.

The primary experiment adds features derived from the 0.05-degree grid to a
district-day dataset. It does not replace the district-level target with a
cell-level target.

## 2. Locked Design Decisions

- Prediction unit: district-day.
- Primary fire-count source:
  `data/processed/v3/district_day_panel_005_centroid_fallback.csv`
- Primary grid source:
  `data/processed/v3/grid_day_panel_005_centroid_fallback_weather.csv`
- Weather source:
  `data/processed/v3/district_daily_weather_v3.csv`
- Primary assignment: 0.05-degree grid, centroid assignment with
  same-state largest-positive-area-overlap fallback.
- Initial forecast horizon: +1 day.
- Elevated target: `fire_count_t_plus_1d > 2`.
- Temporal split: 2023 training, 2024 validation, 2025 frozen test.
- Area-weighted fire counts remain a separate sensitivity experiment.
- Existing V1/V2 scripts, datasets, outputs, and research notes must not be
  modified by the V3 builder.

## 3. Input Validation

Before feature construction, validate:

1. Required input files and columns exist.
2. Dates parse successfully.
3. District-day keys (`state`, `district`, `date`) are unique.
4. Grid-day keys (`cell_id`, `date`) are unique.
5. The district panel has complete coverage for the 183 study dates:
   October-November 2023, 2024, and 2025.
6. Grid-to-district mappings are present and consistent with the primary
   panel's assignment.
7. Weather joins use exact `state`, `district`, and `date` keys.
8. Missing values are not silently converted to zero unless zero has a
   defined physical meaning, such as a zero-fire count.

## 4. Seasonal Boundaries and Temporal Alignment

Treat each October-November block as an independent seasonal block:

- 2023-10-01 through 2023-11-30
- 2024-10-01 through 2024-11-30
- 2025-10-01 through 2025-11-30

Construct historical features and future targets using exact calendar-date
joins, not unvalidated row shifts.

All lag and target dates must belong to the same seasonal block as the
prediction date. Never carry a lag or target across the November-to-October
gap.

A missing source date must remain distinguishable from a true zero-fire date.
The builder must fail validation if expected daily coverage is incomplete.

## 5. Base District-Day Features

Retain V2-compatible features where they can be constructed and validated
from V3 sources:

- Calendar: year, month, day, day of year, sine and cosine of annual
  seasonality.
- Current district fire state: fire count, FRP sum, FRP mean, FRP maximum,
  high-confidence fire count, day/night counts, and available brightness
  statistics.
- Weather: T2M, RH2M, WS2M, PRECTOTCORR.
- Historical fire-count and FRP lags at 1, 2, 3, 5, 7, and 14 days.
- Historical fire and FRP summaries following V2 definitions.

The builder must explicitly record any base feature that cannot be reproduced
from V3 sources. It must not silently substitute a different definition.

## 6. Grid-Derived District-Day Features

Aggregate the primary grid panel by `state`, `district`, and `date` to derive:

- Active-cell count: number of cells with `fire_count > 0`.
- Maximum cell fire count.
- Maximum cell FRP among cells with observed fire activity.
- A documented spatial concentration or dispersion measure.

Define zero-fire behavior explicitly. For example, active-cell count is zero
on a zero-fire district-day. FRP maxima for zero-fire days may remain missing
or use a documented sentinel-free convention; do not invent a physical FRP
value.

For the +1-day experiment, day-t grid-derived aggregates may be used because
day-t observations are assumed available when the forecast is issued.
Historical versions of each grid-derived feature must use exact calendar-date
joins at lags 1, 2, 3, 5, 7, and 14 days.

No grid-derived feature may use observations from after the prediction date.
Feature calculations and aggregation definitions must be recorded in the audit.

## 7. Target Construction

Create:

- `fire_count_t_plus_1d`
- `frp_sum_t_plus_1d`, if required for diagnostic comparison
- `target_elevated_t_plus_1d`, defined as
  `fire_count_t_plus_1d > 2`

The primary classification target uses the centroid/fallback district fire
count. Do not use fractional area-weighted counts as the primary target.

For the first experiment, require the historical features and +1-day target
only. Do not discard rows solely because longer-horizon targets are absent.

## 8. Weather Limitation

Weather is obtained at district representative coordinates and shared by
grid cells assigned to the same district and date. It therefore does not
provide within-district grid-level weather variation.

Document this limitation in the dataset audit. Per-cell weather extraction
may be considered in a later experiment, not silently introduced here.

## 9. Temporal Evaluation Protocol

- Train: 2023.
- Validation: 2024; select the classification threshold by maximizing
  elevated-class F1 over thresholds 0.01 to 0.99 in increments of 0.005.
- Test: 2025; evaluate once using the frozen validation-selected threshold.
- Classification threshold for the label: fire count strictly greater than 2.
- Do not use test results to choose features, models, or thresholds.

Use V2's training protocol as a reference, but implement V3 in separate
scripts and write outputs only to V3-specific paths.

## 10. Reconstruction Gate

Before claiming any benefit from grid-derived features:

1. Compare equivalent Punjab-Haryana district-day populations.
2. Compare base fire counts, weather, calendar features, and historical lags
   wherever definitions and source coverage are genuinely equivalent.
3. Verify exact lag and +1-day target alignment against the V3 source panel.
4. Report unmatched keys and differences caused by boundary versions,
   district coverage, or assignment methods.
5. Do not require every raw district count to match if source boundaries or
   assignment definitions differ; quantify and explain such differences.
6. Do not train or claim forecasting improvement until the reconstruction
   discrepancies are understood and the audit passes.

## 11. Required Dataset Audit

The separate V3 audit must check:

- Required columns and unique column names.
- Unique district-day keys and expected row counts.
- Complete seasonal date coverage.
- Historical lag alignment against exact source dates.
- Future target alignment against exact source dates.
- Explicit seasonal-block membership for every lag and target.
- No future-derived columns in the predictor set.
- Missing values, infinities, negative fire counts, and invalid aggregates.
- Target distribution and elevated-class prevalence.
- Grid aggregation consistency with the primary grid panel.
- Weather matching and row-count preservation.
- Train/validation/test year membership.
- Reproducibility metadata, input paths, and feature definitions.

## 12. Planned Separate Scripts and Outputs

Planned scripts:

- `scripts/v3/build_v3_district_dataset.py`
- `scripts/v3/audit_v3_district_dataset.py`
- `scripts/v3/evaluate_v3_baselines.py`
- `scripts/v3/train_v3_models.py`

Planned primary dataset:

- `data/processed/v3/v3_district_ml_dataset_1d.csv`

Planned audit report:

- `research/v3/v3_district_feature_dataset_audit.md`

All outputs must be V3-specific. Do not overwrite frozen V1/V2 files.

## 13. Interpretation Rules

- Passing data validation does not establish forecasting improvement.
- Better continuous-count allocation does not automatically imply better
  classification at the threshold of 2.
- Changes in district labels caused by spatial assignment must be distinguished
  from improvements due to new predictive features.
- Report baseline metrics before claiming the grid-derived features help.
- Keep the 0.10-degree and area-weighted variants as separate sensitivity
  experiments.

## 14. Implementation Status

The V3 district ML dataset has been built and audited against the checks documented in `research/v3/v3_district_feature_dataset_audit.md`. This does not establish forecasting improvement. Model training and V2 reconstruction must be tracked separately.


---

# Pre-Implementation Addendum — V3 Research Controls

**Revision date:** 17 October 2026
**Status:** Required design constraints before implementation

## A. Shared Dataset for Gate A and Gate B

The V3 builder must generate one full six-state district-day ML dataset:

`data/processed/v3/v3_district_ml_dataset_1d.csv`

Gate A must filter the same dataset to Punjab and Haryana for the V2-compatible reconstruction. Gate B must use the six-state population. Do not generate separate datasets through different construction code paths.

This ensures both experiments share feature definitions, date alignment and eligibility rules. Boundary-source and population differences must still be measured and documented.

## B. Reconstruction-Gate Provenance

The reconstruction comparison artifact will be:

`research/v3/v2_reconstruction_gate_comparison.csv`

Record, at minimum:

- `experiment_id`
- `attempt_id`
- `date_run`
- `reconstruction_method_version`
- `script_path`
- `script_hash` or Git commit when available
- `dataset_version`
- `state_scope`
- `split`
- `metric_name`
- `frozen_v2_value`
- `reconstructed_value`
- `absolute_difference`
- `row_count`
- `positive_label_count`
- `threshold`
- `comparison_status`
- `discrepancy_reason`
- `notes`

Every rerun must receive a distinct `attempt_id`. Preserve earlier attempts; do not silently replace an earlier result with a corrected run.

## C. Neighbor-Context Features Are Deferred

V3.1 initially tests within-district grid-derived features: active-cell count, maximum cell fire count, maximum cell FRP sum and maximum-cell share.

Neighboring-district and neighboring-cell fire activity are deliberately deferred to a separately specified follow-up experiment. This means V3.1 does not directly test whether neighbor activity helps predict fire emergence in a district with little or no current fire activity.

The omission must be acknowledged when interpreting results.

## D. Pre-Registration and Test-Set Discipline

Before accessing 2025 test metrics for a named experiment, record its research question, experiment ID, feature arms, target, eligible population, split, model-selection procedure, threshold-selection rule, metrics and code/dataset version.

The named experiments are:

1. Gate A — Punjab–Haryana V2-compatible reconstruction.
2. Gate B — Six-state base-only versus base-plus-grid ablation.
3. Neighbor-context follow-up — a separately specified experiment.

The 2025 test set may be evaluated at most once per named, pre-registered experiment. Do not repeatedly inspect test results while changing features, models or thresholds.

If a follow-up experiment is motivated by previously inspected 2025 results, label its test result **exploratory/non-confirmatory**. It must not be presented with the same evidentiary status as a genuinely pre-registered confirmatory result.

A corrected implementation must have a new attempt record. Changing code does not reset the evidentiary status if the change was motivated by prior test-set results.

## E. Required Implementation Order

1. Finalize and preserve this specification.
2. Build the single six-state dataset.
3. Run the independent dataset audit.
4. Complete and document Gate A.
5. Proceed to Gate B only after reconstruction discrepancies are understood.
6. Evaluate neighbor-context features later under a separate experiment specification.

The builder must not train models or inspect 2025 model performance.

## F. Research Claim Boundary

Successful data validation establishes data-quality evidence, not forecasting improvement. Gate A measures reconstruction fidelity and explains discrepancies. Gate B tests the contribution of grid-derived features on the defined six-state population. Neither result should be overstated beyond its actual comparison.

All outputs remain V3-specific. Frozen V1/V2 artifacts and raw source datasets must remain unchanged.
