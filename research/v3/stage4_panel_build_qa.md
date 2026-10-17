# StubbleAI V3 — Stage 4 Panel Build QA

**Audit date:** 2026-10-15  
**Status:** Core post-build checks passed  
**Study period:** 1 October–30 November, 2023–2025  
**Study dates:** 183

## 1. Inputs

- FIRMS seasonal candidate records: 463,212
- Detections assigned to the six pilot states: 256,903
- Outside or unmatched candidates: 206,309
- Boundary source: SOI District Boundary shapefile
- Primary grid: state-scoped 0.05-degree cells
- Sensitivity protocol: state-scoped 0.10-degree grid
- Primary assignment: centroid containment, with same-state largest-positive-overlap fallback
- Area-weighted allocation: secondary count-only artifact

## 2. Build results

| Metric | Primary 0.05-degree | Sensitivity 0.10-degree |
|---|---:|---:|
| Occupied state-scoped cells | 31,658 | 8,368 |
| Grid-day rows | 5,793,414 | 1,531,344 (planned; no 0.10-degree grid-day CSV emitted) |
| District-day rows | 45,384 | 45,384 |
| Detections reported by builder | 256,903 | 256,903 |
| Area-weighted total | 256,903.0 | 256,903.0 |
| Maximum absolute area-weight-sum error | 2.220446049250313e-16 | 2.220446049250313e-16 |

### Primary 0.05-degree assignment methods

- Centroid: 29,427 cells
- Largest-overlap fallback: 2,231 cells

### Sensitivity 0.10-degree assignment methods

- Centroid: 7,376 cells
- Largest-overlap fallback: 992 cells

## 3. Saved artifacts

- `data/processed/v3/grid_day_panel_005_centroid_fallback.csv`
- `data/processed/v3/district_day_panel_005_centroid_fallback.csv`
- `data/processed/v3/district_day_panel_010_centroid_fallback_sensitivity.csv`
- `data/processed/v3/district_day_fire_count_005_area_weighted_secondary.csv`
- `data/processed/v3/stage4_panel_build_summary.json`

## 4. Independent post-build QA

The following checks were run against the saved CSV files:

- All four CSV row counts matched expectations.
- All four CSVs covered 2023-10-01 through 2025-11-30.
- No missing key values were found.
- No negative count values were found.
- Count totals were 256,903 for each artifact.
- Zero-fire rows had no non-null fire mean/max fields in the three standard metric panels.
- Primary grid: 5,793,414 rows; no repeated key hashes for `(cell_id, date)`.
- Each of the three district-day artifacts had 45,384 rows and zero duplicate `(state, district, date)` keys.
- The primary grid had 5,662,645 zero-fire rows.
- The primary district panel had 26,650 zero-fire rows.
- The sensitivity district panel had 26,642 zero-fire rows.
- The area-weighted artifact had 24,779 zero-count rows.

**Uniqueness caveat:** primary grid uniqueness was screened using 64-bit hashes of `(cell_id, date)`. No repeated hashes were found. This is strong screening evidence, but not an exact comparison of every original key pair.

## 5. Interpretation and limitations

This audit establishes that the saved panel artifacts pass the listed structural and count checks. It does not establish predictive skill, forecast improvement, or historical V2 reproduction.

The area-weighted artifact contains fractional fire counts. Its exact-match rate must not be compared directly with integer-valued counts without an appropriate evaluation design.

The 0.10-degree grid-day row count above is the builder's planned expansion count; the current build emitted the district-day sensitivity panel, not a separate 0.10-degree grid-day CSV.

## 6. Next research gate

Before training the first V3 forecasting baseline:

1. Reconcile the grid and district count totals directly by state and date.
2. Verify the complete study-date calendar and expected district coverage.
3. Compare the primary district-day panel with the historical V2 audit where definitions and coverage are comparable.
4. Document any differences before model training.
5. Keep V1 and the frozen V2 research baseline unchanged.


## 7. State-date reconciliation

**Result: PASS ? primary grid and district panels reconcile.**

- State-date keys checked: 1,098 (6 states ? 183 dates).
- Unmatched state-date keys: 0.
- Nonzero grid-versus-district fire-count differences: 0.
- Maximum absolute difference: 0.
- District totals matched for all six pilot states.
- Unique districts: 248.
- Study dates: 183.
- Expected and actual district-day rows: 45,384.
- Duplicate `(state, district, date)` keys: 0.
- Districts with incomplete date coverage: 0.
- Date range: 2023-10-01 through 2025-11-30.

This establishes internal aggregation consistency between the primary grid-day
and district-day panels. It does not establish historical V2 reproduction or
forecasting skill.
