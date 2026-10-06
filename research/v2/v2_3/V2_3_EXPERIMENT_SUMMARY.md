# V2.3 Experiment Summary

## 1. Purpose

V2.3 evaluates additional feature families intended to improve the frozen V2 multi-horizon fire-occurrence forecasting system.

Two controlled experiments were conducted:

1. Temporal trend and acceleration features.
2. Spatial neighbor context features derived from validated district GIS boundaries.

The purpose of V2.3 is research evaluation, not replacement of the frozen V2 benchmark.

---

## 2. Research Status

**V2.3 is complete as an experimental feature study.**

The two experiments produced different outcomes:

- **Temporal trend features: REJECTED**
- **Spatial neighbor context: CONDITIONALLY ACCEPTED AS AN EXPERIMENTAL FEATURE FAMILY**

Neither experiment replaces the official frozen V2 benchmark.

The spatial feature family is retained for further investigation because it produced meaningful improvements at selected horizons, particularly +5 days, while showing strong dependence on geography, season, and current fire regime.

---

## 3. Frozen V2 Baseline

All V2.3 experiments were evaluated against the frozen V2 Random Forest benchmark.

The official V2 protocol uses:

- 2023 training
- 2024 validation
- 2025 untouched final test
- five forecast horizons: +1, +2, +3, +5, +7 days
- fire-count target threshold of greater than 2 for Elevated
- Random Forest with 400 trees
- `min_samples_leaf=2`
- `random_state=42`
- no class weighting
- validation-based F1 threshold selection

The frozen V2 benchmark remains the reference point for all subsequent experiments.

---

## 4. Experiment 1 — Temporal Trend Features

### Objective

Test whether short-term changes, ratios, slopes, and acceleration in fire activity and FRP provide additional predictive information beyond the existing V2 lag and rolling features.

### Added feature family

Sixteen temporal features were introduced, including:

- fire changes over 1, 3, and 7 days
- FRP changes over 1, 3, and 7 days
- recent-to-window fire ratios
- recent-to-window FRP ratios
- fire trends
- FRP trends
- fire acceleration
- FRP acceleration

### Result

The trend experiment did not improve the frozen V2 model consistently.

2025 F1:

| Horizon | Frozen V2 | Trend | Delta |
|---|---:|---:|---:|
| +1d | 0.7754 | 0.7663 | -0.0091 |
| +2d | 0.7513 | 0.7498 | -0.0015 |
| +3d | 0.7305 | 0.7228 | -0.0077 |
| +5d | 0.7303 | 0.7287 | -0.0016 |
| +7d | 0.7085 | 0.6933 | -0.0152 |

The experiment also showed lower ROC-AUC and PR-AUC across the evaluated horizons.

### Decision

**REJECTED FOR PROMOTION INTO THE OFFICIAL V2 MODEL.**

The negative result is retained as part of the research record.

The result does not prove that temporal dynamics are useless. It shows that this particular handcrafted trend feature family did not provide sufficient additional predictive value under the current V2 Random Forest protocol.

Full details are recorded in:

`trend/TREND_EXPERIMENT.md`

---

## 5. Experiment 2 — Spatial Neighbor Context

### Objective

Test whether neighboring-district fire activity provides useful information for forecasting fire occurrence in the target district.

### GIS foundation

The spatial experiment used validated Punjab and Haryana district geometries.

The boundary dataset contains:

- 45 districts
- 23 Punjab districts
- 22 Haryana districts
- EPSG:4326 coordinate reference system
- valid Polygon and MultiPolygon geometries

Adjacency was constructed using polygon boundary touching.

The resulting graph contains:

- 208 directed adjacency edges
- 104 unique undirected district pairs
- symmetric relationships
- no district without a neighbor
- all 45 ML districts successfully mapped to GIS districts

The only naming mismatch required an explicit mapping:

`S A S Nagar` ? `S.A.S Nagar`

No ML dataset district name was changed.

### Spatial features

Eight spatial features were added:

- `neighbor_fire_current`
- `neighbor_fire_lag_1d`
- `neighbor_fire_lag_3d`
- `neighbor_fire_lag_7d`
- `neighbor_frp_current`
- `neighbor_frp_lag_1d`
- `neighbor_frp_lag_3d`
- `neighbor_frp_lag_7d`

Neighbor values were constructed using calendar-date-aligned lookups from the complete district fire panel.

The resulting experimental ML dataset contains 5,400 rows and no missing spatial features after merging with the V2 ML dataset.

### Aggregate results

| Horizon | Frozen V2 F1 | Spatial F1 | Delta |
|---|---:|---:|---:|
| +1d | 0.7754 | 0.7696 | -0.0057 |
| +2d | 0.7513 | 0.7490 | -0.0023 |
| +3d | 0.7305 | 0.7309 | +0.0004 |
| +5d | 0.7303 | 0.7390 | +0.0087 |
| +7d | 0.7085 | 0.7013 | -0.0072 |

The strongest aggregate result was +5 days.

At +5 days:

- F1 increased from 0.7303 to 0.7390
- precision increased from 0.6507 to 0.6870
- ROC-AUC increased from 0.8581 to 0.8634
- PR-AUC increased from 0.8022 to 0.8188

At +7 days, emerging-event recall improved from approximately 62.67% to 67.28%, despite a decrease in overall F1.

### Decision

**CONDITIONALLY ACCEPTED AS AN EXPERIMENTAL FEATURE FAMILY.**

Spatial context is not promoted as the universal replacement for the frozen V2 feature set.

Instead, it is retained for additional controlled research.

Full details are recorded in:

`spatial/SPATIAL_EXPERIMENT.md`

---

## 6. What V2.3 Taught Us

### 6.1 Temporal trends are not automatically useful

Existing fire-history features already capture substantial temporal information.

Simple handcrafted changes, ratios, slopes, and acceleration did not produce consistent gains.

Future temporal research should therefore focus on more targeted or interaction-aware representations rather than simply adding more trend variables.

### 6.2 Spatial information contains useful signal

Neighboring-district activity provides additional predictive information at some horizons.

The strongest aggregate evidence occurred at +5 days.

However, the improvement is not universal.

### 6.3 Spatial context appears particularly relevant to continuation and emerging activity

The spatial model maintained strong continuation performance in already active districts and improved emerging-event recall at several horizons.

This suggests that spatial information may capture propagation or regional synchronization effects.

This interpretation remains observational and should not be treated as causal evidence.

### 6.4 Geography matters

The Punjab and Haryana results remain substantially different.

This indicates that model performance depends on regional fire dynamics and/or data characteristics.

The difference should not automatically be described as model bias without further investigation.

### 6.5 Current fire regime matters

Spatial performance varies strongly between:

- districts with no current fire activity
- districts with low activity
- districts with already elevated activity

The model is strongest when fire activity is already elevated.

Therefore, future research should explicitly consider fire-regime-conditioned behavior.

### 6.6 Seasonality matters

Performance changes substantially between early, middle, and late season.

The +5-day spatial improvement is particularly associated with earlier-season performance, while late-season performance remains difficult.

This means aggregate seasonal metrics alone are insufficient.

---

## 7. Research Integrity

V2.3 followed the frozen research protocol.

The following were preserved:

- frozen V2 ML dataset
- frozen V2 benchmark results
- frozen calibration results
- V1 operational system
- 2025 final test period

No 2025 test observations were used for model retraining or feature selection.

The experiments were evaluated against the frozen V2 baseline rather than replacing it.

Negative results are retained rather than discarded.

---

## 8. Limitations

V2.3 remains limited to Punjab and Haryana.

The spatial relationships are based on district adjacency rather than physical transport or meteorological connectivity.

Equal-weight neighbor aggregation may not represent real fire propagation mechanisms.

District-level aggregation can hide substantial subdistrict variation.

FIRMS active-fire detections are observations of thermal anomalies and should not automatically be interpreted as confirmed agricultural-residue burning.

The experiments do not establish causal relationships between neighboring fires, air pollution, or agricultural practices.

---

## 9. V2.4 Research Questions

The next research stage should focus on whether the useful spatial signal can be made more robust without simply increasing feature count.

Candidate questions include:

1. Can spatial and temporal information be combined through controlled interaction features?
2. Do weighted neighbors outperform equal-weight neighbors?
3. Are spatial features more useful at particular forecast horizons?
4. Can spatial context improve emerging-event detection without substantially increasing false alarms?
5. Which spatial features provide independent predictive value?
6. Does performance improve when analysis is conditioned on fire regime?
7. Can geographic generalization be improved while preserving the frozen benchmark?

These questions should be evaluated through controlled ablation experiments.

---

## 10. Path Toward India-Wide Expansion

V2.3 is intentionally not the India-wide expansion.

The current Punjab-Haryana environment is being used as a controlled research laboratory.

The eventual national architecture should support:

`India ? State ? District ? spatial grid`

and integrate multiple data families, including:

- satellite fire detections
- meteorological variables
- air-quality observations
- land-use and crop context
- atmospheric information
- emissions estimation

The eventual emissions and air-quality layers must remain scientifically distinct from fire detection.

Observed pollution should not automatically be attributed causally to detected fires.

Estimated emissions should be clearly labelled as estimates and should document their assumptions.

The national expansion should occur only after the core forecasting methodology has been sufficiently validated.

---

## 11. Recommended Research Sequence

The recommended progression is:

**V2.3**
Feature-family research

?

**V2.4**
Improved forecasting representation and controlled spatial-temporal experiments

?

**V2.5**
Robustness, ablation, verification, and generalization

?

**V3.0**
India-wide expansion and multi-source environmental intelligence

This prevents premature expansion and preserves scientific interpretability.

---

## 12. Reproducibility

V2.3 trend experiment:

- `build_v2_3_trend_features.py`
- `train_v2_3_trend_experiment.py`
- `analyze_v2_3_trend_diagnostics.py`
- `trend/TREND_EXPERIMENT.md`

V2.3 spatial experiment:

- `build_v2_3_spatial_features.py`
- `train_v2_3_spatial_experiment.py`
- `analyze_v2_3_spatial_diagnostics.py`
- `analyze_v2_3_spatial_fire_regimes.py`
- `spatial/SPATIAL_EXPERIMENT.md`

All V2.3 artifacts are stored under:

`research/v2/v2_3/`

---

## 13. Final V2.3 Decision

V2.3 is complete as a feature-family investigation.

**Trend features: REJECTED.**

**Spatial context: CONDITIONALLY ACCEPTED FOR FURTHER RESEARCH.**

**Official V2 baseline: UNCHANGED.**

The next step is V2.4 research design, not India-wide expansion.

---

## 14. Checkpoint Policy

V2.3 artifacts must remain reproducible and must not overwrite frozen V2 benchmark or calibration checkpoints.

Any V2.4 experiment must create separate artifacts and must be compared against the same frozen V2 reference.

The official model should not be changed solely because an experimental feature family performs better at one horizon.

Promotion requires evidence across the relevant horizons, diagnostics, robustness checks, and verification stages.
