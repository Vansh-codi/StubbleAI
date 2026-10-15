# V3 Grid Resolution and Boundary-Assignment Audit

**Status:** Completed diagnostic audit; resolution selection remains provisional  
**Study period:** October-November, 2023-2025  
**Pilot states:** Assam, Bihar, Haryana, Madhya Pradesh, Punjab, Uttar Pradesh  
**Reference:** Direct FIRMS point-to-district assignment using the current SOI district boundaries

## 1. Research question

How much of the observed difference between 0.10-degree and 0.05-degree grid-based district-day counts is associated with grid cells that cross district boundaries?

This audit evaluates district-assignment consistency. It does not evaluate fire-forecasting performance.

## 2. Data and method

- Seasonal FIRMS candidate observations were filtered to October and November in 2023, 2024, and 2025.
- Candidate detections were assigned to the six pilot states and their districts using point-in-polygon `within` joins.
- The direct assignment produced 256,903 detections across 248 districts.
- The comparison calendar contained 45,384 district-day combinations.
- Grid cells were state-scoped, preventing a cell in one state from being assigned to a district in another state.
- Grid-based district assignment used cell-centroid assignment, followed by a same-state largest-positive-area-overlap fallback for unresolved centroids.
- Geometric straddling was defined as positive-area overlap greater than 1 square metre with more than one district polygon within the same state.
- Both grid resolutions retained all 256,903 directly assigned detections.
- No files were written by the audit scripts.

## 3. Overall results

| Metric | 0.10-degree grid | 0.05-degree grid |
|---|---:|---:|
| State-scoped occupied cells | 5,462 | 14,941 |
| Geometrically straddling cells | 2,266 (41.49%) | 3,645 (24.40%) |
| Detections in straddling cells | 108,396 (42.19%) | 61,195 (23.82%) |
| Detections in non-straddling cells | 148,507 (57.81%) | 195,708 (76.18%) |
| Overall exact district-day match | 79.93% | 85.31% |
| Overall mean absolute error | 0.5095 | 0.2935 |
| Total absolute district-day count difference | 23,124 | 13,320 |
| Maximum absolute district-day difference | 75 | 49 |
| Signed total difference | 0 | 0 |
| Detections retained | 256,903 / 256,903 | 256,903 / 256,903 |

The total absolute difference decreased by 9,804, approximately 42.4%, at 0.05 degrees. Exact district-day agreement increased by 5.38 percentage points.

## 4. Non-straddling versus straddling cells

When the reference counts and grid-assigned counts were recalculated using the same subset of detections:

| Metric | 0.10-degree grid | 0.05-degree grid |
|---|---:|---:|
| Non-straddling detections | 148,507 | 195,708 |
| Non-straddling exact-match rate | 100.00% | 100.00% |
| Non-straddling total absolute error | 0 | 0 |
| Straddling detections | 108,396 | 61,195 |
| Straddling-subset total absolute error | 23,124 | 13,320 |

Under the tested assignment method, all observed district-day discrepancies arise from detections in geometrically straddling cells. Non-straddling-cell detections are assigned to the same district as the direct reference.

The finer grid has fewer geometrically straddling occupied cells as a proportion of all occupied cells and a smaller share of detections in those cells. This provides strong evidence that the observed resolution advantage is associated with boundary-related assignment error.

This result is conditional on the current boundary dataset, grid origin, cell definition, and assignment method. It does not establish that 0.05 degrees is better for forecasting or that the same relationship holds under every assignment rule.

## 5. Punjab diagnostic

| Metric | 0.10-degree grid | 0.05-degree grid |
|---|---:|---:|
| Exact district-day match | 54.10% | 61.39% |
| Mean absolute error | 2.1169 | 1.1903 |
| Total absolute error | 8,910 | 5,010 |
| Detections in straddling cells | 48,505 | 29,267 |
| Occupied straddling cells | 240 | 531 |

Punjab's total absolute error decreased by 3,900, approximately 43.8%, at 0.05 degrees.

Districts with large absolute errors at 0.10 degrees included Sangrur, Bathinda, Barnala, Sri Muktsar Sahib, Moga, Mansa, Patiala, Kapurthala, Firozpur, and Faridkot. At 0.05 degrees, Sangrur and Mansa remained among the largest contributors.

These results are consistent with boundary-related assignment error, but they do not by themselves establish that Punjab has the highest boundary density among the six states or that district geometry alone explains its error.

## 6. Interpretation and limitations

1. The 0.05-degree grid reproduces direct district-day counts more closely under the tested centroid-plus-overlap-fallback assignment method.
2. The perfect agreement for non-straddling detections localizes the observed discrepancies to straddling cells in this audit.
3. The finer grid also creates more occupied cells and has previously shown greater spatial fragmentation and lower recurrence across seasons.
4. District-day agreement is an assignment diagnostic, not a forecasting metric.
5. Area-weighted allocation assumes a uniform distribution of activity within a cell. Actual fire detections may cluster on agricultural land or other specific land covers, so geometric area alone may misallocate counts.
6. Historical detection-density weighting may use observed spatial concentration, but must estimate allocation proportions from a separate training period and evaluate on held-out data.
7. The current findings are tied to the SOI boundary dataset and should not be interpreted as proof that the boundaries are authoritative or that all spatial uncertainty has been resolved.

## 7. Next experiments

### Experiment A: existing reference assignment

Retain the current 0.10-degree centroid-plus-overlap-fallback result as the comparison baseline.

### Experiment B: area-weighted allocation at 0.10 degrees

Allocate each straddling cell's detection count among intersecting districts in proportion to the cell's intersection area with each district. Evaluate against the direct point-to-district baseline. Do not assume area weighting is superior before measuring its error.

### Experiment C: historical detection-density-weighted allocation

Estimate district allocation proportions from a separate historical training period. For example, use 2023-2024 to allocate 2025 counts, with all proportions estimated without using the evaluated year's observations. Report the evaluation period and the treatment of cells with insufficient historical detections.

### Cross-state geometry audit

For both resolutions, calculate the proportion of each state's total area covered by geometrically straddling cells. Use a clearly defined denominator and avoid treating the fraction of occupied cells as the same quantity as the fraction of state area.

### Conditional experiment D

Evaluate alternative allocation methods at 0.05 degrees only after the 0.10-degree area-weighted experiment has been assessed.

## 8. Decision

**Grid resolution is not frozen.**

The current evidence supports treating 0.05 degrees as a finer-resolution candidate that reduces boundary-related assignment error under the tested method. The 0.10-degree grid remains a valid comparison baseline.

The resolution decision should consider assignment error, spatial fragmentation, recurrence, computational cost, and ultimately leakage-safe forecasting performance. No forecasting improvement is claimed from this audit.


## 9. Corrected same-state area-weighted audit

**Status:** Completed diagnostic; read-only experiment  
**Correction:** Cross-state cell-district intersections were excluded before intersection areas were calculated and normalized.

### 9.1 Cross-state intersection audit

| Metric | 0.10-degree grid | 0.05-degree grid |
|---|---:|---:|
| Intersection rows before same-state filter | 8,548 | 19,435 |
| Intersection rows after same-state filter | 8,111 | 18,848 |
| Cross-state rows removed | 437 (5.11%) | 587 (3.02%) |
| Occupied cells with allocation weights | 5,462 | 14,941 |

Rows removed, grouped by cell state:

| State | 0.10-degree | 0.05-degree |
|---|---:|---:|
| Assam | 0 | 0 |
| Bihar | 39 | 46 |
| Haryana | 103 | 169 |
| Madhya Pradesh | 72 | 85 |
| Punjab | 71 | 128 |
| Uttar Pradesh | 152 | 159 |
| **Total** | **437** | **587** |

These are intersection-row counts, not detection counts or area percentages. The cell-state grouping identifies which state's cells were involved in cross-state intersections; it does not, by itself, quantify the distortion to a state's final area estimate.

### 9.2 Allocation normalization and conservation

The corrected procedure filtered cell-district intersections to matching states before calculating intersection areas. Area weights were then normalized within each state-scoped cell.

| Check | 0.10-degree grid | 0.05-degree grid |
|---|---:|---:|
| Maximum absolute weight-sum error | 2.22e-16 | 2.22e-16 |
| Direct-assigned detections | 256,903 | 256,903 |
| Area-allocated detections | 256,903.0 | 256,903.0 |
| Total conservation difference | 0 | 0 |
| District-day rows | 45,384 | 45,384 |

The checks support correct weight normalization and conservation for both resolutions. The reported district-day metrics below use the full 248-district by 183-date calendar, including zero-count days.

### 9.3 Corrected area-weighted district-day comparison

| Metric | 0.10-degree grid | 0.05-degree grid |
|---|---:|---:|
| Exact district-day match | 57.78% | 66.05% |
| Mean absolute error | 0.42746 | 0.26096 |
| Total absolute error | 19,399.81 | 11,843.58 |
| Maximum absolute error | 40.36 | 25.89 |
| Signed total difference | Approximately 0 | Approximately 0 |

At 0.05 degrees, area-weighted allocation reduced MAE by approximately 38.9% and total absolute error by approximately 39.0% relative to 0.10 degrees.

At 0.05 degrees, area-weighted MAE (0.26096) and total absolute error (11,843.58) were lower than the previously measured centroid-plus-fallback values (MAE 0.2935; total absolute error 13,320). However, area weighting creates fractional district counts, so exact-match rates are not directly comparable with those from integer-count assignment.

Area weighting assumes uniform distribution within each grid cell. These results do not establish superior forecasting performance and do not change the planned primary assignment method for the first V3 panel.

### 9.4 Corrected occupied straddling-cell area shares

The following percentages use the union of occupied cells that intersect multiple districts within the same state, clipped to the state's boundary, divided by that state's union geometry area.

| State | 0.10-degree grid | 0.05-degree grid |
|---|---:|---:|
| Assam | 17.2585% | 3.7768% |
| Bihar | 23.2119% | 6.6226% |
| Haryana | 40.4402% | 19.7814% |
| Madhya Pradesh | 22.9114% | 8.2939% |
| Punjab | 49.0455% | 27.6232% |
| Uttar Pradesh | 40.2777% | 17.4330% |

These corrected values supersede the provisional area-share estimates from the earlier experiment. Punjab has the largest estimated occupied straddling-cell area share at both resolutions under this definition. This geometric metric is not a measure of fire density or forecasting difficulty.

### 9.5 Decision and scope closure

The same-state intersection bug is corrected in this audit. Both grid resolutions conserve the directly assigned detections, and 0.05 degrees has lower district-day assignment error than 0.10 degrees under both the centroid-plus-fallback comparison and this area-weighted diagnostic.

The area-weighted experiment is closed for this phase. Historical detection-density weighting and further allocation variants are deferred. The first V3 panel will proceed with 0.05 degrees as the provisional primary grid and retain 0.10 degrees as a sensitivity baseline. The V2 reconstruction gate must use a predefined comparison against the established assignment-error baseline rather than requiring an exact match.

No forecasting improvement is claimed by this audit.
