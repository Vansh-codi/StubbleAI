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

