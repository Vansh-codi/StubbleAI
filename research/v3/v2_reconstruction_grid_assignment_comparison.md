\# V3 Research Note: V2 Reconstruction and Grid-Assignment Comparison

**Status:** Comparison and binary-target sensitivity completed; historical key-universe reconciliation remains a limitation



\*\*Date:\*\* 15 October 2026  

\*\*Status:\*\* Preliminary comparison completed; binary-target sensitivity test pending  

\*\*Scope:\*\* Punjab and Haryana, October–November 2023–2025



\## 1. Objective



Determine whether differences between frozen V2 and V3 district-day fire counts arise primarily from boundary-version differences or from V3's grid-cell-to-district assignment method.



The comparison uses the same seasonal detections inside the original Punjab–Haryana study boundary wherever specified. Frozen V2 data and existing V3 outputs remain unchanged.



\## 2. Boundary-version comparison



The original V2 boundary dataset and the newer Survey of India (SOI) district boundaries were compared using the same detections.



\- Seasonal detections inside the old boundary: 104,259.

\- Detections assigned to the same district by both boundary datasets: 104,249.

\- Detections assigned to different districts: 10.

\- All 10 differences occurred in Haryana, mapping from Hisar under the old boundaries to Bhiwani under the SOI boundaries.

\- No unassigned detections or multiple spatial matches were reported.



\*\*Interpretation:\*\* Boundary-version differences are small in this particular point-assignment comparison. They do not explain the larger district-level redistribution observed under V3.



\## 3. Controlled grid-assignment comparison



The same 104,259 seasonal detections were compared under direct SOI district assignment and V3's 0.05° grid-cell assignment.



| Metric | Result |

|---|---:|

| Detections under each method | 104,259 |

| District-day keys in comparison | 5,121 |

| Keys missing on either side | 202 |

| Exactly equal rows | 2,827 |

| Changed rows | 2,294 |

| Sum of absolute count differences | 5,966 |

| Maximum absolute count difference | 49 |



Both methods conserve the same total number of detections, but their district-day distributions differ substantially.



\### Largest district-level differences



| District | Direct total | Grid-assigned total | Net difference |

|---|---:|---:|---:|

| Sangrur | 14,309 | 13,842 | −467 |

| Mansa | 5,600 | 5,970 | +370 |

| Moga | 6,402 | 6,698 | +296 |

| Ferozepur | 8,984 | 8,988 | +4 |

| Patiala | 5,093 | 5,236 | +143 |

| Amritsar | 4,188 | 3,999 | −189 |

| Jalandhar | 2,525 | 2,321 | −204 |



The complete comparison output should be retained separately if a district-by-district audit is required.



\## 4. Frozen V2 comparisons



\### Frozen V2 versus direct SOI assignment



\- Frozen V2 total: 104,259.

\- Direct SOI total: 104,259.

\- Exactly equal rows: 8,221.

\- Changed rows: 14.

\- Sum of absolute count differences: 20.

\- Maximum absolute count difference: 3.



This supports the conclusion that the newer boundary dataset has a small effect on the seasonal district-day counts in this comparison.



\### Frozen V2 versus V3 grid assignment



\- Frozen V2 total: 104,259.

\- V3 grid-assigned total: 104,259.

\- Exactly equal rows: 5,929.

\- Changed rows: 2,306.

\- Sum of absolute count differences: 5,984.

\- Maximum absolute count difference: 49.



These results are consistent with grid-cell assignment being the dominant source of the observed district-level redistribution.



\*\*Comparison limitation:\*\* The reported key counts differ because the comparisons do not all use the same zero-filled key universe. The 202 missing keys in the direct-versus-grid comparison must be audited before treating its row-level statistics as a complete district-day panel comparison.



\## 5. Interpretation and limitations



1\. Detection-count conservation confirms that the assignment process does not lose or create detections in the controlled comparison.

2\. Count conservation does not imply district-level equivalence.

3\. Grid cells can cross district boundaries. Assigning an entire cell to its centroid district, or using a largest-overlap fallback, can shift multiple detections together.

4\. These results establish a substantial difference in spatial allocation; they do not establish that grid assignment improves or harms forecasting performance.

5\. The binary emerging-fire target (`fire\_count > 2`) has not yet been compared between the two assignment methods.

6\. No claim of improved predictive performance should be made until the baseline evaluation is completed.



\## 6. Next steps



1\. Reconcile the full zero-filled district-day key universe for direct and grid assignment.

2\. Quantify binary-target label flips for the established forecast horizons (+1, +2, +3, +5, +7 days).

3\. Document the 21 additional V3 records found in the seasonal source comparison.

4\. Complete the V2 reconstruction gate.

5\. Proceed to the first six-state classical baseline using the frozen temporal split: 2023 training, 2024 validation, and 2025 testing.



\## 7. Preservation



\- Frozen V2 remains unchanged.

\- V1 operational pipeline remains unchanged.

\- Existing V3 panel outputs remain unchanged.

\- This note records a diagnostic comparison, not a model-performance result.




## 8. Binary-Target Sensitivity: Direct District Assignment vs Grid Assignment

**Date:** 15 October 2026  
**Target:** `fire_count > 2` on the future target day  
**Comparison:** Same seasonal detections, district universe, and seasonal date range; direct SOI district assignment versus 0.05-degree grid assignment.

### Results

| Forecast horizon | District-days compared | Direct positives | Grid positives | Label flips | Flip rate |
|---|---:|---:|---:|---:|---:|
| +1 day | 8,100 | 3,328 | 3,326 | 184 | 2.272% |
| +2 days | 7,965 | 3,297 | 3,296 | 181 | 2.272% |
| +3 days | 7,830 | 3,274 | 3,274 | 178 | 2.273% |
| +5 days | 7,560 | 3,220 | 3,220 | 174 | 2.302% |
| +7 days | 7,290 | 3,132 | 3,132 | 168 | 2.305% |

The full seasonal panel contains 45 districts, 183 seasonal dates, and 8,235 district-day rows per method before horizon-specific exclusions.

### Interpretation

1. Grid assignment changes the binary target labels, not merely the numerical district fire counts.
2. Across the tested horizons, approximately 2.27–2.31% of comparable district-day labels differ.
3. Overall positive-label counts remain close. At +3, +5, and +7 days, the positive counts are identical despite individual label flips.
4. Similar aggregate class counts therefore do not establish that the underlying labels are equivalent.
5. These results measure sensitivity to spatial assignment. They do not show that direct assignment or grid assignment produces better forecasts.
6. The number of comparable district-days decreases with horizon because future target dates must remain inside the defined seasonal date set.

### Research implication

Retain this sensitivity result when evaluating the first V3 baseline. Report the chosen spatial assignment method explicitly and avoid interpreting performance differences without considering changes in target labels.

### Next step

Proceed with the frozen V2 reconstruction gate and first six-state classical baseline using 2023 for training, 2024 for validation, and 2025 for testing. Evaluate predictive performance before making claims about improvement.

