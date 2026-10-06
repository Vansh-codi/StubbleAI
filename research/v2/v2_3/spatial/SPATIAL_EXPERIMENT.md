\# V2.3 Spatial Context Experiment



\## 1. Purpose



This experiment evaluates whether spatial neighbor context improves the V2 district-level fire forecasting system.



The experiment extends the frozen V2 machine-learning dataset with spatial features derived from validated district adjacency relationships between Punjab and Haryana.



The experiment is diagnostic/research work and does not replace the frozen V2 benchmark.



\---



\## 2. Research Status



\*\*Status: CONDITIONALLY ACCEPTED AS AN EXPERIMENTAL FEATURE FAMILY\*\*



Spatial features are not adopted as the universal replacement for the frozen V2 model.



The strongest aggregate improvement occurs at the +5 day horizon. The +7 day horizon shows improved emerging-fire detection, but overall F1 decreases because of reduced precision.



The results indicate that spatial context provides horizon- and regime-dependent information rather than a universally beneficial feature family.



\---



\## 3. Frozen V2 Baseline



The official V2 benchmark remains unchanged.



Temporal protocol:



\- Training: 2023

\- Validation: 2024

\- Final test: 2025

\- No tuning on the 2025 test period

\- Random Forest: 400 trees

\- `min\_samples\_leaf=2`

\- `random\_state=42`

\- No class weighting

\- Validation threshold selected by maximum F1

\- Target: `fire\_count\_t\_plus\_H > 2`



All frozen V2 benchmark and calibration artifacts remain untouched.



\---



\## 4. Spatial GIS Foundation



District geometries were validated using:



`districts\_punjab\_haryana.geojson`



Coverage:



\- Punjab: 23 districts

\- Haryana: 22 districts

\- Total: 45 districts



CRS:



`EPSG:4326`



Geometry validation:



\- 45 valid district geometries

\- 0 empty geometries

\- 39 Polygon geometries

\- 6 MultiPolygon geometries



Adjacency was defined using polygon `touches`.



The resulting graph contains:



\- 208 directed adjacency edges

\- 104 unique undirected district pairs

\- 45 mapped districts

\- 0 districts without neighbors



One naming mismatch was explicitly handled:



`S A S Nagar` in the ML dataset corresponds to `S.A.S Nagar` in the GIS dataset.



The ML dataset naming was not changed.



\---



\## 5. Spatial Features



Eight spatial features were added:



1\. `neighbor\_fire\_current`

2\. `neighbor\_fire\_lag\_1d`

3\. `neighbor\_fire\_lag\_3d`

4\. `neighbor\_fire\_lag\_7d`

5\. `neighbor\_frp\_current`

6\. `neighbor\_frp\_lag\_1d`

7\. `neighbor\_frp\_lag\_3d`

8\. `neighbor\_frp\_lag\_7d`



Neighbor values were calculated using the validated district adjacency graph.



Calendar-date alignment was used for neighbor fire/FRP lookups rather than positional dataframe shifting.



The frozen `ml\_dataset\_v2.csv` was not modified.



Spatial dataset:



`ml\_dataset\_v2\_3\_spatial.csv`



Dimensions:



`5400 × 59`



All 5400 ML rows contain complete spatial features.



\---



\## 6. Model Training Protocol



The spatial experiment uses the same protocol as the frozen V2 benchmark.



Temporal split:



\- 2023 → training

\- 2024 → validation

\- 2025 → final test



For each horizon:



\- +1 day

\- +2 days

\- +3 days

\- +5 days

\- +7 days



The model is:



`RandomForestClassifier`



Configuration:



\- `n\_estimators=400`

\- `min\_samples\_leaf=2`

\- `random\_state=42`

\- `n\_jobs=-1`



Categorical encoding uses the same pandas one-hot encoding methodology as V2.



Threshold candidates:



`np.linspace(0.01, 0.99, 199)`



The threshold maximizing validation F1 is selected and then frozen for 2025 evaluation.



No 2025 test tuning was performed.



\---



\## 7. Aggregate Results



| Horizon | V2 F1 | Spatial F1 | Δ F1 | V2 ROC-AUC | Spatial ROC-AUC | V2 PR-AUC | Spatial PR-AUC |

|---|---:|---:|---:|---:|---:|---:|---:|

| +1d | 0.7754 | 0.7696 | -0.0057 | 0.8953 | 0.8934 | 0.8582 | 0.8568 |

| +2d | 0.7513 | 0.7490 | -0.0023 | 0.8795 | 0.8786 | 0.8369 | 0.8374 |

| +3d | 0.7305 | 0.7309 | +0.0004 | 0.8683 | 0.8688 | 0.8193 | 0.8193 |

| +5d | 0.7303 | 0.7390 | +0.0087 | 0.8581 | 0.8634 | 0.8022 | 0.8188 |

| +7d | 0.7085 | 0.7013 | -0.0072 | 0.8504 | 0.8507 | 0.7816 | 0.7836 |



The +5 day horizon provides the strongest aggregate spatial improvement:



\- F1: 0.7303 → 0.7390

\- Precision: 0.6507 → 0.6870

\- ROC-AUC: 0.8581 → 0.8634

\- PR-AUC: 0.8022 → 0.8188



\---



\## 8. Transition Diagnostics



Spatial context performs strongly for continuation of existing elevated activity.



Spatial `E→E` recall:



| Horizon | Recall |

|---|---:|

| +1d | 0.9381 |

| +2d | 0.9326 |

| +3d | 0.9578 |

| +5d | 0.9150 |

| +7d | 0.9519 |



Emerging-event (`N→E`) recall:



| Horizon | Spatial | V2 |

|---|---:|---:|

| +1d | 0.3292 | 0.4037 |

| +2d | 0.4346 | 0.4817 |

| +3d | 0.5865 | 0.6202 |

| +5d | 0.5025 | 0.5635 |

| +7d | 0.6728 | 0.6267 |



Spatial context therefore does not consistently improve emerging-fire detection.



The main positive exception is +7 days, where emerging-event recall increases from approximately 62.7% to 67.3%.



Transition precision values within fixed transition groups are not interpreted as general model precision because the actual class is conditioned by the transition definition.



\---



\## 9. State Dependence



Spatial performance remains substantially different between Punjab and Haryana.



Examples:



\### +5 days



\- Haryana F1: 0.5800

\- Punjab F1: 0.8099



\### +7 days



\- Haryana F1: 0.5671

\- Punjab F1: 0.7642



Spatial context therefore does not eliminate the geographic performance heterogeneity already observed in V2.



The difference should not be interpreted as proof of intrinsic model bias without additional investigation.



Possible explanations include differences in prevalence, temporal dynamics, spatial connectivity, fire regimes, and feature distributions.



\---



\## 10. Seasonal Dependence



Spatial performance is also dependent on season phase.



At +5 days:



\- Early F1: 0.7651

\- Middle F1: 0.7736

\- Late F1: 0.3486



The aggregate +5 improvement is therefore not uniform across the season.



At +7 days:



\- Early F1: 0.7433

\- Middle F1: 0.7093

\- Late F1: 0.2118



The +7 late-season result is relatively better than the corresponding V2 late-season result, but absolute performance remains weak.



\---



\## 11. Current-Fire-Regime Diagnostics



The spatial model is strongly conditioned on current fire activity.



At +5 days:



| Current regime | Spatial F1 |

|---|---:|

| No current fire | 0.4224 |

| Low current fire | 0.5285 |

| Elevated current fire | 0.8312 |



At +7 days:



| Current regime | Spatial F1 |

|---|---:|

| No current fire | 0.4143 |

| Low current fire | 0.5847 |

| Elevated current fire | 0.8086 |



This indicates that neighboring fire information is particularly useful when there is already elevated activity.



The feature family is less effective as a general signal for completely quiet districts.



\---



\## 12. Interpretation



The results suggest that spatial context primarily provides information related to spatial continuation and propagation of existing fire activity.



It does not behave as a universally useful early-warning feature.



The strongest evidence is:



1\. +5 day aggregate improvement.

2\. Strong continuation-event performance.

3\. +7 day improvement in emerging-event recall.

4\. Persistent Punjab/Haryana performance differences.

5\. Strong dependence on current fire regime.

6\. Strong seasonal dependence.



Therefore, spatial information appears to contain useful forecasting information, but its usefulness is conditional on horizon, current activity, geography, and season.



\---



\## 13. Limitations



This experiment has several limitations.



\### 13.1 Geographic scope



The spatial graph currently covers only Punjab and Haryana.



\### 13.2 Adjacency definition



Adjacency is based on polygon touching relationships. Alternative spatial definitions such as distance-based neighbors, k-nearest neighbors, or weighted adjacency were not evaluated in this experiment.



\### 13.3 Spatial aggregation



District-level aggregation can hide sub-district variation.



\### 13.4 Model scope



Only the Random Forest baseline was evaluated for the spatial feature experiment.



\### 13.5 Feature-family attribution



Aggregate improvements do not establish that every spatial feature is individually useful.



\### 13.6 Test-period scope



The final evaluation uses the frozen 2025 test period. No additional independent year has yet been evaluated.



\### 13.7 Interpretation



FIRMS active-fire detections are not equivalent to confirmed stubble-burning events. Spatial association does not establish causality.



\---



\## 14. Research Integrity



The following were preserved:



\- Frozen V2 ML dataset was not overwritten.

\- Frozen V2 benchmark files were not overwritten.

\- Frozen calibration artifacts were not overwritten.

\- 2025 test data were not used for hyperparameter or threshold selection.

\- Spatial features were generated using calendar-date-aligned lookups.

\- The spatial experiment was evaluated against the frozen V2 baseline.

\- Prediction-level outputs were saved for reproducible diagnostics.



\---



\## 15. Experimental Artifacts



Core artifacts:



\- `ml\_dataset\_v2\_3\_spatial.csv`

\- `district\_adjacency.csv`

\- `v2\_3\_spatial\_results.csv`

\- `v2\_3\_spatial\_config.json`

\- `spatial\_rf\_predictions\_plus1d.csv`

\- `spatial\_rf\_predictions\_plus2d.csv`

\- `spatial\_rf\_predictions\_plus3d.csv`

\- `spatial\_rf\_predictions\_plus5d.csv`

\- `spatial\_rf\_predictions\_plus7d.csv`



Diagnostic artifacts:



\- `spatial\_transition\_diagnostics.csv`

\- `spatial\_state\_diagnostics.csv`

\- `spatial\_season\_diagnostics.csv`

\- `spatial\_fire\_regime\_diagnostics.csv`

\- `spatial\_state\_fire\_regime\_diagnostics.csv`



\---



\## 16. Decision



\*\*V2.3 Spatial Context: CONDITIONALLY ACCEPTED AS AN EXPERIMENTAL FEATURE FAMILY\*\*



Spatial features are retained for further research but are not promoted to the official V2 model.



The strongest evidence is concentrated at +5 days, with additional evidence of improved emerging-event recall at +7 days.



Further work should investigate whether spatial features can be improved through:



\- weighted spatial relationships,

\- alternative neighbor definitions,

\- feature selection,

\- spatial lag design,

\- horizon-specific spatial features,

\- and/or interaction with temporal dynamics.



Any future improvement must continue to use the frozen V2 protocol and preserve an untouched final test period.



\---



\## 17. Reproducibility



The experiment can be reconstructed using:



1\. `build\_v2\_3\_spatial\_features.py`

2\. `train\_v2\_3\_spatial\_experiment.py`

3\. `analyze\_v2\_3\_spatial\_diagnostics.py`

4\. `analyze\_v2\_3\_spatial\_fire\_regimes.py`



The spatial experiment uses deterministic Random Forest configuration with `random\_state=42`.



\---



\## 18. Checkpoint Policy



This document records the completed spatial experiment and its diagnostic findings.



The spatial experiment must remain separate from the frozen V2 benchmark unless a future experiment demonstrates consistent, independently validated improvement.



No frozen V2 artifact should be overwritten as part of this experiment.

