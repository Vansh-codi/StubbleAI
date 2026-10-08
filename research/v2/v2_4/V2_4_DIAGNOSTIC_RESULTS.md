\# StubbleAI V2.4 Diagnostic Results

\## Spatio-Temporal Interaction Experiment



\*\*Status:\*\* REJECTED as official V2 replacement

\*\*Official baseline:\*\* Frozen V2

\*\*Experiment family:\*\* V2.4 Spatio-Temporal Interaction

\*\*Evaluation period:\*\* 2025 frozen test

\*\*Purpose:\*\* Diagnostic analysis only



\---



\## 1. Research Question



Can interactions between local fire dynamics and neighboring-district fire activity improve multi-horizon forecasting, particularly emerging-fire detection, without degrading continuation detection?



V2.4 added 16 interaction features:



\- local fire Ã— neighbor fire

\- local FRP Ã— neighbor FRP

\- relative neighbor fire activity

\- relative neighbor FRP



for current, 1-day, 3-day and 7-day lag contexts.



The experiment used the same frozen temporal protocol as V2:



\- 2023 training

\- 2024 validation

\- 2025 untouched test

\- Random Forest

\- 400 trees

\- minimum leaf size 2

\- random\_state=42

\- no class weighting

\- validation-only threshold selection



No 2025 information was used for model or threshold selection.



\---



\## 2. Official V2 Comparison



| Horizon | V2 F1 | V2.3 Spatial F1 | V2.4 Spatio-Temporal F1 | V2.4 Î” vs V2 |

|---|---:|---:|---:|---:|

| +1d | 0.7754 | 0.7636 | 0.7640 | -0.0114 |

| +2d | 0.7513 | 0.7545 | 0.7503 | -0.0010 |

| +3d | 0.7305 | 0.7304 | 0.7358 | +0.0053 |

| +5d | 0.7303 | 0.7362 | 0.7118 | -0.0185 |

| +7d | 0.7085 | 0.7013 | 0.6924 | -0.0161 |



V2.4 improves only one of five horizons.



It violates the predefined acceptance criteria:



1\. Average F1 improvement â‰¥ 0.005: FAIL

2\. At least 3/5 horizons improve: FAIL

3\. No horizon decreases by more than 0.010: FAIL

4\. Emerging-event recall improves at 3/5 horizons: FAIL

5\. Robustness across state/season/regime: FAIL



Therefore V2.4 is rejected as a replacement for V2.



\---



\## 3. Transition Analysis



\### Emerging-fire detection: Nâ†’E



| Horizon | V2 | V2.4 |

|---|---:|---:|

| +1d | 0.4037 | 0.2981 |

| +2d | 0.4817 | 0.4817 |

| +3d | 0.6202 | 0.5337 |

| +5d | 0.5635 | 0.5939 |

| +7d | 0.6267 | 0.6728 |



V2.4 substantially improves emerging-event recall only at +5d and +7d.



At +1d it falls from 40.37% to 29.81%.



\### Continuation: Eâ†’E



| Horizon | V2 | V2.4 |

|---|---:|---:|

| +1d | 0.9599 | 0.9326 |

| +2d | 0.9518 | 0.9461 |

| +3d | 0.9739 | 0.9458 |

| +5d | 0.9368 | 0.9625 |

| +7d | 0.9387 | 0.9562 |



V2.4 improves continuation recall at +5d and +7d but degrades it at shorter horizons.



The interaction feature family therefore appears more promising for longer-range dynamics than immediate next-day prediction.



\---



\## 4. State Robustness



V2.4 does not consistently outperform V2 across Haryana and Punjab.



At +1d:



\- Haryana F1: V2 0.6402 â†’ V2.4 0.6274

\- Punjab F1: V2 0.8324 â†’ V2.4 0.8196



At +5d:



\- Haryana recall: 0.6340 â†’ 0.6809

\- Punjab recall: 0.9316 â†’ 0.9487



The longer-horizon signal is therefore more encouraging, but it is not sufficient for an overall replacement.



\---



\## 5. Seasonal Robustness



V2.4 does not provide uniform seasonal improvement.



At +1d:



\- Early F1: 0.7487 â†’ 0.7266

\- Middle F1: 0.8548 â†’ 0.8544

\- Late F1: 0.5701 â†’ 0.5392



At +5d:



\- Early F1: 0.7444 â†’ 0.7415

\- Middle F1: 0.7744 â†’ 0.7580

\- Late F1: 0.3761 â†’ 0.3289



At +7d:



\- Early F1: 0.7441 â†’ 0.7340

\- Middle F1: 0.7246 â†’ 0.7052

\- Late F1: 0.1389 â†’ 0.2268



The benefit is therefore horizon- and season-dependent rather than uniformly robust.



\---



\## 6. Fire-Regime Analysis



The strongest short-horizon degradation occurs in the low-fire regime.



At +1d:



| Regime | V2 F1 | V2.4 F1 |

|---|---:|---:|

| No fire | 0.2151 | 0.2222 |

| Low | 0.4783 | 0.3762 |

| Elevated | 0.8754 | 0.8722 |



This suggests that interaction features can be less reliable when local fire activity is weak.



At longer horizons, V2.4 retains strong elevated-fire performance and improves emerging-event recall, but not enough to overcome the overall classification losses.



\---



\## 7. Probability / Ranking Analysis



The probability-level analysis produced an important secondary finding.



\### PR-AUC



| Horizon | V2 | V2.4 |

|---|---:|---:|

| +1d | 0.8582 | 0.8569 |

| +2d | 0.8369 | 0.8387 |

| +3d | 0.8193 | 0.8172 |

| +5d | 0.8022 | 0.8210 |

| +7d | 0.7816 | 0.7812 |



The clearest improvement occurs at +5d:



\*\*PR-AUC: 0.8022 â†’ 0.8210\*\*



The +5d ROC-AUC also improves:



\*\*0.8581 â†’ 0.8638\*\*



and Brier score improves:



\*\*0.1484 â†’ 0.1443\*\*



This indicates that the V2.4 interaction features contain useful probability-ranking information at +5d despite the lower official thresholded F1.



\---



\## 8. Threshold-Transfer Diagnostic



A diagnostic scan across the frozen 2025 predictions showed that V2.4 could achieve a higher F1 at a different threshold.



For +5d:



\- V2 official/test F1: 0.7303

\- V2.4 official/test F1: 0.7118

\- V2 diagnostic best 2025 F1: 0.7362

\- V2.4 diagnostic best 2025 F1: 0.7512



However, these 2025-optimal thresholds are diagnostic only.



They MUST NOT be used as official thresholds because doing so would tune on the frozen test set.



Therefore this result is interpreted as evidence of threshold-transfer instability, not as evidence that V2.4 officially outperforms V2.



\---



\## 9. Probability Scale



At +5d:



\- V2 mean probability: 0.3923

\- V2.4 mean probability: 0.3762

\- actual positive rate: 0.3906



V2.4 therefore produces a lower probability scale despite showing better ranking performance.



This provides a plausible explanation for why a validation-derived threshold did not transfer optimally to the 2025 test distribution.



This remains a diagnostic interpretation rather than proof of a causal mechanism.



\---



\## 10. Emerging-Event Probability



At +5d:



\- V2 emerging-event mean probability: 0.4018

\- V2.4 emerging-event mean probability: 0.3693



Yet emerging recall improves:



\- V2: 0.5635

\- V2.4: 0.5939



At +7d:



\- V2 emerging-event mean probability: 0.4304

\- V2.4: 0.4125



while emerging recall improves:



\- V2: 0.6267

\- V2.4: 0.6728



This reinforces the distinction between probability scale and ranking ability.



\---



\## 11. Scientific Interpretation



The V2.4 interaction family should NOT be interpreted as universally ineffective.



Instead, the evidence suggests:



1\. Local temporal dynamics remain strongest for short-horizon forecasting.

2\. Spatial context contains useful information for longer horizons.

3\. Explicit local Ã— neighbor interactions may contain additional longer-range signal.

4\. The signal is not robust enough to replace the V2 classifier.

5\. The clearest evidence appears at +5d, where PR-AUC improves substantially.

6\. Probability scale and threshold transfer are less stable for V2.4.

7\. The interaction features therefore remain a research lead rather than a production feature family.



\---



\## 12. Final Decision



\### V2.4 Spatio-Temporal Interaction



\*\*DECISION: REJECTED as official V2 replacement.\*\*



The experiment does not satisfy the predefined acceptance criteria.



The official V2 model remains unchanged.



V2.4 predictions and diagnostics are retained as research artifacts.



No V2.4 threshold should be promoted to deployment.



No 2025-derived threshold should be used for official evaluation.



\---



\## 13. Research Value



V2.4 produced a useful research finding:



> Spatial information appears to become increasingly useful as the forecast horizon increases, but the benefit is better expressed in probability ranking than in a single fixed-threshold classifier.



This motivates future research into:



\- horizon-specific probability calibration

\- multi-horizon modelling

\- structured spatio-temporal representations

\- graph-based forecasting

\- spatial propagation dynamics

\- probabilistic forecasting

\- decision policies that distinguish ranking from binary alerting



These should be treated as future research directions rather than immediate model changes.



\---



\## 14. Reproducibility



V2.4 diagnostics were performed only on already-generated 2025 prediction files.



Generated diagnostic outputs:



\- v2\_4\_transition\_analysis.csv

\- v2\_4\_state\_analysis.csv

\- v2\_4\_season\_analysis.csv

\- v2\_4\_fire\_regime\_analysis.csv

\- v2\_4\_emerging\_analysis.csv

\- v2\_4\_probability\_quality.csv

\- v2\_4\_threshold\_sensitivity.csv

\- v2\_4\_probability\_bins.csv

\- v2\_4\_emerging\_probability\_analysis.csv



The frozen V2 benchmark remains unchanged.



\---



\## 15. Final Research Status



\*\*Official model:\*\* V2



\*\*V2.3 Spatial:\*\* Experimental / conditionally accepted feature family



\*\*V2.3 Trend:\*\* Rejected



\*\*V2.4 Spatio-Temporal:\*\* Rejected as replacement; retained as diagnostic research



\*\*2025 test integrity:\*\* Preserved



\*\*Next research direction:\*\* Do not tune V2.4 on 2025. Design the next experiment using lessons from V2.3/V2.4.
