\# StubbleAI V2.2 â€” Frozen Test Error Analysis



\## 1. Purpose



This document records the post-benchmark error analysis of the StubbleAI V2 Random Forest forecasting model.



The purpose is to characterize where and when the frozen model succeeds or fails, rather than to optimize the model.



The analysis covers:



\- horizon-wise errors

\- district-level performance

\- state-level performance

\- comparison with persistence

\- current fire-regime dependence

\- fire-state transitions

\- emerging fire-event detection

\- temporal/seasonal variation



This analysis is diagnostic and does not alter the frozen V2 benchmark.



\---



\## 2. Evaluation Protocol



The V2 temporal evaluation protocol is:



| Period | Role |

|---|---|

| 2023 | Training |

| 2024 | Validation / threshold selection |

| 2025 | Frozen final test |



The target is:



`fire\_count\_t\_plus\_Nd > 2 -> Elevated`



where `N` is the forecasting horizon.



Evaluated horizons:



\- +1 day

\- +2 days

\- +3 days

\- +5 days

\- +7 days



The classification threshold for each model/horizon was selected using the 2024 validation period and frozen before evaluating 2025.



No threshold was selected using 2025 test performance.



The persistence baseline uses the fixed rule:



`current fire\_count > 2 -> Elevated`



No persistence threshold tuning was performed.



\---



\## 3. Random Forest Reconstruction



The error-analysis predictions were reconstructed using the same feature preparation and model specification as the frozen V2 benchmark.



The reconstruction uses:



\- the V2 ML dataset

\- 2023 training data

\- 2024 validation data

\- 2025 frozen test data

\- pandas one-hot encoding with the training dummy columns defining the feature space

\- Random Forest

\- 400 trees

\- minimum leaf size of 2

\- random state 42

\- no class weighting



The categorical features are:



\- state

\- district



The numeric features include:



\- calendar/seasonal features

\- current fire activity

\- FRP features

\- weather

\- fire-history lags

\- FRP lags

\- rolling/sparse rolling fire statistics



The reconstructed 2025 test set contains:



\- 1,800 district-day observations per horizon

\- 45 districts

\- 40 test dates

\- 5 forecasting horizons

\- 9,000 horizon-level prediction records in total



The reconstructed metrics match the frozen benchmark.



\---



\## 4. Frozen Random Forest Test Performance



| Horizon | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |

|---|---:|---:|---:|---:|---:|---:|

| +1d | 0.8094 | 0.7246 | 0.8338 | \*\*0.7754\*\* | 0.8953 | 0.8582 |

| +2d | 0.7844 | 0.6894 | 0.8254 | \*\*0.7513\*\* | 0.8795 | 0.8369 |

| +3d | 0.7483 | 0.6297 | 0.8697 | \*\*0.7305\*\* | 0.8683 | 0.8193 |

| +5d | 0.7600 | 0.6507 | 0.8321 | \*\*0.7303\*\* | 0.8581 | 0.8022 |

| +7d | 0.7417 | 0.6135 | 0.8383 | \*\*0.7085\*\* | 0.8504 | 0.7816 |



The model's probability-ranking performance remains strong across all horizons, while thresholded F1 decreases with forecast distance.



The +1-day horizon provides the strongest overall classification performance.



\---



\## 5. Random Forest vs Persistence



Persistence remains a strong baseline because fire activity exhibits substantial temporal continuity.



\### Overall comparison



| Horizon | RF F1 | Persistence F1 | RF âˆ’ Persistence |

|---|---:|---:|---:|

| +1d | 0.7754 | 0.7700 | +0.0054 |

| +2d | 0.7513 | 0.7279 | +0.0234 |

| +3d | 0.7305 | 0.7004 | +0.0301 |

| +5d | 0.7303 | 0.7132 | +0.0171 |

| +7d | 0.7085 | 0.6576 | +0.0509 |



The Random Forest does not dominate persistence on every metric or every horizon, but it provides complementary predictive information.



Across all 9,000 horizon-level predictions:



\- both methods correct: 6,120

\- both methods wrong: 1,226

\- persistence-only correct: 855

\- RF-only correct: 799



Thus, the model should not be described as universally superior to persistence.



At +7 days, however, RF-only correct cases slightly exceed persistence-only correct cases:



\- RF-only: 227

\- persistence-only: 216



This supports the interpretation that the RF contains information beyond simple persistence, particularly at longer horizons, while still sharing many correct predictions with the temporal baseline.



\---



\## 6. Current Fire-Regime Analysis



The current fire state was divided into:



1\. No current activity: `fire\_count == 0`

2\. Low current activity: `1 <= fire\_count <= 2`

3\. Elevated current activity: `fire\_count > 2`



Performance is strongly conditioned on the current fire regime.



\### +1 day



For districts with no current fire activity:



\- RF accuracy: 0.8992

\- persistence accuracy: 0.9240

\- RF F1: 0.2151



For low current activity:



\- RF F1: 0.4783



For currently elevated districts:



\- RF F1: 0.8754

\- RF recall: 0.9599



The same qualitative pattern continues across longer horizons.



At +7 days, currently elevated districts still achieve:



\- RF F1: 0.8140

\- RF recall: 0.9387



while performance is substantially weaker for currently quiet districts.



This indicates that the model is particularly effective when there is already observable fire activity and is less reliable for predicting activity emerging from quiet conditions.



\---



\## 7. Fire-State Transition Analysis



The 2025 test observations were categorized according to current and future fire state.



The four transition types are:



\- Normal -> Normal

\- Normal -> Elevated

\- Elevated -> Normal

\- Elevated -> Elevated



where Normal means `fire\_count <= 2` and Elevated means `fire\_count > 2`.



\### Elevated -> Elevated



The RF is very strong at detecting continuation of elevated activity:



| Horizon | RF correct | Recall |

|---|---:|---:|

| +1d | 527 / 549 | 96.0% |

| +2d | 494 / 519 | 95.2% |

| +3d | 485 / 498 | 97.4% |

| +5d | 474 / 506 | 93.7% |

| +7d | 429 / 457 | 93.9% |



Persistence is naturally strongest for this transition and correctly classifies all continuation cases in this analysis.



\---



\## 8. Emerging Fire Events



An emerging event is defined as:



`Normal -> Elevated`



Persistence cannot detect these events using the current-state rule because it predicts the same state as the current day.



The RF does detect a substantial fraction of emerging events.



| Horizon | Events | RF detected | Recall |

|---|---:|---:|---:|

| +1d | 161 | 65 | 40.4% |

| +2d | 191 | 92 | 48.2% |

| +3d | 208 | 129 | 62.0% |

| +5d | 197 | 111 | 56.3% |

| +7d | 217 | 136 | 62.7% |



Therefore, emerging-event prediction is meaningful but incomplete.



The strongest overall emerging-event recall occurs at +7 days, while +1 day has the lowest emerging-event recall.



This should not be interpreted as evidence that the model is intrinsically better at longer horizons. The result is conditional on the particular 2025 test sample and transition distribution.



\---



\## 9. State Dependence of Emerging Events



Emerging-event detection differs substantially between Punjab and Haryana.



At +7 days:



\- Haryana: 37 / 104 detected, recall 35.6%

\- Punjab: 99 / 113 detected, recall 87.6%



Similar state differences occur at other horizons.



This is an observed model-performance difference in the evaluation dataset.



The analysis does not establish the cause of the difference.



Possible explanations such as differences in prevalence, fire dynamics, geography, agricultural practices, or data characteristics require separate investigation and should not be presented as established causal explanations.



\---



\## 10. Temporal / Season-Phase Analysis



The 2025 test period was divided into:



\- Early season: 15 October â€“ 31 October

\- Middle season: 1 November â€“ 15 November

\- Late season: 16 November â€“ 23 November



This analysis is diagnostic only.



\### Overall temporal performance



| Horizon | Early F1 | Middle F1 | Late F1 |

|---|---:|---:|---:|

| +1d | 0.7487 | \*\*0.8548\*\* | 0.5701 |

| +2d | 0.7215 | \*\*0.8289\*\* | 0.5333 |

| +3d | 0.7194 | \*\*0.8047\*\* | 0.4767 |

| +5d | 0.7444 | \*\*0.7744\*\* | 0.3761 |

| +7d | \*\*0.7441\*\* | 0.7246 | 0.1389 |



\### Early season



The RF provides its clearest improvement over persistence during the early period:



| Horizon | RF F1 | Persistence F1 | Difference |

|---|---:|---:|---:|

| +1d | 0.7487 | 0.7107 | +0.0380 |

| +2d | 0.7215 | 0.6480 | +0.0735 |

| +3d | 0.7194 | 0.6162 | +0.1032 |

| +5d | 0.7444 | 0.6701 | +0.0743 |

| +7d | 0.7441 | 0.6480 | +0.0960 |



The largest observed improvement occurs at +3 days.



This suggests that the additional model features provide useful information beyond current-state persistence during this part of the test period.



It does not establish causal mechanisms.



\### Middle season



The middle season has the highest RF F1 for +1d through +5d:



\- +1d: 0.8548

\- +2d: 0.8289

\- +3d: 0.8047

\- +5d: 0.7744



However, persistence is also very strong during this period.



Consequently, high absolute RF performance should not be interpreted as equivalent to large incremental value over the persistence baseline.



\### Late season



The late period is the clearest weakness.



RF F1 decreases to:



\- +1d: 0.5701

\- +2d: 0.5333

\- +3d: 0.4767

\- +5d: 0.3761

\- +7d: 0.1389



At +7 days:



\- RF F1: 0.1389

\- persistence F1: 0.3000

\- difference: -0.1611



Therefore, long-horizon forecasting during the late test period should be treated as substantially less reliable.



This is an important limitation and should remain visible in research reporting.



\---



\## 11. State Ã— Temporal Dependence



Temporal variation interacts with geography.



For example, at +7 days:



\### Early season



| State | RF F1 | Persistence F1 |

|---|---:|---:|

| Haryana | 0.5069 | 0.4267 |

| Punjab | 0.8262 | 0.7205 |



\### Late season



| State | RF F1 | Persistence F1 |

|---|---:|---:|

| Haryana | 0.1111 | 0.3684 |

| Punjab | 0.1667 | 0.2188 |



This demonstrates that aggregate performance can conceal substantial heterogeneity across both geography and temporal phase.



The analysis does not establish the causal source of this heterogeneity.



\---



\## 12. Main Research Findings



\### Finding 1 â€” Forecast skill decreases with horizon



RF F1 declines from:



`0.7754 (+1d)`



to:



`0.7085 (+7d)`



while ROC-AUC remains above 0.85 at all evaluated horizons.



This indicates that ranking quality remains useful even as thresholded classification becomes harder with longer forecast horizons.



\### Finding 2 â€” Persistence is a serious baseline



Persistence is highly competitive.



The RF should therefore be described as a model that provides additional predictive information rather than as a universally superior replacement for persistence.



\### Finding 3 â€” The RF is particularly strong for continuation



Elevated -> Elevated recall remains approximately 94â€“97% across horizons.



The model is therefore highly effective when elevated activity is already present.



\### Finding 4 â€” Emerging-event detection is meaningful but incomplete



Normal -> Elevated recall ranges from approximately 40% to 63%.



This is an important capability because persistence cannot detect these transitions, but the RF still misses a substantial fraction of emerging events.



\### Finding 5 â€” Performance is strongly state-dependent



Punjab and Haryana show substantial differences in both overall performance and emerging-event detection.



This motivates future India-wide stratified evaluation rather than assuming a single homogeneous fire process.



\### Finding 6 â€” Performance is season-dependent



The model performs strongly during much of the early and middle test period but degrades substantially during the late period, particularly at longer horizons.



\### Finding 7 â€” Aggregate metrics are insufficient



District, state, fire-regime, transition, and temporal analyses reveal patterns that are not visible in the aggregate F1 score.



This supports retaining stratified verification and error analysis as part of the StubbleAI research architecture.



\---



\## 13. Limitations



\### FIRMS detection limitation



Satellite active-fire detections are not equivalent to confirmed agricultural-residue burning events.



The model therefore predicts patterns in the observed fire-detection dataset rather than directly proving that a detected event is stubble burning.



\### Geographic limitation



The current V2 benchmark covers Punjab and Haryana and 45 districts.



It should not be presented as a validated India-wide forecasting model.



\### Temporal limitation



The frozen test period covers the 2025 October-November evaluation window.



The observed seasonal behavior may not generalize to every year.



\### Data-source limitation



Historical weather data and operational forecast weather inputs may not be identical data products.



This distinction must be considered before deployment claims are made.



\### Aggregation limitation



District-level aggregation can hide substantial sub-district spatial variation.



\### Causal limitation



The analyses are predictive.



They do not establish causal relationships between weather, agricultural practices, fire activity, or air quality.



\### Late-season limitation



The strong degradation observed in the late 2025 test period, especially for +5d and +7d forecasting, means long-horizon predictions should not be presented with uniform confidence across the season.



\---



\## 14. Research Integrity



The V2.2 analysis is a post-hoc diagnostic analysis of a frozen test evaluation.



The following were not performed using 2025 test results:



\- threshold optimization

\- hyperparameter tuning

\- model selection

\- retraining

\- feature selection

\- calibration fitting

\- changing the benchmark protocol



The thresholds were determined from the established validation procedure.



The purpose of this analysis is characterization, not optimization.



\---



\## 15. Reproducibility



The principal reconstruction and analysis artifacts are stored under:



`research/v2/error\_analysis/`



Key artifacts include:



\- `rf\_2025\_error\_predictions.csv`

\- `rf\_2025\_horizon\_metrics.csv`

\- `rf\_2025\_error\_summary.csv`

\- `district\_horizon\_error\_analysis.csv`

\- `district\_aggregate\_error\_analysis.csv`

\- `district\_error\_rankings.csv`

\- `state\_horizon\_error\_analysis.csv`

\- `state\_aggregate\_error\_analysis.csv`

\- `rf\_vs\_persistence\_horizon.csv`

\- `rf\_vs\_persistence\_state\_horizon.csv`

\- `rf\_vs\_persistence\_overall.csv`

\- `fire\_regime\_horizon\_analysis.csv`

\- `fire\_regime\_state\_horizon\_analysis.csv`

\- `fire\_transition\_horizon\_analysis.csv`

\- `emerging\_fire\_events\_analysis.csv`

\- `emerging\_fire\_state\_analysis.csv`

\- `temporal\_error\_analysis.csv`

\- `temporal\_state\_error\_analysis.csv`



Analysis scripts are stored at repository root:



\- `analyze\_v2\_errors.py`

\- `analyze\_v2\_district\_errors.py`

\- `analyze\_v2\_state\_errors.py`

\- `analyze\_v2\_persistence\_comparison.py`

\- `analyze\_v2\_fire\_regimes.py`

\- `analyze\_v2\_transitions.py`

\- `analyze\_v2\_temporal\_errors.py`



The frozen benchmark and calibration checkpoints remain separate and are not overwritten by this analysis.



\---



\## 16. Responsible Use



StubbleAI should not be treated as a sole enforcement or penalty system.



Predictions should be interpreted together with:



\- source information

\- forecast horizon

\- probability

\- model version

\- verification status

\- observed fire detections

\- known data limitations



The system should communicate uncertainty and avoid presenting satellite detections as confirmed individual farmer activity.



Air-quality and emissions conclusions require separate atmospheric and emissions modeling rather than being inferred directly from fire predictions.



\---



\## 17. Research Status



\### Completed



\- V2 fire-data pipeline

\- V2 ML dataset

\- leakage and alignment audit

\- temporal train/validation/test protocol

\- persistence baseline

\- climatology baselines

\- LR/RF/XGBoost/LightGBM benchmark

\- five-horizon forecasting benchmark

\- probability calibration study

\- exact RF reconstruction

\- horizon-wise error analysis

\- district-level analysis

\- state-level analysis

\- RF vs persistence comparison

\- fire-regime analysis

\- transition analysis

\- emerging-event analysis

\- temporal/season-phase analysis



\### Current status



The V2.2 error-analysis phase is complete.



The resulting artifacts are diagnostic research outputs and do not replace the frozen V2 benchmark.



The next research stages should be treated as separate experiments and should preserve this checkpoint unchanged.



\---



\## 18. Checkpoint Policy



The V2 benchmark and calibration results are frozen research checkpoints.



Future experiments should:



1\. create new experiment directories or explicitly version new artifacts;

2\. avoid overwriting frozen benchmark results;

3\. preserve the 2025 test set as a final evaluation reference;

4\. document any new feature, model, calibration, or deployment change;

5\. compare new methods against persistence and the existing RF benchmark.



This ensures that future improvements can be evaluated against a stable research baseline.
