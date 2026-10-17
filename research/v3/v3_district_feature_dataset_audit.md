# V3 District Feature Dataset — Independent Audit

- Audit result: **PASS**
- Dataset: `data/processed/v3/v3_district_ml_dataset_1d.csv`
- Source district panel: `data/processed/v3/district_day_panel_005_centroid_fallback.csv`
- Scope: read-only validation; no model training or test-set evaluation.

## Dataset summary
- Dataset rows: 45,384
- Unique district-day keys: 45,384
- States: 6
- Districts: 248
- Date range: 2023-10-01 to 2025-11-30
- Unique seasonal dates: 183
- Missing required columns: []
- Base-feature mismatches by column: {'fire_count': 0, 'frp_sum': 0, 'frp_mean': 0, 'frp_max': 0, 'high_confidence_fire_count': 0, 'day_fire_count': 0, 'night_fire_count': 0, 'bright_ti4_mean': 0, 'bright_ti4_max': 0, 'bright_ti5_mean': 0}
- Lag mismatches by feature: {'fire_count_lag_1d': 0, 'fire_count_lag_2d': 0, 'fire_count_lag_3d': 0, 'fire_count_lag_5d': 0, 'fire_count_lag_7d': 0, 'fire_count_lag_14d': 0, 'frp_sum_lag_1d': 0, 'frp_sum_lag_2d': 0, 'frp_sum_lag_3d': 0, 'frp_sum_lag_5d': 0, 'frp_sum_lag_7d': 0, 'frp_sum_lag_14d': 0, 'active_cell_count_lag_1d': 0, 'active_cell_count_lag_2d': 0, 'active_cell_count_lag_3d': 0, 'active_cell_count_lag_5d': 0, 'active_cell_count_lag_7d': 0, 'active_cell_count_lag_14d': 0, 'max_cell_fire_count_lag_1d': 0, 'max_cell_fire_count_lag_2d': 0, 'max_cell_fire_count_lag_3d': 0, 'max_cell_fire_count_lag_5d': 0, 'max_cell_fire_count_lag_7d': 0, 'max_cell_fire_count_lag_14d': 0, 'max_cell_frp_lag_1d': 0, 'max_cell_frp_lag_2d': 0, 'max_cell_frp_lag_3d': 0, 'max_cell_frp_lag_5d': 0, 'max_cell_frp_lag_7d': 0, 'max_cell_frp_lag_14d': 0, 'max_cell_share_lag_1d': 0, 'max_cell_share_lag_2d': 0, 'max_cell_share_lag_3d': 0, 'max_cell_share_lag_5d': 0, 'max_cell_share_lag_7d': 0, 'max_cell_share_lag_14d': 0}
- Known next-day targets: 44,640
- Missing next-day targets: 744
- Correctly recalculated positive targets (all known rows): 11,264
- Label/source target mismatches: 0
- Rows with complete six-feature lag history: 34,968
- Eligible rows with known +1-day target: 34,224
- Rows excluded by complete-history/known-target rule: 11,160
- Eligible target counts by year:
  - 2023: 11,408 rows; 3,373 positive; 8,035 negative
  - 2024: 11,408 rows; 3,580 positive; 7,828 negative
  - 2025: 11,408 rows; 3,181 positive; 8,227 negative
- Suspicious predictor-like column names: []
- Outcome columns excluded from candidate predictor list: fire_count_t_plus_1d, frp_sum_t_plus_1d, target_elevated_t_plus_1d

## Checks
- [x] required_schema_present
- [x] expected_row_count
- [x] unique_district_day_keys
- [x] six_states
- [x] 248_districts
- [x] expected_seasonal_dates
- [x] complete_183_dates_per_district
- [x] all_dataset_keys_in_source
- [x] base_fire_features_match_source
- [x] all_exact_date_lags_aligned
- [x] fire_target_exactly_aligned
- [x] frp_target_exactly_aligned
- [x] binary_label_matches_gt_2
- [x] target_missing_only_when_next_seasonal_date_absent
- [x] weather_complete
- [x] grid_features_complete
- [x] grid_features_nonnegative
- [x] max_cell_share_in_range
- [x] no_obvious_future_target_predictors
- [x] seasonal_target_boundaries_respected
- [x] source_key_uniqueness

## Interpretation limits
- This audit validates the listed data integrity, alignment, and range checks.
- It does not prove predictive improvement or complete scientific validity.
- Weather is district-representative weather repeated across cells within a district-day.
- The V2 reconstruction gate and model-specific finite-value/feature checks remain separate steps.
