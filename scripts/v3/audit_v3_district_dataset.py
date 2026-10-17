from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "data/processed/v3/v3_district_ml_dataset_1d.csv"
DISTRICT_PATH = ROOT / "data/processed/v3/district_day_panel_005_centroid_fallback.csv"
REPORT_PATH = ROOT / "research/v3/v3_district_feature_dataset_audit.md"

KEYS = ["state", "district", "date"]
LAGS = [1, 2, 3, 5, 7, 14]
BASE_FIRE = [
    "fire_count", "frp_sum", "frp_mean", "frp_max",
    "high_confidence_fire_count", "day_fire_count",
    "night_fire_count", "bright_ti4_mean", "bright_ti4_max",
    "bright_ti5_mean",
]
WEATHER = ["T2M", "RH2M", "WS2M", "PRECTOTCORR"]
GRID = [
    "active_cell_count", "max_cell_fire_count",
    "max_cell_frp", "max_cell_share",
]
LAGGED = [
    "fire_count", "frp_sum", "active_cell_count",
    "max_cell_fire_count", "max_cell_frp", "max_cell_share",
]
TARGET = "fire_count_t_plus_1d"
LABEL = "target_elevated_t_plus_1d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize(df):
    df = df.copy()
    for col in ["state", "district"]:
        df[col] = df[col].astype("string").str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    require(df[KEYS].notna().all().all(), "Null key values found")
    return df


def main():
    require(DATASET_PATH.is_file(), f"Dataset missing: {DATASET_PATH}")
    require(DISTRICT_PATH.is_file(), f"District source missing: {DISTRICT_PATH}")
    require(not REPORT_PATH.exists(), f"Audit report already exists: {REPORT_PATH}")

    checks = {}
    details = []

    print("1/7 Loading dataset and source panel...")
    df = normalize(pd.read_csv(DATASET_PATH))
    source = normalize(pd.read_csv(DISTRICT_PATH))

    required = (
        KEYS + BASE_FIRE + WEATHER + GRID
        + ["year", "month", "day", "day_of_season",
           "day_of_year", "sin_year", "cos_year",
           TARGET, "frp_sum_t_plus_1d", LABEL]
    )
    required += [
        f"{feature}_lag_{lag}d"
        for feature in LAGGED for lag in LAGS
    ]
    required += [
        f"{feature}_mean_3d"
        for feature in ["fire_count", "frp_sum"]
    ]
    required += [
        f"{feature}_sparse_mean_{window}d"
        for feature in ["fire_count", "frp_sum"]
        for window in [7, 14]
    ]

    missing_columns = sorted(set(required) - set(df.columns))
    checks["required_schema_present"] = not missing_columns
    require(not missing_columns, f"Missing required columns: {missing_columns}")

    checks["expected_row_count"] = len(df) == 45_384
    checks["unique_district_day_keys"] = not df.duplicated(KEYS).any()
    checks["six_states"] = df["state"].nunique() == 6
    checks["248_districts"] = df[["state", "district"]].drop_duplicates().shape[0] == 248

    expected_dates = set()
    for year in [2023, 2024, 2025]:
        expected_dates.update(pd.date_range(
            f"{year}-10-01", f"{year}-11-30", freq="D"
        ))
    actual_dates = set(df["date"].unique())
    checks["expected_seasonal_dates"] = actual_dates == expected_dates

    per_district_dates = df.groupby(KEYS[:2])["date"].nunique()
    checks["complete_183_dates_per_district"] = (
        len(per_district_dates) == 248 and (per_district_dates == 183).all()
    )

    details += [
        f"- Dataset rows: {len(df):,}",
        f"- Unique district-day keys: {df[KEYS].drop_duplicates().shape[0]:,}",
        f"- States: {df['state'].nunique()}",
        f"- Districts: {df[['state', 'district']].drop_duplicates().shape[0]}",
        f"- Date range: {df['date'].min().date()} to {df['date'].max().date()}",
        f"- Unique seasonal dates: {len(actual_dates)}",
        f"- Missing required columns: {missing_columns}",
    ]

    print("2/7 Checking base fire features against the source...")
    src = source[KEYS + BASE_FIRE].copy()
    check_unique = not source.duplicated(KEYS).any()
    require(check_unique, "Source district panel has duplicate keys")

    base_join = df[KEYS + BASE_FIRE].merge(
        src, on=KEYS, how="left", validate="one_to_one",
        suffixes=("_dataset", "_source"), indicator=True
    )
    checks["all_dataset_keys_in_source"] = (base_join["_merge"] == "both").all()
    base_mismatches = {}
    for col in BASE_FIRE:
        left = base_join[f"{col}_dataset"]
        right = base_join[f"{col}_source"]
        equal = np.isclose(left, right, rtol=0, atol=1e-10, equal_nan=True)
        base_mismatches[col] = int((~equal).sum())
    checks["base_fire_features_match_source"] = all(
        count == 0 for count in base_mismatches.values()
    )
    details.append(f"- Base-feature mismatches by column: {base_mismatches}")

    print("3/7 Verifying exact-date lag features...")
    # District fire-count and FRP lags are checked against the independent
    # district source. Grid-derived lags are checked against their own
    # current-day values at the exact historical date in this dataset.
    lag_mismatches = {}
    for feature in LAGGED:
        source_frame = (
            src[KEYS + [feature]]
            if feature in ["fire_count", "frp_sum"]
            else df[KEYS + [feature]]
        )
        for lag in LAGS:
            lag_col = f"{feature}_lag_{lag}d"
            hist = source_frame.copy()
            hist["date"] = hist["date"] + pd.Timedelta(days=lag)
            hist = hist.rename(columns={feature: "_expected"})
            probe = df[KEYS + [lag_col]].merge(
                hist[KEYS + ["_expected"]],
                on=KEYS, how="left", validate="one_to_one"
            )
            expected_known = probe["_expected"].notna()
            observed_known = probe[lag_col].notna()
            mismatch = (
                (expected_known != observed_known)
                | (
                    expected_known & observed_known
                    & ~np.isclose(
                        probe[lag_col].fillna(0),
                        probe["_expected"].fillna(0),
                        rtol=0, atol=1e-10
                    )
                )
            )
            lag_mismatches[lag_col] = int(mismatch.sum())

    checks["all_exact_date_lags_aligned"] = all(
        count == 0 for count in lag_mismatches.values()
    )
    details.append(f"- Lag mismatches by feature: {lag_mismatches}")

    print("4/7 Verifying +1-day target and label...")
    future = src[KEYS + ["fire_count", "frp_sum"]].copy()
    future["date"] = future["date"] - pd.Timedelta(days=1)
    future = future.rename(columns={
        "fire_count": "_expected_fire_target",
        "frp_sum": "_expected_frp_target",
    })
    target_probe = df[KEYS + [TARGET, "frp_sum_t_plus_1d", LABEL]].merge(
        future, on=KEYS, how="left", validate="one_to_one"
    )

    target_known = target_probe["_expected_fire_target"].notna()
    fire_target_equal = np.isclose(
        target_probe[TARGET].fillna(-999999),
        target_probe["_expected_fire_target"].fillna(-999999),
        rtol=0, atol=1e-10
    )
    frp_target_equal = np.isclose(
        target_probe["frp_sum_t_plus_1d"].fillna(-999999),
        target_probe["_expected_frp_target"].fillna(-999999),
        rtol=0, atol=1e-10
    )
    checks["fire_target_exactly_aligned"] = fire_target_equal.all()
    checks["frp_target_exactly_aligned"] = frp_target_equal.all()

    label_known = target_probe[LABEL].notna()
    expected_label = (
        target_probe["_expected_fire_target"].gt(2).astype("Int64")
    )
    label_equal = (
        target_probe.loc[target_known, LABEL].astype("Int64").reset_index(drop=True)
        == expected_label.loc[target_known].reset_index(drop=True)
    ).all()
    checks["binary_label_matches_gt_2"] = bool(label_equal)
    checks["target_missing_only_when_next_seasonal_date_absent"] = (
        df[TARGET].isna().equals(~target_known)
    )

    target_positive_total = int((df.loc[target_known, TARGET] > 2).sum())
    details += [
        f"- Known next-day targets: {int(target_known.sum()):,}",
        f"- Missing next-day targets: {int((~target_known).sum()):,}",
        f"- Correctly recalculated positive targets (all known rows): {target_positive_total:,}",
        f"- Label/source target mismatches: {int((~label_equal).sum()) if not label_equal else 0}",
    ]

    print("5/7 Checking feature ranges, missingness, and eligibility...")
    checks["weather_complete"] = df[WEATHER].notna().all().all()
    checks["grid_features_complete"] = df[GRID].notna().all().all()
    checks["grid_features_nonnegative"] = (
        (df[["active_cell_count", "max_cell_fire_count", "max_cell_frp"]] >= 0).all().all()
    )
    checks["max_cell_share_in_range"] = df["max_cell_share"].between(0, 1).all()

    lag_columns = [
        f"{feature}_lag_{lag}d"
        for feature in LAGGED for lag in LAGS
    ]
    complete_history = df[lag_columns].notna().all(axis=1)
    eligible = complete_history & df[TARGET].notna()
    details += [
        f"- Rows with complete six-feature lag history: {int(complete_history.sum()):,}",
        f"- Eligible rows with known +1-day target: {int(eligible.sum()):,}",
        f"- Rows excluded by complete-history/known-target rule: {int((~eligible).sum()):,}",
    ]

    year_balance = []
    for year in [2023, 2024, 2025]:
        subset = df.loc[eligible & (df["date"].dt.year == year)]
        positives = int((subset[TARGET] > 2).sum())
        year_balance.append({
            "year": year,
            "eligible_rows": int(len(subset)),
            "positive_rows": positives,
            "negative_rows": int(len(subset) - positives),
        })
    details.append("- Eligible target counts by year:")
    for row in year_balance:
        details.append(
            f"  - {row['year']}: {row['eligible_rows']:,} rows; "
            f"{row['positive_rows']:,} positive; {row['negative_rows']:,} negative"
        )

    print("6/7 Checking target-derived columns and season boundaries...")
    outcome_columns = {
        TARGET, "frp_sum_t_plus_1d", LABEL
    }
    candidate_predictors = [
        col for col in df.columns
        if col not in outcome_columns
        and not col.startswith("fire_count_t_plus_")
        and not col.startswith("frp_sum_t_plus_")
        and not col.startswith("target_")
    ]
    suspicious_predictors = [
        col for col in candidate_predictors
        if "future" in col.lower() or "target" in col.lower()
        or "_t_plus_" in col.lower()
    ]
    checks["no_obvious_future_target_predictors"] = len(suspicious_predictors) == 0

    # No date in this study should use a lag/target from outside its seasonal block.
    # The exact-date joins above require a source row; absent dates must remain null.
    no_target_without_source = (
        df[TARGET].notna().to_numpy() == target_known.to_numpy()
    ).all()
    checks["seasonal_target_boundaries_respected"] = no_target_without_source
    details.append(f"- Suspicious predictor-like column names: {suspicious_predictors}")
    details.append("- Outcome columns excluded from candidate predictor list: " + ", ".join(sorted(outcome_columns)))

    print("7/7 Summarizing audit...")
    checks["source_key_uniqueness"] = not source.duplicated(KEYS).any()
    passed = all(bool(v) for v in checks.values())

    report_lines = [
        "# V3 District Feature Dataset — Independent Audit",
        "",
        f"- Audit result: **{'PASS' if passed else 'FAIL'}**",
        "- Dataset: `data/processed/v3/v3_district_ml_dataset_1d.csv`",
        "- Source district panel: `data/processed/v3/district_day_panel_005_centroid_fallback.csv`",
        "- Scope: read-only validation; no model training or test-set evaluation.",
        "",
        "## Dataset summary",
        *details,
        "",
        "## Checks",
    ]
    report_lines += [
        f"- [{'x' if bool(value) else ' '}] {name}"
        for name, value in checks.items()
    ]
    report_lines += [
        "",
        "## Interpretation limits",
        "- This audit validates the listed data integrity, alignment, and range checks.",
        "- It does not prove predictive improvement or complete scientific validity.",
        "- Weather is district-representative weather repeated across cells within a district-day.",
        "- The V2 reconstruction gate and model-specific finite-value/feature checks remain separate steps.",
        "",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "x", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\nCHECK RESULTS")
    for name, passed_check in checks.items():
        print(("PASS" if passed_check else "FAIL"), name)
    print(f"\nCorrectly recalculated all-known target positives: {target_positive_total:,}")
    print(f"Eligible rows: {int(eligible.sum()):,}")
    print(f"Report: {REPORT_PATH.relative_to(ROOT)}")
    print("\nOVERALL:", "PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
