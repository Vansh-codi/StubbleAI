from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

GRID_PATH = ROOT / "data/processed/v3/grid_day_panel_005_centroid_fallback_weather.csv"
DISTRICT_PATH = ROOT / "data/processed/v3/district_day_panel_005_centroid_fallback.csv"
WEATHER_PATH = ROOT / "data/processed/v3/district_daily_weather_v3.csv"

OUTPUT_PATH = ROOT / "data/processed/v3/v3_district_ml_dataset_1d.csv"
SUMMARY_PATH = ROOT / "data/processed/v3/v3_district_ml_dataset_1d_build_summary.json"

KEYS = ["state", "district", "date"]
WEATHER_FEATURES = ["T2M", "RH2M", "WS2M", "PRECTOTCORR"]

FIRE_FEATURES = [
    "fire_count", "frp_sum", "frp_mean", "frp_max",
    "high_confidence_fire_count", "day_fire_count",
    "night_fire_count", "bright_ti4_mean", "bright_ti4_max",
    "bright_ti5_mean",
]

GRID_FEATURES = [
    "active_cell_count",
    "max_cell_fire_count",
    "max_cell_frp",
    "max_cell_share",
]

LAGS = [1, 2, 3, 5, 7, 14]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv_checked(path: Path, required: list[str],
                     usecols: list[str] | None = None) -> pd.DataFrame:
    require(path.is_file(), f"Required input not found: {path}")
    df = pd.read_csv(path, usecols=usecols)
    missing = sorted(set(required) - set(df.columns))
    require(not missing, f"{path.name} missing columns: {missing}")
    return df


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["state", "district"]:
        df[col] = df[col].astype("string").str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    require(df[KEYS].notna().all().all(), "Null district/date key found")
    return df


def check_unique(df: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicates = int(df.duplicated(keys).sum())
    require(duplicates == 0, f"{label}: {duplicates:,} duplicate keys for {keys}")


def add_exact_date_lags(
    panel: pd.DataFrame,
    source_features: list[str],
    prefix: str = "",
) -> pd.DataFrame:
    """Attach historical values by exact date; never shift rows."""
    out = panel
    source = panel[KEYS + source_features].copy()

    for lag in LAGS:
        historical = source.copy()
        historical["date"] = historical["date"] + pd.Timedelta(days=lag)
        rename = {
            col: f"{prefix}{col}_lag_{lag}d"
            for col in source_features
        }
        historical = historical.rename(columns=rename)

        out = out.merge(
            historical,
            on=KEYS,
            how="left",
            validate="one_to_one",
        )

    return out


def main() -> None:
    # Never overwrite a previous dataset or build summary.
    require(not OUTPUT_PATH.exists(), f"Output already exists: {OUTPUT_PATH}")
    require(not SUMMARY_PATH.exists(), f"Summary already exists: {SUMMARY_PATH}")

    temp_csv = OUTPUT_PATH.with_suffix(OUTPUT_PATH.suffix + ".tmp")
    temp_json = SUMMARY_PATH.with_suffix(SUMMARY_PATH.suffix + ".tmp")
    require(not temp_csv.exists(), f"Temporary output already exists: {temp_csv}")
    require(not temp_json.exists(), f"Temporary summary already exists: {temp_json}")

    try:
        print("1/6 Reading primary district panel...")
        district = read_csv_checked(
            DISTRICT_PATH,
            KEYS + FIRE_FEATURES + ["year", "month", "day", "day_of_season"],
        )
        district = normalize_keys(district)
        check_unique(district, KEYS, "District panel")

        require(len(district) == 45_384,
                f"Unexpected district panel row count: {len(district):,}")

        expected_dates = set()
        for year in [2023, 2024, 2025]:
            expected_dates.update(
                pd.date_range(
                    f"{year}-10-01", f"{year}-11-30", freq="D"
                )
            )
        expected_dates = set(pd.to_datetime(list(expected_dates)))

        require(set(district["date"].unique()) == expected_dates,
                "District panel has an unexpected seasonal date set")

        district_counts = (
            district.groupby("state")["district"].nunique().to_dict()
        )
        require(len(district_counts) == 6,
                f"Expected six pilot states; found {len(district_counts)}")

        print("2/6 Reading district weather...")
        weather = read_csv_checked(
            WEATHER_PATH,
            KEYS + WEATHER_FEATURES,
            usecols=KEYS + WEATHER_FEATURES,
        )
        weather = normalize_keys(weather)
        check_unique(weather, KEYS, "Weather input")

        weather = weather[KEYS + WEATHER_FEATURES]
        weather_merge = district[KEYS].merge(
            weather, on=KEYS, how="left",
            validate="one_to_one", indicator=True,
        )
        unmatched_weather = int(
            (weather_merge["_merge"] != "both").sum()
        )
        require(unmatched_weather == 0,
                f"{unmatched_weather:,} district-days lack weather")
        require(weather_merge[WEATHER_FEATURES].notna().all().all(),
                "Weather contains missing values after exact-key join")

        print("3/6 Aggregating grid-derived district-day features...")
        grid_usecols = [
            "state", "district", "cell_id", "date",
            "fire_count", "frp_sum",
        ]
        grid = read_csv_checked(
            GRID_PATH,
            grid_usecols,
            usecols=grid_usecols,
        )
        grid = normalize_keys(grid)
        require(grid["cell_id"].notna().all(), "Grid panel has null cell IDs")
        require(grid["fire_count"].notna().all(), "Grid fire_count contains nulls")
        require((grid["fire_count"] >= 0).all(), "Negative grid fire count")
        require(grid["frp_sum"].notna().all(), "Grid frp_sum contains nulls")

        check_unique(grid, ["cell_id", "date"], "Grid panel")

        grid["is_active"] = (grid["fire_count"] > 0).astype("int64")

        grid_features = (
            grid.groupby(KEYS, as_index=False)
            .agg(
                active_cell_count=("is_active", "sum"),
                max_cell_fire_count=("fire_count", "max"),
                max_cell_frp=("frp_sum", "max"),
                grid_fire_count_sum=("fire_count", "sum"),
            )
        )

        # Verify the grid's district-day fire counts against the
        # primary district target source before using the features.
        reconciliation = district[KEYS + ["fire_count"]].merge(
            grid_features[KEYS + ["grid_fire_count_sum"]],
            on=KEYS,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        require((reconciliation["_merge"] == "both").all(),
                "Grid/district district-day key populations differ")

        count_diff = (
            reconciliation["fire_count"]
            - reconciliation["grid_fire_count_sum"]
        ).abs()
        max_count_diff = float(count_diff.max())
        require(max_count_diff == 0,
                f"Grid/district fire counts differ; max difference={max_count_diff}")

        grid_features = grid_features.drop(columns="grid_fire_count_sum")

        # Join grid features to the district-level base panel.
        panel = district.merge(
            grid_features, on=KEYS, how="left", validate="one_to_one"
        )
        require(panel[GRID_FEATURES[:3]].notna().all().all(),
                "Grid aggregation left missing feature values")

        # The primary district count and grid sum must agree.
        positive = panel["fire_count"] > 0
        panel["max_cell_share"] = 0.0
        panel.loc[positive, "max_cell_share"] = (
            panel.loc[positive, "max_cell_fire_count"]
            / panel.loc[positive, "fire_count"]
        )
        require(
            panel["max_cell_share"].between(0, 1).all(),
            "max_cell_share is outside [0, 1]",
        )

        print("4/6 Joining weather and constructing calendar features...")
        panel = panel.merge(
            weather, on=KEYS, how="left", validate="one_to_one"
        )
        require(panel[WEATHER_FEATURES].notna().all().all(),
                "Weather missing after panel merge")

        panel["day_of_year"] = panel["date"].dt.dayofyear
        panel["sin_year"] = np.sin(
            2 * np.pi * panel["day_of_year"] / 365.25
        )
        panel["cos_year"] = np.cos(
            2 * np.pi * panel["day_of_year"] / 365.25
        )

        # Exact calendar-date lags for base and grid-derived features.
        print("5/6 Constructing exact-date lags and +1-day targets...")
        lag_features = ["fire_count", "frp_sum"] + GRID_FEATURES
        panel = add_exact_date_lags(panel, lag_features)

        # Historical summaries use only the explicitly listed lag columns.
        for feature in ["fire_count", "frp_sum"]:
            short_cols = [f"{feature}_lag_{k}d" for k in [1, 2, 3]]
            panel[f"{feature}_mean_3d"] = panel[short_cols].mean(
                axis=1, skipna=False
            )

            seven_cols = [
                f"{feature}_lag_{k}d" for k in [1, 2, 3, 5, 7]
            ]
            fourteen_cols = seven_cols + [f"{feature}_lag_14d"]

            panel[f"{feature}_sparse_mean_7d"] = panel[seven_cols].mean(
                axis=1, skipna=False
            )
            panel[f"{feature}_sparse_mean_14d"] = panel[
                fourteen_cols
            ].mean(axis=1, skipna=False)

        # Target at t+1 is attached by matching source date t+1 to row t.
        future = panel[KEYS + ["fire_count", "frp_sum"]].copy()
        future["date"] = future["date"] - pd.Timedelta(days=1)
        future = future.rename(columns={
            "fire_count": "fire_count_t_plus_1d",
            "frp_sum": "frp_sum_t_plus_1d",
        })

        panel = panel.merge(
            future, on=KEYS, how="left", validate="one_to_one"
        )
        panel["target_elevated_t_plus_1d"] = pd.Series(
            pd.NA, index=panel.index, dtype="Int64"
        )
        known_target = panel["fire_count_t_plus_1d"].notna()
        panel.loc[known_target, "target_elevated_t_plus_1d"] = (
            panel.loc[known_target, "fire_count_t_plus_1d"] > 2
        ).astype("int64")

        panel = panel.sort_values(KEYS).reset_index(drop=True)

        require(len(panel) == len(district),
                "Unexpected row-count change during feature construction")
        check_unique(panel, KEYS, "Final ML dataset")

        # No training, threshold selection, or test evaluation in this script.
        known_targets = panel.loc[known_target]
        summary = {
            "status": "BUILD_PASS",
            "dataset": str(OUTPUT_PATH.relative_to(ROOT)),
            "source_rows": int(len(district)),
            "output_rows": int(len(panel)),
            "districts": int(panel[["state", "district"]].drop_duplicates().shape[0]),
            "states": sorted(panel["state"].unique().tolist()),
            "date_min": str(panel["date"].min().date()),
            "date_max": str(panel["date"].max().date()),
            "target_known_rows": int(known_target.sum()),
            "target_missing_rows": int((~known_target).sum()),
            "target_positive_rows": int(
                known_targets["target_elevated_t_plus_1d"].sum()
            ),
            "grid_district_count_max_abs_difference": max_count_diff,
            "weather_unmatched_rows": unmatched_weather,
            "grid_rows_read": int(len(grid)),
            "features": {
                "base_fire": FIRE_FEATURES,
                "weather": WEATHER_FEATURES,
                "grid": GRID_FEATURES,
                "lags_days": LAGS,
                "target_horizon_days": 1,
            },
            "note": (
                "Dataset construction only. Independent audit and "
                "V2 reconstruction are still required. No model trained."
            ),
        }

        print("6/6 Publishing V3-only outputs...")
        panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")
        panel.to_csv(temp_csv, index=False)

        with open(temp_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Recheck destinations before atomic publication.
        require(not OUTPUT_PATH.exists(), f"Output appeared during build: {OUTPUT_PATH}")
        require(not SUMMARY_PATH.exists(), f"Summary appeared during build: {SUMMARY_PATH}")

        os.replace(temp_csv, OUTPUT_PATH)
        os.replace(temp_json, SUMMARY_PATH)

        print(json.dumps(summary, indent=2))
        print("\nBUILD PASS")

    finally:
        # Clean up only temporary files created by this run.
        for temp_path in [temp_csv, temp_json]:
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    main()
