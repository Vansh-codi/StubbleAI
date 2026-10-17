from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/processed/v3/experiments/grid_sensitivity_010"
GRID_PATH = OUT_DIR / "grid_day_panel_010_centroid_fallback_weather.csv"
OUTPUT_PATH = OUT_DIR / "district_neighbor_features_010.csv"
SUMMARY_PATH = OUT_DIR / "neighbor_features_010_summary.json"

KEYS = ["state", "district", "date"]
OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),             (0, 1),
    (1, -1),  (1, 0),    (1, 1),
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def preflight() -> None:
    require(GRID_PATH.is_file(), f"Missing input: {GRID_PATH}")
    require(not OUTPUT_PATH.exists(), f"Refusing to overwrite: {OUTPUT_PATH}")
    require(not SUMMARY_PATH.exists(), f"Refusing to overwrite: {SUMMARY_PATH}")
    print("PREFLIGHT PASS")
    print("Input:", GRID_PATH)
    print("Features output:", OUTPUT_PATH)
    print("Summary output:", SUMMARY_PATH)
    print("Resolution: 0.10 degrees")
    print("No data loaded and no outputs written.")


def main() -> None:
    preflight()

    columns = [
        "state", "state_key", "grid_x", "grid_y", "cell_id", "date",
        "fire_count", "frp_sum", "district", "district_key",
    ]
    print("\nLoading 0.10-degree weather-enriched grid panel...")
    grid = pd.read_csv(GRID_PATH, usecols=columns)
    grid["date"] = pd.to_datetime(grid["date"], errors="raise").dt.normalize()

    for col in ["state", "state_key", "district", "district_key", "cell_id"]:
        grid[col] = grid[col].astype("string").str.strip()

    require(grid[columns].notna().all().all(), "Null required input values")
    require(not grid.duplicated(["cell_id", "date"]).any(),
            "Duplicate cell/date keys")
    require((grid["fire_count"] >= 0).all(), "Negative fire counts")
    require((grid["frp_sum"] >= 0).all(), "Negative FRP values")

    targets = grid[
        ["state", "state_key", "grid_x", "grid_y", "cell_id",
         "date", "district", "district_key"]
    ].copy()

    existing_cells = set(
        map(
            tuple,
            grid[["state_key", "grid_x", "grid_y"]]
            .drop_duplicates()
            .itertuples(index=False, name=None),
        )
    )

    cell_day = targets[
        ["state", "district", "date", "state_key", "cell_id"]
    ].copy()
    cell_day["neighbor_observed_count"] = 0
    cell_day["neighbor_fire_count_observed_sum"] = 0.0
    cell_day["neighbor_frp_sum_observed_sum"] = 0.0
    cell_day["neighbor_active_observed_count"] = 0
    cell_day["neighbor_max_fire_count"] = 0.0

    source = grid[
        ["state_key", "grid_x", "grid_y", "date", "fire_count", "frp_sum"]
    ].copy()
    source["date"] += pd.Timedelta(days=1)
    source = source.rename(columns={
        "fire_count": "neighbor_fire_count",
        "frp_sum": "neighbor_frp_sum",
    })

    target_cells = targets[
        ["state_key", "grid_x", "grid_y", "date", "cell_id"]
    ]

    for dx, dy in OFFSETS:
        print(f"Processing neighbor offset ({dx}, {dy})...", flush=True)
        part = source.copy()
        part["grid_x"] -= dx
        part["grid_y"] -= dy

        matched = target_cells.merge(
            part,
            on=["state_key", "grid_x", "grid_y", "date"],
            how="left",
            validate="one_to_one",
        )

        update = matched[
            ["cell_id", "date", "neighbor_fire_count", "neighbor_frp_sum"]
        ].copy()
        update["_observed"] = update["neighbor_fire_count"].notna().astype("int8")
        update["_active"] = update["neighbor_fire_count"].gt(0).fillna(False).astype("int8")
        update = update.rename(columns={
            "neighbor_fire_count": "_fires",
            "neighbor_frp_sum": "_frp",
        })

        cell_day = cell_day.merge(
            update,
            on=["cell_id", "date"],
            how="left",
            validate="one_to_one",
        )
        cell_day["neighbor_observed_count"] += cell_day["_observed"].fillna(0).astype("int8")
        cell_day["neighbor_fire_count_observed_sum"] += cell_day["_fires"].fillna(0)
        cell_day["neighbor_frp_sum_observed_sum"] += cell_day["_frp"].fillna(0)
        cell_day["neighbor_active_observed_count"] += cell_day["_active"].fillna(0).astype("int8")
        cell_day["neighbor_max_fire_count"] = cell_day[
            ["neighbor_max_fire_count", "_fires"]
        ].max(axis=1)
        cell_day.drop(columns=["_observed", "_fires", "_frp", "_active"], inplace=True)
        del part, matched, update

    cell_day["neighbor_fire_count_mean"] = (
        cell_day["neighbor_fire_count_observed_sum"]
        / cell_day["neighbor_observed_count"].replace(0, float("nan"))
    )
    cell_day["neighbor_frp_sum_mean"] = (
        cell_day["neighbor_frp_sum_observed_sum"]
        / cell_day["neighbor_observed_count"].replace(0, float("nan"))
    )

    unique_cells = targets[
        ["state_key", "grid_x", "grid_y", "cell_id"]
    ].drop_duplicates()
    physical_counts = []
    for row in unique_cells[["state_key", "grid_x", "grid_y"]].itertuples(index=False):
        state_key, x, y = row
        physical_counts.append(
            sum((state_key, x + dx, y + dy) in existing_cells for dx, dy in OFFSETS)
        )
    physical_lookup = unique_cells[["cell_id"]].copy()
    physical_lookup["physical_neighbor_count"] = physical_counts
    cell_day = cell_day.merge(
        physical_lookup, on="cell_id", how="left", validate="many_to_one"
    )
    cell_day["neighbor_coverage_fraction"] = (
        cell_day["neighbor_observed_count"]
        / cell_day["physical_neighbor_count"].replace(0, float("nan"))
    )

    boundary = cell_day["date"].dt.month.eq(10) & cell_day["date"].dt.day.eq(1)
    boundary_features = [
        "neighbor_fire_count_observed_sum",
        "neighbor_frp_sum_observed_sum",
        "neighbor_active_observed_count",
        "neighbor_max_fire_count",
        "neighbor_fire_count_mean",
        "neighbor_frp_sum_mean",
    ]
    cell_day.loc[boundary, boundary_features] = float("nan")
    cell_day.loc[boundary, "neighbor_coverage_fraction"] = float("nan")

    district_day = cell_day.groupby(KEYS, as_index=False).agg(
        neighbor_fire_count_lag_1d_mean=("neighbor_fire_count_observed_sum", "mean"),
        neighbor_fire_count_lag_1d_max=("neighbor_fire_count_observed_sum", "max"),
        neighbor_fire_count_per_neighbor_mean=("neighbor_fire_count_mean", "mean"),
        neighbor_frp_sum_lag_1d_mean=("neighbor_frp_sum_observed_sum", "mean"),
        neighbor_frp_per_neighbor_mean=("neighbor_frp_sum_mean", "mean"),
        neighbor_active_cells_lag_1d_mean=("neighbor_active_observed_count", "mean"),
        neighbor_max_fire_count_lag_1d_max=("neighbor_max_fire_count", "max"),
        neighbor_coverage_fraction_mean=("neighbor_coverage_fraction", "mean"),
    )

    require(not district_day.duplicated(KEYS).any(), "Duplicate district/day output keys")
    require(district_day[KEYS].notna().all().all(), "Null output keys")
    require(len(district_day) == grid[["state", "district", "date"]].drop_duplicates().shape[0],
            "Output district-day coverage differs from grid input")

    # Publish only after feature generation and validation succeed.
    temp_features = OUT_DIR / "district_neighbor_features_010.tmp.csv"
    temp_summary = OUT_DIR / "neighbor_features_010_summary.tmp.json"
    require(not temp_features.exists(), f"Refusing to overwrite: {temp_features}")
    require(not temp_summary.exists(), f"Refusing to overwrite: {temp_summary}")

    summary = {
        "resolution_degrees": 0.10,
        "input_rows": int(len(grid)),
        "unique_cells": int(grid["cell_id"].nunique()),
        "output_rows": int(len(district_day)),
        "date_min": str(grid["date"].min().date()),
        "date_max": str(grid["date"].max().date()),
        "offsets": [list(x) for x in OFFSETS],
        "feature_timing": "Observed neighboring-cell activity on exact date t-1",
        "cross_state_neighbors": False,
        "absent_spatial_cells_counted_as_neighbors": False,
        "future_target_data_used": False,
        "weather_granularity": "District-day weather repeated across cells in each district",
        "output_columns": district_day.columns.tolist(),
    }

    try:
        district_day.to_csv(temp_features, index=False)
        temp_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        require(not OUTPUT_PATH.exists() and not SUMMARY_PATH.exists(),
                "Final output appeared during build; refusing to overwrite")
        temp_features.replace(OUTPUT_PATH)
        temp_summary.replace(SUMMARY_PATH)
    finally:
        if temp_features.exists():
            temp_features.unlink()
        if temp_summary.exists():
            temp_summary.unlink()

    print("\nNEIGHBOR FEATURE BUILD PASSED")
    print("Input grid-day rows:", f"{len(grid):,}")
    print("Unique cells:", f"{grid['cell_id'].nunique():,}")
    print("District-day rows:", f"{len(district_day):,}")
    print("Neighbor features:", len(district_day.columns) - len(KEYS))
    print("Output:", OUTPUT_PATH)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        main()
