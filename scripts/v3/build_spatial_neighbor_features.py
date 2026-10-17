
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GRID_PATH = ROOT / (
    "data/processed/v3/"
    "grid_day_panel_005_centroid_fallback_weather.csv"
)
OUT_DIR = ROOT / "data/processed/v3/experiments/spatial_neighbors_v3"
OUTPUT_PATH = OUT_DIR / "district_neighbor_features.csv"
SUMMARY_PATH = OUT_DIR / "build_summary.json"

KEYS = ["state", "district", "date"]
CELL_KEYS = ["state_key", "grid_x", "grid_y", "date"]
OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),             (0, 1),
    (1, -1),  (1, 0),    (1, 1),
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    require(GRID_PATH.is_file(), f"Missing input: {GRID_PATH}")
    require(not OUTPUT_PATH.exists(), f"Refusing to overwrite: {OUTPUT_PATH}")
    require(not SUMMARY_PATH.exists(), f"Refusing to overwrite: {SUMMARY_PATH}")

    columns = [
        "state", "state_key", "grid_x", "grid_y", "cell_id", "date",
        "fire_count", "frp_sum", "district", "district_key",
    ]
    grid = pd.read_csv(GRID_PATH, usecols=columns)
    grid["date"] = pd.to_datetime(grid["date"], errors="raise").dt.normalize()

    for col in ["state", "state_key", "district", "district_key", "cell_id"]:
        grid[col] = grid[col].astype("string").str.strip()

    require(grid[columns].notna().all().all(), "Null required input values")
    require(not grid.duplicated(["cell_id", "date"]).any(),
            "Duplicate cell/date keys")
    require((grid["fire_count"] >= 0).all(), "Negative fire counts")
    require((grid["frp_sum"] >= 0).all(), "Negative FRP values")

    # Each cell's observed prior-day activity is shifted onto date t.
    source = grid[
        ["state_key", "grid_x", "grid_y", "date", "fire_count", "frp_sum"]
    ].copy()
    source["date"] += pd.Timedelta(days=1)
    source = source.rename(columns={
        "fire_count": "neighbor_fire_count",
        "frp_sum": "neighbor_frp_sum",
    })

    # Build the physical neighbor count from the cells that actually exist.
    # State-scoped coordinates prevent neighbors crossing state boundaries.
    existing_cells = set(
        map(
            tuple,
            grid[["state_key", "grid_x", "grid_y"]]
            .drop_duplicates()
            .itertuples(index=False, name=None),
        )
    )

    # Process one neighbor offset at a time to limit peak memory.
    targets = grid[
        ["state", "state_key", "grid_x", "grid_y", "cell_id",
         "date", "district", "district_key"]
    ].drop_duplicates()

    cell_keys = ["state", "district", "date", "state_key", "cell_id"]

    cell_day = targets[cell_keys].copy()
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

    for dx, dy in OFFSETS:
        print(f"Processing neighbor offset ({dx}, {dy})...", flush=True)
        part = source.copy()
        part["grid_x"] -= dx
        part["grid_y"] -= dy

        matched = targets[
            ["state_key", "grid_x", "grid_y", "date", "cell_id"]
        ].merge(
            part,
            on=["state_key", "grid_x", "grid_y", "date"],
            how="left",
            validate="one_to_one",
        )

        matched["neighbor_observed"] = (
            matched["neighbor_fire_count"].notna().astype("int8")
        )
        matched["neighbor_active"] = (
            matched["neighbor_fire_count"].gt(0).fillna(False).astype("int8")
        )

        update = matched[
            ["cell_id", "date", "neighbor_observed",
             "neighbor_fire_count", "neighbor_frp_sum", "neighbor_active"]
        ].rename(columns={
            "neighbor_observed": "_observed",
            "neighbor_fire_count": "_fires",
            "neighbor_frp_sum": "_frp",
            "neighbor_active": "_active",
        })

        cell_day = cell_day.merge(
            update,
            on=["cell_id", "date"],
            how="left",
            validate="one_to_one",
        )
        cell_day["neighbor_observed_count"] += (
            cell_day["_observed"].fillna(0).astype("int8")
        )
        cell_day["neighbor_fire_count_observed_sum"] += (
            cell_day["_fires"].fillna(0)
        )
        cell_day["neighbor_frp_sum_observed_sum"] += (
            cell_day["_frp"].fillna(0)
        )
        cell_day["neighbor_active_observed_count"] += (
            cell_day["_active"].fillna(0).astype("int8")
        )
        cell_day["neighbor_max_fire_count"] = cell_day[
            ["neighbor_max_fire_count", "_fires"]
        ].max(axis=1)

        cell_day.drop(
            columns=["_observed", "_fires", "_frp", "_active"],
            inplace=True,
        )
        del part, matched, update

    cell_day = cell_day.rename(columns={"state": "state", "district": "district"})
    cell_day["neighbor_fire_count_mean"] = (
        cell_day["neighbor_fire_count_observed_sum"]
        / cell_day["neighbor_observed_count"].replace(0, float("nan"))
    )
    cell_day["neighbor_frp_sum_mean"] = (
        cell_day["neighbor_frp_sum_observed_sum"]
        / cell_day["neighbor_observed_count"].replace(0, float("nan"))
    )
    # Count physical neighbors independently of date availability.
    physical_counts = []
    for row in targets[["state_key", "grid_x", "grid_y"]].drop_duplicates().itertuples(index=False):
        state_key, x, y = row
        physical_counts.append(
            sum((state_key, x + dx, y + dy) in existing_cells for dx, dy in OFFSETS)
        )

    physical_lookup = (
        targets[["state_key", "grid_x", "grid_y", "cell_id"]]
        .drop_duplicates()
        .assign(physical_neighbor_count=physical_counts)
        [["cell_id", "physical_neighbor_count"]]
    )
    cell_day = cell_day.merge(
        physical_lookup, on="cell_id", how="left", validate="many_to_one"
    )
    cell_day["neighbor_coverage_fraction"] = (
        cell_day["neighbor_observed_count"]
        / cell_day["physical_neighbor_count"].replace(0, float("nan"))
    )

    # Each annual observation window begins on October 1.
    # September 30 is outside the panel, so its neighbor activity is unavailable.
    boundary = (
        cell_day["date"].dt.month.eq(10)
        & cell_day["date"].dt.day.eq(1)
    )
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

    district_day = (
        cell_day.groupby(KEYS, as_index=False)
        .agg(
            neighbor_fire_count_lag_1d_mean=(
                "neighbor_fire_count_observed_sum", "mean"
            ),
            neighbor_fire_count_lag_1d_max=(
                "neighbor_fire_count_observed_sum", "max"
            ),
            neighbor_fire_count_per_neighbor_mean=(
                "neighbor_fire_count_mean", "mean"
            ),
            neighbor_frp_sum_lag_1d_mean=(
                "neighbor_frp_sum_observed_sum", "mean"
            ),
            neighbor_frp_per_neighbor_mean=(
                "neighbor_frp_sum_mean", "mean"
            ),
            neighbor_active_cells_lag_1d_mean=(
                "neighbor_active_observed_count", "mean"
            ),
            neighbor_max_fire_count_lag_1d_max=(
                "neighbor_max_fire_count", "max"
            ),
            neighbor_coverage_fraction_mean=(
                "neighbor_coverage_fraction", "mean"
            ),
        )
    )

    require(not district_day.duplicated(KEYS).any(),
            "Duplicate district/day output keys")
    require(district_day[KEYS].notna().all().all(),
            "Null output keys")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    district_day.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "input_rows": int(len(grid)),
        "unique_cells": int(grid["cell_id"].nunique()),
        "output_rows": int(len(district_day)),
        "date_min": str(grid["date"].min().date()),
        "date_max": str(grid["date"].max().date()),
        "offsets": [list(x) for x in OFFSETS],
        "feature_timing": "Observed neighboring-cell activity on exact date t-1",
        "cross_state_neighbors": False,
        "absent_spatial_cells_counted_as_neighbors": False,
        "neighbor_observation_coverage_reported": True,
        "future_target_data_used": False,
        "output_columns": district_day.columns.tolist(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Input rows: {len(grid):,}")
    print(f"Unique cells: {grid['cell_id'].nunique():,}")
    print(f"District-day rows: {len(district_day):,}")
    print(f"Output: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Summary: {SUMMARY_PATH.relative_to(ROOT)}")
    print("Status: COMPLETE")


if __name__ == "__main__":
    main()
