#!/usr/bin/env python
"""Build StubbleAI V3 grid-day and district-day panels.

Inputs are read-only. Outputs are written only under data/processed/v3/.
Primary grid: 0.05 degrees, origin (0, 0), centroid assignment with a
same-state largest-positive-area-overlap fallback. The 0.10-degree result is
a sensitivity baseline. Area-weighted district counts are retained separately
for a later forecasting-stage classification comparison.

This is a new V3 workflow; it does not modify V1 or frozen V2 artifacts.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

ROOT = Path(__file__).resolve().parents[2]
FIRMS_PATH = ROOT / "data/raw/firms/v3/firms_bbox_candidates_2023_2025.csv"
BOUNDARY_PATH = (
    ROOT / "data/raw/boundaries/v3/soi_extracted/"
    "State_District_Subdistrict_PAN INDIA/"
    "District_Subdistrict_PAN INDIA/District Boundary.shp"
)
OUT_DIR = ROOT / "data/processed/v3"
METRIC_CRS = "EPSG:6933"
GEO_CRS = "EPSG:4326"

STATE_NAMES = {
    "ASSAM": "Assam",
    "BIHAR": "Bihar",
    "HARYANA": "Haryana",
    "MADHYA PRADESH": "Madhya Pradesh",
    "PUNJAB": "Punjab",
    "UTTAR PRADESH": "Uttar Pradesh",
}
YEARS = (2023, 2024, 2025)
SEASON_MONTHS = (10, 11)
RESOLUTIONS = (0.05, 0.10)
AREA_TOLERANCE_M2 = 1.0
TIE_TOLERANCE_M2 = 1.0
CONFIDENCE_HIGH = {"h"}
DAY_VALUES = {"D"}
NIGHT_VALUES = {"N"}

RAW_COLUMNS = [
    "latitude", "longitude", "bright_ti4", "bright_t5", "bright_ti5",
    "frp", "acq_date", "acq_time", "satellite", "confidence", "daynight",
]
# bright_t5 is not a FIRMS field; the loader below selects only fields present.
AGG_NUMERIC = ["frp", "bright_ti4", "bright_ti5"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def study_dates() -> pd.DatetimeIndex:
    pieces = []
    for year in YEARS:
        pieces.append(pd.date_range(
            f"{year}-10-01", f"{year}-11-30", freq="D"
        ))
    dates = pieces[0].append(pieces[1:])
    require(len(dates) == 183, f"Expected 183 study dates, got {len(dates)}")
    return dates


def load_boundaries() -> gpd.GeoDataFrame:
    require(BOUNDARY_PATH.exists(), f"Boundary file not found: {BOUNDARY_PATH}")
    districts = gpd.read_file(BOUNDARY_PATH)
    required = {"STATE_UT", "DISTRICT", "DIST_LGD", "geometry"}
    missing = required - set(districts.columns)
    require(not missing, f"Boundary columns missing: {sorted(missing)}")
    require(districts.crs is not None, "Boundary CRS is missing")
    districts = districts.to_crs(GEO_CRS)
    districts["state_key"] = districts["STATE_UT"].astype(str).str.strip().str.upper()
    districts = districts[districts["state_key"].isin(STATE_NAMES)].copy()
    districts["state"] = districts["state_key"].map(STATE_NAMES)
    districts["district"] = districts["DISTRICT"].astype(str).str.strip()
    districts["district_lgd"] = districts["DIST_LGD"].astype(str).str.strip()
    districts["district_key"] = (
        districts["state_key"] + "|" + districts["district"].str.casefold()
    )
    require(len(districts) > 0, "No pilot-state boundaries found")
    require(districts.geometry.notna().all(), "Null boundary geometry found")
    require(int(districts.geometry.is_empty.sum()) == 0, "Empty boundary geometry found")
    require(districts.geometry.is_valid.all(), "Invalid boundary geometry found")
    counts = districts.groupby("state")["district_key"].nunique().to_dict()
    expected = {
        "Punjab": 23, "Haryana": 22, "Uttar Pradesh": 75,
        "Bihar": 38, "Madhya Pradesh": 55, "Assam": 35,
    }
    require(counts == expected, f"Unexpected pilot district counts: {counts}")
    require(not districts["district_key"].duplicated().any(),
            "Duplicate state/district keys in SOI boundaries")
    return districts.reset_index(drop=True)


def load_detections() -> pd.DataFrame:
    require(FIRMS_PATH.exists(), f"FIRMS CSV not found: {FIRMS_PATH}")
    header = pd.read_csv(FIRMS_PATH, nrows=0).columns.tolist()
    wanted = [
        "latitude", "longitude", "bright_ti4", "bright_ti5", "frp",
        "acq_date", "acq_time", "satellite", "confidence", "daynight",
    ]
    missing = set(wanted) - set(header)
    require(not missing, f"FIRMS columns missing: {sorted(missing)}")
    df = pd.read_csv(FIRMS_PATH, usecols=wanted)
    df["date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df[
        df["date"].dt.year.isin(YEARS)
        & df["date"].dt.month.isin(SEASON_MONTHS)
    ].copy()
    require(len(df) == 463212,
            f"Seasonal candidate count changed: expected 463212, got {len(df)}")
    require(df[["latitude", "longitude", "date", "frp"]].notna().all().all(),
            "Missing required coordinate/date/FRP values")
    df["confidence"] = df["confidence"].astype(str).str.strip().str.casefold()
    df["daynight"] = df["daynight"].astype(str).str.strip().str.upper()
    df["detection_id"] = np.arange(len(df), dtype=np.int64)
    return df


def assign_detection_states(
    detections: pd.DataFrame, districts: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, dict]:
    points = gpd.GeoDataFrame(
        detections.copy(),
        geometry=gpd.points_from_xy(detections["longitude"], detections["latitude"]),
        crs=GEO_CRS,
    )
    # Assign state/district directly first, to establish the state scope for each
    # detection. This direct district assignment is not the final grid assignment.
    joined = gpd.sjoin(
        points,
        districts[["state", "state_key", "district", "district_key", "geometry"]],
        how="left",
        predicate="within",
    )
    matches = joined.loc[joined["index_right"].notna()].copy()
    duplicate_ids = matches["detection_id"].duplicated(keep=False)
    require(not duplicate_ids.any(),
            "A detection falls within multiple pilot district polygons; stop for audit")
    pilot = matches.drop(columns=["index_right"]).copy()
    unmatched = len(detections) - len(pilot)
    expected_pilot = 256903
    require(len(pilot) == expected_pilot,
            f"Direct pilot detections changed: expected {expected_pilot}, got {len(pilot)}")
    require(pilot["state_key"].notna().all(), "Pilot detections have missing state")
    counts = {
        "candidate_detections": int(len(detections)),
        "pilot_detections": int(len(pilot)),
        "outside_or_unmatched_candidates": int(unmatched),
        "state_detection_counts": {
            str(k): int(v) for k, v in pilot["state"].value_counts().sort_index().items()
        },
    }
    return pilot, counts


def grid_ids(points: pd.DataFrame, resolution: float) -> pd.DataFrame:
    out = points.copy()
    out["grid_x"] = np.floor(out["longitude"] / resolution).astype(np.int64)
    out["grid_y"] = np.floor(out["latitude"] / resolution).astype(np.int64)
    out["resolution"] = resolution
    out["cell_id"] = (
        out["state_key"] + "|" + out["grid_x"].astype(str)
        + "|" + out["grid_y"].astype(str)
    )
    return out


def aggregate_detection_metrics(
    df: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    result = grouped.agg(
        fire_count=("detection_id", "size"),
        frp_sum=("frp", "sum"),
        frp_mean=("frp", "mean"),
        frp_max=("frp", "max"),
        high_confidence_fire_count=("confidence", lambda s: int(s.isin(CONFIDENCE_HIGH).sum())),
        day_fire_count=("daynight", lambda s: int(s.isin(DAY_VALUES).sum())),
        night_fire_count=("daynight", lambda s: int(s.isin(NIGHT_VALUES).sum())),
        bright_ti4_mean=("bright_ti4", "mean"),
        bright_ti4_max=("bright_ti4", "max"),
        bright_ti5_mean=("bright_ti5", "mean"),
    ).reset_index()
    result["fire_count"] = result["fire_count"].astype("int64")
    for col in ("high_confidence_fire_count", "day_fire_count", "night_fire_count"):
        result[col] = result[col].astype("int64")
    return result


def cell_polygons(cells: pd.DataFrame, resolution: float) -> gpd.GeoDataFrame:
    records = []
    for row in cells.itertuples(index=False):
        xmin = row.grid_x * resolution
        ymin = row.grid_y * resolution
        records.append({
            "state": row.state,
            "state_key": row.state_key,
            "grid_x": int(row.grid_x),
            "grid_y": int(row.grid_y),
            "cell_id": row.cell_id,
            "geometry": box(xmin, ymin, xmin + resolution, ymin + resolution),
        })
    return gpd.GeoDataFrame(records, geometry="geometry", crs=GEO_CRS)



def build_eligible_cells(
    districts: gpd.GeoDataFrame, resolution: float
) -> pd.DataFrame:
    """Enumerate every state-scoped grid cell with >1 m² pilot-state overlap."""
    import shapely
    import time

    rows = []
    districts_metric = districts.to_crs(METRIC_CRS)
    metric_unions = {}

    t0 = time.time()
    for state_key, part in districts_metric.groupby("state_key", sort=False):
        t_state = time.time()
        metric_unions[state_key] = shapely.union_all(part.geometry.array)
        print(
            f"    [{resolution}] union built: {state_key} "
            f"({len(part)} districts, {time.time() - t_state:.1f}s)"
        )
    print(f"    [{resolution}] all unions built in {time.time() - t0:.1f}s")

    for state_key, part in districts.groupby("state_key", sort=False):
        t_state = time.time()
        state_name = STATE_NAMES[state_key]
        union = shapely.union_all(part.geometry.array)
        minx, miny, maxx, maxy = union.bounds
        x_values = np.arange(
            math.floor(minx / resolution),
            math.floor(maxx / resolution) + 1,
            dtype=np.int64,
        )
        y_values = np.arange(
            math.floor(miny / resolution),
            math.floor(maxy / resolution) + 1,
            dtype=np.int64,
        )
        xx, yy = np.meshgrid(x_values, y_values)
        gx = xx.ravel()
        gy = yy.ravel()
        print(f"    [{resolution}] {state_key}: {len(gx):,} candidate cells before filtering")
        xmin = gx * resolution
        ymin = gy * resolution
        candidates = shapely.box(
            xmin, ymin, xmin + resolution, ymin + resolution
        )
        intersects = shapely.intersects(candidates, union)
        if not np.any(intersects):
            continue
        candidates = candidates[intersects]
        gx = gx[intersects]
        gy = gy[intersects]
        print(f"    [{resolution}] {state_key}: {len(gx):,} candidates after intersects filter")
        candidate_gdf = gpd.GeoDataFrame(
            {"grid_x": gx, "grid_y": gy},
            geometry=gpd.GeoSeries(candidates, crs=GEO_CRS),
            crs=GEO_CRS,
        )
        metric_candidates = candidate_gdf.to_crs(METRIC_CRS)
        overlap_area = metric_candidates.geometry.intersection(
            metric_unions[state_key]
        ).area.to_numpy()
        keep = overlap_area > AREA_TOLERANCE_M2
        for x, y in zip(gx[keep], gy[keep]):
            rows.append({
                "state": state_name,
                "state_key": state_key,
                "grid_x": int(x),
                "grid_y": int(y),
                "cell_id": f"{state_key}|{int(x)}|{int(y)}",
            })
        print(
            f"    [{resolution}] {state_key}: {int(keep.sum()):,} eligible cells "
            f"({time.time() - t_state:.1f}s)"
        )

    result = pd.DataFrame(rows).drop_duplicates(
        ["state_key", "grid_x", "grid_y"]
    )
    require(not result.empty, f"No eligible cells found at {resolution} degrees")
    require(not result.duplicated(["state_key", "grid_x", "grid_y"]).any(),
            "Duplicate eligible state-scoped grid cells")
    return result.sort_values(
        ["state_key", "grid_x", "grid_y"]
    ).reset_index(drop=True)

def resolve_cells_OLD(
    cells: gpd.GeoDataFrame, districts: gpd.GeoDataFrame, resolution: float
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return centroid/fallback assignment and same-state area weights."""
    district_records = []
    weight_records = []

    # Project once for metric area calculations. Grid IDs are state-scoped.
    cells_metric = cells.to_crs(METRIC_CRS)
    districts_metric = districts.to_crs(METRIC_CRS)
    district_by_state = {
        state_key: part.reset_index(drop=True)
        for state_key, part in districts.groupby("state_key", sort=False)
    }
    metric_by_state = {
        state_key: part.reset_index(drop=True)
        for state_key, part in districts_metric.groupby("state_key", sort=False)
    }

    for cell_row, cell_metric_row in zip(
        cells.itertuples(index=False), cells_metric.itertuples(index=False)
    ):
        state_key = cell_row.state_key
        same_state = district_by_state[state_key]
        same_state_metric = metric_by_state[state_key]
        centroid = cell_row.geometry.centroid
        # Strict containment is the primary rule. A centroid on a shared
        # boundary is unresolved and must use the documented overlap fallback.
        within = same_state[same_state.geometry.contains(centroid)]
        if len(within):
            chosen = within.sort_values("district_key").iloc[0]
            method = "centroid"
        else:
            chosen = None
            method = "largest_overlap_fallback"

        cell_metric = cell_metric_row.geometry
        overlaps = []
        for drow in same_state_metric.itertuples(index=False):
            if not drow.geometry.intersects(cell_metric):
                continue
            area = float(cell_metric.intersection(drow.geometry).area)
            if area > AREA_TOLERANCE_M2:
                overlaps.append((drow.district_key, drow.district, drow.state,
                                 area, drow.Index if hasattr(drow, "Index") else None))
        require(bool(overlaps), f"No positive same-state overlap for cell {cell_row.cell_id}")
        overlaps.sort(key=lambda x: x[0])
        total_area = sum(item[3] for item in overlaps)
        require(total_area > 0, f"Invalid overlap total for cell {cell_row.cell_id}")
        for district_key, district_name, state_name, area, _ in overlaps:
            weight_records.append({
                "state": state_name,
                "state_key": state_key,
                "cell_id": cell_row.cell_id,
                "grid_x": int(cell_row.grid_x),
                "grid_y": int(cell_row.grid_y),
                "district": district_name,
                "district_key": district_key,
                "area_m2": area,
                "area_weight": area / total_area,
            })

        if chosen is None:
            max_area = max(item[3] for item in overlaps)
            tied = [
                item for item in overlaps
                if max_area - item[3] <= TIE_TOLERANCE_M2
            ]
            chosen_item = sorted(tied, key=lambda x: x[0])[0]
            district_key, district_name, state_name = chosen_item[:3]
        else:
            district_key = chosen["district_key"]
            district_name = chosen["district"]
            state_name = chosen["state"]

        district_records.append({
            "state": state_name,
            "state_key": state_key,
            "cell_id": cell_row.cell_id,
            "grid_x": int(cell_row.grid_x),
            "grid_y": int(cell_row.grid_y),
            "district": district_name,
            "district_key": district_key,
            "assignment_method": method,
            "same_state_overlap_district_count": len(overlaps),
        })

    assignment = pd.DataFrame(district_records)
    weights = pd.DataFrame(weight_records)
    require(not assignment["cell_id"].duplicated().any(),
            "Cell assignment table contains duplicate cell IDs")
    weight_sums = weights.groupby("cell_id")["area_weight"].sum()
    require(np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-10, rtol=0),
            "Area weights do not sum to one per cell")
    return assignment, weights


def resolve_cells(
    cells: gpd.GeoDataFrame, districts: gpd.GeoDataFrame, resolution: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized assignment and area weights; verify against OLD first."""
    cells_metric = cells.to_crs(METRIC_CRS)
    districts_metric = districts.to_crs(METRIC_CRS)

    assignment_frames = []
    weight_frames = []

    for state_key, state_cells in cells.groupby("state_key", sort=False):
        state_cells_metric = cells_metric.loc[state_cells.index]
        state_districts = districts[districts["state_key"] == state_key]
        state_districts_metric = districts_metric[
            districts_metric["state_key"] == state_key
        ]

        # Intersections may create multiple geometry fragments for one pair.
        overlay = gpd.overlay(
            state_cells_metric[
                ["cell_id", "grid_x", "grid_y", "state", "state_key", "geometry"]
            ],
            state_districts_metric[
                ["district", "district_key", "geometry"]
            ],
            how="intersection",
            keep_geom_type=False,
        )
        overlay["area_m2"] = overlay.geometry.area

        overlay = (
            overlay.groupby(
                [
                    "cell_id", "grid_x", "grid_y", "state", "state_key",
                    "district", "district_key",
                ],
                as_index=False,
            )["area_m2"].sum()
        )
        overlay = overlay[overlay["area_m2"] > AREA_TOLERANCE_M2].copy()

        missing_cells = set(state_cells["cell_id"]) - set(overlay["cell_id"])
        require(
            not missing_cells,
            f"No positive same-state overlap for cells: "
            f"{sorted(missing_cells)[:5]} ({state_key})",
        )

        total_area = overlay.groupby("cell_id")["area_m2"].transform("sum")
        overlay["area_weight"] = overlay["area_m2"] / total_area

        weight_frames.append(
            overlay[
                [
                    "state", "state_key", "cell_id", "grid_x", "grid_y",
                    "district", "district_key", "area_m2", "area_weight",
                ]
            ]
        )

        # Strict centroid containment, matching the original .contains rule.
        centroids = state_cells[
            ["cell_id", "grid_x", "grid_y", "state", "state_key", "geometry"]
        ].copy()
        centroids["geometry"] = state_cells.to_crs(METRIC_CRS).geometry.centroid.to_crs(state_cells.crs)

        centroid_join = gpd.sjoin(
            centroids,
            state_districts[["district", "district_key", "geometry"]],
            how="left",
            predicate="within",
        )

        # The reference implementation picks the lexical first district if
        # multiple polygons contain a centroid. Preserve that behavior.
        resolved = centroid_join[
            centroid_join["index_right"].notna()
        ].copy()
        resolved = resolved.sort_values(["cell_id", "district_key"])
        resolved = resolved.drop_duplicates("cell_id", keep="first")
        resolved["assignment_method"] = "centroid"

        unresolved_ids = (
            set(state_cells["cell_id"]) - set(resolved["cell_id"])
        )

        if unresolved_ids:
            fb = overlay[overlay["cell_id"].isin(unresolved_ids)].copy()
            max_area = fb.groupby("cell_id")["area_m2"].transform("max")
            tied = fb[
                (max_area - fb["area_m2"]) <= TIE_TOLERANCE_M2
            ].copy()
            fallback = (
                tied.sort_values(["cell_id", "district_key"])
                .drop_duplicates("cell_id", keep="first")
                .copy()
            )
            require(
                len(fallback) == len(unresolved_ids),
                f"Fallback did not resolve every unresolved cell in {state_key}",
            )
            fallback["assignment_method"] = "largest_overlap_fallback"
        else:
            fallback = pd.DataFrame(
                columns=list(resolved.columns) + ["assignment_method"]
            )

        cols = [
            "cell_id", "grid_x", "grid_y", "state", "state_key",
            "district", "district_key", "assignment_method",
        ]
        state_assignment = pd.concat(
            [resolved[cols], fallback[cols]], ignore_index=True
        )

        overlap_counts = (
            overlay.groupby("cell_id")
            .size()
            .rename("same_state_overlap_district_count")
        )
        state_assignment = state_assignment.merge(
            overlap_counts,
            on="cell_id",
            how="left",
            validate="one_to_one",
        )

        require(
            len(state_assignment) == len(state_cells),
            f"Assignment row count mismatch for {state_key}",
        )
        assignment_frames.append(state_assignment)

    assignment = pd.concat(assignment_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)

    require(
        not assignment["cell_id"].duplicated().any(),
        "Cell assignment table contains duplicate cell IDs",
    )
    weight_sums = weights.groupby("cell_id")["area_weight"].sum()
    require(
        np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-10, rtol=0),
        "Area weights do not sum to one per cell",
    )

    return assignment, weights

def add_calendar_fields(panel: pd.DataFrame) -> pd.DataFrame:
    panel["date"] = pd.to_datetime(panel["date"])
    panel["year"] = panel["date"].dt.year.astype(int)
    panel["month"] = panel["date"].dt.month.astype(int)
    panel["day"] = panel["date"].dt.day.astype(int)
    # Calculate relative to each year’s October 1 so leap years are handled.
    season_start = pd.to_datetime(panel["year"].astype(str) + "-10-01")
    panel["day_of_season"] = (panel["date"] - season_start).dt.days + 1
    panel["day_of_season"] = panel["day_of_season"].astype(int)
    return panel


def make_zero_filled_panel(
    observed: pd.DataFrame, cell_metadata: pd.DataFrame,
    key_cols: list[str], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    keys = cell_metadata[key_cols].drop_duplicates().copy()
    date_frame = pd.DataFrame({"date": dates})
    keys["_join"] = 1
    date_frame["_join"] = 1
    full = keys.merge(date_frame, on="_join", how="inner").drop(columns="_join")
    panel = full.merge(observed, on=key_cols + ["date"], how="left", validate="one_to_one")
    count_cols = ["fire_count", "high_confidence_fire_count", "day_fire_count", "night_fire_count"]
    for col in count_cols:
        panel[col] = panel[col].fillna(0).astype("int64")
    panel["frp_sum"] = panel["frp_sum"].fillna(0.0)
    # Means/maxima remain NaN on no-fire days: there is no observed fire value.
    return panel

def make_zero_filled_count_panel(
    observed: pd.DataFrame, cell_metadata: pd.DataFrame,
    key_cols: list[str], dates: pd.DatetimeIndex, count_col: str = "fire_count",
) -> pd.DataFrame:
    keys = cell_metadata[key_cols].drop_duplicates().copy()
    date_frame = pd.DataFrame({"date": dates})
    keys["_join"] = 1
    date_frame["_join"] = 1
    full = keys.merge(date_frame, on="_join", how="inner").drop(columns="_join")
    panel = full.merge(observed, on=key_cols + ["date"], how="left", validate="one_to_one")
    panel[count_col] = panel[count_col].fillna(0.0)
    return panel

def build_resolution(
    pilot: gpd.GeoDataFrame,
    districts: gpd.GeoDataFrame,
    resolution: float,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    assigned = grid_ids(pilot, resolution)
    # Build the full spatial grid over pilot-state polygons, including cells
    # with no detections in any of the three study seasons.
    cell_meta = build_eligible_cells(districts, resolution)
    cells = cell_polygons(cell_meta, resolution)
    assignment, weights = resolve_cells(cells, districts, resolution)
    require(len(assignment) == len(cell_meta), "Not every occupied cell was assigned")
    # Cell-level assignment is authoritative; remove any point-level district
    # fields before merging to avoid pandas district_x/district_y suffixes.
    assigned = assigned.drop(
        columns=["district", "district_key", "assignment_method"],
        errors="ignore",
    )
    assigned = assigned.merge(
        assignment[["cell_id", "district", "district_key", "assignment_method"]],
        on="cell_id", how="left", validate="many_to_one",
    )
    require(assigned["district"].notna().all(), "Some detections lack cell district assignment")

    assigned["date"] = pd.to_datetime(assigned["date"])
    grid_observed = aggregate_detection_metrics(
        assigned, ["state", "state_key", "grid_x", "grid_y", "cell_id", "date"]
    )
    grid_meta = cell_meta[["state", "state_key", "grid_x", "grid_y", "cell_id"]]
    grid_panel = make_zero_filled_panel(
        grid_observed, grid_meta,
        ["state", "state_key", "grid_x", "grid_y", "cell_id"], dates,
    )
    grid_panel = add_calendar_fields(grid_panel)
    grid_panel["resolution_degrees"] = resolution
    grid_panel = grid_panel.sort_values(
        ["state_key", "grid_x", "grid_y", "date"]
    ).reset_index(drop=True)

    district_observed = aggregate_detection_metrics(assigned, ["state", "district", "date"])
    # Include every pilot district, including districts with no assigned detections.
    district_meta = (
        districts[["state", "district", "district_key"]]
        .drop_duplicates()
        .sort_values(["state", "district"])
    )
    district_panel = make_zero_filled_panel(
        district_observed, district_meta, ["state", "district"], dates
    )
    district_panel = add_calendar_fields(district_panel)
    district_panel = district_panel.sort_values(["state", "district", "date"]).reset_index(drop=True)

    # Area-weighted fire_count only, retained as a separate secondary artifact.
    counts = assigned.groupby(["cell_id", "date"], as_index=False).size().rename(
        columns={"size": "cell_fire_count"}
    )
    weighted = counts.merge(
        weights[["cell_id", "state", "district", "area_weight"]],
        on="cell_id", how="left", validate="many_to_many",
    )
    weighted["weighted_fire_count"] = weighted["cell_fire_count"] * weighted["area_weight"]
    area_observed = weighted.groupby(
        ["state", "district", "date"], as_index=False
    )["weighted_fire_count"].sum().rename(
        columns={"weighted_fire_count": "fire_count_area_weighted"}
    )
    area_meta = district_meta[["state", "district"]]
    area_panel = make_zero_filled_count_panel(
    area_observed.rename(columns={"fire_count_area_weighted": "fire_count"}),
    area_meta, ["state", "district"], dates,
)
    # area_panel = make_zero_filled_panel(
    #     area_observed.rename(columns={"fire_count_area_weighted": "fire_count"}),
    #     area_meta, ["state", "district"], dates,
    # )
    area_panel = area_panel[["state", "district", "date", "fire_count"]].rename(
        columns={"fire_count": "fire_count_area_weighted"}
    )
    area_panel = add_calendar_fields(area_panel)
    area_panel = area_panel.sort_values(["state", "district", "date"]).reset_index(drop=True)

    input_count = len(assigned)
    grid_count = int(grid_observed["fire_count"].sum())
    district_count = int(district_observed["fire_count"].sum())
    weighted_count = float(area_observed["fire_count_area_weighted"].sum())
    require(input_count == grid_count == district_count,
            f"Detection conservation failed at {resolution}: "
            f"input={input_count}, grid={grid_count}, district={district_count}")
    require(math.isclose(weighted_count, input_count, rel_tol=0, abs_tol=1e-6),
            f"Area-weighted count conservation failed at {resolution}: {weighted_count}")
    require(not grid_panel.duplicated(
        ["state_key", "grid_x", "grid_y", "date"]
    ).any(), "Duplicate grid-day keys")
    require(not district_panel.duplicated(["state", "district", "date"]).any(),
            "Duplicate district-day keys")
    require(len(grid_panel) == len(cell_meta) * len(dates),
            "Grid panel is missing zero-fire dates")
    require(len(district_panel) == len(district_meta) * len(dates),
            "District panel is missing zero-fire dates")

    stats = {
        "resolution_degrees": resolution,
        "occupied_state_scoped_cells": int(len(cell_meta)),
        "cell_assignments_by_method": {
            str(k): int(v) for k, v in assignment["assignment_method"].value_counts().items()
        },
        "detections": input_count,
        "grid_day_rows": int(len(grid_panel)),
        "district_day_rows": int(len(district_panel)),
        "area_weighted_total": weighted_count,
        "area_weight_sum_max_abs_error": float(
            (weights.groupby("cell_id")["area_weight"].sum() - 1.0).abs().max()
        ),
    }
    return grid_panel, district_panel, area_panel, stats

def run_haryana_equivalence_test() -> None:
    """Compare old and vectorized cell assignment on Haryana only."""
    import time

    print("Loading pilot boundaries...")
    districts = load_boundaries()
    haryana = districts[
        districts["state_key"] == "HARYANA"
    ].reset_index(drop=True)

    require(len(haryana) == 22, "Expected 22 Haryana districts")

    print("Building Haryana cells at 0.05 degrees...")
    cell_meta = build_eligible_cells(haryana, 0.05)
    cells = cell_polygons(cell_meta, 0.05)
    print(f"Cells: {len(cells):,}")

    print("Running original implementation...")
    start = time.time()
    old_assignment, old_weights = resolve_cells_OLD(
        cells, haryana, 0.05
    )
    old_seconds = time.time() - start
    print(f"Original runtime: {old_seconds:.2f}s")

    print("Running vectorized implementation...")
    start = time.time()
    new_assignment, new_weights = resolve_cells(
        cells, haryana, 0.05
    )
    new_seconds = time.time() - start
    print(f"Vectorized runtime: {new_seconds:.2f}s")

    failures = []

    # Compare selected district and assignment method for every cell.
    old_a = (
        old_assignment[
            ["cell_id", "district", "assignment_method"]
        ]
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    new_a = (
        new_assignment[
            ["cell_id", "district", "assignment_method"]
        ]
        .sort_values("cell_id")
        .reset_index(drop=True)
    )

    if not old_a.equals(new_a):
        failures.append("Assignment rows differ")
        diff = old_a.merge(
            new_a,
            on="cell_id",
            how="outer",
            suffixes=("_old", "_new"),
            indicator=True,
        )
        mismatched = diff[
            (diff["_merge"] != "both")
            | (diff["district_old"] != diff["district_new"])
            | (
                diff["assignment_method_old"]
                != diff["assignment_method_new"]
            )
        ]
        print("First assignment differences:")
        print(mismatched.head(20).to_string(index=False))

    # Compare the complete cell-district pair sets.
    old_pairs = set(
        zip(old_weights["cell_id"], old_weights["district_key"])
    )
    new_pairs = set(
        zip(new_weights["cell_id"], new_weights["district_key"])
    )

    if old_pairs != new_pairs:
        failures.append("Cell-district overlap pairs differ")
        print("Pairs missing in vectorized result:", len(old_pairs - new_pairs))
        print("Unexpected vectorized pairs:", len(new_pairs - old_pairs))

    # Compare weights for each exact cell-district pair.
    old_w = old_weights[
        ["cell_id", "district_key", "area_weight"]
    ].rename(columns={"area_weight": "weight_old"})
    new_w = new_weights[
        ["cell_id", "district_key", "area_weight"]
    ].rename(columns={"area_weight": "weight_new"})

    merged = old_w.merge(
        new_w,
        on=["cell_id", "district_key"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if not (merged["_merge"] == "both").all():
        failures.append("Weight comparison has missing or extra pairs")
    else:
        max_difference = (
            merged["weight_old"] - merged["weight_new"]
        ).abs().max()
        print(f"Maximum area-weight difference: {max_difference:.12g}")
        if max_difference > 1e-9:
            failures.append("Area weights differ beyond 1e-9")

    # Both implementations must conserve weights.
    for label, weights in [
        ("Original", old_weights),
        ("Vectorized", new_weights),
    ]:
        sums = weights.groupby("cell_id")["area_weight"].sum()
        if not np.allclose(
            sums.to_numpy(), 1.0, atol=1e-10, rtol=0
        ):
            failures.append(f"{label} weights do not sum to one")

    print("\n" + "=" * 55)
    if failures:
        print("HARYANA EQUIVALENCE TEST: FAIL")
        for failure in failures:
            print("-", failure)
        print("=" * 55)
        raise SystemExit(1)

    print("HARYANA EQUIVALENCE TEST: PASS")
    print(f"Cells compared: {len(cells):,}")
    print(f"Original runtime: {old_seconds:.2f}s")
    print(f"Vectorized runtime: {new_seconds:.2f}s")
    if new_seconds > 0:
        print(f"Speedup: {old_seconds / new_seconds:.2f}x")
    print("=" * 55)


def run_six_state_validation() -> None:
    """Validate vectorized assignments for all six pilot states and resolutions."""
    import time

    print("Loading pilot boundaries...")
    districts = load_boundaries()
    states = sorted(districts["state_key"].unique())
    print(f"States: {', '.join(states)}")

    for resolution in RESOLUTIONS:
        print(f"\nValidating resolution {resolution:.2f} degrees")
        for state_key in states:
            state_districts = districts[
                districts["state_key"] == state_key
            ].reset_index(drop=True)

            start = time.time()
            cell_meta = build_eligible_cells(
                state_districts, resolution
            )
            cells = cell_polygons(cell_meta, resolution)

            assignment, weights = resolve_cells(
                cells, state_districts, resolution
            )

            require(
                len(assignment) == len(cells),
                f"{state_key}: assignment count mismatch at {resolution}",
            )
            require(
                not assignment["cell_id"].duplicated().any(),
                f"{state_key}: duplicate cell assignments at {resolution}",
            )

            weight_sums = weights.groupby("cell_id")["area_weight"].sum()
            require(
                len(weight_sums) == len(cells),
                f"{state_key}: missing cell weights at {resolution}",
            )
            require(
                np.allclose(
                    weight_sums.to_numpy(), 1.0, atol=1e-10, rtol=0
                ),
                f"{state_key}: area weights do not sum to one at {resolution}",
            )

            print(
                f"  PASS {state_key}: {len(cells):,} cells, "
                f"{len(weights):,} overlap pairs, "
                f"{time.time() - start:.2f}s"
            )

    print("\nSIX-STATE VALIDATION: PASS")



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Count eligible cells and projected panel rows without loading FIRMS or writing outputs.",
    )
    parser.add_argument(
        "--test-haryana",
        action="store_true",
        help="Compare original and vectorized cell assignment on Haryana only.",
    )
    parser.add_argument(
        "--validate-six-states",
        action="store_true",
        help="Validate vectorized assignments for all pilot states and resolutions.",
    )
    args = parser.parse_args()

    if args.validate_six_states:
        run_six_state_validation()
        return

    if args.test_haryana:
        run_haryana_equivalence_test()
        return

    dates = study_dates()

    print("Loading SOI pilot boundaries...")
    districts = load_boundaries()
    if args.plan_only:
        print("Planning full spatial grids; no FIRMS data loaded and no outputs written.")
        plan = {}
        for resolution in RESOLUTIONS:
            cells = build_eligible_cells(districts, resolution)
            plan[f"{resolution:.2f}"] = {
                "eligible_state_scoped_cells": int(len(cells)),
                "grid_day_rows": int(len(cells) * len(dates)),
                "district_day_rows": int(districts["district_key"].nunique() * len(dates)),
            }
        print(__import__("json").dumps({
            "study_dates": int(len(dates)),
            "resolutions": plan,
            "note": "Counts are estimates of rows, not file-size predictions.",
        }, indent=2))
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading FIRMS seasonal candidates...")
    detections = load_detections()
    print(f"Candidates: {len(detections):,}")
    print("Assigning detections to pilot states...")
    pilot, spatial_summary = assign_detection_states(detections, districts)
    print("Spatial filtering summary:", spatial_summary)

    results = {}
    for resolution in RESOLUTIONS:
        print(f"Building {resolution:.2f}-degree panels...")
        results[resolution] = build_resolution(pilot, districts, resolution, dates)

    grid005, district005, area005, stats005 = results[0.05]
    _, district010, _, stats010 = results[0.10]

    grid_path = OUT_DIR / "grid_day_panel_005_centroid_fallback.csv"
    district005_path = OUT_DIR / "district_day_panel_005_centroid_fallback.csv"
    district010_path = OUT_DIR / "district_day_panel_010_centroid_fallback_sensitivity.csv"
    area_path = OUT_DIR / "district_day_fire_count_005_area_weighted_secondary.csv"
    summary_path = OUT_DIR / "stage4_panel_build_summary.json"

    grid005.to_csv(grid_path, index=False, date_format="%Y-%m-%d")
    district005.to_csv(district005_path, index=False, date_format="%Y-%m-%d")
    district010.to_csv(district010_path, index=False, date_format="%Y-%m-%d")
    area005.to_csv(area_path, index=False, date_format="%Y-%m-%d")

    summary = {
        "stage": "V3 Stage 4 grid-day panel build",
        "study_dates": {
            "count": len(dates),
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
            "months": list(SEASON_MONTHS),
            "years": list(YEARS),
        },
        "primary_method": "0.05-degree state-scoped grid; centroid assignment; "
                          "same-state largest positive-area overlap fallback",
        "sensitivity_method": "0.10-degree grid using the same assignment protocol",
        "area_weighted_status": "secondary count-only artifact; deferred forecasting comparison",
        "boundary_source": str(BOUNDARY_PATH.relative_to(ROOT)),
        "firms_source": str(FIRMS_PATH.relative_to(ROOT)),
        "spatial_filter": spatial_summary,
        "resolution_005": stats005,
        "resolution_010": stats010,
        "outputs": {
            "grid_day_primary": str(grid_path.relative_to(ROOT)),
            "district_day_primary": str(district005_path.relative_to(ROOT)),
            "district_day_sensitivity": str(district010_path.relative_to(ROOT)),
            "district_day_area_weighted_secondary": str(area_path.relative_to(ROOT)),
        },
        "notes": [
            "All state-scoped grid cells with positive-area overlap with pilot-state polygons are expanded across the 183 study dates.",
            "Zero-fire counts are zero; means and maxima remain null on zero-fire days.",
            "Area-weighted artifact contains fractional fire_count only; do not compare its exact-match rate directly with integer counts.",
            "This build is independent and is not claimed to reproduce the historical audit until its metrics are checked.",
        ],
    }
    summary_path.write_text(
        __import__("json").dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nBUILD COMPLETE")
    print(__import__("json").dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
