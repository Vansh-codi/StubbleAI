from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "scripts/v3/build_grid_day_panel.py"
FIRMS_PATH = ROOT / "data/raw/firms/v3/firms_bbox_candidates_2023_2025.csv"
BOUNDARY_PATH = (
    ROOT / "data/raw/boundaries/v3/soi_extracted/"
    "State_District_Subdistrict_PAN INDIA/"
    "District_Subdistrict_PAN INDIA/District Boundary.shp"
)
OUT_DIR = ROOT / "data/processed/v3/experiments/grid_sensitivity_010"
GRID_OUT = OUT_DIR / "grid_day_panel_010_centroid_fallback.csv"
SUMMARY_OUT = OUT_DIR / "grid_build_summary.json"
GRID_TMP = OUT_DIR / "grid_day_panel_010_centroid_fallback.tmp.csv"
SUMMARY_TMP = OUT_DIR / "grid_build_summary.tmp.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "v3_grid_builder_sensitivity_010", BUILDER_PATH
    )
    require(spec is not None and spec.loader is not None,
            f"Cannot load existing builder: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_paths(builder) -> None:
    require(BUILDER_PATH.is_file(), f"Missing builder: {BUILDER_PATH}")
    require(FIRMS_PATH.is_file(), f"Missing FIRMS input: {FIRMS_PATH}")
    require(BOUNDARY_PATH.is_file(), f"Missing boundary input: {BOUNDARY_PATH}")
    require(0.10 in builder.RESOLUTIONS,
            "Existing builder does not declare 0.10-degree support.")
    for path in (GRID_OUT, SUMMARY_OUT, GRID_TMP, SUMMARY_TMP):
        require(not path.exists(), f"Refusing to overwrite existing file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an isolated V3 0.10-degree grid panel."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--build-grid", action="store_true")
    args = parser.parse_args()

    builder = load_builder()
    check_paths(builder)

    print("Output directory:", OUT_DIR)
    print("Grid output:", GRID_OUT)
    print("Summary output:", SUMMARY_OUT)
    print("Resolution: 0.10 degrees")
    print("Existing 0.05-degree artifacts will not be written.")

    if args.preflight:
        print("\nPREFLIGHT PASS — inputs and output paths validated.")
        print("No data loaded, no directories created, no outputs written.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading study dates and district boundaries...", flush=True)
    dates = builder.study_dates()
    districts = builder.load_boundaries()

    print("Loading FIRMS seasonal candidates...", flush=True)
    detections = builder.load_detections()
    print(f"Candidates loaded: {len(detections):,}", flush=True)

    print("Assigning detections to pilot states...", flush=True)
    pilot, spatial_summary = builder.assign_detection_states(
        detections, districts
    )
    print("Spatial filtering summary:", spatial_summary, flush=True)

    print("Building 0.10-degree grid panel...", flush=True)
    grid_panel, _, _, stats = builder.build_resolution(
        pilot, districts, 0.10, dates
    )

    require(len(grid_panel) == stats["grid_day_rows"],
            "Grid panel row count disagrees with builder statistics.")
    require(not grid_panel.duplicated(
        ["state_key", "grid_x", "grid_y", "date"]
    ).any(), "Duplicate grid-day keys.")
    require(set(grid_panel["resolution_degrees"].unique()) == {0.10},
            "Unexpected grid resolution in output.")

    summary = {
        "stage": "V3 isolated 0.10-degree grid sensitivity build",
        "resolution_degrees": 0.10,
        "study_dates": len(dates),
        "start_date": str(dates.min().date()),
        "end_date": str(dates.max().date()),
        "boundary_source": str(BOUNDARY_PATH.relative_to(ROOT)),
        "firms_source": str(FIRMS_PATH.relative_to(ROOT)),
        "spatial_filter": spatial_summary,
        "builder_statistics": stats,
        "grid_output": str(GRID_OUT.relative_to(ROOT)),
        "note": (
            "Built using the existing V3 spatial assignment logic. "
            "No 0.05-degree outputs are written by this script."
        ),
    }

    print("Writing isolated temporary grid output...", flush=True)
    grid_panel.to_csv(GRID_TMP, index=False, date_format="%Y-%m-%d")
    SUMMARY_TMP.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Publish only after both temporary files have been written.
    require(not GRID_OUT.exists() and not SUMMARY_OUT.exists(),
            "Final output appeared during build; refusing to overwrite.")
    os.replace(GRID_TMP, GRID_OUT)
    os.replace(SUMMARY_TMP, SUMMARY_OUT)

    print("\nBUILD COMPLETE")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
