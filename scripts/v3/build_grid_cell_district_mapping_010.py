from pathlib import Path
import argparse
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "scripts" / "v3" / "build_grid_day_panel.py"
OUT_DIR = ROOT / "data" / "processed" / "v3" / "experiments" / "grid_sensitivity_010"
GRID_PANEL = OUT_DIR / "grid_day_panel_010_centroid_fallback.csv"
OUTPUT = OUT_DIR / "grid_cell_district_mapping_010.csv"

def load_builder():
    if not BUILDER_PATH.is_file():
        raise SystemExit(f"Missing builder: {BUILDER_PATH}")
    spec = importlib.util.spec_from_file_location("v3_grid_builder_010_mapping", BUILDER_PATH)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder

def preflight():
    if not GRID_PANEL.is_file():
        raise SystemExit(f"Missing 0.10-degree grid panel: {GRID_PANEL}")
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing mapping: {OUTPUT}")
    if not OUT_DIR.is_dir():
        raise SystemExit(f"Missing isolated output directory: {OUT_DIR}")
    builder = load_builder()
    if 0.10 not in builder.RESOLUTIONS:
        raise SystemExit("Existing builder does not declare 0.10-degree support.")
    print("PREFLIGHT PASS")
    print("Grid panel:", GRID_PANEL)
    print("Mapping output:", OUTPUT)
    print("Resolution: 0.10 degrees")
    print("No data loaded and no outputs written.")

def build():
    preflight()
    builder = load_builder()

    print("\nLoading SOI districts...")
    districts = builder.load_boundaries()

    print("Building eligible 0.10-degree cells...")
    cell_meta = builder.build_eligible_cells(districts, 0.10)
    cells = builder.cell_polygons(cell_meta, 0.10)

    print("Resolving cell-to-district assignments...")
    assignment, _ = builder.resolve_cells(cells, districts, 0.10)

    mapping = assignment[
        ["state", "state_key", "grid_x", "grid_y", "cell_id",
         "district", "district_key", "assignment_method"]
    ].copy()

    if mapping["cell_id"].duplicated().any():
        raise SystemExit("Duplicate cell IDs in mapping.")
    if mapping["district"].isna().any() or mapping["district_key"].isna().any():
        raise SystemExit("Some cells have no district assignment.")

    print("Checking mapping against the isolated 0.10-degree grid panel...")
    grid_cols = pd.read_csv(
        GRID_PANEL,
        usecols=["cell_id", "state", "state_key", "grid_x", "grid_y"]
    )
    grid_cells = grid_cols.drop_duplicates()

    if grid_cells["cell_id"].duplicated().any():
        raise SystemExit("Grid panel has inconsistent cell metadata.")

    comparison = grid_cells[
        ["cell_id", "state", "state_key", "grid_x", "grid_y"]
    ].merge(
        mapping[
            ["cell_id", "state", "state_key", "grid_x", "grid_y",
             "district", "district_key", "assignment_method"]
        ],
        on="cell_id",
        how="left",
        suffixes=("_grid", "_map"),
        validate="one_to_one",
    )

    if comparison["district"].isna().any():
        raise SystemExit(f"Unmapped grid cells: {comparison['district'].isna().sum()}")

    for col in ["state", "state_key", "grid_x", "grid_y"]:
        if not (
            comparison[f"{col}_grid"].astype(str)
            == comparison[f"{col}_map"].astype(str)
        ).all():
            raise SystemExit(f"Grid metadata mismatch for {col}")

    if len(comparison) != len(grid_cells) or len(mapping) != len(grid_cells):
        raise SystemExit(
            f"Cell-count mismatch: grid={len(grid_cells)}, mapping={len(mapping)}"
        )

    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing mapping: {OUTPUT}")

    temp_output = OUT_DIR / "grid_cell_district_mapping_010.tmp.csv"
    if temp_output.exists():
        raise SystemExit(f"Refusing to overwrite temporary file: {temp_output}")

    try:
        mapping.sort_values(["state_key", "grid_x", "grid_y"]).to_csv(
            temp_output, index=False
        )
        check = pd.read_csv(temp_output)
        if len(check) != len(grid_cells) or check["cell_id"].duplicated().any():
            raise SystemExit("Temporary mapping failed post-write validation.")
        if check["district"].isna().any() or check["district_key"].isna().any():
            raise SystemExit("Temporary mapping contains unassigned districts.")
        temp_output.replace(OUTPUT)
    finally:
        if temp_output.exists():
            temp_output.unlink()

    print("\nMAPPING VALIDATION PASSED")
    print("Grid cells:", len(grid_cells))
    print("Mapping rows:", len(mapping))
    print("\nAssignment methods:")
    print(mapping["assignment_method"].value_counts().to_string())
    print("\nCells by state:")
    print(mapping.groupby("state").size().to_string())
    print("Output:", OUTPUT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--build", action="store_true")
    args = parser.parse_args()

    if args.preflight:
        preflight()
    else:
        build()

