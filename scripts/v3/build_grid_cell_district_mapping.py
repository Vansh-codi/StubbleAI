from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "scripts" / "v3" / "build_grid_day_panel.py"
GRID_PANEL = ROOT / "data" / "processed" / "v3" / "grid_day_panel_005_centroid_fallback.csv"
OUTPUT = ROOT / "data" / "processed" / "v3" / "grid_cell_district_mapping_005.csv"

spec = importlib.util.spec_from_file_location("v3_grid_builder", BUILDER_PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

print("Loading SOI districts...")
districts = builder.load_boundaries()

print("Building eligible 0.05-degree cells...")
cell_meta = builder.build_eligible_cells(districts, 0.05)
cells = builder.cell_polygons(cell_meta, 0.05)

print("Resolving cell-to-district assignments using the existing builder...")
assignment, _ = builder.resolve_cells(cells, districts, 0.05)

mapping = assignment[
    ["state", "state_key", "grid_x", "grid_y", "cell_id",
     "district", "district_key", "assignment_method"]
].copy()

if mapping["cell_id"].duplicated().any():
    raise SystemExit("Duplicate cell IDs in mapping.")
if mapping["district"].isna().any():
    raise SystemExit("Some cells have no district assignment.")

print("Checking against the existing grid panel...")
grid_cols = pd.read_csv(GRID_PANEL, usecols=["cell_id", "state", "state_key", "grid_x", "grid_y"])
grid_cells = grid_cols.drop_duplicates()

if grid_cells["cell_id"].duplicated().any():
    raise SystemExit("Grid panel has inconsistent cell metadata.")

comparison = grid_cells[["cell_id", "state", "state_key", "grid_x", "grid_y"]].merge(
    mapping[["cell_id", "state", "state_key", "grid_x", "grid_y", "district", "district_key", "assignment_method"]],
    on="cell_id", how="left", suffixes=("_grid", "_map"), validate="one_to_one",
)

if comparison["district"].isna().any():
    raise SystemExit(f"Unmapped grid cells: {comparison['district'].isna().sum()}")

for col in ["state", "state_key", "grid_x", "grid_y"]:
    if not (comparison[f"{col}_grid"].astype(str) == comparison[f"{col}_map"].astype(str)).all():
        raise SystemExit(f"Grid metadata mismatch for {col}")

if len(comparison) != len(grid_cells):
    raise SystemExit("Mapping and grid cell counts differ.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
mapping.sort_values(["state_key", "grid_x", "grid_y"]).to_csv(OUTPUT, index=False)

print("\nMAPPING VALIDATION PASSED")
print("Grid cells:", len(grid_cells))
print("Mapping rows:", len(mapping))
print("\nAssignment methods:")
print(mapping["assignment_method"].value_counts().to_string())
print("\nCells by state:")
print(mapping.groupby("state").size().to_string())
print("Output:", OUTPUT)
print("Rows written:", len(mapping))

