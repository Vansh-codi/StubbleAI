from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "data/processed/v3/v3_district_ml_dataset_1d.csv"
SPATIAL_005 = ROOT / "data/processed/v3/experiments/spatial_neighbors_v3/district_neighbor_features.csv"
SPATIAL_010 = ROOT / "data/processed/v3/experiments/grid_sensitivity_010/district_neighbor_features_010.csv"
OUT_DIR = ROOT / "data/processed/v3/experiments/grid_sensitivity_010"
OUT_PATH = OUT_DIR / "resolution_eligibility_audit.json"

KEYS = ["state", "district", "date"]
TARGET = "target_elevated_t_plus_1d"
FUTURE = {"fire_count_t_plus_1d", "frp_sum_t_plus_1d", TARGET}
LAGS = [1, 2, 3, 5, 7, 14]
SPATIAL_FEATURES = [
    "neighbor_fire_count_lag_1d_mean",
    "neighbor_fire_count_lag_1d_max",
    "neighbor_fire_count_per_neighbor_mean",
    "neighbor_frp_sum_lag_1d_mean",
    "neighbor_frp_per_neighbor_mean",
    "neighbor_active_cells_lag_1d_mean",
    "neighbor_max_fire_count_lag_1d_max",
    "neighbor_coverage_fraction_mean",
]

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def load_spatial(path, label):
    require(path.is_file(), f"Missing {label} features: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    df["state"] = df["state"].str.strip().str.upper()
    df["district"] = df["district"].str.strip().str.upper()
    require(df[KEYS].notna().all().all(), f"{label}: null keys")
    require(not df.duplicated(KEYS).any(), f"{label}: duplicate keys")
    require(set(SPATIAL_FEATURES).issubset(df.columns),
            f"{label}: missing spatial features")
    return df

def main():
    for path in [BASE_PATH, SPATIAL_005, SPATIAL_010]:
        require(path.is_file(), f"Missing input: {path}")
    require(not OUT_PATH.exists(), f"Refusing to overwrite: {OUT_PATH}")

    base = pd.read_csv(BASE_PATH, parse_dates=["date"])
    base["state"] = base["state"].str.strip().str.upper()
    base["district"] = base["district"].str.strip().str.upper()
    require(base[KEYS].notna().all().all(), "Baseline: null keys")
    require(not base.duplicated(KEYS).any(), "Baseline: duplicate keys")
    require(TARGET in base.columns, f"Missing target: {TARGET}")

    lag_columns = [
        f"{feature}_lag_{lag}d"
        for lag in LAGS
        for feature in [
            "fire_count", "frp_sum", "active_cell_count",
            "max_cell_fire_count", "max_cell_frp", "max_cell_share",
        ]
    ]
    require(set(lag_columns).issubset(base.columns),
            "Baseline lag columns are incomplete")

    s005 = load_spatial(SPATIAL_005, "0.05-degree")
    s010 = load_spatial(SPATIAL_010, "0.10-degree")

    base_keys = set(map(tuple, base[KEYS].itertuples(index=False, name=None)))
    keys_005 = set(map(tuple, s005[KEYS].itertuples(index=False, name=None)))
    keys_010 = set(map(tuple, s010[KEYS].itertuples(index=False, name=None)))

    require(base_keys == keys_005, "Baseline and 0.05-degree key populations differ")
    require(base_keys == keys_010, "Baseline and 0.10-degree key populations differ")

    eligible_base = (
        base[TARGET].notna()
        & base[lag_columns].notna().all(axis=1)
    )
    y = base[TARGET].fillna(0).astype(int)
    require(set(y[eligible_base].unique()).issubset({0, 1}),
            "Eligible target contains values other than 0 and 1")

    def eligible_with(spatial):
        joined = base.merge(
            spatial[KEYS + SPATIAL_FEATURES],
            on=KEYS, how="left", validate="one_to_one", indicator=True
        )
        require((joined["_merge"] == "both").all(),
                "Spatial join has unmatched baseline rows")
        mask = (
            joined[TARGET].notna()
            & joined[lag_columns].notna().all(axis=1)
            & joined[SPATIAL_FEATURES].notna().all(axis=1)
        )
        return joined, mask

    j005, mask005 = eligible_with(s005)
    j010, mask010 = eligible_with(s010)

    require(j005.loc[mask005, KEYS].reset_index(drop=True).equals(
                j010.loc[mask010, KEYS].reset_index(drop=True)),
            "Eligible row keys differ between resolutions")

    require(
        j005.loc[mask005, TARGET].astype(int).reset_index(drop=True).equals(
            j010.loc[mask010, TARGET].astype(int).reset_index(drop=True)
        ),
        "Target labels differ between resolutions"
    )

    def yearly_counts(frame, mask):
        return {
            str(int(year)): {
                "rows": int((mask & frame["date"].dt.year.eq(year)).sum()),
                "positives": int(
                    frame.loc[mask & frame["date"].dt.year.eq(year), TARGET].sum()
                ),
            }
            for year in [2023, 2024, 2025]
        }

    report = {
        "status": "PASS",
        "baseline_rows": int(len(base)),
        "baseline_sha256_note": "Hash not calculated by this key-eligibility audit",
        "key_populations_equal": True,
        "eligible_rows_005": int(mask005.sum()),
        "eligible_rows_010": int(mask010.sum()),
        "eligible_keys_equal": True,
        "target_labels_equal_on_eligible_rows": True,
        "yearly_eligible_005": yearly_counts(j005, mask005),
        "yearly_eligible_010": yearly_counts(j010, mask010),
        "interpretation": (
            "This audit validates key and target comparability. "
            "It does not train models or establish that one resolution performs better."
        ),
    }

    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("RESOLUTION ELIGIBILITY AUDIT PASSED")
    print(json.dumps(report, indent=2))
    print("Audit:", OUT_PATH)

if __name__ == "__main__":
    main()
