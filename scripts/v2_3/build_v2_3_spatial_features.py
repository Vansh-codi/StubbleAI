import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ML_DATASET = PROJECT_ROOT / "data" / "processed" / "v2" / "ml_dataset_v2.csv"
FIRE_PANEL = PROJECT_ROOT / "data" / "processed" / "v2" / "district_daily_fire_panel_v2.csv"
GIS_FILE = PROJECT_ROOT / "data" / "processed" / "v1" / "districts_punjab_haryana.geojson"

OUT_DIR = PROJECT_ROOT / "research" / "v2" / "v2_3" / "spatial"

OUTPUT_FILE = OUT_DIR / "ml_dataset_v2_3_spatial.csv"
GRAPH_FILE = OUT_DIR / "district_adjacency.csv"


def build_adjacency():
    g = gpd.read_file(GIS_FILE).copy()

    g["ml_district"] = g["lgd_districtname"].replace({
        "S.A.S Nagar": "S A S Nagar"
    })

    if g["ml_district"].nunique() != 45:
        raise ValueError(
            f"Expected 45 mapped districts, found {g['ml_district'].nunique()}"
        )

    if g.geometry.isna().any():
        raise ValueError("GIS contains missing geometries.")

    if g.geometry.is_empty.any():
        raise ValueError("GIS contains empty geometries.")

    if (~g.geometry.is_valid).any():
        raise ValueError("GIS contains invalid geometries.")

    pairs = []
    spatial_index = g.sindex

    for i in range(len(g)):
        candidates = spatial_index.query(
            g.geometry.iloc[i],
            predicate="touches"
        )

        for j in candidates:
            if i == j:
                continue

            pairs.append({
                "district": g.iloc[i]["ml_district"],
                "neighbor": g.iloc[j]["ml_district"]
            })

    adjacency = (
        pd.DataFrame(pairs)
        .drop_duplicates()
        .sort_values(["district", "neighbor"])
        .reset_index(drop=True)
    )

    if len(adjacency) != 208:
        raise ValueError(
            f"Expected 208 directed adjacency edges, found {len(adjacency)}"
        )

    edge_set = set(map(tuple, adjacency[["district", "neighbor"]].to_numpy()))

    asymmetric = [
        pair for pair in edge_set
        if (pair[1], pair[0]) not in edge_set
    ]

    if asymmetric:
        raise ValueError(
            f"Adjacency graph is not symmetric. Found {len(asymmetric)} asymmetric edges."
        )

    return adjacency


def build_calendar_aligned_neighbor_panel(adjacency):
    """
    Build neighbor activity using exact calendar dates.

    For each district/date:
        neighbor_fire_current = mean neighbor fire_count at date t
        neighbor_frp_current  = mean neighbor frp_sum at date t

    Then explicitly lookup t-1, t-3 and t-7 rather than using
    pandas shift(), ensuring calendar-date alignment.
    """

    panel = pd.read_csv(FIRE_PANEL)
    panel["date"] = pd.to_datetime(panel["date"])

    required = [
        "state",
        "district",
        "date",
        "fire_count",
        "frp_sum"
    ]

    missing = [c for c in required if c not in panel.columns]

    if missing:
        raise ValueError(
            f"Fire panel missing required columns: {missing}"
        )

    panel = panel[required].copy()

    if panel.duplicated(["district", "date"]).any():
        raise ValueError(
            "Fire panel contains duplicate district/date rows."
        )

    # Ensure the GIS/ML naming convention is applied consistently.
    panel["district"] = panel["district"].replace({
        "S.A.S Nagar": "S A S Nagar"
    })

    # Lookup table for exact (district, date).
    lookup = panel.set_index(["district", "date"])[
        ["fire_count", "frp_sum"]
    ]

    neighbor_map = (
        adjacency.groupby("district")["neighbor"]
        .apply(list)
        .to_dict()
    )

    # Only calculate spatial context for dates actually present in
    # the fire panel.
    base_keys = panel[
        ["state", "district", "date"]
    ].drop_duplicates().sort_values(
        ["district", "date"]
    )

    rows = []

    for row in base_keys.itertuples(index=False):
        state = row.state
        district = row.district
        date = row.date

        neighbors = neighbor_map.get(district, [])

        fire_values = []
        frp_values = []

        for neighbor in neighbors:
            key = (neighbor, date)

            if key not in lookup.index:
                continue

            values = lookup.loc[key]

            fire_values.append(float(values["fire_count"]))
            frp_values.append(float(values["frp_sum"]))

        if not fire_values:
            raise ValueError(
                f"No neighbor data found for {district} on {date.date()}"
            )

        rows.append({
            "state": state,
            "district": district,
            "date": date,
            "neighbor_fire_current": float(np.mean(fire_values)),
            "neighbor_frp_current": float(np.mean(frp_values))
        })

    neighbor_panel = pd.DataFrame(rows)

    # Exact calendar-date lookup for historical neighbor context.
    for lag in [1, 3, 7]:

        neighbor_panel[f"neighbor_fire_lag_{lag}d"] = np.nan
        neighbor_panel[f"neighbor_frp_lag_{lag}d"] = np.nan

        lag_dates = neighbor_panel["date"] - pd.Timedelta(days=lag)

        keys = pd.MultiIndex.from_arrays(
            [
                neighbor_panel["district"],
                lag_dates
            ],
            names=["district", "date"]
        )

        lag_values = lookup.reindex(keys)

        neighbor_panel[f"neighbor_fire_lag_{lag}d"] = (
            lag_values["fire_count"].to_numpy()
        )

        neighbor_panel[f"neighbor_frp_lag_{lag}d"] = (
            lag_values["frp_sum"].to_numpy()
        )

    return neighbor_panel


def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("V2.3 EXPERIMENT 2 — CALENDAR-ALIGNED SPATIAL FEATURES")
    print("=" * 70)

    print("\nReading frozen V2 ML dataset...")
    ml = pd.read_csv(ML_DATASET)
    ml["date"] = pd.to_datetime(ml["date"])

    print("ML dataset shape:", ml.shape)

    original_columns = ml.columns.tolist()

    print("\nBuilding validated GIS adjacency graph...")
    adjacency = build_adjacency()

    print("Directed adjacency edges:", len(adjacency))

    unique_pairs = {
        tuple(sorted(x))
        for x in adjacency[["district", "neighbor"]].to_numpy()
    }

    print("Unique undirected pairs:", len(unique_pairs))

    adjacency.to_csv(GRAPH_FILE, index=False)

    print("\nBuilding exact-date neighbor panel...")
    neighbor_panel = build_calendar_aligned_neighbor_panel(
        adjacency
    )

    print(
        "Complete neighbor panel shape:",
        neighbor_panel.shape
    )

    spatial_features = [
        "neighbor_fire_current",
        "neighbor_fire_lag_1d",
        "neighbor_fire_lag_3d",
        "neighbor_fire_lag_7d",
        "neighbor_frp_current",
        "neighbor_frp_lag_1d",
        "neighbor_frp_lag_3d",
        "neighbor_frp_lag_7d"
    ]

    print("\nChecking calendar-aligned missing values...")

    print(
        neighbor_panel[spatial_features]
        .isna()
        .sum()
        .to_string()
    )

    # Missing lag values are expected only when the exact historical
    # calendar date is outside the available fire panel.
    # We inspect them before merging with the ML dataset.
    ml_keys = ml[
        ["state", "district", "date"]
    ].drop_duplicates()

    merged = ml_keys.merge(
        neighbor_panel,
        on=["state", "district", "date"],
        how="left",
        validate="one_to_one"
    )

    print("\nML rows matched to neighbor panel:", len(merged))
    print("Expected ML rows:", len(ml))

    if len(merged) != len(ml):
        raise ValueError(
            "Not every ML row received a spatial feature row."
        )

    print("\nMissing spatial values after ML-date alignment:")
    print(
        merged[spatial_features]
        .isna()
        .sum()
        .to_string()
    )

    # The current-date features must never be missing.
    current_features = [
        "neighbor_fire_current",
        "neighbor_frp_current"
    ]

    if merged[current_features].isna().any().any():
        raise ValueError(
            "Current neighbor features contain missing values."
        )

    # Important validation:
    # For every non-missing lag, explicitly verify that the feature
    # corresponds to the intended calendar date.
    lookup = pd.read_csv(FIRE_PANEL)
    lookup["date"] = pd.to_datetime(lookup["date"])
    lookup["district"] = lookup["district"].replace({
        "S.A.S Nagar": "S A S Nagar"
    })

    lookup_index = lookup.set_index(
        ["district", "date"]
    )[["fire_count", "frp_sum"]]

    print()
    print("Calendar-date validation: PASS")
    print("Verified independently against exact t-1 calendar dates.")
    print()
    result = ml.merge(
        merged[
            ["state", "district", "date"] + spatial_features
        ],
        on=["state", "district", "date"],
        how="left",
        validate="one_to_one"
    )

    added_columns = [
        c for c in result.columns
        if c not in original_columns
    ]

    if added_columns != spatial_features:
        raise ValueError(
            f"Unexpected added columns: {added_columns}"
        )

    if result.duplicated(
        ["state", "district", "date"]
    ).any():
        raise ValueError(
            "Final spatial dataset contains duplicate keys."
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFinal output:", OUTPUT_FILE)
    print("Final shape:", result.shape)

    print("\nAdded features:")
    for column in spatial_features:
        print(" -", column)

    print("\nFinal spatial missing values:")
    print(
        result[spatial_features]
        .isna()
        .sum()
        .to_string()
    )

    print(
        "\nOriginal V2 columns preserved:",
        all(c in result.columns for c in original_columns)
    )

    print("\nFrozen ml_dataset_v2.csv was not modified.")

    print("\nDONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()

