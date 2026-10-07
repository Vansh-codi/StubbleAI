import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score


ROOT = Path(__file__).resolve().parent

V2_PRED = (
    ROOT / "research" / "v2" / "error_analysis"
    / "rf_2025_error_predictions.csv"
)

SPATIAL_RESULTS = (
    ROOT / "research" / "v2" / "v2_3" / "spatial"
    / "v2_3_spatial_results.csv"
)

OUT_DIR = (
    ROOT / "research" / "v2" / "v2_3" / "spatial"
)


def metric_block(group):
    y = group["actual_elevated"]
    p = group["spatial_prediction"]

    return {
        "n": len(group),
        "actual_elevated_rate": y.mean(),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
    }


def normalize_horizon(series):
    return (
        series
        .astype(str)
        .str.replace("+", "", regex=False)
        .str.replace("d", "", regex=False)
        .astype(int)
    )


def main():

    print("=" * 70)
    print("V2.3 SPATIAL ERROR DIAGNOSTICS")
    print("=" * 70)

    v2 = pd.read_csv(V2_PRED)
    results = pd.read_csv(SPATIAL_RESULTS)

    # ---------------------------------------------------------------
    # LOAD SPATIAL PREDICTIONS
    # ---------------------------------------------------------------

    spatial_files = []

    for horizon in [1, 2, 3, 5, 7]:

        path = (
            OUT_DIR
            / f"spatial_rf_predictions_plus{horizon}d.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Expected prediction file not found: {path}"
            )

        spatial_files.append(
            pd.read_csv(path)
        )

    spatial = pd.concat(
        spatial_files,
        ignore_index=True
    )

    print("V2 rows:", v2.shape)
    print("Spatial prediction rows:", spatial.shape)

    print("\nSpatial columns:")
    print(spatial.columns.tolist())

    # ---------------------------------------------------------------
    # NORMALIZE MERGE KEYS
    # ---------------------------------------------------------------

    keys = [
        "state",
        "district",
        "date",
        "horizon",
    ]

    v2["date"] = pd.to_datetime(v2["date"])
    spatial["date"] = pd.to_datetime(spatial["date"])

    # V2 uses +1d, +2d, ...
    # Spatial predictions use 1, 2, ...
    v2["horizon"] = normalize_horizon(
        v2["horizon"]
    )

    spatial["horizon"] = normalize_horizon(
        spatial["horizon"]
    )

    # ---------------------------------------------------------------
    # MERGE
    # ---------------------------------------------------------------

    merged = v2.merge(
        spatial,
        on=keys,
        how="inner",
        suffixes=("_v2", "_spatial"),
        validate="one_to_one",
    )

    print("\nMerged rows:", len(merged))

    if len(merged) != 9000:
        raise ValueError(
            f"Expected 9000 merged rows, got {len(merged)}"
        )

    # ---------------------------------------------------------------
    # RESTORE CANONICAL V2 TRUTH COLUMNS
    # ---------------------------------------------------------------
    #
    # Both V2 and spatial prediction files contain fire_count,
    # target_fire_count and actual_elevated. Because the merge
    # applies suffixes, use the V2 versions as the authoritative
    # ground truth for diagnostics.

    merged["fire_count"] = merged["fire_count_v2"]
    merged["target_fire_count"] = merged["target_fire_count_v2"]
    merged["actual_elevated"] = merged["actual_elevated_v2"]

    # ---------------------------------------------------------------
    # SPATIAL PREDICTIONS

    # ---------------------------------------------------------------
    # SPATIAL PREDICTIONS
    # ---------------------------------------------------------------

    threshold_map = dict(
        zip(
            results["horizon"].astype(int),
            results["threshold"],
        )
    )

    if "spatial_probability" in merged.columns:
        probability_column = "spatial_probability"
    elif "spatial_probability_spatial" in merged.columns:
        probability_column = "spatial_probability_spatial"
    else:
        raise KeyError(
            "Could not find spatial probability column."
        )

    merged["spatial_probability_used"] = (
        merged[probability_column]
    )

    merged["spatial_prediction"] = 0

    for horizon, threshold in threshold_map.items():

        mask = merged["horizon"] == horizon

        merged.loc[mask, "spatial_prediction"] = (
            merged.loc[
                mask,
                "spatial_probability_used",
            ]
            >= threshold
        ).astype(int)

    # ---------------------------------------------------------------
    # TRANSITIONS
    # ---------------------------------------------------------------

    merged["transition"] = (
        merged["fire_count_v2"]
        .gt(2)
        .map({
            True: "E",
            False: "N",
        })
        + "→"
        + merged["target_fire_count"]
        .gt(2)
        .map({
            True: "E",
            False: "N",
        })
    )

    transition_rows = []

    for horizon in [1, 2, 3, 5, 7]:

        h = merged[
            merged["horizon"] == horizon
        ]

        for transition in [
            "E→E",
            "E→N",
            "N→E",
            "N→N",
        ]:

            g = h[
                h["transition"] == transition
            ]

            if len(g) == 0:
                continue

            y = g["actual_elevated"]

            transition_rows.append({
                "horizon": horizon,
                "transition": transition,
                "n": len(g),
                "spatial_recall": recall_score(
                    y,
                    g["spatial_prediction"],
                    zero_division=0,
                ),
                "spatial_precision": precision_score(
                    y,
                    g["spatial_prediction"],
                    zero_division=0,
                ),
                "spatial_f1": f1_score(
                    y,
                    g["spatial_prediction"],
                    zero_division=0,
                ),
            })

    transition_df = pd.DataFrame(
        transition_rows
    )

    # ---------------------------------------------------------------
    # STATE
    # ---------------------------------------------------------------

    state_rows = []

    for horizon in [1, 2, 3, 5, 7]:

        h = merged[
            merged["horizon"] == horizon
        ]

        for state in [
            "Haryana",
            "Punjab",
        ]:

            g = h[
                h["state"] == state
            ]

            m = metric_block(g)

            state_rows.append({
                "horizon": horizon,
                "state": state,
                **{
                    f"spatial_{k}": v
                    for k, v in m.items()
                },
            })

    state_df = pd.DataFrame(
        state_rows
    )

    # ---------------------------------------------------------------
    # SEASON
    # ---------------------------------------------------------------

    merged["season_phase"] = "Other"

    merged.loc[
        merged["date"].dt.month.eq(10)
        & merged["date"].dt.day.between(15, 31),
        "season_phase",
    ] = "Early"

    merged.loc[
        merged["date"].dt.month.eq(11)
        & merged["date"].dt.day.between(1, 15),
        "season_phase",
    ] = "Middle"

    merged.loc[
        merged["date"].dt.month.eq(11)
        & merged["date"].dt.day.between(16, 30),
        "season_phase",
    ] = "Late"

    season_rows = []

    for horizon in [1, 2, 3, 5, 7]:

        h = merged[
            merged["horizon"] == horizon
        ]

        for phase in [
            "Early",
            "Middle",
            "Late",
        ]:

            g = h[
                h["season_phase"] == phase
            ]

            m = metric_block(g)

            season_rows.append({
                "horizon": horizon,
                "season_phase": phase,
                **{
                    f"spatial_{k}": v
                    for k, v in m.items()
                },
            })

    season_df = pd.DataFrame(
        season_rows
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    transition_file = (
        OUT_DIR / "spatial_transition_diagnostics.csv"
    )

    state_file = (
        OUT_DIR / "spatial_state_diagnostics.csv"
    )

    season_file = (
        OUT_DIR / "spatial_season_diagnostics.csv"
    )

    transition_df.to_csv(
        transition_file,
        index=False,
    )

    state_df.to_csv(
        state_file,
        index=False,
    )

    season_df.to_csv(
        season_file,
        index=False,
    )

    # ---------------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------------

    print("\nTRANSITION DIAGNOSTICS")
    print(
        transition_df.to_string(index=False)
    )

    print("\nSTATE DIAGNOSTICS")
    print(
        state_df.to_string(index=False)
    )

    print("\nSEASON DIAGNOSTICS")
    print(
        season_df.to_string(index=False)
    )

    print("\nSaved:")
    print(transition_file)
    print(state_file)
    print(season_file)

    print("\nDONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()