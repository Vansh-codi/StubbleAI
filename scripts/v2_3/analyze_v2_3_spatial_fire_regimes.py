import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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


HORIZONS = [1, 2, 3, 5, 7]


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
    print("V2.3 SPATIAL CURRENT-FIRE-REGIME DIAGNOSTICS")
    print("=" * 70)

    v2 = pd.read_csv(V2_PRED)
    results = pd.read_csv(SPATIAL_RESULTS)

    spatial_files = []

    for horizon in HORIZONS:

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

    # ---------------------------------------------------------------
    # NORMALIZE KEYS
    # ---------------------------------------------------------------

    keys = [
        "state",
        "district",
        "date",
        "horizon",
    ]

    v2["date"] = pd.to_datetime(v2["date"])
    spatial["date"] = pd.to_datetime(spatial["date"])

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

    print("Merged rows:", len(merged))

    if len(merged) != 9000:
        raise ValueError(
            f"Expected 9000 merged rows, got {len(merged)}"
        )

    # ---------------------------------------------------------------
    # RESTORE V2 TRUTH
    # ---------------------------------------------------------------

    merged["fire_count"] = merged["fire_count_v2"]
    merged["target_fire_count"] = merged["target_fire_count_v2"]
    merged["actual_elevated"] = merged["actual_elevated_v2"]

    # ---------------------------------------------------------------
    # RECONSTRUCT SPATIAL PREDICTION
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
    # CURRENT FIRE REGIME
    # ---------------------------------------------------------------

    merged["current_fire_regime"] = "No current fire"

    merged.loc[
        merged["fire_count"].between(1, 2),
        "current_fire_regime"
    ] = "Low current fire"

    merged.loc[
        merged["fire_count"] > 2,
        "current_fire_regime"
    ] = "Elevated current fire"

    regime_order = [
        "No current fire",
        "Low current fire",
        "Elevated current fire",
    ]

    # ---------------------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------------------

    rows = []

    for horizon in HORIZONS:

        h = merged[
            merged["horizon"] == horizon
        ]

        for regime in regime_order:

            g = h[
                h["current_fire_regime"] == regime
            ]

            if len(g) == 0:
                continue

            y = g["actual_elevated"]
            p = g["spatial_prediction"]

            rows.append({
                "horizon": horizon,
                "current_fire_regime": regime,
                "n": len(g),
                "actual_elevated_rate": y.mean(),
                "spatial_accuracy": accuracy_score(
                    y,
                    p,
                ),
                "spatial_precision": precision_score(
                    y,
                    p,
                    zero_division=0,
                ),
                "spatial_recall": recall_score(
                    y,
                    p,
                    zero_division=0,
                ),
                "spatial_f1": f1_score(
                    y,
                    p,
                    zero_division=0,
                ),
            })

    regime_df = pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # STATE × REGIME
    # ---------------------------------------------------------------

    state_rows = []

    for horizon in HORIZONS:

        h = merged[
            merged["horizon"] == horizon
        ]

        for state in [
            "Haryana",
            "Punjab",
        ]:

            for regime in regime_order:

                g = h[
                    (h["state"] == state)
                    & (
                        h["current_fire_regime"]
                        == regime
                    )
                ]

                if len(g) == 0:
                    continue

                y = g["actual_elevated"]
                p = g["spatial_prediction"]

                state_rows.append({
                    "horizon": horizon,
                    "state": state,
                    "current_fire_regime": regime,
                    "n": len(g),
                    "actual_elevated_rate": y.mean(),
                    "spatial_accuracy": accuracy_score(
                        y,
                        p,
                    ),
                    "spatial_precision": precision_score(
                        y,
                        p,
                        zero_division=0,
                    ),
                    "spatial_recall": recall_score(
                        y,
                        p,
                        zero_division=0,
                    ),
                    "spatial_f1": f1_score(
                        y,
                        p,
                        zero_division=0,
                    ),
                })

    state_regime_df = pd.DataFrame(
        state_rows
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    regime_file = (
        OUT_DIR
        / "spatial_fire_regime_diagnostics.csv"
    )

    state_regime_file = (
        OUT_DIR
        / "spatial_state_fire_regime_diagnostics.csv"
    )

    regime_df.to_csv(
        regime_file,
        index=False,
    )

    state_regime_df.to_csv(
        state_regime_file,
        index=False,
    )

    # ---------------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------------

    print("\nCURRENT-FIRE-REGIME DIAGNOSTICS")
    print(
        regime_df.to_string(index=False)
    )

    print("\nSTATE × CURRENT-FIRE-REGIME DIAGNOSTICS")
    print(
        state_regime_df.to_string(index=False)
    )

    print("\nSaved:")
    print(regime_file)
    print(state_regime_file)

    print("\nDONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()