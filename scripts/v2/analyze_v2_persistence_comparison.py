from pathlib import Path

import pandas as pd


# ============================================================
# STUBBLEAI V2.2 - RF VS PERSISTENCE COMPARISON
# ============================================================

INPUT_FILE = Path(
    "research/v2/error_analysis/rf_2025_error_predictions.csv"
)

OUTPUT_DIR = Path(
    "research/v2/error_analysis"
)


def main():

    print("=" * 75)
    print("STUBBLEAI V2.2 - RF VS PERSISTENCE COMPARISON")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[1/4] Loading frozen predictions...")

    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------------
    # CORRECTNESS FLAGS
    # --------------------------------------------------------

    df["rf_correct"] = (
        df["rf_prediction"]
        == df["actual_elevated"]
    )

    df["persistence_correct"] = (
        df["persistence_prediction"]
        == df["actual_elevated"]
    )

    df["comparison_category"] = "Both wrong"

    df.loc[
        df["rf_correct"] &
        df["persistence_correct"],
        "comparison_category"
    ] = "Both correct"

    df.loc[
        df["rf_correct"] &
        ~df["persistence_correct"],
        "comparison_category"
    ] = "RF only correct"

    df.loc[
        ~df["rf_correct"] &
        df["persistence_correct"],
        "comparison_category"
    ] = "Persistence only correct"

    # --------------------------------------------------------
    # HORIZON SUMMARY
    # --------------------------------------------------------

    print("\n[2/4] Horizon comparison...")

    horizon_rows = []

    for horizon, group in df.groupby(
        "horizon",
        sort=True,
    ):

        total = len(group)

        counts = (
            group["comparison_category"]
            .value_counts()
            .to_dict()
        )

        horizon_rows.append({
            "horizon": horizon,
            "total": total,
            "both_correct": counts.get(
                "Both correct", 0
            ),
            "rf_only_correct": counts.get(
                "RF only correct", 0
            ),
            "persistence_only_correct": counts.get(
                "Persistence only correct", 0
            ),
            "both_wrong": counts.get(
                "Both wrong", 0
            ),
            "rf_accuracy": group["rf_correct"].mean(),
            "persistence_accuracy":
                group["persistence_correct"].mean(),
        })

    horizon_df = pd.DataFrame(
        horizon_rows
    )

    horizon_file = (
        OUTPUT_DIR /
        "rf_vs_persistence_horizon.csv"
    )

    horizon_df.to_csv(
        horizon_file,
        index=False,
    )

    print(f"Saved: {horizon_file}")

    # --------------------------------------------------------
    # STATE × HORIZON
    # --------------------------------------------------------

    print("\n[3/4] State × horizon comparison...")

    state_rows = []

    for (
        horizon,
        state,
    ), group in df.groupby(
        ["horizon", "state"],
        sort=True,
    ):

        counts = (
            group["comparison_category"]
            .value_counts()
            .to_dict()
        )

        state_rows.append({
            "horizon": horizon,
            "state": state,
            "total": len(group),
            "both_correct": counts.get(
                "Both correct", 0
            ),
            "rf_only_correct": counts.get(
                "RF only correct", 0
            ),
            "persistence_only_correct":
                counts.get(
                    "Persistence only correct", 0
                ),
            "both_wrong": counts.get(
                "Both wrong", 0
            ),
            "rf_accuracy":
                group["rf_correct"].mean(),
            "persistence_accuracy":
                group["persistence_correct"].mean(),
        })

    state_df = pd.DataFrame(
        state_rows
    )

    state_file = (
        OUTPUT_DIR /
        "rf_vs_persistence_state_horizon.csv"
    )

    state_df.to_csv(
        state_file,
        index=False,
    )

    print(f"Saved: {state_file}")

    # --------------------------------------------------------
    # CATEGORY TOTALS
    # --------------------------------------------------------

    print("\n[4/4] Overall comparison...")

    overall = (
        df["comparison_category"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )

    overall["percentage"] = (
        overall["count"] / len(df) * 100
    )

    overall_file = (
        OUTPUT_DIR /
        "rf_vs_persistence_overall.csv"
    )

    overall.to_csv(
        overall_file,
        index=False,
    )

    print(f"Saved: {overall_file}")

    print("\nHORIZON SUMMARY")
    print(
        horizon_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nOVERALL")
    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "=" * 75)
    print("RF VS PERSISTENCE ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()