from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# STUBBLEAI V2.2 - FIRE TRANSITION ANALYSIS
# ============================================================

INPUT_FILE = Path(
    "research/v2/error_analysis/rf_2025_error_predictions.csv"
)

OUTPUT_DIR = Path(
    "research/v2/error_analysis"
)

ELEVATED_THRESHOLD = 2


def main():

    print("=" * 75)
    print("STUBBLEAI V2.2 - FIRE TRANSITION ANALYSIS")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[1/5] Loading frozen RF predictions...")

    df = pd.read_csv(INPUT_FILE)

    # --------------------------------------------------------
    # CURRENT / FUTURE STATE
    # --------------------------------------------------------

    print("\n[2/5] Creating fire-state transitions...")

    df["current_elevated"] = (
        df["fire_count"] > ELEVATED_THRESHOLD
    ).astype(int)

    df["future_elevated"] = (
        df["target_fire_count"] > ELEVATED_THRESHOLD
    ).astype(int)

    def transition(row):

        current = row["current_elevated"]
        future = row["future_elevated"]

        if current == 0 and future == 0:
            return "Normal -> Normal"

        if current == 0 and future == 1:
            return "Normal -> Elevated"

        if current == 1 and future == 0:
            return "Elevated -> Normal"

        return "Elevated -> Elevated"

    df["transition"] = df.apply(
        transition,
        axis=1,
    )

    # --------------------------------------------------------
    # RF / PERSISTENCE CORRECTNESS
    # --------------------------------------------------------

    df["rf_correct"] = (
        df["rf_prediction"]
        == df["actual_elevated"]
    )

    df["persistence_correct"] = (
        df["persistence_prediction"]
        == df["actual_elevated"]
    )

    df["rf_emerging_correct"] = (
        (df["transition"] == "Normal -> Elevated")
        & (df["rf_prediction"] == 1)
    )

    # --------------------------------------------------------
    # TRANSITION × HORIZON
    # --------------------------------------------------------

    print("\n[3/5] Calculating transition × horizon metrics...")

    rows = []

    for (
        horizon,
        transition,
    ), group in df.groupby(
        ["horizon", "transition"],
        sort=True,
    ):

        total = len(group)

        actual_elevated = int(
            group["actual_elevated"].sum()
        )

        rf_predicted_elevated = int(
            group["rf_prediction"].sum()
        )

        persistence_predicted_elevated = int(
            group["persistence_prediction"].sum()
        )

        rf_correct = int(
            group["rf_correct"].sum()
        )

        persistence_correct = int(
            group["persistence_correct"].sum()
        )

        rf_only_correct = int(
            (
                group["rf_correct"]
                & ~group["persistence_correct"]
            ).sum()
        )

        persistence_only_correct = int(
            (
                ~group["rf_correct"]
                & group["persistence_correct"]
            ).sum()
        )

        rows.append({
            "horizon": horizon,
            "transition": transition,
            "total": total,
            "actual_elevated": actual_elevated,
            "actual_elevated_rate":
                actual_elevated / total
                if total else np.nan,
            "rf_predicted_elevated":
                rf_predicted_elevated,
            "persistence_predicted_elevated":
                persistence_predicted_elevated,
            "rf_correct": rf_correct,
            "persistence_correct":
                persistence_correct,
            "rf_accuracy":
                rf_correct / total
                if total else np.nan,
            "persistence_accuracy":
                persistence_correct / total
                if total else np.nan,
            "rf_only_correct":
                rf_only_correct,
            "persistence_only_correct":
                persistence_only_correct,
        })

    transition_horizon = pd.DataFrame(rows)

    transition_file = (
        OUTPUT_DIR /
        "fire_transition_horizon_analysis.csv"
    )

    transition_horizon.to_csv(
        transition_file,
        index=False,
    )

    print(
        f"Saved: {transition_file}"
    )

    # --------------------------------------------------------
    # EMERGING EVENTS
    # --------------------------------------------------------

    print("\n[4/5] Analyzing emerging Elevated events...")

    emerging = df[
        df["transition"] == "Normal -> Elevated"
    ].copy()

    emerging_rows = []

    for horizon, group in emerging.groupby(
        "horizon",
        sort=True,
    ):

        actual_events = len(group)

        rf_caught = int(
            (group["rf_prediction"] == 1).sum()
        )

        persistence_caught = int(
            (group["persistence_prediction"] == 1).sum()
        )

        rf_missed = actual_events - rf_caught

        persistence_missed = (
            actual_events - persistence_caught
        )

        rf_recall = (
            rf_caught / actual_events
            if actual_events else np.nan
        )

        persistence_recall = (
            persistence_caught / actual_events
            if actual_events else np.nan
        )

        rf_only = int(
            (
                (group["rf_prediction"] == 1)
                & (group["persistence_prediction"] == 0)
            ).sum()
        )

        persistence_only = int(
            (
                (group["rf_prediction"] == 0)
                & (group["persistence_prediction"] == 1)
            ).sum()
        )

        emerging_rows.append({
            "horizon": horizon,
            "emerging_events": actual_events,
            "rf_caught": rf_caught,
            "rf_missed": rf_missed,
            "rf_recall": rf_recall,
            "persistence_caught":
                persistence_caught,
            "persistence_missed":
                persistence_missed,
            "persistence_recall":
                persistence_recall,
            "rf_only_caught": rf_only,
            "persistence_only_caught":
                persistence_only,
        })

    emerging_df = pd.DataFrame(
        emerging_rows
    )

    emerging_file = (
        OUTPUT_DIR /
        "emerging_fire_events_analysis.csv"
    )

    emerging_df.to_csv(
        emerging_file,
        index=False,
    )

    print(
        f"Saved: {emerging_file}"
    )

    # --------------------------------------------------------
    # STATE × EMERGING EVENTS
    # --------------------------------------------------------

    state_rows = []

    for (
        horizon,
        state,
    ), group in emerging.groupby(
        ["horizon", "state"],
        sort=True,
    ):

        events = len(group)

        rf_caught = int(
            (group["rf_prediction"] == 1).sum()
        )

        persistence_caught = int(
            (group["persistence_prediction"] == 1).sum()
        )

        state_rows.append({
            "horizon": horizon,
            "state": state,
            "emerging_events": events,
            "rf_caught": rf_caught,
            "rf_missed": events - rf_caught,
            "rf_recall":
                rf_caught / events
                if events else np.nan,
            "persistence_caught":
                persistence_caught,
            "persistence_missed":
                events - persistence_caught,
            "persistence_recall":
                persistence_caught / events
                if events else np.nan,
        })

    state_emerging = pd.DataFrame(
        state_rows
    )

    state_file = (
        OUTPUT_DIR /
        "emerging_fire_state_analysis.csv"
    )

    state_emerging.to_csv(
        state_file,
        index=False,
    )

    print(
        f"Saved: {state_file}"
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n[5/5] Main transition results")

    print("\nTRANSITION × HORIZON")

    print(
        transition_horizon.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nEMERGING ELEVATED EVENTS")

    print(
        emerging_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nEMERGING EVENTS BY STATE")

    print(
        state_emerging.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 75)
    print("FIRE TRANSITION ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()