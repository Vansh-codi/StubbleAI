from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# STUBBLEAI V2.2 - CURRENT FIRE REGIME ANALYSIS
# ============================================================

INPUT_FILE = Path(
    "research/v2/error_analysis/rf_2025_error_predictions.csv"
)

OUTPUT_DIR = Path(
    "research/v2/error_analysis"
)


def classify_fire_regime(fire_count):

    if fire_count == 0:
        return "No current activity"

    if fire_count <= 2:
        return "Low current activity"

    return "Elevated current activity"


def calculate_metrics(group):

    y_true = group["actual_elevated"].astype(int)
    y_pred = group["rf_prediction"].astype(int)

    persistence_pred = (
        group["persistence_prediction"].astype(int)
    )

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    rf_correct = (
        y_pred == y_true
    )

    persistence_correct = (
        persistence_pred == y_true
    )

    return pd.Series({
        "total": len(group),

        "actual_elevated":
            int(y_true.sum()),

        "actual_elevated_rate":
            float(y_true.mean()),

        "predicted_elevated":
            int(y_pred.sum()),

        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),

        "rf_accuracy":
            float(rf_correct.mean()),

        "persistence_accuracy":
            float(persistence_correct.mean()),

        "rf_only_correct":
            int(
                (
                    rf_correct
                    & ~persistence_correct
                ).sum()
            ),

        "persistence_only_correct":
            int(
                (
                    ~rf_correct
                    & persistence_correct
                ).sum()
            ),

        "both_correct":
            int(
                (
                    rf_correct
                    & persistence_correct
                ).sum()
            ),

        "both_wrong":
            int(
                (
                    ~rf_correct
                    & ~persistence_correct
                ).sum()
            ),
    })


def main():

    print("=" * 75)
    print("STUBBLEAI V2.2 - CURRENT FIRE REGIME ANALYSIS")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[1/5] Loading frozen RF predictions...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df)}")

    # --------------------------------------------------------
    # CURRENT FIRE REGIME
    # --------------------------------------------------------

    print("\n[2/5] Classifying current fire regimes...")

    df["current_fire_regime"] = (
        df["fire_count"]
        .apply(classify_fire_regime)
    )

    regime_order = [
        "No current activity",
        "Low current activity",
        "Elevated current activity",
    ]

    df["current_fire_regime"] = pd.Categorical(
        df["current_fire_regime"],
        categories=regime_order,
        ordered=True,
    )

    # --------------------------------------------------------
    # REGIME × HORIZON
    # --------------------------------------------------------

    print("\n[3/5] Calculating regime × horizon metrics...")

    regime_horizon = (
        df.groupby(
            [
                "horizon",
                "current_fire_regime",
            ],
            sort=True,
            observed=True,
        )
        .apply(calculate_metrics)
        .reset_index()
    )

    output_file = (
        OUTPUT_DIR /
        "fire_regime_horizon_analysis.csv"
    )

    regime_horizon.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved: {output_file}")

    # --------------------------------------------------------
    # REGIME × STATE × HORIZON
    # --------------------------------------------------------

    print(
        "\n[4/5] Calculating "
        "regime × state × horizon metrics..."
    )

    regime_state_horizon = (
        df.groupby(
            [
                "horizon",
                "state",
                "current_fire_regime",
            ],
            sort=True,
            observed=True,
        )
        .apply(calculate_metrics)
        .reset_index()
    )

    state_output_file = (
        OUTPUT_DIR /
        "fire_regime_state_horizon_analysis.csv"
    )

    regime_state_horizon.to_csv(
        state_output_file,
        index=False,
    )

    print(
        f"Saved: {state_output_file}"
    )

    # --------------------------------------------------------
    # PRINT MAIN RESULT
    # --------------------------------------------------------

    print("\n[5/5] Main fire-regime comparison")

    display_cols = [
        "horizon",
        "current_fire_regime",
        "total",
        "actual_elevated_rate",
        "rf_accuracy",
        "persistence_accuracy",
        "precision",
        "recall",
        "f1",
        "rf_only_correct",
        "persistence_only_correct",
        "both_wrong",
    ]

    print(
        regime_horizon[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 75)
    print("FIRE REGIME ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()