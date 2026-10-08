from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# STUBBLEAI V2.2 - STATE ERROR ANALYSIS
# ============================================================

INPUT_FILE = Path(
    "research/v2/error_analysis/rf_2025_error_predictions.csv"
)

OUTPUT_DIR = Path(
    "research/v2/error_analysis"
)


def calculate_metrics(group):

    y_true = group["actual_elevated"].astype(int)
    y_pred = group["rf_prediction"].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    return pd.Series({
        "total": len(group),
        "actual_elevated": int(y_true.sum()),
        "predicted_elevated": int(y_pred.sum()),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


def add_error_rates(df):

    df["fp_rate"] = (
        df["FP"] / (df["FP"] + df["TN"])
    ).replace([np.inf, -np.inf], np.nan)

    df["fn_rate"] = (
        df["FN"] / (df["FN"] + df["TP"])
    ).replace([np.inf, -np.inf], np.nan)

    return df


def main():

    print("=" * 75)
    print("STUBBLEAI V2.2 - STATE ERROR ANALYSIS")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[1/4] Loading frozen RF predictions...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df)}")
    print(
        f"States: {df['state'].unique().tolist()}"
    )

    # --------------------------------------------------------
    # STATE × HORIZON
    # --------------------------------------------------------

    print("\n[2/4] Calculating state × horizon metrics...")

    state_horizon = (
        df.groupby(
            ["horizon", "state"],
            sort=True,
        )
        .apply(calculate_metrics)
        .reset_index()
    )

    state_horizon = add_error_rates(
        state_horizon
    )

    output_file = (
        OUTPUT_DIR /
        "state_horizon_error_analysis.csv"
    )

    state_horizon.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved: {output_file}")

    # --------------------------------------------------------
    # STATE AGGREGATE
    # --------------------------------------------------------

    print("\n[3/4] Calculating state aggregate metrics...")

    state_aggregate = (
        df.groupby(
            ["state"],
            sort=True,
        )
        .apply(calculate_metrics)
        .reset_index()
    )

    state_aggregate = add_error_rates(
        state_aggregate
    )

    aggregate_file = (
        OUTPUT_DIR /
        "state_aggregate_error_analysis.csv"
    )

    state_aggregate.to_csv(
        aggregate_file,
        index=False,
    )

    print(f"Saved: {aggregate_file}")

    # --------------------------------------------------------
    # PRINT COMPARISON
    # --------------------------------------------------------

    print("\n[4/4] State comparison")

    display_cols = [
        "horizon",
        "state",
        "total",
        "actual_elevated",
        "predicted_elevated",
        "TP",
        "FP",
        "FN",
        "precision",
        "recall",
        "f1",
    ]

    print(
        state_horizon[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n" + "=" * 75)
    print("STATE ERROR ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()