from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# STUBBLEAI V2.2 - DISTRICT ERROR ANALYSIS
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

    actual_elevated = int(y_true.sum())
    predicted_elevated = int(y_pred.sum())
    total = len(group)

    return pd.Series({
        "total": total,
        "actual_elevated": actual_elevated,
        "predicted_elevated": predicted_elevated,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


def main():

    print("=" * 75)
    print("STUBBLEAI V2.2 - DISTRICT ERROR ANALYSIS")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[1/4] Loading frozen RF predictions...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df)}")
    print(f"Horizons: {sorted(df['horizon'].unique())}")
    print(f"Districts: {df['district'].nunique()}")
    print(f"States: {df['state'].unique().tolist()}")

    # --------------------------------------------------------
    # DISTRICT × HORIZON
    # --------------------------------------------------------

    print("\n[2/4] Calculating district × horizon metrics...")

    district_metrics = (
        df.groupby(
            ["horizon", "state", "district"],
            sort=True,
        )
        .apply(calculate_metrics)
        .reset_index()
    )

    district_metrics["fp_rate"] = (
        district_metrics["FP"]
        / (district_metrics["FP"] + district_metrics["TN"])
    ).replace([np.inf, -np.inf], np.nan)

    district_metrics["fn_rate"] = (
        district_metrics["FN"]
        / (district_metrics["FN"] + district_metrics["TP"])
    ).replace([np.inf, -np.inf], np.nan)

    output_file = (
        OUTPUT_DIR /
        "district_horizon_error_analysis.csv"
    )

    district_metrics.to_csv(
        output_file,
        index=False,
    )

    print(f"Saved: {output_file}")

    # --------------------------------------------------------
    # DISTRICT AGGREGATED ACROSS ALL HORIZONS
    # --------------------------------------------------------

    print("\n[3/4] Calculating district aggregate metrics...")

    district_aggregate = (
        df.groupby(
            ["state", "district"],
            sort=True,
        )
        .apply(calculate_metrics)
        .reset_index()
    )

    district_aggregate["fp_rate"] = (
        district_aggregate["FP"]
        / (
            district_aggregate["FP"]
            + district_aggregate["TN"]
        )
    ).replace([np.inf, -np.inf], np.nan)

    district_aggregate["fn_rate"] = (
        district_aggregate["FN"]
        / (
            district_aggregate["FN"]
            + district_aggregate["TP"]
        )
    ).replace([np.inf, -np.inf], np.nan)

    aggregate_file = (
        OUTPUT_DIR /
        "district_aggregate_error_analysis.csv"
    )

    district_aggregate.to_csv(
        aggregate_file,
        index=False,
    )

    print(f"Saved: {aggregate_file}")

    # --------------------------------------------------------
    # TOP / BOTTOM DISTRICTS
    # --------------------------------------------------------

    print("\n[4/4] Creating diagnostic rankings...")

    rankings = []

    for horizon in sorted(
        district_metrics["horizon"].unique()
    ):

        subset = district_metrics[
            district_metrics["horizon"] == horizon
        ].copy()

        subset = subset.sort_values(
            ["f1", "actual_elevated"],
            ascending=[False, False],
        )

        for rank, (_, row) in enumerate(
            subset.iterrows(),
            start=1,
        ):

            rankings.append({
                "horizon": horizon,
                "rank_by_f1": rank,
                "state": row["state"],
                "district": row["district"],
                "total": row["total"],
                "actual_elevated": row["actual_elevated"],
                "predicted_elevated": row["predicted_elevated"],
                "TP": row["TP"],
                "TN": row["TN"],
                "FP": row["FP"],
                "FN": row["FN"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "fp_rate": row["fp_rate"],
                "fn_rate": row["fn_rate"],
            })

    rankings_df = pd.DataFrame(rankings)

    rankings_file = (
        OUTPUT_DIR /
        "district_error_rankings.csv"
    )

    rankings_df.to_csv(
        rankings_file,
        index=False,
    )

    print(f"Saved: {rankings_file}")

    print("\n" + "=" * 75)
    print("DISTRICT ERROR ANALYSIS COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()