from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)


BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_DIR = (
    BASE_DIR
    / "research"
    / "v2"
    / "v2_4"
    / "predictions"
)

OUTPUT_DIR = (
    BASE_DIR
    / "research"
    / "v2"
    / "v2_4"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARMS = {
    "v2": "V2",
    "spatial": "V2.3 Spatial",
    "spatiotemporal": "V2.4 Spatio-Temporal",
}

HORIZONS = [1, 2, 3, 5, 7]


def load_predictions(arm, horizon):

    path = (
        PREDICTIONS_DIR
        / f"{arm}_predictions_plus{horizon}d.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [
        "state",
        "district",
        "date",
        "target_fire_count",
        "actual_elevated",
        "rf_probability",
        "rf_prediction",
        "threshold",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} missing: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])

    return df


def probability_bins(prob):

    bins = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]

    return pd.cut(
        prob,
        bins=bins,
        include_lowest=True,
        right=True,
    )


def main():

    print("=" * 80)
    print("STUBBLEAI V2.4 PROBABILITY / RANKING DIAGNOSTIC")
    print("=" * 80)

    overall_rows = []
    threshold_rows = []
    bin_rows = []
    emerging_rows = []

    loaded = {}

    # ========================================================
    # LOAD
    # ========================================================

    print("\n[1/5] Loading frozen 2025 predictions...")

    for arm, label in ARMS.items():

        loaded[arm] = {}

        for horizon in HORIZONS:

            df = load_predictions(
                arm,
                horizon,
            )

            loaded[arm][horizon] = df

            print(
                f"{label:25s} "
                f"+{horizon}d: {len(df)} rows"
            )

    # ========================================================
    # OVERALL PROBABILITY QUALITY
    # ========================================================

    print("\n[2/5] Probability quality...")

    for arm, label in ARMS.items():

        for horizon in HORIZONS:

            df = loaded[arm][horizon]

            y = df["actual_elevated"].astype(int)
            p = df["rf_probability"].astype(float)

            auc = roc_auc_score(y, p)
            pr_auc = average_precision_score(y, p)
            brier = brier_score_loss(y, p)
            logloss = log_loss(
                y,
                np.clip(p, 1e-7, 1 - 1e-7),
            )

            overall_rows.append({
                "arm": arm,
                "arm_label": label,
                "horizon": f"+{horizon}d",
                "roc_auc": auc,
                "pr_auc": pr_auc,
                "brier": brier,
                "log_loss": logloss,
                "mean_probability": p.mean(),
                "actual_rate": y.mean(),
                "probability_minus_actual_rate":
                    p.mean() - y.mean(),
            })

    overall_df = pd.DataFrame(
        overall_rows
    )

    overall_df.to_csv(
        OUTPUT_DIR
        / "v2_4_probability_quality.csv",
        index=False,
    )

    # ========================================================
    # THRESHOLD CURVES
    # ========================================================

    print("\n[3/5] Threshold-sensitivity analysis...")

    thresholds = np.linspace(
        0.05,
        0.95,
        181,
    )

    for arm, label in ARMS.items():

        for horizon in HORIZONS:

            df = loaded[arm][horizon]

            y = df["actual_elevated"].astype(int).to_numpy()
            p = df["rf_probability"].astype(float).to_numpy()

            for threshold in thresholds:

                pred = (
                    p >= threshold
                ).astype(int)

                tp = np.sum(
                    (pred == 1)
                    & (y == 1)
                )

                fp = np.sum(
                    (pred == 1)
                    & (y == 0)
                )

                fn = np.sum(
                    (pred == 0)
                    & (y == 1)
                )

                tn = np.sum(
                    (pred == 0)
                    & (y == 0)
                )

                precision = (
                    tp / (tp + fp)
                    if tp + fp > 0
                    else 0.0
                )

                recall = (
                    tp / (tp + fn)
                    if tp + fn > 0
                    else 0.0
                )

                f1 = (
                    2 * precision * recall
                    / (precision + recall)
                    if precision + recall > 0
                    else 0.0
                )

                accuracy = (
                    (tp + tn) / len(y)
                )

                threshold_rows.append({
                    "arm": arm,
                    "arm_label": label,
                    "horizon": f"+{horizon}d",
                    "threshold": threshold,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "accuracy": accuracy,
                    "predicted_positive_rate":
                        pred.mean(),
                })

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_df.to_csv(
        OUTPUT_DIR
        / "v2_4_threshold_sensitivity.csv",
        index=False,
    )

    # ========================================================
    # PROBABILITY BIN ANALYSIS
    # ========================================================

    print("\n[4/5] Probability-bin reliability analysis...")

    for arm, label in ARMS.items():

        for horizon in HORIZONS:

            df = loaded[arm][horizon].copy()

            df["probability_bin"] = (
                probability_bins(
                    df["rf_probability"]
                )
            )

            for probability_bin, group in (
                df.groupby(
                    "probability_bin",
                    observed=False,
                )
            ):

                if len(group) == 0:
                    continue

                mean_probability = (
                    group["rf_probability"]
                    .mean()
                )

                actual_rate = (
                    group["actual_elevated"]
                    .mean()
                )

                bin_rows.append({
                    "arm": arm,
                    "arm_label": label,
                    "horizon": f"+{horizon}d",
                    "probability_bin":
                        str(probability_bin),
                    "n": len(group),
                    "mean_probability":
                        mean_probability,
                    "actual_rate":
                        actual_rate,
                    "calibration_gap":
                        mean_probability
                        - actual_rate,
                })

    bins_df = pd.DataFrame(
        bin_rows
    )

    bins_df.to_csv(
        OUTPUT_DIR
        / "v2_4_probability_bins.csv",
        index=False,
    )

    # ========================================================
    # EMERGING-EVENT PROBABILITY ANALYSIS
    # ========================================================

    print("\n[5/5] Emerging-event probability analysis...")

    for arm, label in ARMS.items():

        for horizon in HORIZONS:

            df = loaded[arm][horizon].copy()

            df["current_elevated"] = (
                df["fire_count"] > 2
            ).astype(int)

            emerging = df[
                (df["current_elevated"] == 0)
                & (df["actual_elevated"] == 1)
            ]

            non_emerging = df[
                ~(
                    (df["current_elevated"] == 0)
                    & (df["actual_elevated"] == 1)
                )
            ]

            if len(emerging) > 0:

                emerging_rows.append({
                    "arm": arm,
                    "arm_label": label,
                    "horizon": f"+{horizon}d",
                    "group":
                        "Emerging N->E",
                    "n": len(emerging),
                    "mean_probability":
                        emerging["rf_probability"]
                        .mean(),
                    "median_probability":
                        emerging["rf_probability"]
                        .median(),
                    "p75_probability":
                        emerging["rf_probability"]
                        .quantile(.75),
                    "p90_probability":
                        emerging["rf_probability"]
                        .quantile(.90),
                })

            if len(non_emerging) > 0:

                emerging_rows.append({
                    "arm": arm,
                    "arm_label": label,
                    "horizon": f"+{horizon}d",
                    "group":
                        "All other cases",
                    "n": len(non_emerging),
                    "mean_probability":
                        non_emerging["rf_probability"]
                        .mean(),
                    "median_probability":
                        non_emerging["rf_probability"]
                        .median(),
                    "p75_probability":
                        non_emerging["rf_probability"]
                        .quantile(.75),
                    "p90_probability":
                        non_emerging["rf_probability"]
                        .quantile(.90),
                })

    emerging_df = pd.DataFrame(
        emerging_rows
    )

    emerging_df.to_csv(
        OUTPUT_DIR
        / "v2_4_emerging_probability_analysis.csv",
        index=False,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("PROBABILITY QUALITY")
    print("=" * 80)

    print(
        overall_df[
            [
                "arm_label",
                "horizon",
                "roc_auc",
                "pr_auc",
                "brier",
                "log_loss",
                "mean_probability",
                "actual_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("BEST 2025 DIAGNOSTIC F1 BY THRESHOLD")
    print("=" * 80)

    best_rows = []

    for (arm, horizon), group in (
        threshold_df.groupby(
            ["arm", "horizon"]
        )
    ):

        best = group.loc[
            group["f1"].idxmax()
        ]

        best_rows.append(best)

    best_df = pd.DataFrame(
        best_rows
    )

    print(
        best_df[
            [
                "arm_label",
                "horizon",
                "threshold",
                "f1",
                "precision",
                "recall",
                "predicted_positive_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("EMERGING N->E PROBABILITY")
    print("=" * 80)

    print(
        emerging_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FILES WRITTEN")
    print("=" * 80)

    for path in sorted(
        OUTPUT_DIR.glob(
            "v2_4_*probability*.csv"
        )
    ):
        print(path)

    print(
        OUTPUT_DIR
        / "v2_4_threshold_sensitivity.csv"
    )

    print()
    print("Probability/ranking diagnostic complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()