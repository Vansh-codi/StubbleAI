from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent

PRED_DIR = (
    ROOT
    / "research"
    / "v2"
    / "v2_6"
    / "predictions"
)

OUT_DIR = (
    ROOT
    / "research"
    / "v2"
    / "v2_6"
    / "diagnostics"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 3, 5, 7]


def probability_summary(df, probability_col, transition):
    x = df.loc[
        df["transition_actual"] == transition,
        probability_col,
    ].dropna()

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan,
        }

    return {
        "n": len(x),
        "mean": x.mean(),
        "median": x.median(),
        "p10": x.quantile(0.10),
        "p25": x.quantile(0.25),
        "p75": x.quantile(0.75),
        "p90": x.quantile(0.90),
    }


all_probability_rows = []
all_error_rows = []
all_summary_rows = []


print("=" * 75)
print("V2.6 TRANSITION-AWARE DIAGNOSTICS")
print("=" * 75)

for horizon in HORIZONS:

    path = (
        PRED_DIR
        / f"v2_6_predictions_plus{horizon}d.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    print("\n" + "=" * 75)
    print(f"HORIZON +{horizon}D")
    print("=" * 75)

    # --------------------------------------------------------
    # Basic integrity
    # --------------------------------------------------------

    required = [
        "state",
        "district",
        "date",
        "fire_count",
        "actual_elevated",
        "v2_probability",
        "v2_prediction",
        "transition_probability",
        "transition_prediction",
        "current_regime",
        "transition_actual",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns for +{horizon}: {missing}"
        )

    print("Rows:", len(df))
    print(
        "Transitions:",
        df["transition_actual"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Error labels
    # --------------------------------------------------------

    actual = df["actual_elevated"].astype(int)

    df["v2_error"] = np.select(
        [
            (df["v2_prediction"] == 1)
            & (actual == 1),
            (df["v2_prediction"] == 0)
            & (actual == 1),
            (df["v2_prediction"] == 1)
            & (actual == 0),
            (df["v2_prediction"] == 0)
            & (actual == 0),
        ],
        [
            "TP",
            "FN",
            "FP",
            "TN",
        ],
        default="UNKNOWN",
    )

    df["transition_error"] = np.select(
        [
            (df["transition_prediction"] == 1)
            & (actual == 1),
            (df["transition_prediction"] == 0)
            & (actual == 1),
            (df["transition_prediction"] == 1)
            & (actual == 0),
            (df["transition_prediction"] == 0)
            & (actual == 0),
        ],
        [
            "TP",
            "FN",
            "FP",
            "TN",
        ],
        default="UNKNOWN",
    )

    # --------------------------------------------------------
    # Probability distribution by transition
    # --------------------------------------------------------

    for transition in [
        "N->N",
        "N->E",
        "E->N",
        "E->E",
    ]:

        for model_name, probability_col in [
            ("frozen_v2_rf", "v2_probability"),
            ("transition_rf", "transition_probability"),
        ]:

            stats = probability_summary(
                df,
                probability_col,
                transition,
            )

            all_probability_rows.append(
                {
                    "horizon": horizon,
                    "model": model_name,
                    "transition": transition,
                    **stats,
                }
            )

    # --------------------------------------------------------
    # Error counts
    # --------------------------------------------------------

    for model_name, error_col in [
        ("frozen_v2_rf", "v2_error"),
        ("transition_rf", "transition_error"),
    ]:

        counts = (
            df.groupby(
                ["transition_actual", error_col]
            )
            .size()
            .reset_index(name="n")
        )

        counts["horizon"] = horizon
        counts["model"] = model_name

        counts = counts.rename(
            columns={
                error_col: "error_type"
            }
        )

        all_error_rows.append(counts)

    # --------------------------------------------------------
    # Brier / LogLoss
    # Descriptive only — NOT used for tuning.
    # --------------------------------------------------------

    for model_name, probability_col in [
        ("frozen_v2_rf", "v2_probability"),
        ("transition_rf", "transition_probability"),
    ]:

        probability = df[
            probability_col
        ].clip(1e-7, 1 - 1e-7)

        all_summary_rows.append(
            {
                "horizon": horizon,
                "model": model_name,
                "brier": brier_score_loss(
                    actual,
                    probability,
                ),
                "logloss": log_loss(
                    actual,
                    probability,
                ),
            }
        )

    # --------------------------------------------------------
    # Print useful diagnostic tables
    # --------------------------------------------------------

    print("\nTransition probability means:")

    print(
        df.groupby("transition_actual")[
            [
                "v2_probability",
                "transition_probability",
            ]
        ]
        .mean()
        .round(4)
        .to_string()
    )

    print("\nError counts by actual transition:")

    error_table = pd.crosstab(
        df["transition_actual"],
        [
            df["v2_error"],
            df["transition_error"],
        ],
    )

    print(error_table.to_string())

    # --------------------------------------------------------
    # Specifically inspect N->E
    # --------------------------------------------------------

    onset = df[
        df["transition_actual"] == "N->E"
    ].copy()

    print("\nN->E diagnostic:")

    if len(onset) > 0:

        print(
            pd.DataFrame(
                {
                    "V2": [
                        onset["v2_prediction"].mean(),
                    ],
                    "Transition": [
                        onset["transition_prediction"].mean(),
                    ],
                }
            ).round(4).to_string(index=False)
        )

        print(
            "\nN->E probability summary:"
        )

        print(
            onset[
                [
                    "v2_probability",
                    "transition_probability",
                ]
            ]
            .describe(
                percentiles=[
                    0.10,
                    0.25,
                    0.50,
                    0.75,
                    0.90,
                ]
            )
            .round(4)
            .to_string()
        )

    # --------------------------------------------------------
    # Specifically inspect N->N false positives
    # --------------------------------------------------------

    normal = df[
        df["transition_actual"] == "N->N"
    ].copy()

    print("\nN->N false-positive diagnostic:")

    if len(normal) > 0:

        v2_fp_rate = normal[
            "v2_prediction"
        ].mean()

        transition_fp_rate = normal[
            "transition_prediction"
        ].mean()

        print(
            f"V2 false-positive rate: "
            f"{v2_fp_rate:.4f}"
        )

        print(
            f"Transition false-positive rate: "
            f"{transition_fp_rate:.4f}"
        )

    # --------------------------------------------------------
    # State diagnostic
    # --------------------------------------------------------

    print("\nState-level F1:")

    state_rows = []

    for state in sorted(
        df["state"].dropna().unique()
    ):

        state_df = df[
            df["state"] == state
        ]

        state_rows.append(
            {
                "horizon": horizon,
                "state": state,
                "v2_f1": f1_score(
                    state_df["actual_elevated"],
                    state_df["v2_prediction"],
                    zero_division=0,
                ),
                "transition_f1": f1_score(
                    state_df["actual_elevated"],
                    state_df["transition_prediction"],
                    zero_division=0,
                ),
                "v2_recall": recall_score(
                    state_df["actual_elevated"],
                    state_df["v2_prediction"],
                    zero_division=0,
                ),
                "transition_recall": recall_score(
                    state_df["actual_elevated"],
                    state_df["transition_prediction"],
                    zero_division=0,
                ),
                "v2_precision": precision_score(
                    state_df["actual_elevated"],
                    state_df["v2_prediction"],
                    zero_division=0,
                ),
                "transition_precision": precision_score(
                    state_df["actual_elevated"],
                    state_df["transition_prediction"],
                    zero_division=0,
                ),
            }
        )

    state_df_out = pd.DataFrame(state_rows)

    state_df_out.to_csv(
        OUT_DIR
        / f"v2_6_state_diagnostics_plus{horizon}d.csv",
        index=False,
    )

    print(
        state_df_out.round(4).to_string(
            index=False
        )
    )


# ============================================================
# Save combined outputs
# ============================================================

probability_df = pd.DataFrame(
    all_probability_rows
)

error_df = pd.concat(
    all_error_rows,
    ignore_index=True,
)

summary_df = pd.DataFrame(
    all_summary_rows
)

probability_df.to_csv(
    OUT_DIR / "v2_6_probability_distributions.csv",
    index=False,
)

error_df.to_csv(
    OUT_DIR / "v2_6_error_breakdown.csv",
    index=False,
)

summary_df.to_csv(
    OUT_DIR / "v2_6_probability_quality.csv",
    index=False,
)


# ============================================================
# Overall comparison
# ============================================================

comparison_rows = []

for horizon in HORIZONS:

    path = (
        PRED_DIR
        / f"v2_6_predictions_plus{horizon}d.csv"
    )

    df = pd.read_csv(path)

    actual = df["actual_elevated"]

    for model_name, prediction_col in [
        ("frozen_v2_rf", "v2_prediction"),
        ("transition_rf", "transition_prediction"),
    ]:

        comparison_rows.append(
            {
                "horizon": horizon,
                "model": model_name,
                "f1": f1_score(
                    actual,
                    df[prediction_col],
                    zero_division=0,
                ),
                "precision": precision_score(
                    actual,
                    df[prediction_col],
                    zero_division=0,
                ),
                "recall": recall_score(
                    actual,
                    df[prediction_col],
                    zero_division=0,
                ),
            }
        )

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    OUT_DIR / "v2_6_overall_comparison.csv",
    index=False,
)

print("\n" + "=" * 75)
print("V2.6 DIAGNOSTICS COMPLETE")
print("=" * 75)

print("\nProbability quality:")
print(
    summary_df.round(5).to_string(
        index=False
    )
)

print("\nOverall comparison:")
print(
    comparison_df.round(5).to_string(
        index=False
    )
)

print("\nArtifacts:")
print(OUT_DIR)
