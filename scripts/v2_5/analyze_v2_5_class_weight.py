
from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
V25_DIR = BASE_DIR / "research" / "v2" / "v2_5"
PRED_DIR = V25_DIR / "predictions"
DIAG_DIR = V25_DIR / "diagnostics"

DIAG_DIR.mkdir(parents=True, exist_ok=True)

ARMS = [
    "baseline",
    "balanced",
    "positive_1_5",
    "positive_2",
    "positive_3",
]

HORIZONS = [1, 2, 3, 5, 7]

V2_F1 = {
    1: 0.775377,
    2: 0.751282,
    3: 0.730518,
    5: 0.730337,
    7: 0.708464,
}

V2_EMERGING_RECALL = {
    1: 0.403727,
    2: 0.481675,
    3: 0.620192,
    5: 0.563452,
    7: 0.626728,
}

V2_EE_RECALL = {
    1: 0.959927,
    2: 0.951830,
    3: 0.973896,
    5: 0.936759,
    7: 0.938994,
}


def load_prediction(arm, horizon):
    path = PRED_DIR / f"{arm}_predictions_plus{horizon}d.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {
        "state",
        "district",
        "date",
        "fire_count",
        "target_fire_count",
        "actual_elevated",
        "prediction",
        "probability",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} missing columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"])

    # Current-day regime.
    df["current_regime"] = pd.cut(
        df["fire_count"],
        bins=[-np.inf, 0, 2, np.inf],
        labels=["no_fire", "low", "elevated"],
        right=True,
    )

    # Target regime.
    df["target_elevated"] = (
        df["target_fire_count"] > 2
    ).astype(int)

    # Transition class.
    current_elevated = (
        df["fire_count"] > 2
    ).astype(int)

    target_elevated = df["target_elevated"]

    df["transition"] = np.select(
    [
        (current_elevated == 0) & (target_elevated == 0),
        (current_elevated == 0) & (target_elevated == 1),
        (current_elevated == 1) & (target_elevated == 0),
        (current_elevated == 1) & (target_elevated == 1),
    ],
    [
        "N_to_N",
        "N_to_E",
        "E_to_N",
        "E_to_E",
    ],
    default="UNKNOWN",
)

    # Season windows used by V2.2 diagnostics.
    day = df["date"].dt.day

    df["season_period"] = np.select(
        [
            (df["date"].dt.month == 10)
            & (day <= 31),
            (df["date"].dt.month == 11)
            & (day <= 15),
            (df["date"].dt.month == 11)
            & (day >= 16),
        ],
        [
            "Early",
            "Middle",
            "Late",
        ],
        default="Other",
    )

    return df


def recall_for_subset(df):
    if len(df) == 0:
        return np.nan

    return float(
        (
            df["prediction"]
            == df["target_elevated"]
        ).where(
            df["target_elevated"] == 1
        ).dropna().mean()
    )


def transition_metrics(df):
    rows = []

    for transition in [
        "N_to_N",
        "N_to_E",
        "E_to_N",
        "E_to_E",
    ]:
        subset = df[df["transition"] == transition]

        if len(subset) == 0:
            continue

        rows.append({
            "transition": transition,
            "n": len(subset),
            "prediction_correct": int(
                (subset["prediction"] == subset["target_elevated"]).sum()
            ),
            "prediction_accuracy": float(
                (subset["prediction"] == subset["target_elevated"]).mean()
            ),
        })

    return rows


def main():
    all_predictions = {}
    all_results = []

    print("=" * 90)
    print("V2.5 COST-SENSITIVE DIAGNOSTIC ANALYSIS")
    print("=" * 90)

    for arm in ARMS:
        all_predictions[arm] = {}

        for horizon in HORIZONS:
            df = load_prediction(arm, horizon)
            all_predictions[arm][horizon] = df

            print(
                f"{arm:12s} +{horizon}d "
                f"rows={len(df)}"
            )

    # ------------------------------------------------------------------
    # 1. Transition analysis
    # ------------------------------------------------------------------
    transition_rows = []

    for arm in ARMS:
        for horizon in HORIZONS:
            df = all_predictions[arm][horizon]

            for transition in [
                "N_to_N",
                "N_to_E",
                "E_to_N",
                "E_to_E",
            ]:
                subset = df[
                    df["transition"] == transition
                ]

                if len(subset) == 0:
                    continue

                transition_rows.append({
                    "arm": arm,
                    "horizon_days": horizon,
                    "transition": transition,
                    "n": len(subset),
                    "correct": int(
                        (
                            subset["prediction"]
                            == subset["target_elevated"]
                        ).sum()
                    ),
                    "accuracy": float(
                        (
                            subset["prediction"]
                            == subset["target_elevated"]
                        ).mean()
                    ),
                })

    transition_df = pd.DataFrame(transition_rows)

    transition_file = (
        DIAG_DIR / "v2_5_transition_analysis.csv"
    )

    transition_df.to_csv(
        transition_file,
        index=False,
    )

    # ------------------------------------------------------------------
    # 2. Emerging-event analysis
    # ------------------------------------------------------------------
    emerging_rows = []

    for arm in ARMS:
        for horizon in HORIZONS:
            df = all_predictions[arm][horizon]

            subset = df[
                df["transition"] == "N_to_E"
            ]

            recall = (
                float(
                    (subset["prediction"] == 1).mean()
                )
                if len(subset)
                else np.nan
            )

            emerging_rows.append({
                "arm": arm,
                "horizon_days": horizon,
                "n_emerging": len(subset),
                "emerging_recall": recall,
                "v2_emerging_recall": V2_EMERGING_RECALL[horizon],
                "delta_vs_v2": (
                    recall
                    - V2_EMERGING_RECALL[horizon]
                ),
            })

    emerging_df = pd.DataFrame(emerging_rows)

    emerging_file = (
        DIAG_DIR / "v2_5_emerging_analysis.csv"
    )

    emerging_df.to_csv(
        emerging_file,
        index=False,
    )

    # ------------------------------------------------------------------
    # 3. Continuation E->E analysis
    # ------------------------------------------------------------------
    continuation_rows = []

    for arm in ARMS:
        for horizon in HORIZONS:
            df = all_predictions[arm][horizon]

            subset = df[
                df["transition"] == "E_to_E"
            ]

            recall = (
                float(
                    (subset["prediction"] == 1).mean()
                )
                if len(subset)
                else np.nan
            )

            continuation_rows.append({
                "arm": arm,
                "horizon_days": horizon,
                "n_continuation": len(subset),
                "ee_recall": recall,
                "v2_ee_recall": V2_EE_RECALL[horizon],
                "delta_vs_v2": (
                    recall
                    - V2_EE_RECALL[horizon]
                ),
            })

    continuation_df = pd.DataFrame(
        continuation_rows
    )

    continuation_file = (
        DIAG_DIR / "v2_5_continuation_analysis.csv"
    )

    continuation_df.to_csv(
        continuation_file,
        index=False,
    )

    # ------------------------------------------------------------------
    # 4. State analysis
    # ------------------------------------------------------------------
    state_rows = []

    for arm in ARMS:
        for horizon in HORIZONS:
            df = all_predictions[arm][horizon]

            for state in sorted(df["state"].unique()):
                subset = df[
                    df["state"] == state
                ]

                tp = int(
                    (
                        (subset["prediction"] == 1)
                        & (subset["target_elevated"] == 1)
                    ).sum()
                )

                fp = int(
                    (
                        (subset["prediction"] == 1)
                        & (subset["target_elevated"] == 0)
                    ).sum()
                )

                fn = int(
                    (
                        (subset["prediction"] == 0)
                        & (subset["target_elevated"] == 1)
                    ).sum()
                )

                precision = (
                    tp / (tp + fp)
                    if tp + fp
                    else 0.0
                )

                recall = (
                    tp / (tp + fn)
                    if tp + fn
                    else 0.0
                )

                f1 = (
                    2 * precision * recall
                    / (precision + recall)
                    if precision + recall
                    else 0.0
                )

                state_rows.append({
                    "arm": arm,
                    "horizon_days": horizon,
                    "state": state,
                    "n": len(subset),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                })

    state_df = pd.DataFrame(state_rows)

    state_file = (
        DIAG_DIR / "v2_5_state_analysis.csv"
    )

    state_df.to_csv(
        state_file,
        index=False,
    )

    # ------------------------------------------------------------------
    # 5. Season analysis
    # ------------------------------------------------------------------
    season_rows = []

    for arm in ARMS:
        for horizon in HORIZONS:
            df = all_predictions[arm][horizon]

            for season in [
                "Early",
                "Middle",
                "Late",
            ]:
                subset = df[
                    df["season_period"] == season
                ]

                if len(subset) == 0:
                    continue

                tp = int(
                    (
                        (subset["prediction"] == 1)
                        & (subset["target_elevated"] == 1)
                    ).sum()
                )

                fp = int(
                    (
                        (subset["prediction"] == 1)
                        & (subset["target_elevated"] == 0)
                    ).sum()
                )

                fn = int(
                    (
                        (subset["prediction"] == 0)
                        & (subset["target_elevated"] == 1)
                    ).sum()
                )

                precision = (
                    tp / (tp + fp)
                    if tp + fp
                    else 0.0
                )

                recall = (
                    tp / (tp + fn)
                    if tp + fn
                    else 0.0
                )

                f1 = (
                    2 * precision * recall
                    / (precision + recall)
                    if precision + recall
                    else 0.0
                )

                season_rows.append({
                    "arm": arm,
                    "horizon_days": horizon,
                    "season": season,
                    "n": len(subset),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                })

    season_df = pd.DataFrame(season_rows)

    season_file = (
        DIAG_DIR / "v2_5_season_analysis.csv"
    )

    season_df.to_csv(
        season_file,
        index=False,
    )

    # ------------------------------------------------------------------
    # 6. Current fire-regime analysis
    # ------------------------------------------------------------------
    regime_rows = []

    for arm in ARMS:
        for horizon in HORIZONS:
            df = all_predictions[arm][horizon]

            for regime in [
                "no_fire",
                "low",
                "elevated",
            ]:
                subset = df[
                    df["current_regime"] == regime
                ]

                if len(subset) == 0:
                    continue

                tp = int(
                    (
                        (subset["prediction"] == 1)
                        & (subset["target_elevated"] == 1)
                    ).sum()
                )

                fp = int(
                    (
                        (subset["prediction"] == 1)
                        & (subset["target_elevated"] == 0)
                    ).sum()
                )

                fn = int(
                    (
                        (subset["prediction"] == 0)
                        & (subset["target_elevated"] == 1)
                    ).sum()
                )

                precision = (
                    tp / (tp + fp)
                    if tp + fp
                    else 0.0
                )

                recall = (
                    tp / (tp + fn)
                    if tp + fn
                    else 0.0
                )

                f1 = (
                    2 * precision * recall
                    / (precision + recall)
                    if precision + recall
                    else 0.0
                )

                regime_rows.append({
                    "arm": arm,
                    "horizon_days": horizon,
                    "current_regime": regime,
                    "n": len(subset),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                })

    regime_df = pd.DataFrame(regime_rows)

    regime_file = (
        DIAG_DIR / "v2_5_fire_regime_analysis.csv"
    )

    regime_df.to_csv(
        regime_file,
        index=False,
    )

    # ------------------------------------------------------------------
    # 7. Summary / acceptance analysis
    # ------------------------------------------------------------------
    result_file = (
        V25_DIR / "v2_5_class_weight_results.csv"
    )

    results_df = pd.read_csv(result_file)

    summary_rows = []

    for arm in ARMS:
        arm_results = results_df[
            results_df["arm"] == arm
        ].copy()

        avg_f1 = float(
            arm_results["f1"].mean()
        )

        f1_deltas = [
            float(
                arm_results.loc[
                    arm_results["horizon_days"] == h,
                    "f1",
                ].iloc[0]
                - V2_F1[h]
            )
            for h in HORIZONS
        ]

        emerging_arm = emerging_df[
            emerging_df["arm"] == arm
        ]

        emerging_deltas = (
            emerging_arm["delta_vs_v2"]
            .to_numpy()
        )

        continuation_arm = continuation_df[
            continuation_df["arm"] == arm
        ]

        continuation_deltas = (
            continuation_arm["delta_vs_v2"]
            .to_numpy()
        )

        horizons_emerging_improved = int(
            (emerging_deltas > 0).sum()
        )

        max_f1_drop = float(
            min(f1_deltas)
        )

        min_ee_delta = float(
            min(continuation_deltas)
        )

        avg_v2_f1 = float(
            np.mean(list(V2_F1.values()))
        )

        criteria_avg_f1 = (
            avg_f1 >= avg_v2_f1
        )

        criteria_emerging = (
            horizons_emerging_improved >= 3
        )

        criteria_f1_drop = (
            max_f1_drop >= -0.010
        )

        criteria_ee = (
            min_ee_delta >= -0.020
        )

        accepted = all([
            criteria_avg_f1,
            criteria_emerging,
            criteria_f1_drop,
            criteria_ee,
        ])

        summary_rows.append({
            "arm": arm,
            "average_f1": avg_f1,
            "v2_average_f1": avg_v2_f1,
            "average_f1_delta": avg_f1 - avg_v2_f1,
            "emerging_horizons_improved": horizons_emerging_improved,
            "max_f1_delta": max(f1_deltas),
            "min_f1_delta": min(f1_deltas),
            "min_ee_recall_delta": min_ee_delta,
            "criterion_average_f1": criteria_avg_f1,
            "criterion_emerging_3_of_5": criteria_emerging,
            "criterion_no_f1_drop_gt_010": criteria_f1_drop,
            "criterion_ee_drop_le_2pp": criteria_ee,
            "candidate_passes_core_criteria": accepted,
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_file = (
        DIAG_DIR / "v2_5_acceptance_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print()
    print("=" * 90)
    print("EMERGING-FIRE RECALL")
    print("=" * 90)

    print(
        emerging_df[
            [
                "arm",
                "horizon_days",
                "emerging_recall",
                "v2_emerging_recall",
                "delta_vs_v2",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 90)
    print("CONTINUATION E->E RECALL")
    print("=" * 90)

    print(
        continuation_df[
            [
                "arm",
                "horizon_days",
                "ee_recall",
                "v2_ee_recall",
                "delta_vs_v2",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 90)
    print("ACCEPTANCE SUMMARY")
    print("=" * 90)

    print(
        summary_df.to_string(index=False)
    )

    print()
    print("Files written:")
    print(emerging_file)
    print(continuation_file)
    print(transition_file)
    print(state_file)
    print(season_file)
    print(regime_file)
    print(summary_file)


if __name__ == "__main__":
    main()

