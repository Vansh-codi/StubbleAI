from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# STUBBLEAI V2.4
# DIAGNOSTIC ANALYSIS
# ============================================================
#
# Compare:
#   V2
#   V2.3 Spatial
#   V2.4 Spatio-Temporal
#
# Diagnostics:
#   1. Transition analysis
#   2. Emerging-fire recall
#   3. State analysis
#   4. Season analysis
#   5. Current fire-regime analysis
#
# IMPORTANT:
#   This script ONLY analyzes already-generated predictions.
#   It does not retrain models.
#   It does not tune thresholds.
#   It does not modify frozen benchmark files.
# ============================================================


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


# ============================================================
# LOAD
# ============================================================

def load_predictions(arm, horizon):
    path = (
        PREDICTIONS_DIR
        / f"{arm}_predictions_plus{horizon}d.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file:\n{path}"
        )

    df = pd.read_csv(path)

    required = [
        "state",
        "district",
        "date",
        "fire_count",
        "frp_sum",
        "horizon",
        "target_fire_count",
        "actual_elevated",
        "rf_probability",
        "rf_prediction",
        "threshold",
        "error_type",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} missing columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])

    return df


# ============================================================
# COMMON DIAGNOSTIC LABELS
# ============================================================

def add_labels(df):

    # Current state
    df["current_elevated"] = (
        df["fire_count"] > 2
    ).astype(int)

    df["actual_elevated"] = (
        df["actual_elevated"]
        .astype(int)
    )

    df["rf_prediction"] = (
        df["rf_prediction"]
        .astype(int)
    )

    # Transition
    df["transition"] = (
        df["current_elevated"]
        .map({0: "N", 1: "E"})
        + "→"
        + df["actual_elevated"]
        .map({0: "N", 1: "E"})
    )

    # Season
    month = df["date"].dt.month
    day = df["date"].dt.day

    def season_label(row):
        m = row["month"]
        d = row["day"]

        if m == 10 and d <= 31:
            return "Early"

        if m == 11 and d <= 15:
            return "Middle"

        return "Late"

    temp = pd.DataFrame({
        "month": month,
        "day": day,
    })

    df["season"] = temp.apply(
        season_label,
        axis=1,
    )

    # Current fire regime
    def regime(x):
        if x == 0:
            return "No fire"

        if x <= 2:
            return "Low"

        return "Elevated"

    df["fire_regime"] = (
        df["fire_count"]
        .apply(regime)
    )

    return df


# ============================================================
# METRICS
# ============================================================

def metric_summary(group):

    y = group["actual_elevated"].to_numpy()
    p = group["rf_prediction"].to_numpy()

    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / len(group)
        if len(group) > 0
        else np.nan
    )

    return {
        "n": len(group),
        "actual_elevated_rate": y.mean(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ============================================================
# TRANSITION ANALYSIS
# ============================================================

def transition_analysis(df):

    rows = []

    for transition, group in df.groupby(
        "transition",
        sort=False,
    ):

        metrics = metric_summary(group)

        rows.append({
            "transition": transition,
            **metrics,
        })

    return pd.DataFrame(rows)


# ============================================================
# STATE ANALYSIS
# ============================================================

def state_analysis(df):

    rows = []

    for state, group in df.groupby(
        "state",
        sort=False,
    ):

        metrics = metric_summary(group)

        rows.append({
            "state": state,
            **metrics,
        })

    return pd.DataFrame(rows)


# ============================================================
# SEASON ANALYSIS
# ============================================================

def season_analysis(df):

    order = [
        "Early",
        "Middle",
        "Late",
    ]

    rows = []

    for season in order:

        group = df[
            df["season"] == season
        ]

        if len(group) == 0:
            continue

        metrics = metric_summary(group)

        rows.append({
            "season": season,
            **metrics,
        })

    return pd.DataFrame(rows)


# ============================================================
# FIRE REGIME ANALYSIS
# ============================================================

def fire_regime_analysis(df):

    order = [
        "No fire",
        "Low",
        "Elevated",
    ]

    rows = []

    for regime in order:

        group = df[
            df["fire_regime"] == regime
        ]

        if len(group) == 0:
            continue

        metrics = metric_summary(group)

        rows.append({
            "fire_regime": regime,
            **metrics,
        })

    return pd.DataFrame(rows)


# ============================================================
# EMERGING-FIRE ANALYSIS
# ============================================================

def emerging_analysis(df):

    # Emerging event = current Normal,
    # future Elevated.
    emerging = df[
        (df["current_elevated"] == 0)
        & (df["actual_elevated"] == 1)
    ]

    detected = emerging[
        emerging["rf_prediction"] == 1
    ]

    total = len(emerging)
    detected_count = len(detected)

    recall = (
        detected_count / total
        if total > 0
        else np.nan
    )

    return pd.DataFrame([{
        "n_emerging_events": total,
        "detected_emerging_events": detected_count,
        "emerging_recall": recall,
    }])


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STUBBLEAI V2.4 DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    all_transition = []
    all_state = []
    all_season = []
    all_regime = []
    all_emerging = []

    loaded = {}

    # --------------------------------------------------------
    # LOAD ALL 15 PREDICTION FILES
    # --------------------------------------------------------

    print("\n[1/6] Loading predictions...")

    for arm, arm_label in ARMS.items():

        loaded[arm] = {}

        for horizon in HORIZONS:

            df = load_predictions(
                arm,
                horizon,
            )

            df = add_labels(df)

            loaded[arm][horizon] = df

            print(
                f"{arm_label:25s} "
                f"+{horizon}d: "
                f"{len(df)} rows"
            )

    # --------------------------------------------------------
    # TRANSITION
    # --------------------------------------------------------

    print("\n[2/6] Transition analysis...")

    for arm, arm_label in ARMS.items():

        for horizon in HORIZONS:

            df = loaded[arm][horizon]

            result = transition_analysis(df)

            result.insert(
                0,
                "arm",
                arm,
            )

            result.insert(
                1,
                "arm_label",
                arm_label,
            )

            result.insert(
                2,
                "horizon",
                f"+{horizon}d",
            )

            all_transition.append(result)

    transition_df = pd.concat(
        all_transition,
        ignore_index=True,
    )

    transition_df.to_csv(
        OUTPUT_DIR
        / "v2_4_transition_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    print("\n[3/6] State analysis...")

    for arm, arm_label in ARMS.items():

        for horizon in HORIZONS:

            df = loaded[arm][horizon]

            result = state_analysis(df)

            result.insert(
                0,
                "arm",
                arm,
            )

            result.insert(
                1,
                "arm_label",
                arm_label,
            )

            result.insert(
                2,
                "horizon",
                f"+{horizon}d",
            )

            all_state.append(result)

    state_df = pd.concat(
        all_state,
        ignore_index=True,
    )

    state_df.to_csv(
        OUTPUT_DIR
        / "v2_4_state_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SEASON
    # --------------------------------------------------------

    print("\n[4/6] Season analysis...")

    for arm, arm_label in ARMS.items():

        for horizon in HORIZONS:

            df = loaded[arm][horizon]

            result = season_analysis(df)

            result.insert(
                0,
                "arm",
                arm,
            )

            result.insert(
                1,
                "arm_label",
                arm_label,
            )

            result.insert(
                2,
                "horizon",
                f"+{horizon}d",
            )

            all_season.append(result)

    season_df = pd.concat(
        all_season,
        ignore_index=True,
    )

    season_df.to_csv(
        OUTPUT_DIR
        / "v2_4_season_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------
    # FIRE REGIME
    # --------------------------------------------------------

    print("\n[5/6] Fire-regime analysis...")

    for arm, arm_label in ARMS.items():

        for horizon in HORIZONS:

            df = loaded[arm][horizon]

            result = fire_regime_analysis(df)

            result.insert(
                0,
                "arm",
                arm,
            )

            result.insert(
                1,
                "arm_label",
                arm_label,
            )

            result.insert(
                2,
                "horizon",
                f"+{horizon}d",
            )

            all_regime.append(result)

    regime_df = pd.concat(
        all_regime,
        ignore_index=True,
    )

    regime_df.to_csv(
        OUTPUT_DIR
        / "v2_4_fire_regime_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------
    # EMERGING
    # --------------------------------------------------------

    print("\n[6/6] Emerging-event analysis...")

    for arm, arm_label in ARMS.items():

        for horizon in HORIZONS:

            df = loaded[arm][horizon]

            result = emerging_analysis(df)

            result.insert(
                0,
                "arm",
                arm,
            )

            result.insert(
                1,
                "arm_label",
                arm_label,
            )

            result.insert(
                2,
                "horizon",
                f"+{horizon}d",
            )

            all_emerging.append(result)

    emerging_df = pd.concat(
        all_emerging,
        ignore_index=True,
    )

    emerging_df.to_csv(
        OUTPUT_DIR
        / "v2_4_emerging_analysis.csv",
        index=False,
    )

    # ========================================================
    # PRINT IMPORTANT RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("V2.4 EMERGING-FIRE RECALL")
    print("=" * 80)

    emerging_pivot = (
        emerging_df[
            [
                "arm_label",
                "horizon",
                "n_emerging_events",
                "detected_emerging_events",
                "emerging_recall",
            ]
        ]
        .pivot(
            index="horizon",
            columns="arm_label",
            values="emerging_recall",
        )
    )

    print(
        emerging_pivot.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print()
    print("=" * 80)
    print("TRANSITION ANALYSIS")
    print("=" * 80)

    transition_summary = transition_df[
        [
            "arm_label",
            "horizon",
            "transition",
            "n",
            "recall",
            "f1",
        ]
    ]

    print(
        transition_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 80)
    print("STATE ANALYSIS")
    print("=" * 80)

    print(
        state_df[
            [
                "arm_label",
                "horizon",
                "state",
                "n",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 80)
    print("SEASON ANALYSIS")
    print("=" * 80)

    print(
        season_df[
            [
                "arm_label",
                "horizon",
                "season",
                "n",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 80)
    print("FIRE REGIME ANALYSIS")
    print("=" * 80)

    print(
        regime_df[
            [
                "arm_label",
                "horizon",
                "fire_regime",
                "n",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=" * 80)
    print("FILES WRITTEN")
    print("=" * 80)

    for path in sorted(
        OUTPUT_DIR.glob("v2_4_*.csv")
    ):
        print(path)

    print()
    print("Diagnostic analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()