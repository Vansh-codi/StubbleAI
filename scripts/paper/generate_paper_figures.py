from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# STUBBLEAI — PAPER FIGURE GENERATOR
# Generates publication figures directly from research artifacts
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
V2_DIR = BASE_DIR / "research" / "v2"
OUT_DIR = BASE_DIR / "research" / "paper" / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


HORIZONS = [1, 2, 3, 5, 7]


def save_figure(fig, filename):
    """Save both PNG and PDF."""
    png = OUT_DIR / f"{filename}.png"
    pdf = OUT_DIR / f"{filename}.pdf"

    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    plt.close(fig)

    print(f"  ✓ {png}")
    print(f"  ✓ {pdf}")


# ============================================================
# LOAD FROZEN V2 RF RESULTS
# ============================================================

def load_v2_rf_results():
    """
    Load official frozen V2 RF results.

    Primary source:
        research/v2/plus*/v2_baseline_results.csv

    +2d fallback:
        research/v2/v2_7/results/v2_7_learner_results.csv

    The +2d fallback is intentional because the original
    plus2d baseline CSV contains no RF row.
    """

    rows = []

    for h in HORIZONS:

        baseline_file = (
            V2_DIR
            / f"plus{h}d"
            / "v2_baseline_results.csv"
        )

        found = False

        if baseline_file.exists():

            df = pd.read_csv(baseline_file)

            row = df[
                (df["method"] == "Random Forest") &
                (df["split"] == "test") &
                (df["horizon"] == f"+{h}d")
            ]

            if not row.empty:

                r = row.iloc[0]

                rows.append({
                    "horizon": h,
                    "rf_f1": float(r["f1"]),
                    "rf_accuracy": float(r["accuracy"]),
                    "rf_precision": float(r["precision"]),
                    "rf_recall": float(r["recall"]),
                    "rf_roc_auc": float(r["roc_auc"]),
                    "rf_pr_auc": float(r["pr_auc"]),
                    "source": str(baseline_file),
                })

                found = True

        # ----------------------------------------------------
        # +2d fallback
        # ----------------------------------------------------

        if not found:

            v27_file = (
                V2_DIR
                / "v2_7"
                / "results"
                / "v2_7_learner_results.csv"
            )

            if v27_file.exists():

                df_v27 = pd.read_csv(v27_file)

                row = df_v27[
                    (df_v27["model"] == "rf") &
                    (df_v27["horizon"] == h)
                ]

                if not row.empty:

                    r = row.iloc[0]

                    rows.append({
                        "horizon": h,
                        "rf_f1": float(r["test_f1"]),
                        "rf_accuracy": float(r["test_accuracy"]),
                        "rf_precision": float(r["test_precision"]),
                        "rf_recall": float(r["test_recall"]),
                        "rf_roc_auc": float(r["test_roc_auc"]),
                        "rf_pr_auc": float(r["test_pr_auc"]),
                        "source": str(v27_file),
                    })

                    found = True

        if not found:
            raise FileNotFoundError(
                f"Could not find RF result for +{h}d"
            )

    return pd.DataFrame(rows)


# ============================================================
# LOAD PERSISTENCE RESULTS
# ============================================================

def load_persistence_results():
    """Load persistence F1 for every horizon."""

    rows = []

    for h in HORIZONS:

        baseline_file = (
            V2_DIR
            / f"plus{h}d"
            / "v2_baseline_results.csv"
        )

        if not baseline_file.exists():
            raise FileNotFoundError(baseline_file)

        df = pd.read_csv(baseline_file)

        row = df[
            (df["method"] == "Persistence") &
            (df["split"] == "test") &
            (df["horizon"] == f"+{h}d")
        ]

        if row.empty:
            raise ValueError(
                f"Persistence result missing for +{h}d"
            )

        rows.append({
            "horizon": h,
            "persistence_f1": float(row.iloc[0]["f1"]),
        })

    return pd.DataFrame(rows)


# ============================================================
# FIGURE 2
# MULTI-HORIZON F1: RF VS PERSISTENCE
# ============================================================

def figure_2_rf_vs_persistence():

    print("\nGenerating Figure 2...")

    rf = load_v2_rf_results()
    persistence = load_persistence_results()

    df = rf.merge(
        persistence,
        on="horizon",
        how="inner"
    )

    print("\nFigure 2 source values:")
    print(
        df[
            [
                "horizon",
                "rf_f1",
                "persistence_f1",
                "source"
            ]
        ].to_string(index=False)
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.6)
    )

    x = range(len(df))

    ax.plot(
        x,
        df["rf_f1"],
        marker="o",
        linewidth=2.2,
        markersize=5.5,
        label="Random Forest",
    )

    ax.plot(
        x,
        df["persistence_f1"],
        marker="s",
        linewidth=2.0,
        markersize=5.0,
        label="Persistence",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"+{h} day" if h == 1 else f"+{h} days"
         for h in df["horizon"]]
    )

    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("F1-score")

    ax.set_ylim(0.60, 0.82)

    ax.set_title(
        "Multi-horizon forecasting performance on the 2025 test set"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25
    )

    ax.legend(
        frameon=False,
        loc="lower left"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(
        fig,
        "Figure_2_RF_vs_Persistence_F1"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STUBBLEAI PAPER FIGURE GENERATOR")
    print("=" * 70)

    print(f"\nProject: {BASE_DIR}")
    print(f"V2 data: {V2_DIR}")
    print(f"Output:  {OUT_DIR}")

    figure_2_rf_vs_persistence()

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)