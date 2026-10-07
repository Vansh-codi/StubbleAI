from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# ============================================================
# STUBBLEAI — PAPER FIGURE GENERATOR
# Generates publication figures directly from research artifacts
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
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
# FIGURE 1
# STUDY AREA MAP
# ============================================================

def figure_1_study_area():
    print("\nGenerating Figure 1...")

    geojson_file = BASE_DIR / "data" / "processed" / "v1" / "districts_punjab_haryana.geojson"

    if not geojson_file.exists():
        raise FileNotFoundError(geojson_file)

    gdf = gpd.read_file(geojson_file)

    required = {"name", "state"}
    missing = required - set(gdf.columns)
    if missing:
        raise ValueError(f"Missing GeoJSON columns: {sorted(missing)}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    print(f"  Districts: {len(gdf)}")
    print(f"  States: {sorted(gdf['state'].dropna().unique())}")

    fig, ax = plt.subplots(figsize=(7.2, 6.5))

    # District boundaries
    gdf.boundary.plot(
        ax=ax,
        linewidth=0.55,
        edgecolor="black"
    )

    # Light state-level distinction
    for state in sorted(gdf["state"].unique()):
        subset = gdf[gdf["state"] == state]
        subset.plot(
            ax=ax,
            alpha=0.18,
            edgecolor="black",
            linewidth=0.55,
            label=state
        )

    # District labels
    for _, row in gdf.iterrows():
        point = row.geometry.representative_point()
        ax.text(
            point.x,
            point.y,
            row["name"],
            fontsize=5.5,
            ha="center",
            va="center"
        )

    # State labels
    for state in sorted(gdf["state"].unique()):
        subset = gdf[gdf["state"] == state]
        point = subset.geometry.unary_union.representative_point()

        ax.text(
            point.x,
            point.y,
            state,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center"
        )

    # North arrow
    ax.annotate(
        "N",
        xy=(0.96, 0.88),
        xytext=(0.96, 0.75),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=1.2
        )
    )

    ax.set_title("Study area: Punjab and Haryana, India")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.legend(
        frameon=False,
        loc="lower left"
    )

    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(
        fig,
        "Figure_1_Study_Area_Map"
    )
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
# FIGURE 3
# EMERGING VS CONTINUING FIRE DETECTION
# ============================================================

def figure_3_emerging_vs_continuing():

    print("\nGenerating Figure 3...")

    transition_file = (
        V2_DIR
        / "error_analysis"
        / "fire_transition_horizon_analysis.csv"
    )

    if not transition_file.exists():
        raise FileNotFoundError(transition_file)

    df = pd.read_csv(transition_file)

    # Keep only the two transitions relevant to the paper.
    df = df[
        df["transition"].isin([
            "Normal -> Elevated",
            "Elevated -> Elevated",
        ])
    ].copy()

    # Verify all five horizons are present.
    expected_horizons = {"+1d", "+2d", "+3d", "+5d", "+7d"}

    if set(df["horizon"]) != expected_horizons:
        raise ValueError(
            "Transition artifact does not contain exactly the expected "
            "five horizons."
        )

    # RF accuracy is equivalent to transition recall here because:
    # - Normal -> Elevated rows have actual_elevated = 1
    # - Elevated -> Elevated rows have actual_elevated = 1
    #
    # We calculate recall explicitly from rf_correct / total so the
    # figure is directly tied to the stored transition counts.
    df["rf_recall"] = df["rf_correct"] / df["total"]

    emerging = df[
        df["transition"] == "Normal -> Elevated"
    ].copy()

    continuing = df[
        df["transition"] == "Elevated -> Elevated"
    ].copy()

    horizon_order = {
        "+1d": 1,
        "+2d": 2,
        "+3d": 3,
        "+5d": 5,
        "+7d": 7,
    }

    emerging["horizon_num"] = emerging["horizon"].map(horizon_order)
    continuing["horizon_num"] = continuing["horizon"].map(horizon_order)

    emerging = emerging.sort_values("horizon_num")
    continuing = continuing.sort_values("horizon_num")

    print("\nFigure 3 source values:")

    for h in horizon_order:
        e = emerging[emerging["horizon"] == h].iloc[0]
        c = continuing[continuing["horizon"] == h].iloc[0]

        print(
            f"{h}: "
            f"N->E recall={e['rf_recall']:.4f}, "
            f"E->E recall={c['rf_recall']:.4f}"
        )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.6)
    )

    x = range(len(HORIZONS))

    ax.plot(
        x,
        emerging["rf_recall"],
        marker="o",
        linewidth=2.2,
        markersize=5.5,
        label="Normal → Elevated",
    )

    ax.plot(
        x,
        continuing["rf_recall"],
        marker="s",
        linewidth=2.2,
        markersize=5.5,
        label="Elevated → Elevated",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [
            f"+{h} day" if h == 1 else f"+{h} days"
            for h in HORIZONS
        ]
    )

    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Recall")

    ax.set_ylim(0.30, 1.02)

    ax.set_title(
        "Detection of emerging and continuing elevated-fire episodes"
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
        "Figure_3_Emerging_vs_Continuing_Recall"
    )

# ============================================================
# FIGURE 4
# +1-DAY RANDOM FOREST CONFUSION MATRIX
# ============================================================

def figure_4_plus1_confusion_matrix():

    print("\nGenerating Figure 4...")

    prediction_file = (
        V2_DIR
        / "error_analysis"
        / "rf_2025_error_predictions.csv"
    )

    if not prediction_file.exists():
        raise FileNotFoundError(prediction_file)

    df = pd.read_csv(prediction_file)

    # Use only the +1-day forecast horizon.
    df = df[df["horizon"] == "+1d"].copy()

    if df.empty:
        raise ValueError("No +1d predictions found.")

    # Make sure the required columns exist.
    required_columns = {
        "actual_elevated",
        "rf_prediction",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Convert to integer labels.
    actual = df["actual_elevated"].astype(int)
    predicted = df["rf_prediction"].astype(int)

    # Explicitly calculate the four cells.
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tp = int(((actual == 1) & (predicted == 1)).sum())

    matrix = [
        [tn, fp],
        [fn, tp],
    ]

    print("\nFigure 4 confusion matrix:")
    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TP = {tp}")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(5.8, 5.0)
    )

    image = ax.imshow(matrix)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "Normal",
        "Elevated",
    ])

    ax.set_yticklabels([
        "Normal",
        "Elevated",
    ])

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")

    ax.set_title(
        "Random Forest confusion matrix for +1-day forecasting"
    )

    # Add counts inside cells.
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{matrix[i][j]:,}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

    # Color scale.
    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Number of predictions",
    )

    ax.set_aspect("equal")

    save_figure(
        fig,
        "Figure_4_Plus1_Confusion_Matrix"
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
    figure_1_study_area()
    figure_2_rf_vs_persistence()
    figure_3_emerging_vs_continuing()
    figure_4_plus1_confusion_matrix()
    figure_2_rf_vs_persistence()
    figure_3_emerging_vs_continuing()
    figure_4_plus1_confusion_matrix()
    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)