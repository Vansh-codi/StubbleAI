import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "research"
    / "v2"
    / "error_analysis"
    / "rf_2025_error_predictions.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "research"
    / "v2"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = ["+1d", "+2d", "+3d", "+5d", "+7d"]


# ============================================================
# TEMPORAL WINDOWS
# ============================================================

PERIODS = {
    "Early season": (
        pd.Timestamp("2025-10-15"),
        pd.Timestamp("2025-10-31"),
    ),
    "Middle season": (
        pd.Timestamp("2025-11-01"),
        pd.Timestamp("2025-11-15"),
    ),
    "Late season": (
        pd.Timestamp("2025-11-16"),
        pd.Timestamp("2025-11-23"),
    ),
}


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("STUBBLEAI V2.2 — TEMPORAL ERROR ANALYSIS")
print("=" * 75)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Missing input file:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

print(f"Input shape: {df.shape}")
print(
    f"Input date range: "
    f"{df['date'].min().date()} -> "
    f"{df['date'].max().date()}"
)


# ============================================================
# ASSIGN TEMPORAL PHASE
# ============================================================

def assign_phase(date):
    for phase, (start, end) in PERIODS.items():
        if start <= date <= end:
            return phase

    return None


df["season_phase"] = df["date"].apply(assign_phase)

df = df[df["season_phase"].notna()].copy()

print(
    f"Analysis rows retained: {len(df)}"
)


# ============================================================
# OVERALL TEMPORAL ANALYSIS
# ============================================================

rows = []

for horizon in HORIZONS:

    hdf = df[df["horizon"] == horizon].copy()

    for phase in PERIODS:

        sub = hdf[
            hdf["season_phase"] == phase
        ].copy()

        if sub.empty:
            continue

        y_true = sub["actual_elevated"].astype(int)

        rf_pred = sub["rf_prediction"].astype(int)

        persistence_pred = (
            sub["persistence_prediction"]
            .astype(int)
        )

        rf_accuracy = accuracy_score(
            y_true,
            rf_pred,
        )

        rf_precision = precision_score(
            y_true,
            rf_pred,
            zero_division=0,
        )

        rf_recall = recall_score(
            y_true,
            rf_pred,
            zero_division=0,
        )

        rf_f1 = f1_score(
            y_true,
            rf_pred,
            zero_division=0,
        )

        persistence_accuracy = accuracy_score(
            y_true,
            persistence_pred,
        )

        persistence_precision = precision_score(
            y_true,
            persistence_pred,
            zero_division=0,
        )

        persistence_recall = recall_score(
            y_true,
            persistence_pred,
            zero_division=0,
        )

        persistence_f1 = f1_score(
            y_true,
            persistence_pred,
            zero_division=0,
        )

        rf_fp = int(
            (
                (rf_pred == 1)
                & (y_true == 0)
            ).sum()
        )

        rf_fn = int(
            (
                (rf_pred == 0)
                & (y_true == 1)
            ).sum()
        )

        persistence_fp = int(
            (
                (persistence_pred == 1)
                & (y_true == 0)
            ).sum()
        )

        persistence_fn = int(
            (
                (persistence_pred == 0)
                & (y_true == 1)
            ).sum()
        )

        rows.append(
            {
                "horizon": horizon,
                "season_phase": phase,
                "n": len(sub),

                "actual_elevated_count":
                    int(y_true.sum()),

                "actual_elevated_rate":
                    float(y_true.mean()),

                "rf_accuracy":
                    rf_accuracy,

                "rf_precision":
                    rf_precision,

                "rf_recall":
                    rf_recall,

                "rf_f1":
                    rf_f1,

                "rf_fp":
                    rf_fp,

                "rf_fn":
                    rf_fn,

                "persistence_accuracy":
                    persistence_accuracy,

                "persistence_precision":
                    persistence_precision,

                "persistence_recall":
                    persistence_recall,

                "persistence_f1":
                    persistence_f1,

                "persistence_fp":
                    persistence_fp,

                "persistence_fn":
                    persistence_fn,

                "rf_f1_minus_persistence":
                    rf_f1 - persistence_f1,

                "rf_accuracy_minus_persistence":
                    rf_accuracy - persistence_accuracy,
            }
        )


result = pd.DataFrame(rows)


# ============================================================
# SAVE OVERALL RESULTS
# ============================================================

output_file = (
    OUTPUT_DIR
    / "temporal_error_analysis.csv"
)

result.to_csv(
    output_file,
    index=False,
)


# ============================================================
# STATE × TEMPORAL ANALYSIS
# ============================================================

state_rows = []

for horizon in HORIZONS:

    hdf = df[df["horizon"] == horizon].copy()

    for state in sorted(
        hdf["state"].unique()
    ):

        sdf = hdf[
            hdf["state"] == state
        ]

        for phase in PERIODS:

            sub = sdf[
                sdf["season_phase"] == phase
            ]

            if sub.empty:
                continue

            y_true = (
                sub["actual_elevated"]
                .astype(int)
            )

            rf_pred = (
                sub["rf_prediction"]
                .astype(int)
            )

            persistence_pred = (
                sub["persistence_prediction"]
                .astype(int)
            )

            rf_f1 = f1_score(
                y_true,
                rf_pred,
                zero_division=0,
            )

            persistence_f1 = f1_score(
                y_true,
                persistence_pred,
                zero_division=0,
            )

            state_rows.append(
                {
                    "state": state,
                    "horizon": horizon,
                    "season_phase": phase,
                    "n": len(sub),

                    "actual_elevated_rate":
                        float(y_true.mean()),

                    "rf_precision":
                        precision_score(
                            y_true,
                            rf_pred,
                            zero_division=0,
                        ),

                    "rf_recall":
                        recall_score(
                            y_true,
                            rf_pred,
                            zero_division=0,
                        ),

                    "rf_f1":
                        rf_f1,

                    "persistence_f1":
                        persistence_f1,

                    "rf_f1_minus_persistence":
                        rf_f1 - persistence_f1,
                }
            )


state_result = pd.DataFrame(
    state_rows
)

state_output_file = (
    OUTPUT_DIR
    / "temporal_state_error_analysis.csv"
)

state_result.to_csv(
    state_output_file,
    index=False,
)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("TEMPORAL ERROR ANALYSIS")
print("=" * 75)

display_cols = [
    "horizon",
    "season_phase",
    "n",
    "actual_elevated_rate",
    "rf_precision",
    "rf_recall",
    "rf_f1",
    "persistence_f1",
    "rf_f1_minus_persistence",
    "rf_fp",
    "rf_fn",
]

print(
    result[display_cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n" + "=" * 75)
print("STATE × TEMPORAL ANALYSIS")
print("=" * 75)

print(
    state_result.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n" + "=" * 75)
print("FILES WRITTEN")
print("=" * 75)

print(output_file)
print(state_output_file)

print("\nAnalysis complete.")
print(
    "No model training or threshold tuning "
    "was performed."
)