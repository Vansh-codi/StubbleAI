import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


# ============================================================
# STUBBLEAI V2 - ML DATASET AUDIT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ML_FILE = PROJECT_ROOT / "data" / "processed" / "v2" / "ml_dataset_v2.csv"
FIRE_FILE = PROJECT_ROOT / "data" / "processed" / "v2" / "district_daily_fire_panel_v2.csv"

HORIZONS = [1, 2, 3, 5, 7]

ELEVATED_THRESHOLD = 2

results = {}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def record_check(number, name, passed, reason):
    results[number] = {
        "name": name,
        "passed": passed,
        "reason": reason,
    }

    status = "PASS" if passed else "FAIL"

    print()
    print(f"[CHECK {number}] {name}")
    print(f"Status: {status}")
    print(f"Reason: {reason}")


def fail_and_stop(message):
    print()
    print("=" * 70)
    print("AUDIT STOPPED")
    print("=" * 70)
    print(message)
    sys.exit(1)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STUBBLEAI V2 - ML DATASET AUDIT")
print("=" * 70)

print("\nLoading datasets...")

if not ML_FILE.exists():
    fail_and_stop(
        f"Missing ML dataset:\n{ML_FILE}"
    )

if not FIRE_FILE.exists():
    fail_and_stop(
        f"Missing V2 fire panel:\n{FIRE_FILE}"
    )


df = pd.read_csv(ML_FILE)
fire = pd.read_csv(FIRE_FILE)

df["date"] = pd.to_datetime(df["date"])
fire["date"] = pd.to_datetime(fire["date"])


print(f"ML dataset shape: {df.shape}")
print(f"Fire panel shape: {fire.shape}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "state",
    "district",
    "date",
    "fire_count",
    "frp_sum",
    "day_of_year",
    "sin_year",
    "cos_year",

    # Fire lags
    "fire_lag_1d",
    "fire_lag_2d",
    "fire_lag_3d",
    "fire_lag_5d",
    "fire_lag_7d",
    "fire_lag_14d",

    # FRP lags
    "frp_lag_1d",
    "frp_lag_2d",
    "frp_lag_3d",
    "frp_lag_5d",
    "frp_lag_7d",
    "frp_lag_14d",

    # Fire summaries
    "fire_mean_3d",
    "fire_sparse_mean_7d",
    "fire_sparse_mean_14d",

    # FRP summaries
    "frp_mean_3d",
    "frp_sparse_mean_7d",
    "frp_sparse_mean_14d",
]


for horizon in HORIZONS:

    required_columns.extend(
        [
            f"fire_count_t_plus_{horizon}d",
            f"frp_sum_t_plus_{horizon}d",
        ]
    )


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    fail_and_stop(
        "Required columns are missing:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# FIRE PANEL LOOKUP
# ============================================================

fire_lookup = fire.set_index(
    ["state", "district", "date"]
)


# ============================================================
# CHECK 1 - LAG ALIGNMENT
# ============================================================

print("\n" + "=" * 70)
print("CHECK 1 - LAG ALIGNMENT")
print("=" * 70)

lag_failures = []


for lag in [1, 2, 3, 5, 7, 14]:

    column = f"fire_lag_{lag}d"

    keys = pd.MultiIndex.from_arrays(
        [
            df["state"],
            df["district"],
            df["date"] - pd.Timedelta(days=lag),
        ],
        names=["state", "district", "date"],
    )

    expected = (
        fire_lookup["fire_count"]
        .reindex(keys)
        .to_numpy(dtype=float)
    )

    actual = df[column].to_numpy(dtype=float)

    valid = ~np.isnan(expected)

    mismatches = np.sum(
        valid & (actual != expected)
    )

    if mismatches > 0:

        lag_failures.append(
            f"{column}: {mismatches} mismatches"
        )


lag_pass = len(lag_failures) == 0


record_check(
    1,
    "Lag alignment",
    lag_pass,
    (
        "All fire lag columns match the V2 fire panel "
        "using exact calendar-day offsets."
        if lag_pass
        else "; ".join(lag_failures)
    ),
)


# ============================================================
# CHECK 2 - TARGET ALIGNMENT
# ============================================================

print("\n" + "=" * 70)
print("CHECK 2 - TARGET ALIGNMENT")
print("=" * 70)

target_failures = []


for horizon in HORIZONS:

    fire_column = (
        f"fire_count_t_plus_{horizon}d"
    )

    frp_column = (
        f"frp_sum_t_plus_{horizon}d"
    )


    keys = pd.MultiIndex.from_arrays(
        [
            df["state"],
            df["district"],
            df["date"] + pd.Timedelta(days=horizon),
        ],
        names=["state", "district", "date"],
    )


    expected_fire = (
        fire_lookup["fire_count"]
        .reindex(keys)
        .to_numpy(dtype=float)
    )

    expected_frp = (
        fire_lookup["frp_sum"]
        .reindex(keys)
        .to_numpy(dtype=float)
    )


    actual_fire = (
        df[fire_column]
        .to_numpy(dtype=float)
    )

    actual_frp = (
        df[frp_column]
        .to_numpy(dtype=float)
    )


    fire_valid = ~np.isnan(expected_fire)
    frp_valid = ~np.isnan(expected_frp)


    fire_mismatch = np.sum(
        fire_valid
        & (actual_fire != expected_fire)
    )

    frp_mismatch = np.sum(
        frp_valid
        & (actual_frp != expected_frp)
    )


    if fire_mismatch > 0:

        target_failures.append(
            f"{fire_column}: "
            f"{fire_mismatch} mismatches"
        )


    if frp_mismatch > 0:

        target_failures.append(
            f"{frp_column}: "
            f"{frp_mismatch} mismatches"
        )


target_pass = len(target_failures) == 0


record_check(
    2,
    "Target alignment",
    target_pass,
    (
        "All +1/+2/+3/+5/+7 fire and FRP targets match "
        "the exact future calendar dates."
        if target_pass
        else "; ".join(target_failures)
    ),
)


# ============================================================
# CHECK 3 - LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("CHECK 3 - LEAKAGE CHECK")
print("=" * 70)


target_columns = []


for horizon in HORIZONS:

    target_columns.extend(
        [
            f"fire_count_t_plus_{horizon}d",
            f"frp_sum_t_plus_{horizon}d",
        ]
    )


future_named_features = [
    col
    for col in df.columns
    if (
        ("t_plus" in col)
        or ("future" in col.lower())
    )
    and col not in target_columns
]


duplicate_columns = (
    df.columns[
        df.columns.duplicated()
    ].tolist()
)


leakage_reasons = []


if future_named_features:

    leakage_reasons.append(
        "Future-looking columns found outside "
        "target set: "
        + ", ".join(
            future_named_features
        )
    )


if duplicate_columns:

    leakage_reasons.append(
        "Duplicate column names found: "
        + ", ".join(
            duplicate_columns
        )
    )


feature_columns = [
    col
    for col in df.columns
    if col not in target_columns
]


target_as_feature = [
    col
    for col in feature_columns
    if col in target_columns
]


if target_as_feature:

    leakage_reasons.append(
        "Target columns included in feature set: "
        + ", ".join(
            target_as_feature
        )
    )


leakage_pass = len(leakage_reasons) == 0


record_check(
    3,
    "Leakage check",
    leakage_pass,
    (
        "No future target columns or future-looking "
        "feature columns were found outside the "
        "designated target set."
        if leakage_pass
        else "; ".join(leakage_reasons)
    ),
)


# ============================================================
# CHECK 4 - SEASONAL BOUNDARY
# ============================================================

print("\n" + "=" * 70)
print("CHECK 4 - SEASONAL BOUNDARY")
print("=" * 70)

boundary_failures = []


# ------------------------------------------------------------
# Historical lag year validation
# ------------------------------------------------------------

for lag in [1, 2, 3, 5, 7, 14]:

    expected_year = (
        df["date"]
        - pd.Timedelta(days=lag)
    ).dt.year


    keys = pd.MultiIndex.from_arrays(
        [
            df["state"],
            df["district"],
            df["date"] - pd.Timedelta(days=lag),
        ],
        names=["state", "district", "date"],
    )


    source_year = (
        fire_lookup["year"]
        .reindex(keys)
        .to_numpy()
    )


    expected_year_arr = (
        expected_year
        .to_numpy()
    )


    # IMPORTANT:
    # Ignore missing lookup keys.
    # NaN != integer would otherwise create
    # false mismatches.

    valid = ~pd.isna(source_year)


    mismatch = np.sum(
        valid
        & (
            source_year[valid]
            != expected_year_arr[valid]
        )
    )


    if mismatch > 0:

        boundary_failures.append(
            f"fire_lag_{lag}d: "
            f"{mismatch} year mismatches"
        )


# ------------------------------------------------------------
# Future target year validation
# ------------------------------------------------------------

for horizon in HORIZONS:

    expected_year = (
        df["date"]
        + pd.Timedelta(days=horizon)
    ).dt.year


    keys = pd.MultiIndex.from_arrays(
        [
            df["state"],
            df["district"],
            df["date"] + pd.Timedelta(days=horizon),
        ],
        names=["state", "district", "date"],
    )


    source_year = (
        fire_lookup["year"]
        .reindex(keys)
        .to_numpy()
    )


    expected_year_arr = (
        expected_year
        .to_numpy()
    )


    # IMPORTANT:
    # Compare only where source data exists.

    valid = ~pd.isna(source_year)


    mismatch = np.sum(
        valid
        & (
            source_year[valid]
            != expected_year_arr[valid]
        )
    )


    if mismatch > 0:

        boundary_failures.append(
            f"+{horizon}d target: "
            f"{mismatch} year mismatches"
        )


boundary_pass = len(boundary_failures) == 0


record_check(
    4,
    "Seasonal boundary",
    boundary_pass,
    (
        "Exact date-based joins preserve the correct "
        "calendar-year relationship; no cross-year "
        "lag/target misalignment detected."
        if boundary_pass
        else "; ".join(boundary_failures)
    ),
)


# ============================================================
# CHECK 5 - DISTRICT / STATE CONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("CHECK 5 - DISTRICT/STATE CONSISTENCY")
print("=" * 70)

consistency_failures = []


ml_keys = (
    df[
        [
            "state",
            "district",
            "date",
        ]
    ]
    .drop_duplicates()
)


source_keys = (
    fire[
        [
            "state",
            "district",
            "date",
        ]
    ]
    .drop_duplicates()
)


merged_keys = ml_keys.merge(
    source_keys,
    on=[
        "state",
        "district",
        "date",
    ],
    how="left",
    indicator=True,
)


unmatched_keys = (
    merged_keys["_merge"] == "left_only"
).sum()


if unmatched_keys > 0:

    consistency_failures.append(
        f"{unmatched_keys} ML district-days "
        "do not exist in the V2 fire panel."
    )


district_state_counts = (
    fire.groupby("district")["state"]
    .nunique()
)


ambiguous_districts = (
    district_state_counts[
        district_state_counts > 1
    ]
)


if len(ambiguous_districts) > 0:

    consistency_failures.append(
        "Districts mapped to multiple states: "
        + ", ".join(
            ambiguous_districts.index.tolist()
        )
    )


consistency_pass = (
    len(consistency_failures) == 0
)


record_check(
    5,
    "District/state consistency",
    consistency_pass,
    (
        "Every ML district-day maps to the V2 fire "
        "panel and each district belongs to one state."
        if consistency_pass
        else "; ".join(
            consistency_failures
        )
    ),
)


# ============================================================
# CHECK 6 - PER-DISTRICT TEMPORAL COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("CHECK 6 - PER-DISTRICT TEMPORAL COVERAGE")
print("=" * 70)


coverage_failures = []


# Derive year directly from date.
df["_audit_year"] = (
    df["date"].dt.year
)


coverage = (
    df.groupby(
        [
            "state",
            "district",
            "_audit_year",
        ]
    )["date"]
    .agg(
        rows="count",
        min_date="min",
        max_date="max",
    )
    .reset_index()
)


# ------------------------------------------------------------
# Check for missing calendar days between min/max
# ------------------------------------------------------------

for row in coverage.itertuples(
    index=False
):

    expected_dates = pd.date_range(
        row.min_date,
        row.max_date,
        freq="D",
    )


    if row.rows != len(expected_dates):

        coverage_failures.append(
            f"{row.state}/"
            f"{row.district}/"
            f"{row._audit_year}: "
            f"{row.rows} rows but "
            f"{len(expected_dates)} calendar "
            "days between min/max dates"
        )


# ------------------------------------------------------------
# Within each year, all districts should have
# identical usable row counts.
#
# Different years are allowed to have different
# counts because of the intentional 14-day
# history warmup and 7-day future tail.
# ------------------------------------------------------------

counts_per_district_year = (
    df.groupby(
        [
            "_audit_year",
            "district",
        ]
    )["date"]
    .count()
)


year_consistency_failures = []


for year in sorted(
    df["_audit_year"].unique()
):

    year_counts = (
        counts_per_district_year
        .loc[year]
    )


    if year_counts.nunique() > 1:

        year_consistency_failures.append(
            f"Year {year}: unequal district "
            "row counts — "
            + str(
                year_counts
                .value_counts()
                .to_dict()
            )
        )


if year_consistency_failures:

    coverage_failures.extend(
        year_consistency_failures
    )


coverage_pass = (
    len(coverage_failures) == 0
)


record_check(
    6,
    "Per-district temporal coverage",
    coverage_pass,
    (
        "All district-year groups have continuous "
        "daily coverage and identical usable "
        "row counts within each year."
        if coverage_pass
        else "; ".join(
            coverage_failures[:10]
        )
    ),
)


# Remove audit helper column.
df.drop(
    columns=["_audit_year"],
    inplace=True,
)


# ============================================================
# CHECK 7 - TARGET DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECK 7 - TARGET DISTRIBUTIONS")
print("=" * 70)


distribution_rows = []

baseline_elevated_rate = None


for horizon in HORIZONS:

    column = (
        f"fire_count_t_plus_{horizon}d"
    )


    target = df[column]


    elevated = (
        target > ELEVATED_THRESHOLD
    )


    mean_value = target.mean()

    median_value = target.median()

    zero_pct = (
        (target == 0).mean()
        * 100
    )

    elevated_pct = (
        elevated.mean()
        * 100
    )


    if horizon == 1:

        baseline_elevated_rate = (
            elevated_pct
        )

        relative_rate = 100.0

    else:

        relative_rate = (
            elevated_pct
            / baseline_elevated_rate
            * 100
            if baseline_elevated_rate > 0
            else np.nan
        )


    distribution_rows.append(
        {
            "horizon": f"+{horizon}d",
            "mean": mean_value,
            "median": median_value,
            "zero_pct": zero_pct,
            "elevated_pct": elevated_pct,
            "relative_to_+1d_pct": relative_rate,
        }
    )


distribution_table = pd.DataFrame(
    distribution_rows
)


print(
    distribution_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# ------------------------------------------------------------
# Validate target values
# ------------------------------------------------------------

distribution_failures = []


for horizon in HORIZONS:

    column = (
        f"fire_count_t_plus_{horizon}d"
    )


    negative_count = (
        df[column] < 0
    ).sum()


    if negative_count > 0:

        distribution_failures.append(
            f"{column}: "
            f"{negative_count} negative values"
        )


if (
    baseline_elevated_rate is None
    or baseline_elevated_rate == 0
):

    distribution_failures.append(
        "No Elevated (+1d) examples exist; "
        "classification baseline cannot be "
        "meaningfully evaluated."
    )


distribution_pass = (
    len(distribution_failures) == 0
)


record_check(
    7,
    "Target distributions",
    distribution_pass,
    (
        "Target distributions are valid. The table "
        "above shows mean, median, zero-fire percentage, "
        "Elevated percentage, and Elevated rate relative "
        "to +1d."
        if distribution_pass
        else "; ".join(
            distribution_failures
        )
    ),
)


# ============================================================
# CHECK 8 - PERSISTENCE BASELINE
# ============================================================

print("\n" + "=" * 70)
print("CHECK 8 - PERSISTENCE BASELINE")
print("=" * 70)


print(
    "\nPersistence rule:"
)

print(
    "Predicted fire count at every horizon "
    "= today's observed fire_count."
)


print(
    f"Classification rule: "
    f"fire_count > {ELEVATED_THRESHOLD} "
    "= Elevated"
)


persistence_rows = []

persistence_failures = []


today_fire = (
    df["fire_count"]
    .to_numpy()
)


today_class = (
    today_fire > ELEVATED_THRESHOLD
).astype(int)


for horizon in HORIZONS:

    target_column = (
        f"fire_count_t_plus_{horizon}d"
    )


    actual = (
        df[target_column]
        .to_numpy()
    )


    # --------------------------------------------------------
    # Regression metric
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual,
        today_fire,
    )


    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    actual_class = (
        actual > ELEVATED_THRESHOLD
    ).astype(int)


    accuracy = accuracy_score(
        actual_class,
        today_class,
    )


    f1 = f1_score(
        actual_class,
        today_class,
        zero_division=0,
    )


    persistence_rows.append(
        {
            "horizon": f"+{horizon}d",
            "MAE": mae,
            "Accuracy": accuracy,
            "F1_Elevated": f1,
        }
    )


persistence_table = pd.DataFrame(
    persistence_rows
)


print(
    persistence_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ------------------------------------------------------------
# Validate metrics
# ------------------------------------------------------------

metric_values = (
    persistence_table[
        [
            "MAE",
            "Accuracy",
            "F1_Elevated",
        ]
    ]
    .to_numpy()
)


if not np.isfinite(
    metric_values
).all():

    persistence_failures.append(
        "Persistence metrics contain "
        "non-finite values."
    )


persistence_pass = (
    len(persistence_failures) == 0
)


record_check(
    8,
    "Persistence baseline",
    persistence_pass,
    (
        "Persistence MAE, Elevated-class accuracy, "
        "and Elevated-class F1 were successfully "
        "calculated for all horizons."
        if persistence_pass
        else "; ".join(
            persistence_failures
        )
    ),
)


# ============================================================
# CHECK 9 - FINAL GO / NO-GO SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CHECK 9 - FINAL AUDIT SUMMARY")
print("=" * 70)


all_passed = True


for number in range(1, 9):

    result = results[number]


    status = (
        "PASS"
        if result["passed"]
        else "FAIL"
    )


    print(
        f"Check {number}: {status} — "
        f"{result['name']}"
    )


    if not result["passed"]:

        all_passed = False


print("\n" + "-" * 70)


# ============================================================
# TARGET IMBALANCE INTERPRETATION
# ============================================================

print("\nTARGET IMBALANCE INTERPRETATION")
print("-" * 70)


for row in distribution_rows:

    horizon = row["horizon"]

    elevated = row["elevated_pct"]

    relative = row[
        "relative_to_+1d_pct"
    ]


    print(
        f"{horizon}: "
        f"Elevated = {elevated:.2f}% "
        f"({relative:.1f}% of +1d rate)"
    )


# ============================================================
# PERSISTENCE FLOOR
# ============================================================

print("\nPERSISTENCE FLOOR")
print("-" * 70)


for row in persistence_rows:

    print(
        f"{row['horizon']}: "
        f"MAE={row['MAE']:.4f}, "
        f"Accuracy={row['Accuracy']:.4f}, "
        f"F1={row['F1_Elevated']:.4f}"
    )


# ============================================================
# FINAL DECISION
# ============================================================

print("\n" + "=" * 70)


if all_passed:

    print("FINAL DECISION: GO")

    print("=" * 70)


    print(
        "\nDataset audit passed."
    )


    print(
        "The V2 dataset is cleared for "
        "baseline/model development."
    )


    print(
        "\nImportant:"
    )


    print(
        "The persistence metrics above are the "
        "minimum baseline that future V2 models "
        "should be compared against."
    )


    print(
        "A model should not be considered "
        "deployment-worthy merely because it "
        "beats a random classifier; it should "
        "demonstrate meaningful improvement "
        "over persistence."
    )


    sys.exit(0)


else:

    print("FINAL DECISION: NO-GO")

    print("=" * 70)


    print(
        "\nAt least one audit check failed."
    )


    print(
        "Do NOT train or deploy V2 until the "
        "failed data issue is investigated."
    )


    sys.exit(1)