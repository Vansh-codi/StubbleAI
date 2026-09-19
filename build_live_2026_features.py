import pandas as pd


# ============================================================
# STUBBLEAI - LIVE 2026 FIRE FEATURE ENGINE
# ============================================================

HISTORICAL_FILE = "district_daily_fire_counts.csv"
LIVE_FILE = "live_district_fire_2026.csv"
OUTPUT_FILE = "live_2026_fire_features.csv"


print("=" * 70)
print("STUBBLEAI - LIVE 2026 FIRE FEATURE ENGINE")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load historical data
# ------------------------------------------------------------

historical = pd.read_csv(HISTORICAL_FILE)

historical["date"] = pd.to_datetime(
    historical["date"]
)
if historical["date"].isna().any():
    raise ValueError(
        "Historical fire data contains invalid dates."
    )

historical["district"] = (
    historical["district"]
    .astype(str)
    .str.strip()
    .str.replace(
        "S.A.S Nagar",
        "S A S Nagar",
        regex=False
    )
)

historical["state"] = (
    historical["state"]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# 2. Load live 2026 data
# ------------------------------------------------------------

live = pd.read_csv(LIVE_FILE)
required_columns = [
    "date",
    "state",
    "district",
    "fire_count",
]

for file_name, df in [
    (HISTORICAL_FILE, historical),
    (LIVE_FILE, live),
]:
    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{file_name} is missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError(
            f"{file_name} is empty."
        )

live["date"] = pd.to_datetime(
    live["date"]
)
if not (live["date"].dt.year == 2026).all():
    raise ValueError(
        "LIVE_FILE contains dates outside 2026."
    )

live["district"] = (
    live["district"]
    .astype(str)
    .str.strip()
    .str.replace(
        "S.A.S Nagar",
        "S A S Nagar",
        regex=False
    )
)

live["state"] = (
    live["state"]
    .astype(str)
    .str.strip()
)
historical["fire_count"] = pd.to_numeric(
    historical["fire_count"],
    errors="coerce"
)

live["fire_count"] = pd.to_numeric(
    live["fire_count"],
    errors="coerce"
)

if historical["fire_count"].isna().any():
    raise ValueError(
        "Historical fire data contains invalid fire_count values."
    )

if live["fire_count"].isna().any():
    raise ValueError(
        "Live fire data contains invalid fire_count values."
    )

if (historical["fire_count"] < 0).any():
    raise ValueError(
        "Historical fire data contains negative fire_count values."
    )

if (live["fire_count"] < 0).any():
    raise ValueError(
        "Live fire data contains negative fire_count values."
    )

# ------------------------------------------------------------
# 3. Combine historical + live
# ------------------------------------------------------------

historical = historical[
    ["date", "state", "district", "fire_count"]
].copy()

live = live[
    ["date", "state", "district", "fire_count"]
].copy()

combined = pd.concat(
    [historical, live],
    ignore_index=True
)


# ------------------------------------------------------------
# 4. Remove duplicate district/date records
# ------------------------------------------------------------

# combined = (
#     combined
#     .groupby(
#         ["date", "state", "district"],
#         as_index=False
#     )["fire_count"]
#     .sum()
# )
combined = combined[
    [
        "date",
        "state",
        "district",
        "fire_count",
    ]
].copy()

# ------------------------------------------------------------
# 5. Identify actual 2026 observation period
# ------------------------------------------------------------

live_start = live["date"].min()
live_end = live["date"].max()

print()
print("Historical period:")
print(
    historical["date"].min().date(),
    "to",
    historical["date"].max().date()
)

print()
print("Actual 2026 FIRMS period:")
print(
    live_start.date(),
    "to",
    live_end.date()
)
# ------------------------------------------------------------
# Validate continuous 2026 observation period
# ------------------------------------------------------------

expected_dates = pd.date_range(
    live_start,
    live_end,
    freq="D"
)

actual_dates = pd.DatetimeIndex(
    live["date"].drop_duplicates()
).sort_values()

missing_dates = expected_dates.difference(
    actual_dates
)

if len(missing_dates) > 0:
    print()
    print("WARNING: Missing live FIRMS dates detected:")

    for missing_date in missing_dates:
        print(" ", missing_date.date())

    raise SystemExit(1)

if len(actual_dates) < 8:
    raise SystemExit(
        "Not enough continuous 2026 FIRMS history. "
        "At least 8 consecutive days are required "
        "to calculate the full 7-day feature history."
    )


# ------------------------------------------------------------
# 6. Build ONLY continuous panels inside each
#    actual observation period
# ------------------------------------------------------------

districts = (
    combined[
        ["state", "district"]
    ]
    .drop_duplicates()
)


# Historical seasonal data already contains complete
# district/date panels for Oct-Nov 2023-2025.
#
# For 2026, only create dates where live observations
# have actually been collected.

live_dates = pd.DataFrame({
    "date": pd.date_range(
        live_start,
        live_end,
        freq="D"
    )
})
expected_districts = (
    districts[
        ["state", "district"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

expected_district_count = len(expected_districts)

if expected_district_count != 45:
    raise ValueError(
        f"Expected 45 Punjab/Haryana districts, "
        f"but found {expected_district_count}."
    )

live_panel = (
    districts.assign(key=1)
    .merge(
        live_dates.assign(key=1),
        on="key"
    )
    .drop(columns="key")
)


live_panel = live_panel.merge(
    live,
    on=["date", "state", "district"],
    how="left"
)

live_panel["fire_count"] = (
    live_panel["fire_count"]
    .fillna(0)
)
expected_live_rows = (
    expected_district_count
    * len(live_dates)
)

if len(live_panel) != expected_live_rows:
    raise ValueError(
        "Live district-date panel is incomplete. "
        f"Expected {expected_live_rows} rows, "
        f"found {len(live_panel)}."
    )

if live_panel[
    ["state", "district", "date"]
].duplicated().any():
    raise ValueError(
        "Duplicate rows found in the live district-date panel."
    )

# ------------------------------------------------------------
# 7. Historical data
# ------------------------------------------------------------

historical_panel = historical.copy()


# ------------------------------------------------------------
# 8. Combine panels
# ------------------------------------------------------------

panel = pd.concat(
    [
        historical_panel,
        live_panel
    ],
    ignore_index=True
)

duplicate_combined = combined.duplicated(
    subset=[
        "date",
        "state",
        "district",
    ],
    keep=False,
)

if duplicate_combined.any():
    print()
    print(
        "Duplicate district-date fire observations detected."
    )

    print(
        combined.loc[
            duplicate_combined,
            [
                "date",
                "state",
                "district",
                "fire_count",
            ],
        ]
        .head(20)
        .to_string(index=False)
    )

    raise SystemExit(1)
panel = (
    panel
    .sort_values(
        ["state", "district", "date"]
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 9. IMPORTANT:
#    Do NOT calculate rolling features across the
#    2025 -> 2026 missing period.
#
#    Calculate features separately for each contiguous
#    observation period.
# ------------------------------------------------------------

feature_parts = []


# Historical seasons
for year in [2023, 2024, 2025]:

    part = panel[
        panel["date"].dt.year == year
    ].copy()

    part = part.sort_values(
        ["state", "district", "date"]
    )

    part["fire_lag_1d"] = (
        part
        .groupby(["state", "district"])["fire_count"]
        .shift(1)
    )

    part["fire_lag_3d"] = (
        part
        .groupby(["state", "district"])["fire_count"]
        .shift(3)
    )

    part["fire_lag_7d"] = (
        part
        .groupby(["state", "district"])["fire_count"]
        .shift(7)
    )

    part["fire_mean_3d"] = (
        part
        .groupby(["state", "district"])["fire_count"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(3)
             .mean()
        )
    )

    part["fire_mean_7d"] = (
        part
        .groupby(["state", "district"])["fire_count"]
        .transform(
            lambda x:
            x.shift(1)
             .rolling(7)
             .mean()
        )
    )

    feature_parts.append(part)


# 2026 live period
part = live_panel.copy()

part = part.sort_values(
    ["state", "district", "date"]
)

part["fire_lag_1d"] = (
    part
    .groupby(["state", "district"])["fire_count"]
    .shift(1)
)

part["fire_lag_3d"] = (
    part
    .groupby(["state", "district"])["fire_count"]
    .shift(3)
)

part["fire_lag_7d"] = (
    part
    .groupby(["state", "district"])["fire_count"]
    .shift(7)
)

part["fire_mean_3d"] = (
    part
    .groupby(["state", "district"])["fire_count"]
    .transform(
        lambda x:
        x.shift(1)
         .rolling(3)
         .mean()
    )
)

part["fire_mean_7d"] = (
    part
    .groupby(["state", "district"])["fire_count"]
    .transform(
        lambda x:
        x.shift(1)
         .rolling(7)
         .mean()
    )
)

feature_parts.append(part)


# ------------------------------------------------------------
# 10. Combine feature datasets
# ------------------------------------------------------------

features = pd.concat(
    feature_parts,
    ignore_index=True
)


features = features.sort_values(
    ["date", "state", "district"]
).reset_index(drop=True)
required_feature_columns = [
    "fire_lag_1d",
    "fire_lag_3d",
    "fire_lag_7d",
    "fire_mean_3d",
    "fire_mean_7d",
]

latest_live_date = features[
    features["date"].dt.year == 2026
]["date"].max()

latest_live_features = features[
    features["date"] == latest_live_date
].copy()

missing_latest = latest_live_features[
    required_feature_columns
].isna().any(axis=1)

if latest_live_features.empty:
    raise ValueError(
        "No features were generated for the latest "
        "2026 live date."
    )

if missing_latest.any():
    print()
    print(
        "ERROR: Some districts are missing required "
        "features on the latest live date."
    )

    print(
        latest_live_features.loc[
            missing_latest,
            [
                "state",
                "district",
                *required_feature_columns,
            ],
        ].to_string(index=False)
    )

    raise SystemExit(1)

# ------------------------------------------------------------
# 11. Save
# ------------------------------------------------------------



if features[required_feature_columns].isna().any().any():
    raise ValueError(
        "Generated fire features contain missing values."
    )
features.to_csv(
    OUTPUT_FILE,
    index=False
)
# ------------------------------------------------------------
# 12. Summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("LIVE 2026 FIRE FEATURES CREATED")
print("=" * 70)

print("Total rows:", len(features))

print(
    "Overall date range:",
    features["date"].min().date(),
    "to",
    features["date"].max().date()
)

live_features = features[
    features["date"].dt.year == 2026
].copy()

print()
print("2026 rows:", len(live_features))

print(
    "2026 districts:",
    live_features["district"].nunique()
)

print()
print("2026 feature availability:")

print(
    live_features[
        [
            "fire_lag_1d",
            "fire_lag_3d",
            "fire_lag_7d",
            "fire_mean_3d",
            "fire_mean_7d"
        ]
    ]
    .notna()
    .sum()
)

print()
print("Latest 2026 observations:")

print(
    live_features[
        [
            "date",
            "state",
            "district",
            "fire_count",
            "fire_lag_1d",
            "fire_lag_3d",
            "fire_lag_7d",
            "fire_mean_3d",
            "fire_mean_7d"
        ]
    ]
    .sort_values(
        ["date", "state", "district"]
    )
    .tail(20)
    .to_string(index=False)
)

print()
print("Saved to:")
print(OUTPUT_FILE)

print("=" * 70)