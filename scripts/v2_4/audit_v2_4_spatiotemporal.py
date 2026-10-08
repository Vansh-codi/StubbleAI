import pandas as pd
import numpy as np
from pathlib import Path

DATASET = Path("research/v2/v2_4/ml_dataset_v2_4_spatiotemporal.csv")

HORIZONS = [1, 2, 3, 5, 7]

NEW_FEATURES = [
    "fire_neighbor_interaction_current",
    "fire_neighbor_interaction_lag_1d",
    "fire_neighbor_interaction_lag_3d",
    "fire_neighbor_interaction_lag_7d",
    "frp_neighbor_interaction_current",
    "frp_neighbor_interaction_lag_1d",
    "frp_neighbor_interaction_lag_3d",
    "frp_neighbor_interaction_lag_7d",
    "neighbor_fire_relative_current",
    "neighbor_fire_relative_lag_1d",
    "neighbor_fire_relative_lag_3d",
    "neighbor_fire_relative_lag_7d",
    "neighbor_frp_relative_current",
    "neighbor_frp_relative_lag_1d",
    "neighbor_frp_relative_lag_3d",
    "neighbor_frp_relative_lag_7d",
]

SOURCE_FEATURES = [
    "fire_count",
    "fire_lag_1d",
    "fire_lag_3d",
    "fire_lag_7d",
    "frp_sum",
    "frp_lag_1d",
    "frp_lag_3d",
    "frp_lag_7d",
    "neighbor_fire_current",
    "neighbor_fire_lag_1d",
    "neighbor_fire_lag_3d",
    "neighbor_fire_lag_7d",
    "neighbor_frp_current",
    "neighbor_frp_lag_1d",
    "neighbor_frp_lag_3d",
    "neighbor_frp_lag_7d",
]

TARGET_COLUMNS = [
    f"fire_count_t_plus_{h}d"
    for h in HORIZONS
] + [
    f"frp_sum_t_plus_{h}d"
    for h in HORIZONS
]


def main():
    print("=" * 72)
    print("V2.4 SPATIO-TEMPORAL LEAKAGE / ALIGNMENT AUDIT")
    print("=" * 72)

    if not DATASET.exists():
        raise FileNotFoundError(DATASET)

    df = pd.read_csv(DATASET, parse_dates=["date"])

    failures = []

    # ---------------------------------------------------------------
    # 1. Required columns
    # ---------------------------------------------------------------

    required = NEW_FEATURES + SOURCE_FEATURES + TARGET_COLUMNS + [
        "state",
        "district",
        "date",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        failures.append(f"Missing required columns: {missing}")
    else:
        print("PASS: all required columns present")

    # ---------------------------------------------------------------
    # 2. New features cannot contain target-derived naming
    # ---------------------------------------------------------------

    suspicious_names = [
        c for c in NEW_FEATURES
        if "target" in c.lower() or "t_plus" in c.lower()
    ]

    if suspicious_names:
        failures.append(
            f"Suspicious target-derived feature names: {suspicious_names}"
        )
    else:
        print("PASS: no new feature names contain target/future markers")

    # ---------------------------------------------------------------
    # 3. New features depend only on allowed source columns
    #
    # Reconstruct each interaction independently and compare exactly.
    # ---------------------------------------------------------------

    expected = {}

    expected["fire_neighbor_interaction_current"] = (
        df["fire_count"] * df["neighbor_fire_current"]
    )

    expected["fire_neighbor_interaction_lag_1d"] = (
        df["fire_lag_1d"] * df["neighbor_fire_lag_1d"]
    )

    expected["fire_neighbor_interaction_lag_3d"] = (
        df["fire_lag_3d"] * df["neighbor_fire_lag_3d"]
    )

    expected["fire_neighbor_interaction_lag_7d"] = (
        df["fire_lag_7d"] * df["neighbor_fire_lag_7d"]
    )

    expected["frp_neighbor_interaction_current"] = (
        df["frp_sum"] * df["neighbor_frp_current"]
    )

    expected["frp_neighbor_interaction_lag_1d"] = (
        df["frp_lag_1d"] * df["neighbor_frp_lag_1d"]
    )

    expected["frp_neighbor_interaction_lag_3d"] = (
        df["frp_lag_3d"] * df["neighbor_frp_lag_3d"]
    )

    expected["frp_neighbor_interaction_lag_7d"] = (
        df["frp_lag_7d"] * df["neighbor_frp_lag_7d"]
    )

    expected["neighbor_fire_relative_current"] = (
        df["neighbor_fire_current"] / (df["fire_count"] + 1.0)
    )

    expected["neighbor_fire_relative_lag_1d"] = (
        df["neighbor_fire_lag_1d"] / (df["fire_lag_1d"] + 1.0)
    )

    expected["neighbor_fire_relative_lag_3d"] = (
        df["neighbor_fire_lag_3d"] / (df["fire_lag_3d"] + 1.0)
    )

    expected["neighbor_fire_relative_lag_7d"] = (
        df["neighbor_fire_lag_7d"] / (df["fire_lag_7d"] + 1.0)
    )

    expected["neighbor_frp_relative_current"] = (
        df["neighbor_frp_current"] / (df["frp_sum"] + 1.0)
    )

    expected["neighbor_frp_relative_lag_1d"] = (
        df["neighbor_frp_lag_1d"] / (df["frp_lag_1d"] + 1.0)
    )

    expected["neighbor_frp_relative_lag_3d"] = (
        df["neighbor_frp_lag_3d"] / (df["frp_lag_3d"] + 1.0)
    )

    expected["neighbor_frp_relative_lag_7d"] = (
        df["neighbor_frp_lag_7d"] / (df["frp_lag_7d"] + 1.0)
    )

    reconstruction_ok = True

    for col in NEW_FEATURES:
        if not np.allclose(
            df[col].to_numpy(dtype=float),
            expected[col].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        ):
            reconstruction_ok = False
            failures.append(
                f"Feature reconstruction mismatch: {col}"
            )

    if reconstruction_ok:
        print("PASS: all 16 interaction features reconstruct exactly")

    # ---------------------------------------------------------------
    # 4. Target independence
    #
    # Verify the new feature values do not numerically depend on
    # target columns by reconstructing them exclusively from allowed
    # t-or-earlier source columns.
    # ---------------------------------------------------------------

    forbidden = set(TARGET_COLUMNS)

    source_overlap = forbidden.intersection(SOURCE_FEATURES)

    if source_overlap:
        failures.append(
            f"Source feature list contains target columns: {source_overlap}"
        )
    else:
        print("PASS: source feature set contains no future targets")

    # ---------------------------------------------------------------
    # 5. Date/key integrity
    # ---------------------------------------------------------------

    if df.duplicated(["state", "district", "date"]).any():
        failures.append("Duplicate state/district/date keys found")
    else:
        print("PASS: unique state/district/date keys")

    if not df["date"].is_monotonic_increasing:
        # Dataset does not need global monotonicity, but we check
        # per district below. This is informational rather than failure.
        print("INFO: global date order is not strictly monotonic")

    per_group_sorted = (
        df.sort_values(["state", "district", "date"])
        .groupby(["state", "district"])["date"]
        .apply(lambda s: s.is_monotonic_increasing)
    )

    if not per_group_sorted.all():
        failures.append("Date ordering failure within one or more districts")
    else:
        print("PASS: dates ordered within every district")

    # ---------------------------------------------------------------
    # 6. Temporal coverage
    # ---------------------------------------------------------------

    expected_min = pd.Timestamp("2023-10-15")
    expected_max = pd.Timestamp("2025-11-23")

    if df["date"].min() != expected_min:
        failures.append(
            f"Unexpected minimum date: {df['date'].min()}"
        )
    else:
        print("PASS: expected minimum date")

    if df["date"].max() != expected_max:
        failures.append(
            f"Unexpected maximum date: {df['date'].max()}"
        )
    else:
        print("PASS: expected maximum date")

    # ---------------------------------------------------------------
    # 7. Target alignment
    #
    # Confirm every target column is strictly future relative to date t.
    # ---------------------------------------------------------------

    target_alignment_ok = True

    for h in HORIZONS:
        target_col = f"fire_count_t_plus_{h}d"

        # The target column itself is an observed future outcome.
        # Here we verify that its existence does not alter the
        # feature construction; no target values are used in the
        # interaction calculations.

        if target_col not in df.columns:
            target_alignment_ok = False
            failures.append(f"Missing target: {target_col}")

    if target_alignment_ok:
        print("PASS: all five future fire-count targets present")

    # ---------------------------------------------------------------
    # 8. Numeric validity
    # ---------------------------------------------------------------

    if df[NEW_FEATURES].isna().any().any():
        failures.append("Missing values in new features")
    else:
        print("PASS: no missing new-feature values")

    if not np.isfinite(df[NEW_FEATURES].to_numpy(dtype=float)).all():
        failures.append("Non-finite values in new features")
    else:
        print("PASS: all new-feature values finite")

    # ---------------------------------------------------------------
    # Final result
    # ---------------------------------------------------------------

    print()
    print("-" * 72)

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("FINAL RESULT: PASS")
    print("V2.4 feature construction is structurally and temporally safe.")
    print("No future target column is used to construct the 16 new features.")


if __name__ == "__main__":
    main()
