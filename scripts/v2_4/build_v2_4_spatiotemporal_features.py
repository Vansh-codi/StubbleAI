import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("research/v2/v2_3/spatial/ml_dataset_v2_3_spatial.csv")
OUTPUT_DIR = Path("research/v2/v2_4")
OUTPUT_PATH = OUTPUT_DIR / "ml_dataset_v2_4_spatiotemporal.csv"
CONFIG_PATH = OUTPUT_DIR / "v2_4_spatiotemporal_config.json"

TARGET_HORIZONS = [1, 2, 3, 5, 7]

LOCAL_FIRE_FEATURES = [
    "fire_count",
    "fire_lag_1d",
    "fire_lag_3d",
    "fire_lag_7d",
]

NEIGHBOR_FIRE_FEATURES = [
    "neighbor_fire_current",
    "neighbor_fire_lag_1d",
    "neighbor_fire_lag_3d",
    "neighbor_fire_lag_7d",
]

LOCAL_FRP_FEATURES = [
    "frp_sum",
    "frp_lag_1d",
    "frp_lag_3d",
    "frp_lag_7d",
]

NEIGHBOR_FRP_FEATURES = [
    "neighbor_frp_current",
    "neighbor_frp_lag_1d",
    "neighbor_frp_lag_3d",
    "neighbor_frp_lag_7d",
]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    print("=" * 70)
    print("V2.4 SPATIO-TEMPORAL FEATURE BUILDER")
    print("=" * 70)
    print(f"Input: {INPUT_PATH}")
    print(f"Input shape: {df.shape}")

    required = (
        [
            "state",
            "district",
            "date",
            "fire_count",
            "frp_sum",
        ]
        + LOCAL_FIRE_FEATURES
        + NEIGHBOR_FIRE_FEATURES
        + LOCAL_FRP_FEATURES
        + NEIGHBOR_FRP_FEATURES
        + [
            f"fire_count_t_plus_{h}d"
            for h in TARGET_HORIZONS
        ]
        + [
            f"frp_sum_t_plus_{h}d"
            for h in TARGET_HORIZONS
        ]
    )

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    original_columns = df.columns.tolist()

    # ---------------------------------------------------------------
    # 1. Fire activity interaction features
    # ---------------------------------------------------------------

    fire_pairs = list(zip(LOCAL_FIRE_FEATURES, NEIGHBOR_FIRE_FEATURES))

    for local_col, neighbor_col in fire_pairs:
        suffix = local_col.replace("fire_", "").replace("_", "")
        neighbor_suffix = neighbor_col.replace("neighbor_fire_", "")
        name = f"fire_neighbor_interaction_{neighbor_suffix}"

        df[name] = df[local_col] * df[neighbor_col]

    # ---------------------------------------------------------------
    # 2. FRP interaction features
    # ---------------------------------------------------------------

    frp_pairs = list(zip(LOCAL_FRP_FEATURES, NEIGHBOR_FRP_FEATURES))

    for local_col, neighbor_col in frp_pairs:
        neighbor_suffix = neighbor_col.replace("neighbor_frp_", "")
        name = f"frp_neighbor_interaction_{neighbor_suffix}"

        df[name] = df[local_col] * df[neighbor_col]

    # ---------------------------------------------------------------
    # 3. Relative spatial activity
    #
    # +1 prevents division by zero.
    # These features describe surrounding activity relative to
    # local activity and use information available at time t or
    # earlier only.
    # ---------------------------------------------------------------

    df["neighbor_fire_relative_current"] = (
        df["neighbor_fire_current"] / (df["fire_count"] + 1.0)
    )

    df["neighbor_fire_relative_lag_1d"] = (
        df["neighbor_fire_lag_1d"] / (df["fire_lag_1d"] + 1.0)
    )

    df["neighbor_fire_relative_lag_3d"] = (
        df["neighbor_fire_lag_3d"] / (df["fire_lag_3d"] + 1.0)
    )

    df["neighbor_fire_relative_lag_7d"] = (
        df["neighbor_fire_lag_7d"] / (df["fire_lag_7d"] + 1.0)
    )

    df["neighbor_frp_relative_current"] = (
        df["neighbor_frp_current"] / (df["frp_sum"] + 1.0)
    )

    df["neighbor_frp_relative_lag_1d"] = (
        df["neighbor_frp_lag_1d"] / (df["frp_lag_1d"] + 1.0)
    )

    df["neighbor_frp_relative_lag_3d"] = (
        df["neighbor_frp_lag_3d"] / (df["frp_lag_3d"] + 1.0)
    )

    df["neighbor_frp_relative_lag_7d"] = (
        df["neighbor_frp_lag_7d"] / (df["frp_lag_7d"] + 1.0)
    )

    # ---------------------------------------------------------------
    # 4. Structural checks
    # ---------------------------------------------------------------

    interaction_columns = [
        c for c in df.columns
        if c.startswith("fire_neighbor_interaction_")
        or c.startswith("frp_neighbor_interaction_")
        or c.startswith("neighbor_fire_relative_")
        or c.startswith("neighbor_frp_relative_")
    ]

    if len(interaction_columns) != 16:
        raise AssertionError(
            f"Expected 16 new interaction features, found {len(interaction_columns)}"
        )

    if df[interaction_columns].isna().any().any():
        missing_counts = df[interaction_columns].isna().sum()
        raise AssertionError(
            f"Missing values in interaction features:\n{missing_counts[missing_counts > 0]}"
        )

    if not np.isfinite(df[interaction_columns].to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite values found in interaction features.")

    if df.duplicated(["state", "district", "date"]).any():
        raise AssertionError("Duplicate state/district/date keys detected.")

    if df["date"].min() != pd.Timestamp("2023-10-15"):
        raise AssertionError(f"Unexpected minimum date: {df['date'].min()}")

    if df["date"].max() != pd.Timestamp("2025-11-23"):
        raise AssertionError(f"Unexpected maximum date: {df['date'].max()}")

    if df["district"].nunique() != 45:
        raise AssertionError(
            f"Expected 45 districts, found {df['district'].nunique()}"
        )

    if df["state"].nunique() != 2:
        raise AssertionError(
            f"Expected 2 states, found {df['state'].nunique()}"
        )

    # ---------------------------------------------------------------
    # 5. Leakage audit
    #
    # New features may depend only on local/neighbor information
    # already available at date t or earlier.
    # They must not depend on any target column.
    # ---------------------------------------------------------------

    target_columns = [
        f"fire_count_t_plus_{h}d"
        for h in TARGET_HORIZONS
    ] + [
        f"frp_sum_t_plus_{h}d"
        for h in TARGET_HORIZONS
    ]

    feature_source_columns = set(
        LOCAL_FIRE_FEATURES
        + NEIGHBOR_FIRE_FEATURES
        + LOCAL_FRP_FEATURES
        + NEIGHBOR_FRP_FEATURES
    )

    if feature_source_columns.intersection(target_columns):
        raise AssertionError(
            "Target columns unexpectedly included in source feature columns."
        )

    # Explicitly ensure no interaction feature contains target naming.
    suspicious = [
        c for c in interaction_columns
        if "t_plus" in c.lower()
        or "target" in c.lower()
    ]

    if suspicious:
        raise AssertionError(
            f"Potential target-derived interaction feature names: {suspicious}"
        )

    # ---------------------------------------------------------------
    # 6. Verify original V2.3 columns remain unchanged
    # ---------------------------------------------------------------

    for col in original_columns:
        if col not in df.columns:
            raise AssertionError(f"Original column disappeared: {col}")

    # ---------------------------------------------------------------
    # 7. Save
    # ---------------------------------------------------------------

    df.to_csv(OUTPUT_PATH, index=False)

    config = {
        "experiment": "V2.4 spatio-temporal interaction experiment",
        "input": str(INPUT_PATH),
        "output": str(OUTPUT_PATH),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "new_features": interaction_columns,
        "target_horizons_days": TARGET_HORIZONS,
        "source_information_cutoff": "date t or earlier",
        "frozen_v2_dataset_modified": False,
        "frozen_v2_3_spatial_dataset_modified": False,
        "leakage_policy": (
            "Interactions use only local and neighboring fire/FRP "
            "features at t or earlier; future target columns are never used."
        ),
    }

    CONFIG_PATH.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print()
    print("PASS: feature construction completed")
    print(f"Output shape: {df.shape}")
    print(f"New interaction features: {len(interaction_columns)}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Config: {CONFIG_PATH}")
    print()
    print("New features:")
    for col in interaction_columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()
