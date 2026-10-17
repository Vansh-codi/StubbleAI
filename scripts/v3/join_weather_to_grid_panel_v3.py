from pathlib import Path
import os

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "v3"

GRID_FILE = DATA_DIR / "grid_day_panel_005_centroid_fallback.csv"
MAPPING_FILE = DATA_DIR / "grid_cell_district_mapping_005.csv"
WEATHER_FILE = DATA_DIR / "district_daily_weather_v3.csv"

OUTPUT_FILE = DATA_DIR / "grid_day_panel_005_centroid_fallback_weather.csv"
TEMP_FILE = DATA_DIR / "grid_day_panel_005_centroid_fallback_weather.tmp.csv"
AUDIT_FILE = DATA_DIR / "grid_day_panel_005_centroid_fallback_weather_audit.txt"

CHUNK_SIZE = 200_000
WEATHER_COLS = ["T2M", "RH2M", "WS2M", "PRECTOTCORR"]
MAP_COLS = [
    "cell_id", "district", "district_key", "assignment_method"
]
KEYS = ["state", "district", "date"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    for path in [GRID_FILE, MAPPING_FILE, WEATHER_FILE]:
        require(path.exists(), f"Required input file not found: {path}")

    # Never overwrite an existing result or stale temporary output silently.
    require(not OUTPUT_FILE.exists(), f"Output already exists: {OUTPUT_FILE}")
    require(not TEMP_FILE.exists(), f"Temporary output already exists: {TEMP_FILE}")

    print("Loading and validating cell-to-district mapping...")
    mapping = pd.read_csv(MAPPING_FILE, dtype={"cell_id": str})
    require(not mapping["cell_id"].duplicated().any(),
            "Mapping contains duplicate cell IDs.")
    require(mapping["district"].notna().all(),
            "Mapping contains missing districts.")

    mapping = mapping[MAP_COLS]
    mapping["cell_id"] = mapping["cell_id"].astype(str)

    print("Loading and validating district weather...")
    weather = pd.read_csv(
        WEATHER_FILE,
        dtype={"state": str, "district": str},
    )
    weather["date"] = pd.to_datetime(
        weather["date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    weather_keys = ["state", "district", "date"]
    require(not weather.duplicated(weather_keys).any(),
            "Weather contains duplicate state-district-date keys.")
    require(weather[WEATHER_COLS].notna().all().all(),
            "Weather contains missing values.")

    weather = weather[weather_keys + WEATHER_COLS].copy()
    weather["state"] = weather["state"].astype(str)
    weather["district"] = weather["district"].astype(str)

    expected_rows = 0
    expected_fire_total = 0
    written_rows = 0
    matched_rows = 0
    unmatched_rows = 0
    output_fire_total = 0
    output_columns = None

    print("Joining grid-day records in chunks...")
    try:
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                GRID_FILE,
                chunksize=CHUNK_SIZE,
                dtype={"cell_id": str, "date": str},
            ),
            start=1,
        ):
            chunk_rows = len(chunk)
            expected_rows += chunk_rows
            expected_fire_total += pd.to_numeric(
                chunk["fire_count"], errors="raise"
            ).sum()

            # Attach exactly one district assignment to each grid cell.
            chunk = chunk.merge(
                mapping,
                on="cell_id",
                how="left",
                validate="many_to_one",
            )

            require(chunk["district"].notna().all(),
                    f"Unmapped cells in chunk {chunk_number}.")

            # Use the grid panel's state and date, plus mapped district.
            chunk["date"] = pd.to_datetime(
                chunk["date"], errors="raise"
            ).dt.strftime("%Y-%m-%d")

            chunk = chunk.merge(
                weather,
                on=KEYS,
                how="left",
                validate="many_to_one",
                indicator="_weather_match",
            )

            matched = int((chunk["_weather_match"] == "both").sum())
            unmatched = int((chunk["_weather_match"] == "left_only").sum())
            matched_rows += matched
            unmatched_rows += unmatched

            require(unmatched == 0,
                    f"{unmatched} weather join failures in chunk {chunk_number}.")

            chunk = chunk.drop(columns="_weather_match")
            output_fire_total += pd.to_numeric(
                chunk["fire_count"], errors="raise"
            ).sum()

            if output_columns is None:
                output_columns = chunk.columns.tolist()

            chunk.to_csv(
                TEMP_FILE,
                mode="w" if chunk_number == 1 else "a",
                header=(chunk_number == 1),
                index=False,
            )

            written_rows += len(chunk)
            print(
                f"  Chunk {chunk_number}: {len(chunk):,} rows; "
                f"total written {written_rows:,}"
            )

        require(written_rows == expected_rows,
                f"Row count changed: read {expected_rows}, wrote {written_rows}.")
        require(matched_rows == expected_rows,
                f"Weather coverage mismatch: {matched_rows}/{expected_rows}.")
        require(unmatched_rows == 0,
                f"Unmatched weather records: {unmatched_rows}.")
        require(abs(output_fire_total - expected_fire_total) < 1e-8,
                "Fire-count total changed during join.")

        # Publish the final file only after all checks pass.
        os.replace(TEMP_FILE, OUTPUT_FILE)

        audit = [
            "V3 weather-enriched grid-day panel audit",
            f"Input grid file: {GRID_FILE.name}",
            f"Mapping file: {MAPPING_FILE.name}",
            f"Weather file: {WEATHER_FILE.name}",
            f"Rows read: {expected_rows}",
            f"Rows written: {written_rows}",
            f"Weather matches: {matched_rows}",
            f"Unmatched weather rows: {unmatched_rows}",
            f"Input fire_count total: {expected_fire_total}",
            f"Output fire_count total: {output_fire_total}",
            f"Columns: {output_columns}",
            "Result: PASS",
        ]
        AUDIT_FILE.write_text("\n".join(audit) + "\n", encoding="utf-8")

        print("\nWEATHER JOIN VALIDATION PASSED")
        print("Rows:", written_rows)
        print("Weather matches:", matched_rows)
        print("Unmatched weather rows:", unmatched_rows)
        print("Fire-count total preserved:", output_fire_total)
        print("Output:", OUTPUT_FILE)
        print("Audit:", AUDIT_FILE)

    except Exception:
        print("\nJoin failed. The final output has not been published.")
        print(f"Inspect or remove the temporary file only after reviewing: {TEMP_FILE}")
        raise


if __name__ == "__main__":
    main()
