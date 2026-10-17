from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "v3"
OUT_DIR = DATA_DIR / "experiments" / "grid_sensitivity_010"

GRID_FILE = OUT_DIR / "grid_day_panel_010_centroid_fallback.csv"
MAPPING_FILE = OUT_DIR / "grid_cell_district_mapping_010.csv"
WEATHER_FILE = DATA_DIR / "district_daily_weather_v3.csv"

OUTPUT_FILE = OUT_DIR / "grid_day_panel_010_centroid_fallback_weather.csv"
TEMP_FILE = OUT_DIR / "grid_day_panel_010_centroid_fallback_weather.tmp.csv"
AUDIT_FILE = OUT_DIR / "grid_day_panel_010_centroid_fallback_weather_audit.txt"

CHUNK_SIZE = 200_000
WEATHER_COLS = ["T2M", "RH2M", "WS2M", "PRECTOTCORR"]
MAP_COLS = ["cell_id", "district", "district_key", "assignment_method"]
KEYS = ["state", "district", "date"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def preflight():
    for path in [GRID_FILE, MAPPING_FILE, WEATHER_FILE]:
        require(path.is_file(), f"Required input file not found: {path}")
    for path in [OUTPUT_FILE, TEMP_FILE, AUDIT_FILE]:
        require(not path.exists(), f"Refusing to overwrite existing file: {path}")
    print("PREFLIGHT PASS")
    print("Grid:", GRID_FILE)
    print("Mapping:", MAPPING_FILE)
    print("Weather:", WEATHER_FILE)
    print("Output:", OUTPUT_FILE)
    print("No data loaded and no outputs written.")


def main():
    preflight()

    print("\nLoading and validating cell-to-district mapping...")
    mapping = pd.read_csv(MAPPING_FILE, dtype={"cell_id": str})
    require(not mapping["cell_id"].duplicated().any(),
            "Mapping contains duplicate cell IDs.")
    require(mapping["district"].notna().all(),
            "Mapping contains missing districts.")
    require(mapping["district_key"].notna().all(),
            "Mapping contains missing district keys.")
    mapping = mapping[MAP_COLS].copy()
    mapping["cell_id"] = mapping["cell_id"].astype(str)

    print("Loading and validating district weather...")
    weather = pd.read_csv(WEATHER_FILE, dtype={"state": str, "district": str})
    required_weather = ["state", "district", "date"] + WEATHER_COLS
    missing_cols = sorted(set(required_weather) - set(weather.columns))
    require(not missing_cols, f"Weather columns missing: {missing_cols}")

    weather["date"] = pd.to_datetime(
        weather["date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    weather_keys = ["state", "district", "date"]
    require(not weather.duplicated(weather_keys).any(),
            "Weather contains duplicate state-district-date keys.")
    require(weather[WEATHER_COLS].notna().all().all(),
            "Weather contains missing values.")
    weather = weather[weather_keys + WEATHER_COLS].copy()

    expected_rows = 0
    written_rows = 0
    matched_rows = 0
    unmatched_rows = 0
    expected_fire_total = 0.0
    output_fire_total = 0.0

    print("\nJoining grid-day records in chunks...")
    try:
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                GRID_FILE,
                chunksize=CHUNK_SIZE,
                dtype={"cell_id": str, "date": str},
            ),
            start=1,
        ):
            expected_rows += len(chunk)
            expected_fire_total += pd.to_numeric(
                chunk["fire_count"], errors="raise"
            ).sum()

            chunk = chunk.merge(
                mapping, on="cell_id", how="left", validate="many_to_one"
            )
            require(chunk["district"].notna().all(),
                    f"Unmapped cells in chunk {chunk_number}.")

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

            chunk.to_csv(
                TEMP_FILE,
                mode="w" if chunk_number == 1 else "a",
                header=(chunk_number == 1),
                index=False,
            )
            written_rows += len(chunk)
            print(f"  Chunk {chunk_number}: {len(chunk):,} rows; "
                  f"total written {written_rows:,}")

        require(written_rows == expected_rows,
                f"Row count changed: read {expected_rows}, wrote {written_rows}.")
        require(matched_rows == expected_rows,
                f"Weather coverage mismatch: {matched_rows}/{expected_rows}.")
        require(unmatched_rows == 0,
                f"Unmatched weather records: {unmatched_rows}.")
        require(abs(output_fire_total - expected_fire_total) < 1e-8,
                "Fire-count total changed during join.")

        # Publish only after all data checks pass.
        require(not OUTPUT_FILE.exists(),
                f"Refusing to overwrite existing output: {OUTPUT_FILE}")
        os.replace(TEMP_FILE, OUTPUT_FILE)

        audit = [
            "V3 0.10-degree weather-enriched grid-day panel audit",
            f"Input grid file: {GRID_FILE.name}",
            f"Mapping file: {MAPPING_FILE.name}",
            f"Weather file: {WEATHER_FILE.name}",
            f"Rows read: {expected_rows}",
            f"Rows written: {written_rows}",
            f"Weather matches: {matched_rows}",
            f"Unmatched weather rows: {unmatched_rows}",
            f"Input fire_count total: {expected_fire_total}",
            f"Output fire_count total: {output_fire_total}",
            "Weather granularity: district-day; repeated across cells in a district",
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
        print("\nJoin failed. The final output may not have been published.")
        print("Review the temporary file before deciding whether it is safe to remove:",
              TEMP_FILE)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        main()
