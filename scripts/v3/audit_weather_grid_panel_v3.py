
from pathlib import Path
import json
import sqlite3

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "v3"
    / "grid_day_panel_005_centroid_fallback_weather.csv"
)
AUDIT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "v3"
    / "grid_day_panel_005_centroid_fallback_weather_qa.json"
)
SQLITE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "v3"
    / "weather_panel_key_audit.sqlite"
)

CHUNK_SIZE = 200_000

WEATHER_COLUMNS = ["T2M", "RH2M", "WS2M", "PRECTOTCORR"]
KEY_COLUMNS = ["cell_id", "date"]
REQUIRED_COLUMNS = [
    "state",
    "cell_id",
    "date",
    "fire_count",
    "district",
    "assignment_method",
    *WEATHER_COLUMNS,
]

EXPECTED_ROWS = 5_793_414
EXPECTED_FIRE_TOTAL = 256_903


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    # Initialize QA accumulators.
    rows = 0
    fire_total = 0
    missing_required = {
        column: 0 for column in REQUIRED_COLUMNS
    }
    weather_min = {column: None for column in WEATHER_COLUMNS}
    weather_max = {column: None for column in WEATHER_COLUMNS}
    weather_missing = {column: 0 for column in WEATHER_COLUMNS}

    negative_fire_rows = 0
    zero_fire_rows = 0
    date_min = None
    date_max = None
    duplicate_key_rows = 0

    # A fresh SQLite database provides exact key comparison using a
    # composite primary key. It is temporary and separate from the panel.
    if SQLITE_FILE.exists():
        raise FileExistsError(
            f"Temporary audit database already exists: {SQLITE_FILE}. "
            "Inspect it before removing it."
        )

    db = None

    try:
        db = sqlite3.connect(SQLITE_FILE)
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("""
            CREATE TABLE grid_keys (
                cell_id TEXT NOT NULL,
                date TEXT NOT NULL,
                PRIMARY KEY (cell_id, date)
            ) WITHOUT ROWID
        """)

        reader = pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            usecols=lambda column: column in REQUIRED_COLUMNS,
            parse_dates=["date"],
        )

        for chunk_number, chunk in enumerate(reader, start=1):
            rows += len(chunk)

            for column in REQUIRED_COLUMNS:
                if column not in chunk.columns:
                    raise ValueError(
                        f"Missing required column: {column}"
                    )
                missing_required[column] += int(
                    chunk[column].isna().sum()
                )

            fire_total += int(chunk["fire_count"].sum())
            negative_fire_rows += int(
                (chunk["fire_count"] < 0).sum()
            )
            zero_fire_rows += int(
                (chunk["fire_count"] == 0).sum()
            )

            chunk_date_min = chunk["date"].min()
            chunk_date_max = chunk["date"].max()

            date_min = (
                chunk_date_min
                if date_min is None
                else min(date_min, chunk_date_min)
            )
            date_max = (
                chunk_date_max
                if date_max is None
                else max(date_max, chunk_date_max)
            )

            for column in WEATHER_COLUMNS:
                values = pd.to_numeric(
                    chunk[column], errors="coerce"
                )
                weather_missing[column] += int(values.isna().sum())

                current_min = values.min()
                current_max = values.max()

                if pd.notna(current_min):
                    weather_min[column] = (
                        float(current_min)
                        if weather_min[column] is None
                        else min(
                            weather_min[column],
                            float(current_min),
                        )
                    )

                if pd.notna(current_max):
                    weather_max[column] = (
                        float(current_max)
                        if weather_max[column] is None
                        else max(
                            weather_max[column],
                            float(current_max),
                        )
                    )

            # Reject null keys explicitly rather than silently converting
            # them into text.
            if chunk[KEY_COLUMNS].isna().any().any():
                raise ValueError(
                    f"Null grid key encountered in chunk {chunk_number}"
                )

            key_rows = list(zip(
                chunk["cell_id"].astype(str),
                chunk["date"].dt.strftime("%Y-%m-%d"),
            ))

            before_count = db.execute(
                "SELECT COUNT(*) FROM grid_keys"
            ).fetchone()[0]

            db.executemany(
                """
                INSERT OR IGNORE INTO grid_keys (cell_id, date)
                VALUES (?, ?)
                """,
                key_rows,
            )
            db.commit()

            after_count = db.execute(
                "SELECT COUNT(*) FROM grid_keys"
            ).fetchone()[0]

            newly_inserted = after_count - before_count
            duplicate_key_rows += len(key_rows) - newly_inserted

            print(
                f"Audited chunk {chunk_number}: {rows:,} rows; "
                f"duplicate keys so far: {duplicate_key_rows:,}"
            )

        unique_key_rows = db.execute(
            "SELECT COUNT(*) FROM grid_keys"
        ).fetchone()[0]

    finally:
        if db is not None:
            db.close()

        # Remove only this script's temporary database and its SQLite
        # sidecar files after closing the connection.
        for suffix in ("", "-journal", "-wal", "-shm"):
            temporary_path = Path(str(SQLITE_FILE) + suffix)
            if temporary_path.exists():
                temporary_path.unlink()

    report = {
        "input_file": str(INPUT_FILE),
        "scope": (
            "Weather-join integrity. Fire-detail columns such as frp_sum, "
            "frp_mean and high_confidence_fire_count are not re-verified."
        ),
        "duplicate_check_method": (
            "Exact SQLite composite PRIMARY KEY on (cell_id, date); "
            "duplicate rows counted as input rows not newly inserted."
        ),
        "rows": rows,
        "expected_rows": EXPECTED_ROWS,
        "unique_grid_day_keys": unique_key_rows,
        "fire_count_total": fire_total,
        "expected_fire_count_total": EXPECTED_FIRE_TOTAL,
        "date_min": (
            str(date_min.date()) if pd.notna(date_min) else None
        ),
        "date_max": (
            str(date_max.date()) if pd.notna(date_max) else None
        ),
        "missing_required_values": missing_required,
        "weather_missing_or_non_numeric": weather_missing,
        "weather_min": weather_min,
        "weather_max": weather_max,
        "duplicate_cell_date_rows": duplicate_key_rows,
        "negative_fire_count_rows": negative_fire_rows,
        "zero_fire_count_rows": zero_fire_rows,
    }

    checks = {
        "row_count_matches": rows == EXPECTED_ROWS,
        "fire_total_preserved": fire_total == EXPECTED_FIRE_TOTAL,
        "required_values_complete": all(
            value == 0 for value in missing_required.values()
        ),
        "weather_values_complete": all(
            value == 0 for value in weather_missing.values()
        ),
        "grid_day_keys_unique": duplicate_key_rows == 0,
        "no_negative_fire_counts": negative_fire_rows == 0,
        "date_coverage_expected": (
            date_min is not None
            and date_min.date() == pd.Timestamp("2023-10-01").date()
            and date_max.date() == pd.Timestamp("2025-11-30").date()
        ),
    }

    report["checks"] = checks
    report["result"] = (
        "PASS" if all(checks.values()) else "FAIL"
    )

    AUDIT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nAudit report saved to: {AUDIT_FILE}")

    if report["result"] != "PASS":
        raise SystemExit("V3 weather grid-panel QA FAILED")


if __name__ == "__main__":
    main()
