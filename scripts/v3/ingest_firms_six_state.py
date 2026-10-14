
"""Historical FIRMS candidate-data collector for StubbleAI V3.

Fetches October-November 2023-2025 observations for a broad bounding box
covering the six-state pilot. It does NOT assign detections to states.
That spatial filtering and its QA are separate downstream steps.

The canonical CSV is written only if every requested window succeeds.
Failed runs retain a window manifest and do not publish a partial dataset
under the canonical filename.
"""

import os
import time
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "raw" / "firms" / "v3"

SOURCES = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"]
YEARS = [2023, 2024, 2025]
SEASON_START = (10, 1)
SEASON_END = (11, 30)
DAY_RANGE = 5

# west, south, east, north
# Broad candidate box; spatial filtering happens downstream.
BBOX = "73,17.5,96,32.5"

BASE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{key}/{source}/{bbox}/{days}/{start}"
)
AVAILABILITY_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/"
    "{key}/ALL"
)

REQUIRED_COLUMNS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
]

DEDUP_COLUMNS = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def seasonal_windows(year):
    """Return non-overlapping 5-day request windows for one season.

    FIRMS returns DATE through DATE + DAY_RANGE - 1.
    The final request may extend beyond November 30; those extra records
    are excluded after retrieval.
    """
    start = date(year, *SEASON_START)
    end = date(year, *SEASON_END)
    windows = []

    current = start
    while current <= end:
        request_end = current + timedelta(days=DAY_RANGE - 1)
        windows.append({
            "start": current,
            "end": min(request_end, end),
        })
        current += timedelta(days=DAY_RANGE)

    return windows


def parse_csv_response(response_text, context):
    """Parse a successful CSV response or raise a clear error."""
    text = response_text.strip()

    if not text:
        raise ValueError(f"{context}: empty response body")

    first_line = text.splitlines()[0].strip().lower()
    if first_line.startswith("invalid") or "error" in first_line:
        raise ValueError(
            f"{context}: FIRMS returned an error: {text[:200]}"
        )

    frame = pd.read_csv(StringIO(text))

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{context}: response missing columns {missing}"
        )

    return frame


def fetch_with_retries(session, url, context):
    """Return a parsed response; never silently skip a failed window."""
    last_error = None

    for attempt in range(3):
        try:
            response = session.get(url, timeout=90)
            response.raise_for_status()
            return parse_csv_response(response.text, context)

        except (
            requests.RequestException,
            pd.errors.ParserError,
            ValueError,
        ) as exc:
            last_error = exc

            if attempt < 2:
                wait_seconds = 2 ** attempt
                print(
                    f"  Retry {attempt + 1}/2 for {context}: {exc}"
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"{context}: failed after 3 attempts: {last_error}"
    )


def check_source_availability(session, key):
    """Check that each requested source spans the study period."""
    url = AVAILABILITY_URL.format(key=key)
    response = session.get(url, timeout=60)
    response.raise_for_status()

    availability = pd.read_csv(StringIO(response.text))
    required = {"data_id", "min_date", "max_date"}

    if not required.issubset(availability.columns):
        raise RuntimeError(
            "FIRMS availability response has unexpected columns: "
            f"{availability.columns.tolist()}"
        )

    availability["min_date"] = pd.to_datetime(
        availability["min_date"], errors="coerce"
    )
    availability["max_date"] = pd.to_datetime(
        availability["max_date"], errors="coerce"
    )

    study_start = pd.Timestamp(f"{YEARS[0]}-10-01")
    study_end = pd.Timestamp(f"{YEARS[-1]}-11-30")
    rows = []

    for source in SOURCES:
        match = availability[availability["data_id"].eq(source)]

        if len(match) != 1:
            rows.append({
                "source": source,
                "study_start": study_start.date(),
                "study_end": study_end.date(),
                "available_min_date": None,
                "available_max_date": None,
                "covers_full_period": False,
                "status": "SOURCE_MISSING_OR_AMBIGUOUS",
            })
            continue

        record = match.iloc[0]
        minimum = record["min_date"]
        maximum = record["max_date"]

        covered = (
            pd.notna(minimum)
            and pd.notna(maximum)
            and minimum <= study_start
            and maximum >= study_end
        )

        rows.append({
            "source": source,
            "study_start": study_start.date(),
            "study_end": study_end.date(),
            "available_min_date": (
                minimum.date() if pd.notna(minimum) else None
            ),
            "available_max_date": (
                maximum.date() if pd.notna(maximum) else None
            ),
            "covers_full_period": bool(covered),
            "status": "OK" if covered else "INSUFFICIENT_DATE_RANGE",
        })

    result = pd.DataFrame(rows)
    result.to_csv(
        OUT_DIR / "firms_source_availability_2023_2025.csv",
        index=False,
    )

    if not result["covers_full_period"].all():
        raise RuntimeError(
            "One or more FIRMS sources do not cover the full study "
            "period. Inspect firms_source_availability_2023_2025.csv."
        )


# -------------------------------------------------------------------
# Main collection
# -------------------------------------------------------------------

def main():
    load_dotenv(ROOT / ".env")
    map_key = os.getenv("FIRMS_MAP_KEY")

    if not map_key or not map_key.strip():
        raise SystemExit("FIRMS_MAP_KEY is not configured.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    candidate_frames = []

    with requests.Session() as session:
        # Preflight source-level date-range check.
        check_source_availability(session, map_key)

        for source in SOURCES:
            for year in YEARS:
                windows = seasonal_windows(year)

                print(
                    f"\n{source} | {year} | "
                    f"{len(windows)} requested windows"
                )

                for window in windows:
                    start = window["start"]
                    end = window["end"]
                    context = f"{source} {start} to {end}"

                    url = BASE_URL.format(
                        key=map_key,
                        source=source,
                        bbox=BBOX,
                        days=DAY_RANGE,
                        start=start.isoformat(),
                    )

                    record = {
                        "source": source,
                        "year": year,
                        "requested_start": start.isoformat(),
                        "requested_end": end.isoformat(),
                        "status": "FAILED",
                        "rows_returned": 0,
                        "season_rows": 0,
                        "error": "",
                    }

                    try:
                        frame = fetch_with_retries(
                            session, url, context
                        )

                        frame["acq_date"] = pd.to_datetime(
                            frame["acq_date"], errors="coerce"
                        )

                        invalid_dates = int(frame["acq_date"].isna().sum())
                        frame = frame.dropna(subset=["acq_date"]).copy()

                        # The API window can extend beyond the season.
                        seasonal_start = pd.Timestamp(
                            f"{year}-10-01"
                        )
                        seasonal_end = pd.Timestamp(
                            f"{year}-11-30"
                        )

                        frame = frame[
                            frame["acq_date"].between(
                                seasonal_start, seasonal_end
                            )
                        ].copy()

                        frame["query_source"] = source

                        record["status"] = "SUCCESS"
                        record["rows_returned"] = len(frame)
                        record["season_rows"] = len(frame)
                        record["invalid_dates_removed"] = invalid_dates

                        candidate_frames.append(frame)

                        print(
                            f"  {start}: SUCCESS; "
                            f"{len(frame):,} seasonal rows"
                        )

                    except Exception as exc:
                        record["error"] = str(exc)
                        print(f"  {start}: FAILED — {exc}")

                    manifest_rows.append(record)

                    # Save progress after each window so a failed run
                    # still leaves a complete audit trail.
                    manifest = pd.DataFrame(manifest_rows)
                    manifest.to_csv(
                        OUT_DIR / "firms_api_window_manifest_2023_2025.csv",
                        index=False,
                    )

                    time.sleep(1)

    # ----------------------------------------------------------------
    # Mandatory coverage report
    # ----------------------------------------------------------------

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = (
        OUT_DIR / "firms_api_window_manifest_2023_2025.csv"
    )
    summary_path = (
        OUT_DIR / "firms_api_coverage_summary_2023_2025.csv"
    )
    missing_path = (
        OUT_DIR / "firms_api_missing_windows_2023_2025.csv"
    )

    expected_windows = len(SOURCES) * sum(
        len(seasonal_windows(year)) for year in YEARS
    )
    failed = manifest[manifest["status"].ne("SUCCESS")]

    summary_rows = []
    for source in SOURCES:
        for year in YEARS:
            subset = manifest[
                manifest["source"].eq(source)
                & manifest["year"].eq(year)
            ]

            failed_dates = []
            for _, row in subset[subset["status"].ne("SUCCESS")].iterrows():
                current = date.fromisoformat(row["requested_start"])
                last = date.fromisoformat(row["requested_end"])
                while current <= last:
                    failed_dates.append(current.isoformat())
                    current += timedelta(days=1)

            summary_rows.append({
                "source": source,
                "year": year,
                "requested_windows": len(subset),
                "returned_windows": int(
                    subset["status"].eq("SUCCESS").sum()
                ),
                "failed_windows": int(
                    subset["status"].ne("SUCCESS").sum()
                ),
                "missing_dates_due_to_failed_requests": ";".join(
                    sorted(set(failed_dates))
                ),
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    failed.to_csv(missing_path, index=False)

    print("\n=== API WINDOW COVERAGE ===")
    print(summary.to_string(index=False))
    print(f"\nExpected windows: {expected_windows}")
    print(f"Manifest: {manifest_path}")
    print(f"Coverage summary: {summary_path}")
    print(f"Failed windows: {missing_path}")

    # No canonical dataset is written if even one requested window failed.
    if len(manifest) != expected_windows or not failed.empty:
        raise SystemExit(
            "Collection incomplete. Canonical CSV was NOT written. "
            "Inspect the coverage summary and failed-window manifest."
        )

    if not candidate_frames:
        raise SystemExit(
            "All windows returned successfully but no seasonal rows "
            "were collected. Canonical CSV was NOT written."
        )

    combined = pd.concat(candidate_frames, ignore_index=True)

    # Validate coordinates and dates before deduplication.
    combined["latitude"] = pd.to_numeric(
        combined["latitude"], errors="coerce"
    )
    combined["longitude"] = pd.to_numeric(
        combined["longitude"], errors="coerce"
    )

    bad_coordinates = (
        combined["latitude"].isna()
        | combined["longitude"].isna()
        | ~combined["latitude"].between(-90, 90)
        | ~combined["longitude"].between(-180, 180)
    )

    if bad_coordinates.any():
        raise SystemExit(
            f"{int(bad_coordinates.sum())} invalid coordinate rows found. "
            "Canonical CSV was NOT written."
        )

    before = len(combined)
    available_keys = [
        col for col in DEDUP_COLUMNS if col in combined.columns
    ]

    if len(available_keys) != len(DEDUP_COLUMNS):
        raise SystemExit(
            "Deduplication key incomplete. Canonical CSV was NOT written."
        )

    combined = combined.drop_duplicates(
        subset=available_keys
    ).reset_index(drop=True)

    duplicates_removed = before - len(combined)

    # This file contains bounding-box candidates, NOT just six-state rows.
    canonical_path = (
        OUT_DIR / "firms_bbox_candidates_2023_2025.csv"
    )
    combined.to_csv(canonical_path, index=False)

    print("\n=== COLLECTION SUMMARY ===")
    print("Windows succeeded:", len(manifest))
    print("Rows before deduplication:", before)
    print("Duplicate records removed:", duplicates_removed)
    print("Rows saved:", len(combined))
    print("Source/year counts:")
    print(
        combined.groupby(
            ["query_source", combined["acq_date"].dt.year]
        ).size().to_string()
    )
    print(f"\nSaved candidate data to: {canonical_path}")
    print(
        "Reminder: these are bounding-box candidates. "
        "Six-state spatial filtering and outside-pilot QA remain required."
    )


if __name__ == "__main__":
    main()
