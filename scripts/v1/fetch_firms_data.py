import os
import time
from datetime import date, timedelta
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv

# ============================================================
# STUBBLEAI — NASA FIRMS DATA FETCHER
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

load_dotenv(
    os.path.join(PROJECT_ROOT, ".env")
)

MAP_KEY = os.getenv("FIRMS_MAP_KEY")

AREA = "73.5,27.5,77.5,32.5"

SOURCES = [
    "VIIRS_SNPP_SP",
    "VIIRS_NOAA20_SP",
]

YEARS = [2023, 2024, 2025]

DAY_RANGE = 5

BASE_URL = (
    "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
    "{key}/{source}/{area}/{day_range}/{date}"
)

def date_chunks(start, end, step_days):
    """Generate dates for consecutive 5-day API windows."""
    current = start

    while current <= end:
        yield current
        current += timedelta(days=step_days)


def fetch_chunk(source, start_date):
    """Download one FIRMS API chunk with retry handling."""

    url = BASE_URL.format(
        key=MAP_KEY,
        source=source,
        area=AREA,
        day_range=DAY_RANGE,
        date=start_date.isoformat(),
    )

    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                timeout=60,
            )

            response.raise_for_status()

            text = response.text.strip()

            if not text:
                return pd.DataFrame()

            first_line = text.splitlines()[0].lower()

            if (
                first_line.startswith("invalid")
                or "error" in first_line
            ):
                print(
                    f"  WARNING: FIRMS returned: {text[:200]}"
                )
                return pd.DataFrame()

            return pd.read_csv(
                StringIO(text)
            )

        except (
            requests.RequestException,
            pd.errors.ParserError,
        ) as e:
            last_error = e

            if attempt < 2:
                wait_seconds = 2 ** attempt

                print(
                    f"  Retry {attempt + 1}/2 "
                    f"after error: {e}"
                )

                time.sleep(wait_seconds)

    raise RuntimeError(
        f"FIRMS request failed after 3 attempts: "
        f"{last_error}"
    )

def main():

    if not MAP_KEY:
     raise SystemExit(
        "ERROR: FIRMS_MAP_KEY is not configured."
    )

    all_frames = []
    summary = {}
    failures = []

    for year in YEARS:

        season_start = date(year, 10, 1)
        season_end = date(year, 11, 30)

        year_frames = []

        for source in SOURCES:

            for chunk_start in date_chunks(
                season_start,
                season_end,
                DAY_RANGE
            ):

                print(
                    f"Fetching {source} "
                    f"{year} starting {chunk_start} ..."
                )

                try:

                    df = fetch_chunk(
                        source,
                        chunk_start
                    )

                except requests.RequestException as e:

                    print(f"  ERROR: {e}")

                    failures.append(
                        (source, year, chunk_start, str(e))
                    )

                    continue

                if not df.empty:

                    df["query_source"] = source
                    year_frames.append(df)

                time.sleep(1)

        if year_frames:

            year_df = pd.concat(
                year_frames,
                ignore_index=True
            )

            all_frames.append(year_df)
            summary[year] = len(year_df)

        else:

            summary[year] = 0

    # --------------------------------------------------------
    # Report failures
    # --------------------------------------------------------

    if failures:

        print("\n--- FAILED REQUESTS ---")

        for source, year, chunk_start, error in failures:

            print(
                f"{source} | "
                f"{year} | "
                f"{chunk_start} | "
                f"{error}"
            )

        print(
            f"\nTotal failed requests: {len(failures)}"
        )
        if failures:
          raise SystemExit(
        1
    )

    # --------------------------------------------------------
    # No data
    # --------------------------------------------------------

    if not all_frames:

        print(
            "\nNo data returned."
            "\nCheck your MAP_KEY and API settings."
        )

        return

    # --------------------------------------------------------
    # Combine all years
    # --------------------------------------------------------

    combined = pd.concat(
        all_frames,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Remove duplicate detections
    # --------------------------------------------------------

    dedup_cols = [
        c
        for c in [
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
            "satellite"
        ]
        if c in combined.columns
    ]

    if dedup_cols:

        before = len(combined)

        combined = combined.drop_duplicates(
            subset=dedup_cols
        )

        removed = before - len(combined)

        print(
            f"\nDeduplicated {removed} rows"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        "firms_punjab_haryana_2023_2025.csv"
    )

    combined.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "--- SUMMARY: ROWS PER YEAR ---"
    )

    for year, count in summary.items():

        print(
            f"  {year}: {count:,} rows"
        )

    print(
        "\nTotal after dedup: "
        f"{len(combined):,} rows"
    )

    print(
        f"\nSaved to: {output_file}"
    )

    print(
        "\nColumns:"
    )

    print(
        list(combined.columns)
    )

    print(
        "\n========================================"
    )


if __name__ == "__main__":
    main()
    