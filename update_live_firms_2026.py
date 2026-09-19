import os
import time
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv

# ============================================================
# STUBBLEAI - PERSISTENT 2026 FIRMS COLLECTOR
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)

MAP_KEY = os.getenv("FIRMS_MAP_KEY")

if not MAP_KEY:
    raise ValueError(
        "FIRMS_MAP_KEY environment variable is not set."
    )
BBOX = "73.5,27.5,77.5,32.5"

SOURCES = [
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
]

HISTORY_FILE = "live_firms_2026_history.csv"


print("=" * 70)
print("STUBBLEAI - UPDATE 2026 FIRMS HISTORY")
print("=" * 70)


all_rows = []
failures = []

for source in SOURCES:

    print()
    print(f"Downloading: {source}")

    # ------------------------------------------------------------
    # Query:
    # 1. Latest available live FIRMS window
    # 2. Historical backfill for feature warm-up
    # ------------------------------------------------------------

    query_starts = [
        None,
        "2026-09-11",
    ]

    for query_start in query_starts:

        if query_start is None:

            url = (
                f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                f"{MAP_KEY}/{source}/{BBOX}/5"
            )

            print(
                "Querying latest 5-day FIRMS window..."
            )

        else:

            url = (
                f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                f"{MAP_KEY}/{source}/{BBOX}/5/{query_start}"
            )

            print(
                "Querying historical FIRMS window "
                f"starting {query_start}..."
            )

        response = None

        # --------------------------------------------------------
        # Retry request
        # --------------------------------------------------------

        for attempt in range(3):

            try:

                response = requests.get(
                    url,
                    timeout=60
                )

                response.raise_for_status()
                break

            except requests.RequestException as e:

                if attempt == 2:

                    failures.append(
                        (
                            source,
                            query_start,
                            str(e)
                        )
                    )

                    print(
                        f"  FAILED after 3 attempts: {e}"
                    )

                    response = None
                    break

                wait_seconds = 2 ** attempt

                print(
                    f"  Retry {attempt + 1}/2 "
                    f"after error: {e}"
                )

                
                time.sleep(wait_seconds)

        if response is None:
            continue

        print(
            "HTTP status:",
            response.status_code
        )

        # --------------------------------------------------------
        # Parse response
        # --------------------------------------------------------

        try:

            df = pd.read_csv(
                StringIO(response.text)
            )

        except Exception as e:

            failures.append(
                (
                    source,
                    query_start,
                    f"CSV parsing failed: {e}"
                )
            )

            print(
                f"  CSV parsing failed: {e}"
            )

            continue

        print(
            "Rows received:",
            len(df)
        )

        if df.empty:
            print(
                "  No observations returned."
            )
            continue

        # --------------------------------------------------------
        # Validate required FIRMS columns
        # --------------------------------------------------------

        required_columns = [
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
            "satellite",
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            failures.append(
                (
                    source,
                    query_start,
                    f"Missing columns: {missing_columns}"
                )
            )

            print(
                "  Missing required columns:",
                missing_columns
            )

            continue

        # --------------------------------------------------------
        # Date handling
        # --------------------------------------------------------

        df["acq_date"] = pd.to_datetime(
            df["acq_date"],
            errors="coerce"
        )

        invalid_dates = df["acq_date"].isna().sum()

        if invalid_dates:

            print(
                f"  WARNING: {invalid_dates} invalid "
                "acquisition dates removed."
            )

            df = df.dropna(
                subset=["acq_date"]
            )

        # --------------------------------------------------------
        # Keep only 2026 observations
        # --------------------------------------------------------

        df = df[
            (df["acq_date"] >= "2026-01-01")
            &
            (df["acq_date"] <= "2026-12-31")
        ].copy()

        if df.empty:

            print(
                "  No 2026 observations in this response."
            )

            continue

        all_rows.append(df)

        print(
            "  Accepted 2026 rows:",
            len(df)
        )

if failures:

    print("\n" + "=" * 70)
    print("FAILED FIRMS REQUESTS")
    print("=" * 70)

    for source, query_start, error in failures:

        window = (
            "latest"
            if query_start is None
            else query_start
        )

        print(
            f"{source} | {window} | {error}"
        )

    raise RuntimeError(
        f"FIRMS collection failed for "
        f"{len(failures)} request(s)."
    )

if not all_rows:

    raise RuntimeError(
        "No FIRMS observations were returned."
    )


new_data = pd.concat(all_rows, ignore_index=True)


# ============================================================
# LOAD EXISTING HISTORY IF AVAILABLE
# ============================================================

if os.path.exists(HISTORY_FILE):

    old_data = pd.read_csv(HISTORY_FILE)

    old_data["acq_date"] = pd.to_datetime(old_data["acq_date"])

    print()
    print("Existing history rows:", len(old_data))

    combined = pd.concat(
        [old_data, new_data],
        ignore_index=True
    )

else:

    print()
    print("No existing history found.")
    print("Creating new history file.")

    combined = new_data


# ============================================================
# REMOVE DUPLICATES
# ============================================================

duplicate_columns = [
    "latitude",
    "longitude",
    "acq_date",
    "acq_time",
    "satellite",
]

available_duplicate_columns = [
    col for col in duplicate_columns
    if col in combined.columns
]

combined = combined.drop_duplicates(
    subset=available_duplicate_columns
)


# ============================================================
# SORT
# ============================================================

combined = combined.sort_values(
    ["acq_date", "acq_time"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

combined.to_csv(
    HISTORY_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("2026 FIRMS HISTORY UPDATED")
print("=" * 70)

print("Total unique rows:", len(combined))

print(
    "Date range:",
    combined["acq_date"].min().date(),
    "to",
    combined["acq_date"].max().date()
)

print()
print("Rows by date:")

print(
    combined["acq_date"]
    .dt.strftime("%Y-%m-%d")
    .value_counts()
    .sort_index()
)

print()
print("Saved to:")
print(HISTORY_FILE)

print("=" * 70)