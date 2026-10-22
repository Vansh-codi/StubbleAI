from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
COORD_FILE = ROOT / "data" / "processed" / "v3" / "district_coordinates_v3.csv"
OUT_DIR = ROOT / "data" / "live" / "v3"
PARAMETERS = ["T2M", "RH2M", "WS2M", "PRECTOTCORR"]
API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# NASA POWER near-real-time data typically lags a few days behind today.
# This buffer is a provisional starting assumption based on observed
# behavior (Oct 15 available, Oct 16+ returned -999 as of Oct 17-18, 2026).
# Tighten this once more days of real observation confirm the actual lag.
LAG_DAYS = 4
WINDOW_DAYS = 38
REQUEST_TIMEOUT = 30
MAX_ATTEMPTS = 3


def fetch_with_retry(session: requests.Session, params: dict) -> dict:
    """Fetch one district's weather window, retrying transient failures.

    Uses a shorter per-attempt timeout with exponential backoff, rather
    than one long timeout with no retry. A hung connection or transient
    network error should not cost the entire 248-district run.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
    raise last_error


def main() -> None:
    if not COORD_FILE.exists():
        raise SystemExit(f"Coordinate file missing: {COORD_FILE}")

    coords = pd.read_csv(COORD_FILE)
    required = {"district", "state", "latitude", "longitude"}
    missing = required - set(coords.columns)
    if missing:
        raise RuntimeError(f"Missing coordinate columns: {sorted(missing)}")
    if coords[list(required)].isna().any().any():
        raise RuntimeError("Missing district or coordinate values")
    if coords.duplicated(["state", "district"]).any():
        raise RuntimeError("Duplicate state-district coordinates")

    expected_pairs = 248
    if coords[["state", "district"]].drop_duplicates().shape[0] != expected_pairs:
        raise RuntimeError("Expected exactly 248 pilot state-district pairs")

    today = datetime.now(timezone.utc).date()
    end_date = today - timedelta(days=LAG_DAYS)
    start_date = end_date - timedelta(days=WINDOW_DAYS - 1)
    print(f"Requesting window: {start_date.isoformat()} to {end_date.isoformat()} "
          f"(today={today.isoformat()}, lag_days={LAG_DAYS})")

    session = requests.Session()
    records = []
    failures = []
    latest_by_district = {}

    for i, row in enumerate(coords.itertuples(index=False), start=1):
        params = {
            "parameters": ",".join(PARAMETERS),
            "community": "AG",
            "longitude": float(row.longitude),
            "latitude": float(row.latitude),
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "format": "JSON",
        }

        try:
            data = fetch_with_retry(session, params)
            parameter_data = data["properties"]["parameter"]

            for name in PARAMETERS:
                if name not in parameter_data:
                    raise ValueError(f"Missing parameter: {name}")

            date_sets = [set(parameter_data[name]) for name in PARAMETERS]
            if not all(keys == date_sets[0] for keys in date_sets):
                raise ValueError("Weather parameters have inconsistent dates")

            date_keys = sorted(date_sets[0])
            if not date_keys:
                raise ValueError("API returned no daily dates")

            district_rows = []
            for date_key in date_keys:
                item = {
                    "date": pd.to_datetime(date_key, format="%Y%m%d").strftime("%Y-%m-%d"),
                    "state": row.state,
                    "district": row.district,
                    "latitude": float(row.latitude),
                    "longitude": float(row.longitude),
                }
                for name in PARAMETERS:
                    value = parameter_data[name][date_key]
                    numeric_value = pd.to_numeric(value, errors="coerce")
                    # Keep NASA POWER fill values (-999) as NaN rather than
                    # failing the whole district. Coverage is reported
                    # explicitly below instead of being silently discarded.
                    if pd.isna(numeric_value) or numeric_value <= -900:
                        item[name] = pd.NA
                    else:
                        item[name] = numeric_value
                district_rows.append(item)

            records.extend(district_rows)
            latest_by_district[f"{row.state}|{row.district}"] = max(
                x["date"] for x in district_rows
            )

        except Exception as exc:
            failures.append({
                "state": row.state,
                "district": row.district,
                "error": str(exc),
            })

        if i % 20 == 0 or i == len(coords):
            print(f"Processed {i}/{len(coords)} districts; failures={len(failures)}")

        time.sleep(0.15)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    failure_path = OUT_DIR / f"weather_live_failures_{stamp}.csv"
    pd.DataFrame(failures, columns=["state", "district", "error"]).to_csv(
        failure_path, index=False
    )

    if not records:
        print("\nWeather collection produced zero usable records.")
        print("Failure report:", failure_path)
        raise SystemExit(1)

    weather = pd.DataFrame(records)
    weather = weather[
        ["date", "state", "district", "latitude", "longitude"] + PARAMETERS
    ].sort_values(["date", "state", "district"]).reset_index(drop=True)

    if weather.duplicated(["date", "state", "district"]).any():
        raise RuntimeError("Duplicate weather date/state/district keys")

    # Always write what was retrieved, complete or not. A failed/incomplete
    # run should not discard real API responses that were already paid for.
    output = OUT_DIR / f"weather_live_candidates_{stamp}.csv"
    weather.to_csv(output, index=False)

    missing_by_parameter = weather[PARAMETERS].isna().sum()
    complete_rows = int(weather[PARAMETERS].notna().all(axis=1).sum())

    pairs = coords[["state", "district"]].drop_duplicates()
    expected_dates = set(weather["date"].unique())
    expected_rows = len(pairs) * len(expected_dates)
    row_panel_complete = len(weather) == expected_rows

    audit = OUT_DIR / f"weather_live_audit_{stamp}.txt"
    audit_lines = [
        f"retrieved_at_utc={datetime.now(timezone.utc).isoformat()}",
        f"requested_window={start_date.isoformat()} to {end_date.isoformat()}",
        f"lag_days_assumed={LAG_DAYS}",
        f"district_pairs_expected={pairs.shape[0]}",
        f"district_requests_failed={len(failures)}",
        f"rows_retrieved={len(weather)}",
        f"rows_expected_if_complete={expected_rows}",
        f"district_date_panel_complete={row_panel_complete}",
        f"rows_with_all_parameters_available={complete_rows}/{len(weather)}",
        f"date_range={weather['date'].min()} to {weather['date'].max()}",
        f"distinct_dates={weather['date'].nunique()}",
        f"duplicate_keys={weather.duplicated(['date', 'state', 'district']).sum()}",
        "",
        "missing_values_by_parameter:",
        missing_by_parameter.to_string(),
        "",
        "rows_by_state:",
        weather.groupby("state").size().to_string(),
        "",
        "latest_date_by_state:",
        weather.groupby("state")["date"].max().to_string(),
    ]
    audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    print(f"\nSaved {len(weather)} candidate rows to: {output}")
    print(f"Rows with all parameters available: {complete_rows}/{len(weather)}")
    print("Missing values by parameter:")
    print(missing_by_parameter.to_string())
    print("Audit:", audit)

    if failures:
        print(f"\nWeather collection INCOMPLETE: {len(failures)}/{len(coords)} "
              f"district requests failed after {MAX_ATTEMPTS} attempts each.")
        print("Failure report:", failure_path)
        print("This candidate file is PARTIAL â€” do not use for inference "
              "without re-checking coverage against the failure report.")
        raise SystemExit(1)

    if not row_panel_complete:
        print(f"\nWeather panel INCOMPLETE: expected {expected_rows} rows, "
              f"got {len(weather)}.")
        print("This candidate file is PARTIAL â€” do not use for inference.")
        raise SystemExit(1)

    if missing_by_parameter.sum() > 0:
        print("\nWeather panel has missing parameter values on some "
              "district-dates (see breakdown above). This candidate file "
              "is retrieved but NOT fully clean â€” resolve before using "
              "for inference.")
        raise SystemExit(1)

    print("\nLIVE WEATHER COLLECTION COMPLETE AND CLEAN")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
