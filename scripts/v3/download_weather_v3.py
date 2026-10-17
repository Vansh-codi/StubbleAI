import time
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[2]

COORD_FILE = BASE_DIR / "data" / "processed" / "v3" / "district_coordinates_v3.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "v3" / "district_daily_weather_v3.csv"
AUDIT_FILE = BASE_DIR / "data" / "processed" / "v3" / "district_daily_weather_v3_audit.txt"

START_DATE = "20231001"
END_DATE = "20251130"
EXPECTED_DAYS = len(pd.date_range("2023-10-01", "2025-11-30", freq="D"))

PARAMETERS = ["T2M", "RH2M", "WS2M", "PRECTOTCORR"]
API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
REQUIRED_COORDS = ["district", "state", "latitude", "longitude"]


def main():
    if not COORD_FILE.exists():
        raise SystemExit(f"Coordinate file not found: {COORD_FILE}")

    districts = pd.read_csv(COORD_FILE)

    missing = sorted(set(REQUIRED_COORDS) - set(districts.columns))
    if missing:
        raise SystemExit(f"Missing coordinate columns: {missing}")
    if districts.empty:
        raise SystemExit("Coordinate file is empty.")
    if districts.duplicated(["state", "district"]).any():
        raise SystemExit("Duplicate state-district pairs in coordinate file.")
    if districts[["latitude", "longitude"]].isna().any().any():
        raise SystemExit("Missing latitude/longitude values.")
    if not districts["latitude"].between(-90, 90).all():
        raise SystemExit("Latitude outside valid range.")
    if not districts["longitude"].between(-180, 180).all():
        raise SystemExit("Longitude outside valid range.")

    expected_pairs = set(zip(districts["state"], districts["district"]))
    all_weather = []
    failures = []
    session = requests.Session()

    print(f"Districts: {len(districts)}")
    print(f"Expected days per district: {EXPECTED_DAYS}")
    print(f"Expected total rows: {len(districts) * EXPECTED_DAYS:,}")

    for i, row in districts.reset_index(drop=True).iterrows():
        district = str(row["district"])
        state = str(row["state"])
        print(f"\n[{i + 1}/{len(districts)}] {district}, {state}")

        params = {
            "parameters": ",".join(PARAMETERS),
            "community": "AG",
            "longitude": float(row["longitude"]),
            "latitude": float(row["latitude"]),
            "start": START_DATE,
            "end": END_DATE,
            "format": "JSON",
        }

        try:
            data = None
            for attempt in range(4):
                try:
                    response = session.get(API_URL, params=params, timeout=90)
                    response.raise_for_status()
                    data = response.json()

                    if "properties" not in data or "parameter" not in data["properties"]:
                        raise ValueError(
                            f"Unexpected NASA POWER response: {str(data)[:300]}"
                        )

                    parameter_data = data["properties"]["parameter"]
                    missing_params = set(PARAMETERS) - set(parameter_data)
                    if missing_params:
                        raise ValueError(f"Missing API parameters: {sorted(missing_params)}")

                    break
                except (requests.RequestException, ValueError, KeyError) as exc:
                    if attempt == 3:
                        raise
                    wait = 2 ** attempt
                    print(f"  Attempt {attempt + 1} failed: {exc}")
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)

            parameter_data = data["properties"]["parameter"]
            date_keys = sorted(parameter_data[PARAMETERS[0]].keys())

            if len(date_keys) != EXPECTED_DAYS:
                raise ValueError(
                    f"Expected {EXPECTED_DAYS} dates, received {len(date_keys)}"
                )

            expected_dates = set(pd.date_range(
                "2023-10-01", "2025-11-30", freq="D"
            ).strftime("%Y%m%d"))
            if set(date_keys) != expected_dates:
                raise ValueError("Returned dates do not match the complete requested date range.")

            for parameter in PARAMETERS:
                if set(parameter_data[parameter].keys()) != set(date_keys):
                    raise ValueError(f"Incomplete date coverage for {parameter}")

            weather = pd.DataFrame({
                "date": pd.to_datetime(date_keys, format="%Y%m%d")
            })

            for parameter in PARAMETERS:
                weather[parameter] = [parameter_data[parameter][d] for d in date_keys]

            weather["district"] = district
            weather["state"] = state
            weather["latitude"] = float(row["latitude"])
            weather["longitude"] = float(row["longitude"])

            all_weather.append(weather)
            print(f"  Validated {len(weather)} daily records")

        except Exception as exc:
            failures.append((state, district, str(exc)))
            print(f"  ERROR: {exc}")

        time.sleep(0.5)

    session.close()

    if failures:
        print("\nFAILED DISTRICTS:")
        for state, district, error in failures:
            print(f"  {state}, {district}: {error}")
        raise SystemExit(
            f"{len(failures)} district request(s) failed. No output CSV written."
        )

    weather_df = pd.concat(all_weather, ignore_index=True)
    weather_df = weather_df[
        ["date", "state", "district", "latitude", "longitude"] + PARAMETERS
    ].sort_values(["date", "state", "district"]).reset_index(drop=True)

    actual_pairs = set(zip(weather_df["state"], weather_df["district"]))
    if actual_pairs != expected_pairs:
        raise SystemExit("Returned district coverage does not match coordinate file.")

    expected_rows = len(districts) * EXPECTED_DAYS
    if len(weather_df) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} rows, got {len(weather_df)}")

    if weather_df.duplicated(["state", "district", "date"]).any():
        raise SystemExit("Duplicate state-district-date records detected.")

    date_counts = weather_df.groupby(["state", "district"])["date"].agg(["count", "min", "max"])
    if not (date_counts["count"] == EXPECTED_DAYS).all():
        raise SystemExit("Incomplete per-district date coverage.")
    if not (date_counts["min"] == pd.Timestamp("2023-10-01")).all():
        raise SystemExit("Incorrect start date for one or more districts.")
    if not (date_counts["max"] == pd.Timestamp("2025-11-30")).all():
        raise SystemExit("Incorrect end date for one or more districts.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    weather_df.to_csv(OUTPUT_FILE, index=False)

    audit = [
        "V3 NASA POWER weather download audit",
        f"Districts requested: {len(districts)}",
        f"Expected days per district: {EXPECTED_DAYS}",
        f"Rows written: {len(weather_df)}",
        f"Date range: {weather_df['date'].min().date()} to {weather_df['date'].max().date()}",
        f"Duplicate state-district-date keys: {weather_df.duplicated(['state', 'district', 'date']).sum()}",
        f"Missing values by parameter: {weather_df[PARAMETERS].isna().sum().to_dict()}",
        "",
        "Rows by state:",
        weather_df.groupby("state").size().to_string(),
        "",
        "Missing-value note: NASA POWER may use documented fill values for unavailable observations; inspect parameter values before modelling.",
    ]
    AUDIT_FILE.write_text("\n".join(audit) + "\n", encoding="utf-8")

    print("\nDOWNLOAD AND STRUCTURAL VALIDATION PASSED")
    print("Output:", OUTPUT_FILE)
    print("Rows:", len(weather_df))
    print("Date range:", weather_df["date"].min().date(), "to", weather_df["date"].max().date())
    print("Missing values:")
    print(weather_df[PARAMETERS].isna().sum().to_string())
    print("Audit:", AUDIT_FILE)


if __name__ == "__main__":
    main()
