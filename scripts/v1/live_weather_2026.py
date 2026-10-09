import time
import requests
import pandas as pd
from pathlib import Path


# ============================================================
# STUBBLEAI - 2026 LIVE WEATHER FORECAST
# ============================================================

COORD_FILE = str(Path(__file__).resolve().parents[2] / "data" / "processed" / "v1" / "district_coordinates.csv")
OUTPUT_FILE = str(Path(__file__).resolve().parents[2] / "data" / "live" / "live_weather_2026.csv")

BASE_URL = "https://api.open-meteo.com/v1/forecast"


print("=" * 70)
print("STUBBLEAI - 2026 WEATHER FORECAST COLLECTOR")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load district coordinates
# ------------------------------------------------------------

districts = pd.read_csv(COORD_FILE)
required_columns = [
    "district",
    "state",
    "latitude",
    "longitude",
]

missing_columns = [
    col
    for col in required_columns
    if col not in districts.columns
]

if missing_columns:
    raise ValueError(
        "Coordinate file is missing required columns: "
        f"{missing_columns}"
    )

if districts.empty:
    raise ValueError(
        "District coordinate file is empty."
    )

if districts[
    ["latitude", "longitude"]
].isna().any().any():
    raise ValueError(
        "District coordinate file contains missing "
        "latitude/longitude values."
    )

districts["district"] = (
    districts["district"]
    .astype(str)
    .str.strip()
    .str.replace(
        "S.A.S Nagar",
        "S A S Nagar",
        regex=False
    )
)

districts["state"] = (
    districts["state"]
    .astype(str)
    .str.strip()
)
if len(districts) != 45:
    raise ValueError(
        f"Expected 45 Punjab/Haryana districts, "
        f"but found {len(districts)}."
    )

district_pairs = districts[
    ["state", "district"]
].drop_duplicates()

if len(district_pairs) != len(districts):
    raise ValueError(
        "Duplicate state/district entries found "
        "in district_coordinates.csv."
    )

print()
print("Districts loaded:", len(districts))


# ------------------------------------------------------------
# 2. Download next-day weather
# ------------------------------------------------------------

# ------------------------------------------------------------
# 2. Download next-day weather
# ------------------------------------------------------------

all_weather = []
failures = []

BATCH_SIZE = 5
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT = (10, 60)

required_forecast_fields = [
    "time",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "precipitation_sum",
]

weather_features = ["T2M", "RH2M", "WS2M", "PRECTOTCORR"]


def fetch_weather_batch(batch, depth=0):
    """Fetch a batch; split it into smaller batches if requests fail."""

    names = [
        f"{r.district}, {r.state}"
        for r in batch.itertuples()
    ]

    print(
        f"\nRequesting {len(batch)} district(s): "
        + ", ".join(names),
        flush=True,
    )

    params = {
        "latitude": ",".join(map(str, batch["latitude"])),
        "longitude": ",".join(map(str, batch["longitude"])),
        "daily": (
            "temperature_2m_mean,"
            "relative_humidity_2m_mean,"
            "wind_speed_10m_mean,"
            "precipitation_sum"
        ),
        "forecast_days": 3,
        "timezone": "Asia/Kolkata",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }

    response_data = None
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(
                BASE_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            response_data = response.json()

            # Open-Meteo may return a dictionary for one location
            # and a list for multiple locations.
            if isinstance(response_data, dict) and len(batch) == 1:
                response_data = [response_data]

            if not isinstance(response_data, list):
                raise ValueError(
                    "Unexpected response format from Open-Meteo."
                )

            if len(response_data) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} responses; "
                    f"received {len(response_data)}."
                )

            break

        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
            response_data = None

            if attempt < MAX_ATTEMPTS - 1:
                wait_seconds = 2 ** (attempt + 1)
                print(
                    f"   Attempt {attempt + 1} failed: {exc}. "
                    f"Retrying in {wait_seconds}s.",
                    flush=True,
                )
                time.sleep(wait_seconds)

    # If a multi-district request keeps failing, split it and retry.
    if response_data is None:
        if len(batch) > 1:
            print(
                f"   Splitting failed batch of {len(batch)} "
                "into smaller requests.",
                flush=True,
            )
            midpoint = len(batch) // 2
            fetch_weather_batch(batch.iloc[:midpoint], depth + 1)
            fetch_weather_batch(batch.iloc[midpoint:], depth + 1)
        else:
            row = batch.iloc[0]
            failures.append(
                (row["district"], row["state"], last_error or
                 "Weather request failed.")
            )
            print(
                f"   FINAL FAILURE: {row['district']}, "
                f"{row['state']}: {last_error}",
                flush=True,
            )
        return

    # Validate and save each district's result in memory.
    for row, data in zip(batch.itertuples(), response_data):
        try:
            daily = data.get("daily")

            if daily is None:
                raise ValueError("No daily forecast returned.")

            missing_fields = [
                field for field in required_forecast_fields
                if field not in daily
            ]
            if missing_fields:
                raise ValueError(
                    f"Missing forecast fields: {missing_fields}"
                )

            weather = pd.DataFrame({
                "date": daily["time"],
                "T2M": daily["temperature_2m_mean"],
                "RH2M": daily["relative_humidity_2m_mean"],
                "WS2M": (
                    pd.Series(
                        daily["wind_speed_10m_mean"],
                        dtype="float64",
                    ) * 0.748
                ),
                "PRECTOTCORR": daily["precipitation_sum"],
            })

            weather["date"] = pd.to_datetime(
                weather["date"], errors="raise"
            )

            if len(weather) != 3:
                raise ValueError(
                    f"Expected 3 forecast days; got {len(weather)}."
                )

            if weather[weather_features].isna().any().any():
                raise ValueError(
                    "Weather forecast contains missing values."
                )

            weather["district"] = row.district
            weather["state"] = row.state

            all_weather.append(
                weather[
                    [
                        "date", "state", "district",
                        "T2M", "RH2M", "WS2M", "PRECTOTCORR",
                    ]
                ]
            )

            print(
                f"   OK: {row.district}, {row.state}",
                flush=True,
            )

        except (ValueError, TypeError, KeyError) as exc:
            failures.append(
                (row.district, row.state, str(exc))
            )
            print(
                f"   INVALID DATA: {row.district}, "
                f"{row.state}: {exc}",
                flush=True,
            )


for start_idx in range(0, len(districts), BATCH_SIZE):
    batch = districts.iloc[start_idx:start_idx + BATCH_SIZE]
    fetch_weather_batch(batch)
    time.sleep(1)



# ------------------------------------------------------------
# 3. Check results
# ------------------------------------------------------------

if failures:

    print("\n" + "=" * 70)
    print("FAILED WEATHER REQUESTS")
    print("=" * 70)

    for district, state, error in failures:
        print(
            f"{district}, {state}: {error}"
        )

    raise RuntimeError(
        f"Weather download failed for "
        f"{len(failures)} district(s)."
    )

if not all_weather:

    raise RuntimeError(
        "No weather data was downloaded."
    )


weather = pd.concat(
    all_weather,
    ignore_index=True
)

# ------------------------------------------------------------
# 4. Remove duplicates
# ------------------------------------------------------------

weather = weather.drop_duplicates(
    subset=[
        "date",
        "state",
        "district"
    ]
)


weather = weather.sort_values(
    [
        "date",
        "state",
        "district"
    ]
).reset_index(drop=True)


# ------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------

weather.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 6. Summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("2026 WEATHER DOWNLOAD COMPLETE")
print("=" * 70)

print("Rows:", len(weather))

print(
    "Districts:",
    weather["district"].nunique()
)

print(
    "Date range:",
    weather["date"].min().date(),
    "to",
    weather["date"].max().date()
)

print(
    "Missing values:",
    weather[
        [
            "T2M",
            "RH2M",
            "WS2M",
            "PRECTOTCORR"
        ]
    ].isna().sum().sum()
)

print()
print("Sample:")

print(
    weather.head(10).to_string(
        index=False
    )
)

print()
print("Saved to:")
print(OUTPUT_FILE)

print("=" * 70)