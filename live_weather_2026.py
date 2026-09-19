import time
import requests
import pandas as pd


# ============================================================
# STUBBLEAI - 2026 LIVE WEATHER FORECAST
# ============================================================

COORD_FILE = "district_coordinates.csv"
OUTPUT_FILE = "live_weather_2026.csv"

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

for i, row in districts.iterrows():

    district = row["district"]
    state = row["state"]

    latitude = row["latitude"]
    longitude = row["longitude"]

    print(
        f"[{i + 1:02d}/{len(districts):02d}] "
        f"{district}, {state}"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "daily": (
            "temperature_2m_mean,"
            "relative_humidity_2m_mean,"
            "wind_speed_10m_mean,"
            "precipitation_sum"
        ),

        "forecast_days": 3,

        "timezone": "Asia/Kolkata",

        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    try:

        response = None

        for attempt in range(3):

            try:

                response = requests.get(
                    BASE_URL,
                    params=params,
                    timeout=30
                )

                response.raise_for_status()
                break

            except requests.RequestException as e:

                if attempt == 2:
                    raise

                wait_seconds = 2 ** attempt

                print(
                    f"   Retry {attempt + 1}/2 "
                    f"after error: {e}"
                )

                time.sleep(wait_seconds)

        data = response.json()

        daily = data.get("daily")

        if daily is None:
            raise ValueError(
                "No daily forecast returned."
            )

        required_forecast_fields = [
            "time",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "wind_speed_10m_mean",
            "precipitation_sum",
        ]

        missing_forecast_fields = [
            field
            for field in required_forecast_fields
            if field not in daily
        ]

        if missing_forecast_fields:
            raise ValueError(
                "Open-Meteo response is missing fields: "
                f"{missing_forecast_fields}"
            )

        weather = pd.DataFrame({
            "date": daily["time"],

            "T2M": daily[
                "temperature_2m_mean"
            ],

            "RH2M": daily[
                "relative_humidity_2m_mean"
            ],

            "WS2M": daily[
                "wind_speed_10m_mean"
            ],

            "PRECTOTCORR": daily[
                "precipitation_sum"
            ]
        })

        weather["date"] = pd.to_datetime(
            weather["date"],
            errors="raise"
        )

        if weather.empty:
            raise ValueError(
                "Open-Meteo returned no weather records."
            )

        weather_features = [
            "T2M",
            "RH2M",
            "WS2M",
            "PRECTOTCORR",
        ]

        missing_values = weather[
            weather_features
        ].isna().sum()

        if missing_values.any():
            raise ValueError(
                "Weather data contains missing values: "
                f"{missing_values[missing_values > 0]}"
            )

        if len(weather) != 3:
            raise ValueError(
                f"Expected 3 forecast days for "
                f"{district}, received {len(weather)}."
            )

        weather["district"] = district
        weather["state"] = state

        weather = weather[
            [
                "date",
                "state",
                "district",
                "T2M",
                "RH2M",
                "WS2M",
                "PRECTOTCORR"
            ]
        ]

        all_weather.append(weather)

        print(
            "   OK:",
            weather["date"].min().date(),
            "to",
            weather["date"].max().date()
        )

    except Exception as e:

        print(
            "   ERROR:",
            str(e)
        )

        failures.append(
            (district, state, str(e))
        )

    # Small delay between requests
    time.sleep(0.2)


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