import pandas as pd
import requests
import time
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

COORD_FILE = BASE_DIR / "district_coordinates.csv"
OUTPUT_FILE = BASE_DIR / "district_daily_weather.csv"

START_DATE = "20231001"
END_DATE = "20251130"

PARAMETERS = [
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR"
]


# NASA POWER Daily API
API_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
)


# ============================================================
# LOAD DISTRICT COORDINATES
# ============================================================

print("Loading district coordinates...")

districts = pd.read_csv(COORD_FILE)

print(f"Districts: {len(districts)}")
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
        f"Coordinate file is missing required columns: "
        f"{missing_columns}"
    )

if districts.empty:
    raise ValueError(
        "District coordinate file is empty."
    )

# ============================================================
# DOWNLOAD WEATHER
# ============================================================

all_weather = []
failures = []

for i, row in districts.iterrows():

    district = row["district"]
    state = row["state"]

    latitude = row["latitude"]
    longitude = row["longitude"]

    print(
        f"\n[{i + 1}/{len(districts)}] "
        f"{district}, {state}"
    )

    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON"
    }

    try:

        response = None

        for attempt in range(3):

            try:
                response = requests.get(
                    API_URL,
                    params=params,
                    timeout=60
                )

                response.raise_for_status()
                break

            except requests.RequestException as e:

                if attempt == 2:
                    raise

                wait_seconds = 2 ** attempt

                print(
                    f"  Retry {attempt + 1}/2 after error: {e}"
                )

                time.sleep(wait_seconds)

        data = response.json()

        properties = data["properties"]

        parameter_data = properties["parameter"]

        dates = list(
            parameter_data[PARAMETERS[0]].keys()
        )

        weather = pd.DataFrame({
            "date": pd.to_datetime(dates)
        })

        for parameter in PARAMETERS:

            values = parameter_data[parameter]

            weather[parameter] = [
                values[d]
                for d in dates
            ]

        weather["district"] = district
        weather["state"] = state

        weather["latitude"] = latitude
        weather["longitude"] = longitude

        all_weather.append(weather)

        print(
            f"  Downloaded {len(weather)} daily records"
        )

    except Exception as e:

        print(
            f"  ERROR for {district}: {e}"
        )

        failures.append(
            (district, state, str(e))
        )

    # Small pause between requests
    time.sleep(0.5)
    

# ============================================================
# VALIDATE DOWNLOAD
# ============================================================

if failures:
    print("\n--- FAILED WEATHER REQUESTS ---")

    for district, state, error in failures:
        print(
            f"{district}, {state}: {error}"
        )

    raise SystemExit(1)

if not all_weather:
    raise SystemExit(
        "No weather data was downloaded."
    )

# ============================================================
# COMBINE
# ============================================================

print("\nCombining weather data...")

weather_df = pd.concat(
    all_weather,
    ignore_index=True
)
expected_districts = set(
    zip(
        districts["state"],
        districts["district"]
    )
)

actual_districts = set(
    zip(
        weather_df["state"],
        weather_df["district"]
    )
)

missing_districts = expected_districts - actual_districts

if missing_districts:
    print("\nMissing districts:")

    for state, district in sorted(missing_districts):
        print(
            f"  {district}, {state}"
        )

    raise SystemExit(1)


# ============================================================
# CLEAN / SORT
# ============================================================

weather_df = weather_df[
    [
        "date",
        "state",
        "district",
        "latitude",
        "longitude",
        "T2M",
        "RH2M",
        "WS2M",
        "PRECTOTCORR"
    ]
]

weather_df = weather_df.sort_values(
    ["date", "state", "district"]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

weather_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("WEATHER DATA VALIDATION")
print("=" * 60)

print(
    f"\nTotal rows: {len(weather_df):,}"
)

print(
    f"Districts: {weather_df['district'].nunique()}"
)

print(
    f"States: {weather_df['state'].nunique()}"
)

print(
    f"Date range: "
    f"{weather_df['date'].min().date()} → "
    f"{weather_df['date'].max().date()}"
)

print("\nMissing values:")

missing_values = weather_df[
    PARAMETERS
].isna().sum()

print(missing_values)

if missing_values.any():
    raise ValueError(
        "Weather dataset contains missing values:\n"
        f"{missing_values[missing_values > 0]}"
    )

print("\nWeather statistics:")

print(
    weather_df[
        PARAMETERS
    ].describe()
)

print("\nRows per district:")

print(
    weather_df.groupby(
        ["state", "district"]
    ).size().describe()
)

print("\nSaved to:")

print(OUTPUT_FILE)

print(
    "\nWeather download completed."
)