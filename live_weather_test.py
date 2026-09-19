import pandas as pd
import requests
import time

# ============================================================
# STUBBLEAI - 2026 WEATHER FORECAST TEST
# ============================================================

DISTRICT = "Amritsar"
FORECAST_DAYS = 3


print("=" * 70)
print("STUBBLEAI 2026 WEATHER FORECAST TEST")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load district coordinates
# ------------------------------------------------------------

coords = pd.read_csv("district_coordinates.csv")
required_columns = [
    "district",
    "state",
    "latitude",
    "longitude",
]

missing_columns = [
    col for col in required_columns
    if col not in coords.columns
]

if missing_columns:
    raise ValueError(
        f"Coordinate file is missing required columns: "
        f"{missing_columns}"
    )

if coords.empty:
    raise ValueError(
        "District coordinate file is empty."
    )

print("\nCoordinate columns:")
print(list(coords.columns))


# ------------------------------------------------------------
# 2. Find Amritsar
# ------------------------------------------------------------

row = coords[
    coords["district"].astype(str).str.strip() == DISTRICT
]

if row.empty:
    raise ValueError(
        f"District '{DISTRICT}' not found in district_coordinates.csv"
    )

latitude = float(row.iloc[0]["latitude"])
longitude = float(row.iloc[0]["longitude"])

print(f"\nDistrict: {DISTRICT}")
print(f"Latitude: {latitude}")
print(f"Longitude: {longitude}")


# ------------------------------------------------------------
# 3. Request forecast
# ------------------------------------------------------------

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "daily": ",".join([
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_mean",
        "precipitation_sum",
    ]),
    "forecast_days": FORECAST_DAYS,
    "timezone": "Asia/Kolkata",
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",
}

response = None

for attempt in range(3):
    try:
        response = requests.get(
            url,
            params=params,
            timeout=30,
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

print("\nHTTP status:", response.status_code)

data = response.json()

if "daily" not in data:
    raise ValueError(
        "Open-Meteo response does not contain daily forecast data."
    )

data = response.json()

if "daily" not in data:
    raise ValueError(
        "Open-Meteo response does not contain daily forecast data."
    )

daily = data["daily"]

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


# ------------------------------------------------------------
# 4. Build forecast dataframe
# ------------------------------------------------------------

weather = pd.DataFrame({
    "date": daily["time"],
    "T2M": daily["temperature_2m_mean"],
    "RH2M": daily["relative_humidity_2m_mean"],
    "WS2M": daily["wind_speed_10m_mean"],
    "PRECTOTCORR": daily["precipitation_sum"],
})

weather["date"] = pd.to_datetime(
    weather["date"],
    errors="raise"
)

if weather.empty:
    raise ValueError(
        "Open-Meteo returned no forecast records."
    )

weather_columns = [
    "T2M",
    "RH2M",
    "WS2M",
    "PRECTOTCORR",
]

missing_values = weather[weather_columns].isna().sum()

if missing_values.any():
    raise ValueError(
        "Forecast contains missing weather values:\n"
        f"{missing_values[missing_values > 0]}"
    )

if len(weather) != FORECAST_DAYS:
    raise ValueError(
        f"Expected {FORECAST_DAYS} forecast days, "
        f"but received {len(weather)}."
    )


print("\nForecast:")
print(weather.to_string(index=False))


# ------------------------------------------------------------
# 5. Save test result
# ------------------------------------------------------------

weather.to_csv(
    "live_weather_test.csv",
    index=False
)

print("\nSaved:")
print("live_weather_test.csv")

print("\n" + "=" * 70)
print("WEATHER FORECAST TEST COMPLETE")
print("=" * 70)