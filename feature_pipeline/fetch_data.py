"""
fetch_data.py
─────────────
Fetches raw AQI + weather data from:
  1. AQICN   → real AQI + individual pollutant levels (PM2.5, PM10, O3, NO2, SO2, CO)
  2. OpenWeather → temperature, humidity, wind speed, pressure

Returns a single merged dictionary for the current hour.
"""

import os
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

AQICN_KEY      = os.getenv("AQICN_API_KEY")
OW_KEY         = os.getenv("OPENWEATHER_API_KEY")
CITY_NAME      = os.getenv("CITY_NAME", "Karachi")
<<<<<<< HEAD
LAT            = float(os.getenv("CITY_LAT", 28.2435))
LON            = float(os.getenv("CITY_LON", 69.1832))
=======
LAT            = float(os.getenv("CITY_LAT", 24.8607))
LON            = float(os.getenv("CITY_LON", 67.0011))
>>>>>>> 9b33817ba7e626b3796cebcd8f80b182a332677a


# ─── AQICN ────────────────────────────────────────────────────────────────────

def fetch_aqicn() -> dict:
    """
    Calls AQICN feed API for the given city.
    Returns a flat dict with AQI and available pollutant levels.
    """
    url = f"https://api.waqi.info/feed/{CITY_NAME}/?token={AQICN_KEY}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        raise ValueError(f"AQICN error: {data.get('data')}")

    feed = data["data"]
    iaqi = feed.get("iaqi", {})   # individual AQI components

    result = {
        "aqi":        feed.get("aqi"),            # overall AQI (target variable)
        "pm25":       iaqi.get("pm25", {}).get("v"),
        "pm10":       iaqi.get("pm10", {}).get("v"),
        "o3":         iaqi.get("o3",   {}).get("v"),
        "no2":        iaqi.get("no2",  {}).get("v"),
        "so2":        iaqi.get("so2",  {}).get("v"),
        "co":         iaqi.get("co",   {}).get("v"),
    }
    print(f"[AQICN]  AQI={result['aqi']}  PM2.5={result['pm25']}  PM10={result['pm10']}")
    return result


# ─── OpenWeather ──────────────────────────────────────────────────────────────

def fetch_openweather() -> dict:
    """
    Calls OpenWeather Current Weather API.
    Returns temperature, humidity, wind, pressure, weather description.
    """
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={LAT}&lon={LON}&appid={OW_KEY}&units=metric"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    result = {
        "temperature":   data["main"]["temp"],
        "humidity":      data["main"]["humidity"],
        "pressure":      data["main"]["pressure"],
        "wind_speed":    data["wind"]["speed"],
        "wind_deg":      data["wind"].get("deg", 0),
        "weather_desc":  data["weather"][0]["description"],
        "clouds":        data["clouds"]["all"],          # cloud coverage %
    }
    print(f"[OW]     Temp={result['temperature']}°C  Humidity={result['humidity']}%  Wind={result['wind_speed']}m/s")
    return result


# ─── Merge ────────────────────────────────────────────────────────────────────

def fetch_all() -> dict:
    """
    Merges both API responses into one flat record stamped with current UTC time.
    This is the raw record that goes into the feature pipeline next.
    """
    now = datetime.now(timezone.utc)

    aqicn_data = fetch_aqicn()
    ow_data    = fetch_openweather()

    record = {
        "timestamp":  now.isoformat(),
        "city":       CITY_NAME,
        **aqicn_data,
        **ow_data,
    }

    print(f"\n✅ Raw record ready @ {now.strftime('%Y-%m-%d %H:%M UTC')}")
    return record


# ─── Run standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    record = fetch_all()
    print("\n── Full record ──")
    print(json.dumps(record, indent=2))
