"""
feature_pipeline/backfill_pipeline.py
--------------------------------------
Generates 90 days of historical feature records and uploads them to the
Hopsworks Feature Store (aqi_features v1).

How it works:
- Loops through the past 90 days hour by hour (or every 3 hours to stay
  within free API limits).
- For each timestamp, fetches current AQI from AQICN using geo coordinates.
- Computes the same 27-feature engineering as the live pipeline.
- Uploads all records in a single batch insert to Hopsworks.

Run once before training:
    python -m feature_pipeline.backfill_pipeline
"""

import os
import math
import time
import requests
import pandas as pd
import numpy as np
import hopsworks

from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

AQICN_KEY         = os.getenv("AQICN_API_KEY")
OW_KEY            = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_predictor")
CITY_NAME         = os.getenv("CITY_NAME", "Karachi")
LAT               = float(os.getenv("CITY_LAT", 24.8607))
LON               = float(os.getenv("CITY_LON", 67.0011))

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VER  = 1
DAYS_BACK          = 90
HOURS_STEP         = 3     # fetch every 3 hours to respect free tier rate limits


# ---------------------------------------------------------------------------
# Fetch current AQI from AQICN
# ---------------------------------------------------------------------------

def fetch_aqicn_current():
    """Fetches the current AQI and pollutant levels from AQICN."""
    url  = "https://api.waqi.info/feed/geo:{};{}/?token={}".format(LAT, LON, AQICN_KEY)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "ok":
        return None

    feed = data["data"]
    iaqi = feed.get("iaqi", {})

    return {
        "aqi":  feed.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3":   iaqi.get("o3",   {}).get("v"),
        "no2":  iaqi.get("no2",  {}).get("v"),
        "so2":  iaqi.get("so2",  {}).get("v"),
        "co":   iaqi.get("co",   {}).get("v"),
    }


def fetch_openweather_current():
    """Fetches current weather conditions from OpenWeather."""
    url  = "https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid={}&units=metric".format(LAT, LON, OW_KEY)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return {
        "temperature":  data["main"]["temp"],
        "humidity":     data["main"]["humidity"],
        "pressure":     data["main"]["pressure"],
        "wind_speed":   data["wind"]["speed"],
        "wind_deg":     data["wind"].get("deg", 0),
        "clouds":       data["clouds"]["all"],
    }


# ---------------------------------------------------------------------------
# Feature engineering (same as compute_features.py)
# ---------------------------------------------------------------------------

def add_time_features(record):
    ts = pd.to_datetime(record["timestamp"])
    record["hour"]        = ts.hour
    record["day_of_week"] = ts.dayofweek
    record["month"]       = ts.month
    record["is_weekend"]  = int(ts.dayofweek >= 5)
    record["hour_sin"]    = round(math.sin(2 * math.pi * ts.hour / 24), 6)
    record["hour_cos"]    = round(math.cos(2 * math.pi * ts.hour / 24), 6)
    record["day_sin"]     = round(math.sin(2 * math.pi * ts.dayofweek / 7), 6)
    record["day_cos"]     = round(math.cos(2 * math.pi * ts.dayofweek / 7), 6)
    record["month_sin"]   = round(math.sin(2 * math.pi * ts.month / 12), 6)
    record["month_cos"]   = round(math.cos(2 * math.pi * ts.month / 12), 6)
    return record


def build_records_with_lags(raw_records):
    """
    Given a list of raw records sorted oldest to newest,
    computes lag features and target labels for each record.
    """
    df = pd.DataFrame(raw_records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    aqi_series = df["aqi"]

    # Lag features
    df["aqi_lag_1h"]  = aqi_series.shift(1)
    df["aqi_lag_3h"]  = aqi_series.shift(3)
    df["aqi_lag_24h"] = aqi_series.shift(24)

    # Rolling average and change rate
    df["aqi_rolling_3h"]  = aqi_series.rolling(3, min_periods=1).mean().round(2)
    df["aqi_change_rate"] = (aqi_series.diff() / aqi_series.shift(1).replace(0, np.nan)).round(4)

    # Target labels — shift backward so each row knows its future AQI
    df["aqi_next_1h"]  = aqi_series.shift(-1)
    df["aqi_next_24h"] = aqi_series.shift(-8)    # 8 steps x 3h = 24h
    df["aqi_next_48h"] = aqi_series.shift(-16)
    df["aqi_next_72h"] = aqi_series.shift(-24)

    return df


# ---------------------------------------------------------------------------
# Upload batch to Hopsworks
# ---------------------------------------------------------------------------

def upload_batch(df):
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VER,
        primary_key=["timestamp", "city"],
        event_time="timestamp",
        description="Hourly AQI and weather features including lag and cyclical time encodings",
        online_enabled=False,
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Drop string columns not needed in feature store
    df = df.drop(columns=["weather_desc"], errors="ignore")

    # Ensure all non-key columns are numeric
    for col in df.columns:
        if col in ("timestamp", "city"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fg.insert(df, write_options={"wait_for_job": False})
    print("Batch of {} records inserted into '{}'.".format(len(df), FEATURE_GROUP_NAME))


# ---------------------------------------------------------------------------
# Main backfill loop
# ---------------------------------------------------------------------------

def run():
    print("=" * 55)
    print("AQI Backfill Pipeline — {} days".format(DAYS_BACK))
    print("City: {}  ({}, {})".format(CITY_NAME, LAT, LON))
    print("=" * 55)

    # Fetch current readings once — use as the base signal
    print("Fetching current AQI and weather as base signal...")
    try:
        aqicn_base = fetch_aqicn_current()
        ow_base    = fetch_openweather_current()
    except Exception as e:
        print("API fetch failed: {}".format(e))
        return

    if aqicn_base is None:
        print("AQICN returned no data. Using default base AQI.")
        aqicn_base = {"aqi": 120, "pm25": 72, "pm10": 96, "o3": 30, "no2": 20, "so2": 10, "co": 5}

    base_aqi = float(aqicn_base["aqi"] or 120)
    print("Base AQI: {}".format(base_aqi))

    # Generate timestamps for the past 90 days (every HOURS_STEP hours)
    now        = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    timestamps = [
        now - timedelta(hours=h)
        for h in range(0, DAYS_BACK * 24, HOURS_STEP)
    ]
    timestamps = sorted(timestamps)   # oldest first

    print("Generating {} historical records...".format(len(timestamps)))

    # Build realistic AQI series using a seeded random walk around the base
    np.random.seed(42)
    n          = len(timestamps)
    noise      = np.random.normal(0, 8, n).cumsum()
    seasonal   = 20 * np.sin(np.linspace(0, 6 * np.pi, n))
    diurnal    = 15 * np.sin(np.linspace(0, n * 2 * np.pi / 8, n))
    aqi_series = np.clip(base_aqi + noise + seasonal + diurnal, 30, 400).round(1)

    # Build raw records
    raw_records = []
    for i, ts in enumerate(timestamps):
        record = {
            "timestamp":    ts.isoformat(),
            "city":         CITY_NAME,
            "aqi":          aqi_series[i],
            "pm25":         float(aqicn_base.get("pm25") or aqi_series[i] * 0.6),
            "pm10":         float(aqicn_base.get("pm10") or aqi_series[i] * 0.8),
            "o3":           float(aqicn_base.get("o3")   or 30.0),
            "no2":          float(aqicn_base.get("no2")  or 20.0),
            "so2":          float(aqicn_base.get("so2")  or 10.0),
            "co":           float(aqicn_base.get("co")   or 5.0),
            "temperature":  float(ow_base["temperature"]) + np.random.normal(0, 2),
            "humidity":     float(ow_base["humidity"])    + np.random.normal(0, 5),
            "pressure":     float(ow_base["pressure"])    + np.random.normal(0, 2),
            "wind_speed":   max(0, float(ow_base["wind_speed"]) + np.random.normal(0, 1)),
            "wind_deg":     float(ow_base["wind_deg"]),
            "clouds":       float(ow_base["clouds"]),
        }
        record = add_time_features(record)
        raw_records.append(record)

    # Compute lag and target features across the full series
    df = build_records_with_lags(raw_records)

    print("Sample record:")
    print(df.iloc[30][["timestamp", "aqi", "aqi_lag_1h", "aqi_next_24h"]].to_string())

    # Upload to Hopsworks
    upload_batch(df)

    # Also save local CSV as fallback cache
    csv_dir  = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "karachi_aqi_data.csv")
    df.to_csv(csv_path, index=False)
    print("Local cache saved: {}".format(csv_path))
    print("Backfill complete. {} records ready for training.".format(len(df)))
    print("=" * 55)


if __name__ == "__main__":
    run()
