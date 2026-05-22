"""
compute_features.py
────────────────────
Takes the raw record from fetch_data.py and engineers features:

  Time-based features:
    hour, day_of_week, month, is_weekend, hour_sin, hour_cos
    (sin/cos encoding avoids the jump from 23 → 0)

  Lag features (needs previous records):
    aqi_lag_1h   → AQI 1 hour ago
    aqi_lag_3h   → AQI 3 hours ago
    aqi_lag_24h  → AQI 24 hours ago

  Derived features:
    aqi_change_rate  → how fast AQI changed vs 1 hour ago
    aqi_rolling_3h   → 3-hour rolling average

  Target (for training):
    aqi_next_1h, aqi_next_24h, aqi_next_48h, aqi_next_72h
    (set to None for live inference — we don't know the future)
"""

import math
import pandas as pd
from datetime import datetime, timezone


# ─── Time features ────────────────────────────────────────────────────────────

def add_time_features(record: dict) -> dict:
    """Adds hour/day/month + cyclical encodings from the record's timestamp."""
    ts = pd.to_datetime(record["timestamp"])

    record["hour"]        = ts.hour
    record["day_of_week"] = ts.dayofweek          # 0=Monday … 6=Sunday
    record["month"]       = ts.month
    record["is_weekend"]  = int(ts.dayofweek >= 5)

    # Cyclical encoding — so model knows hour 23 is close to hour 0
    record["hour_sin"] = round(math.sin(2 * math.pi * ts.hour / 24), 6)
    record["hour_cos"] = round(math.cos(2 * math.pi * ts.hour / 24), 6)

    record["day_sin"]  = round(math.sin(2 * math.pi * ts.dayofweek / 7), 6)
    record["day_cos"]  = round(math.cos(2 * math.pi * ts.dayofweek / 7), 6)

    record["month_sin"] = round(math.sin(2 * math.pi * ts.month / 12), 6)
    record["month_cos"] = round(math.cos(2 * math.pi * ts.month / 12), 6)

    return record


# ─── Lag + rolling features ───────────────────────────────────────────────────

def add_lag_features(record: dict, history: list[dict]) -> dict:
    """
    history = list of past records (dicts), sorted oldest → newest.
    Adds aqi_lag_1h, aqi_lag_3h, aqi_lag_24h, aqi_change_rate, aqi_rolling_3h.

    If not enough history exists, fills with None (safe for training, skipped for inference).
    """

    def get_aqi_n_hours_ago(n: int):
        """Find the record closest to n hours ago."""
        target_ts = pd.to_datetime(record["timestamp"]) - pd.Timedelta(hours=n)
        for past in reversed(history):                  # newest first
            past_ts = pd.to_datetime(past["timestamp"])
            if abs((past_ts - target_ts).total_seconds()) <= 1800:  # ±30 min window
                return past.get("aqi")
        return None

    aqi_1h  = get_aqi_n_hours_ago(1)
    aqi_3h  = get_aqi_n_hours_ago(3)
    aqi_24h = get_aqi_n_hours_ago(24)

    record["aqi_lag_1h"]  = aqi_1h
    record["aqi_lag_3h"]  = aqi_3h
    record["aqi_lag_24h"] = aqi_24h

    # AQI change rate: (current - 1h ago) / 1h ago  (% change)
    current_aqi = record.get("aqi")
    if current_aqi and aqi_1h:
        record["aqi_change_rate"] = round((current_aqi - aqi_1h) / max(aqi_1h, 1), 4)
    else:
        record["aqi_change_rate"] = None

    # 3-hour rolling average (current + last 2 hourly records)
    recent_aqis = [r.get("aqi") for r in history[-2:]] + [current_aqi]
    valid_aqis  = [v for v in recent_aqis if v is not None]
    record["aqi_rolling_3h"] = round(sum(valid_aqis) / len(valid_aqis), 2) if valid_aqis else None

    return record


# ─── Target labels (for training only) ───────────────────────────────────────

def add_target_labels(record: dict, future_records: list[dict]) -> dict:
    """
    For training: look into future records to set target AQI values.
    For live inference: leave as None.

    future_records = list of records AFTER this record's timestamp.
    """

    def get_aqi_n_hours_later(n: int):
        target_ts = pd.to_datetime(record["timestamp"]) + pd.Timedelta(hours=n)
        for fut in future_records:
            fut_ts = pd.to_datetime(fut["timestamp"])
            if abs((fut_ts - target_ts).total_seconds()) <= 1800:
                return fut.get("aqi")
        return None

    record["aqi_next_1h"]  = get_aqi_n_hours_later(1)
    record["aqi_next_24h"] = get_aqi_n_hours_later(24)
    record["aqi_next_48h"] = get_aqi_n_hours_later(48)
    record["aqi_next_72h"] = get_aqi_n_hours_later(72)

    return record


# ─── Full pipeline for a single record ───────────────────────────────────────

def compute_features(raw_record: dict, history: list[dict] = None) -> dict:
    """
    Master function. Call this after fetch_all().

    Args:
        raw_record : dict from fetch_data.fetch_all()
        history    : list of past raw records (for lag features). Pass [] if first run.

    Returns:
        Feature-engineered record ready for Hopsworks upload.
    """
    if history is None:
        history = []

    record = raw_record.copy()
    record = add_time_features(record)
    record = add_lag_features(record, history)

    # Target labels are None during live pipeline (filled during backfill)
    record["aqi_next_1h"]  = None
    record["aqi_next_24h"] = None
    record["aqi_next_48h"] = None
    record["aqi_next_72h"] = None

    print(f"\n✅ Features computed:")
    print(f"   hour={record['hour']}  day={record['day_of_week']}  is_weekend={record['is_weekend']}")
    print(f"   aqi_lag_1h={record['aqi_lag_1h']}  aqi_change_rate={record['aqi_change_rate']}")
    print(f"   aqi_rolling_3h={record['aqi_rolling_3h']}")

    return record


# ─── All features list (for reference in training) ───────────────────────────

FEATURE_COLUMNS = [
    # Pollutants
    "pm25", "pm10", "o3", "no2", "so2", "co",
    # Weather
    "temperature", "humidity", "pressure", "wind_speed", "wind_deg", "clouds",
    # Time
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
    # Lags
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    # Derived
    "aqi_change_rate", "aqi_rolling_3h",
]

TARGET_COLUMNS = ["aqi_next_1h", "aqi_next_24h", "aqi_next_48h", "aqi_next_72h"]


# ─── Run standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from feature_pipeline.fetch_data import fetch_all

    raw = fetch_all()
    features = compute_features(raw, history=[])
    print("\n── Feature record ──")
    print(json.dumps(features, indent=2, default=str))
