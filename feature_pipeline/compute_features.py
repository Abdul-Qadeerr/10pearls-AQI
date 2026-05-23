"""
compute_features.py
-------------------
Takes the raw record produced by fetch_data.py and builds model-ready features.

Time-based features:
    hour, day_of_week, month, is_weekend
    hour_sin / hour_cos, day_sin / day_cos, month_sin / month_cos
    Cyclical (sin/cos) encoding prevents the model from treating hour 23
    and hour 0 as distant values.

Lag features (require historical records):
    aqi_lag_1h   - AQI reading one hour ago
    aqi_lag_3h   - AQI reading three hours ago
    aqi_lag_24h  - AQI reading twenty-four hours ago

Derived features:
    aqi_change_rate - percentage change in AQI compared to one hour ago
    aqi_rolling_3h  - rolling average of the last three hourly AQI values

Target columns (populated during backfill, set to None during live inference):
    aqi_next_1h, aqi_next_24h, aqi_next_48h, aqi_next_72h
"""

import math
import pandas as pd


# --- Time features -----------------------------------------------------------

def add_time_features(record: dict) -> dict:
    """
    Extracts calendar and cyclical time features from the record timestamp.
    """
    ts = pd.to_datetime(record["timestamp"])

    record["hour"]        = ts.hour
    record["day_of_week"] = ts.dayofweek   # 0 = Monday, 6 = Sunday
    record["month"]       = ts.month
    record["is_weekend"]  = int(ts.dayofweek >= 5)

    record["hour_sin"]  = round(math.sin(2 * math.pi * ts.hour / 24), 6)
    record["hour_cos"]  = round(math.cos(2 * math.pi * ts.hour / 24), 6)
    record["day_sin"]   = round(math.sin(2 * math.pi * ts.dayofweek / 7), 6)
    record["day_cos"]   = round(math.cos(2 * math.pi * ts.dayofweek / 7), 6)
    record["month_sin"] = round(math.sin(2 * math.pi * ts.month / 12), 6)
    record["month_cos"] = round(math.cos(2 * math.pi * ts.month / 12), 6)

    return record


# --- Lag and rolling features ------------------------------------------------

def add_lag_features(record: dict, history: list) -> dict:
    """
    Computes lag and rolling features using previously collected records.

    Parameters:
        record  - the current raw record
        history - list of past records sorted from oldest to newest

    If insufficient history is available, the corresponding values are set
    to None. This is handled gracefully during model training.
    """

    def get_aqi_n_hours_ago(n: int):
        target_ts = pd.to_datetime(record["timestamp"]) - pd.Timedelta(hours=n)
        for past in reversed(history):
            past_ts = pd.to_datetime(past["timestamp"])
            if abs((past_ts - target_ts).total_seconds()) <= 1800:
                return past.get("aqi")
        return None

    aqi_1h  = get_aqi_n_hours_ago(1)
    aqi_3h  = get_aqi_n_hours_ago(3)
    aqi_24h = get_aqi_n_hours_ago(24)

    record["aqi_lag_1h"]  = aqi_1h
    record["aqi_lag_3h"]  = aqi_3h
    record["aqi_lag_24h"] = aqi_24h

    current_aqi = record.get("aqi")
    if current_aqi and aqi_1h:
        record["aqi_change_rate"] = round((current_aqi - aqi_1h) / max(aqi_1h, 1), 4)
    else:
        record["aqi_change_rate"] = None

    recent_aqis = [r.get("aqi") for r in history[-2:]] + [current_aqi]
    valid_aqis  = [v for v in recent_aqis if v is not None]
    record["aqi_rolling_3h"] = round(sum(valid_aqis) / len(valid_aqis), 2) if valid_aqis else None

    return record


# --- Target labels -----------------------------------------------------------

def add_target_labels(record: dict, future_records: list) -> dict:
    """
    Assigns future AQI values as training targets.
    Used during the backfill process only.
    During live inference these fields remain None.

    Parameters:
        record         - the current record
        future_records - list of records collected after the current timestamp
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


# --- Main entry point --------------------------------------------------------

def compute_features(raw_record: dict, history: list = None) -> dict:
    """
    Applies all feature engineering steps to a single raw record.

    Parameters:
        raw_record - dictionary returned by fetch_data.fetch_all()
        history    - list of previous raw records used for lag computation.
                     Pass an empty list on the first run.

    Returns a feature-engineered record ready for upload to the feature store.
    """
    if history is None:
        history = []

    record = raw_record.copy()
    record = add_time_features(record)
    record = add_lag_features(record, history)

    record["aqi_next_1h"]  = None
    record["aqi_next_24h"] = None
    record["aqi_next_48h"] = None
    record["aqi_next_72h"] = None

    print(f"Features computed: hour={record['hour']}  weekend={record['is_weekend']}  "
          f"lag_1h={record['aqi_lag_1h']}  change_rate={record['aqi_change_rate']}  "
          f"rolling_3h={record['aqi_rolling_3h']}")

    return record


# --- Column reference --------------------------------------------------------

FEATURE_COLUMNS = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed", "wind_deg", "clouds",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    "aqi_change_rate", "aqi_rolling_3h",
]

TARGET_COLUMNS = ["aqi_next_1h", "aqi_next_24h", "aqi_next_48h", "aqi_next_72h"]


# --- Run standalone ----------------------------------------------------------

if __name__ == "__main__":
    import json
    from feature_pipeline.fetch_data import fetch_all

    raw      = fetch_all()
    features = compute_features(raw, history=[])
    print("\nFeature record:")
    print(json.dumps(features, indent=2, default=str))