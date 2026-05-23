"""
upload_to_hopsworks.py
----------------------
Connects to the Hopsworks Feature Store and performs two operations:
  1. Creates the feature group on the first run (skipped if it already exists).
  2. Inserts the engineered feature record as a new row.

Feature group : aqi_features  (version 1)
Primary key   : timestamp + city
Event time    : timestamp
"""

import os
import tempfile
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from feature_pipeline.compute_features import FEATURE_COLUMNS, TARGET_COLUMNS

os.makedirs(tempfile.gettempdir(), exist_ok=True)

load_dotenv()

HOPSWORKS_API_KEY  = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT  = os.getenv("HOPSWORKS_PROJECT", "aqi_predictor")
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VER  = 1


# --- Connect to Hopsworks ----------------------------------------------------

def get_feature_store():
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()
    print("Connected to Hopsworks project: {}".format(HOPSWORKS_PROJECT))
    return fs


# --- Get or create the feature group -----------------------------------------

def get_or_create_feature_group(fs):
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VER,
        primary_key=["timestamp", "city"],
        event_time="timestamp",
        description="Hourly AQI and weather features including lag and cyclical time encodings",
        online_enabled=True,
    )
    return fg


# --- Insert a single record --------------------------------------------------

def upload_record(feature_record: dict):
    all_columns = ["timestamp", "city", "aqi"] + FEATURE_COLUMNS + TARGET_COLUMNS
    row = {col: feature_record.get(col) for col in all_columns}
    df  = pd.DataFrame([row])

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Drop string columns that are not needed in the feature store
    df = df.drop(columns=["weather_desc"], errors="ignore")

    # Convert all remaining non-timestamp, non-city columns to numeric
    for col in df.columns:
        if col in ("timestamp", "city"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)

    fg.insert(df, write_options={"wait_for_job": True})

    print("Record inserted into '{}' v{}".format(FEATURE_GROUP_NAME, FEATURE_GROUP_VER))
    print("Timestamp: {}  AQI: {}".format(df["timestamp"].iloc[0], df["aqi"].iloc[0]))


# --- Read all stored features ------------------------------------------------

def fetch_all_features() -> pd.DataFrame:
    fs = get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VER)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    print("Retrieved {} rows from the feature store".format(len(df)))
    return df


# --- Run standalone ----------------------------------------------------------

if __name__ == "__main__":
    from feature_pipeline.fetch_data import fetch_all
    from feature_pipeline.compute_features import compute_features

    raw      = fetch_all()
    features = compute_features(raw, history=[])
    upload_record(features)