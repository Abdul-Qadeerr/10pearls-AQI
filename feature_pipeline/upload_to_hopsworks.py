<<<<<<< HEAD
import os
import tempfile

# Windows fix: Hopsworks tries to use /tmp which doesn't exist on Windows
os.makedirs("/tmp", exist_ok=True)
os.environ["TMPDIR"] = "/tmp"
os.environ["TEMP"] = "/tmp"
os.environ["TMP"] = "/tmp"

=======
>>>>>>> 9b33817ba7e626b3796cebcd8f80b182a332677a
"""
upload_to_hopsworks.py
───────────────────────
Connects to Hopsworks Feature Store and:
  1. Creates the feature group (first run only — skipped if already exists)
  2. Uploads the engineered feature record as a new row

Feature group name : aqi_features
Version            : 1
Primary key        : timestamp + city
Event time         : timestamp
"""

<<<<<<< HEAD
=======
import os
>>>>>>> 9b33817ba7e626b3796cebcd8f80b182a332677a
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from feature_pipeline.compute_features import FEATURE_COLUMNS, TARGET_COLUMNS

<<<<<<< HEAD
load_dotenv(".env")

HOPSWORKS_API_KEY  = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT  = os.getenv("HOPSWORKS_PROJECT", "aqi_predictor_kk")
=======
load_dotenv()

HOPSWORKS_API_KEY  = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT  = os.getenv("HOPSWORKS_PROJECT", "aqi_predictor")
>>>>>>> 9b33817ba7e626b3796cebcd8f80b182a332677a
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VER  = 1


# ─── Connect to Hopsworks ─────────────────────────────────────────────────────

def get_feature_store():
    """Login to Hopsworks and return feature store object."""
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()
    print(f"✅ Connected to Hopsworks project: {HOPSWORKS_PROJECT}")
    return fs


# ─── Get or create feature group ─────────────────────────────────────────────

def get_or_create_feature_group(fs):
    """
    Returns the feature group. Creates it on first run.
    Schema is inferred automatically from the DataFrame on first insert.
    """
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VER,
        primary_key=["timestamp", "city"],
        event_time="timestamp",
        description="Hourly AQI + weather features with lag and time encodings",
<<<<<<< HEAD
        online_enabled=True,
=======
        online_enabled=True,       # enables real-time inference lookups
>>>>>>> 9b33817ba7e626b3796cebcd8f80b182a332677a
    )
    return fg


# ─── Upload one record ────────────────────────────────────────────────────────

def upload_record(feature_record: dict):
    """
    Takes the dict from compute_features() and inserts it into Hopsworks.
    """
<<<<<<< HEAD
    all_columns = ["timestamp", "city", "aqi"] + FEATURE_COLUMNS + TARGET_COLUMNS

=======
    # All columns we want to store
    all_columns = ["timestamp", "city", "aqi"] + FEATURE_COLUMNS + TARGET_COLUMNS

    # Build DataFrame with only existing keys (some may be None — that's fine)
>>>>>>> 9b33817ba7e626b3796cebcd8f80b182a332677a
    row = {col: feature_record.get(col) for col in all_columns}
    df  = pd.DataFrame([row])

    # Convert timestamp to datetime (Hopsworks needs datetime, not string)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

<<<<<<< HEAD
    # Fix: Hopsworks does not support null dtype columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("unknown").astype(str)
    else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
=======
>>>>>>> 9b33817ba7e626b3796cebcd8f80b182a332677a
    fs = get_feature_store()
    fg = get_or_create_feature_group(fs)

    fg.insert(df, write_options={"wait_for_job": True})

    print(f"✅ Record inserted into feature group '{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VER}")
    print(f"   timestamp={df['timestamp'].iloc[0]}  aqi={df['aqi'].iloc[0]}")


# ─── Fetch historical records (used by training pipeline) ────────────────────

def fetch_all_features() -> pd.DataFrame:
    """
    Reads ALL rows from the feature group back as a DataFrame.
    Used by training_pipeline to get the full historical dataset.
    """
    fs = get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VER)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"✅ Fetched {len(df)} rows from feature store")
    return df


# ─── Run standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from feature_pipeline.fetch_data import fetch_all
    from feature_pipeline.compute_features import compute_features

    # 1. Fetch raw data
    raw = fetch_all()

    # 2. Compute features (no history on first run)
    features = compute_features(raw, history=[])

    # 3. Upload to Hopsworks
<<<<<<< HEAD
    upload_record(features)
=======
    upload_record(features)
>>>>>>> 9b33817ba7e626b3796cebcd8f80b182a332677a
