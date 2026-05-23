"""
inference/predict.py
--------------------
Loads the trained model and scaler from the Hopsworks Model Registry,
fetches the latest feature record from the Feature Store, and returns
AQI predictions for the next 24h, 48h, and 72h.

Called by the Streamlit dashboard to populate the forecast charts.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import hopsworks
import tempfile

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY  = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT  = os.getenv("HOPSWORKS_PROJECT", "aqi_predictor")
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VER  = 1
MODEL_NAME         = "aqi_forecast_model"

FEATURE_COLUMNS = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed", "wind_deg", "clouds",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    "aqi_change_rate", "aqi_rolling_3h",
]

TARGET_COLUMNS = ["aqi_next_24h", "aqi_next_48h", "aqi_next_72h"]

# Local cache path so we do not re-download on every dashboard refresh
_MODEL_CACHE_DIR = Path(tempfile.gettempdir()) / "aqi_model_cache"


# ---------------------------------------------------------------------------
# Download model from registry
# ---------------------------------------------------------------------------

def download_model() -> Path:
    """
    Downloads the latest registered model from Hopsworks to a local temp folder.
    Returns the path to the downloaded model directory.
    Skips download if a cached version already exists.
    """
    if _MODEL_CACHE_DIR.exists() and any(_MODEL_CACHE_DIR.iterdir()):
        print("Using cached model artifacts.")
        return _MODEL_CACHE_DIR

    print("Downloading model from Hopsworks Model Registry...")
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    mr    = project.get_model_registry()
    model = mr.get_best_model(name=MODEL_NAME, metric="rmse", direction="min")

    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model.download(_MODEL_CACHE_DIR)
    print(f"Model v{model.version} downloaded.")
    return _MODEL_CACHE_DIR


# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------

def load_model_artifacts():
    """
    Reads model_info.json to determine which model type was saved (rf or lstm),
    then loads the appropriate artifact(s) and the scaler.

    Returns:
        model_type  - 'rf' or 'lstm'
        models      - dict of {target: sklearn model} for RF, or Keras model for LSTM
        scaler      - fitted StandardScaler
        model_info  - full metadata dict
    """
    model_dir = download_model()

    with open(model_dir / "model_info.json") as f:
        model_info = json.load(f)

    model_type = model_info.get("model_type", "rf")
    scaler     = joblib.load(model_dir / "scaler.pkl")

    if model_type == "rf":
        models = {}
        for target in TARGET_COLUMNS:
            suffix = target.split("_")[-1]
            path   = model_dir / f"model_rf_{suffix}.pkl"
            models[target] = joblib.load(path)
    else:
        from tensorflow.keras.models import load_model
        models = load_model(model_dir / "model_lstm.keras")

    return model_type, models, scaler, model_info


# ---------------------------------------------------------------------------
# Fetch latest features
# ---------------------------------------------------------------------------

def get_latest_features() -> pd.Series:
    """
    Reads the most recent row from the Hopsworks Feature Store.
    This represents the current conditions used as model input.
    """
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VER)
    df = fg.read()
    df = df.sort_values("timestamp").reset_index(drop=True)
    latest = df.iloc[-1]
    print(f"Latest feature row: {latest.get('timestamp')}  AQI={latest.get('aqi')}")
    return latest


# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------

def predict() -> dict:
    """
    End-to-end inference function.

    1. Loads the trained model and scaler.
    2. Fetches the most recent feature row from Hopsworks.
    3. Scales the features and runs inference.
    4. Returns a dict:
       {
           "current_aqi":  float,
           "aqi_next_24h": float,
           "aqi_next_48h": float,
           "aqi_next_72h": float,
           "model_type":   str,
           "timestamp":    str,
       }
    """
    model_type, models, scaler, model_info = load_model_artifacts()
    latest = get_latest_features()

    # Build feature vector — fill missing values with 0
    feature_vector = np.array([
        float(latest.get(col, 0) or 0) for col in FEATURE_COLUMNS
    ]).reshape(1, -1)

    feature_scaled = scaler.transform(feature_vector)

    if model_type == "rf":
        preds = {
            target: float(models[target].predict(feature_scaled)[0])
            for target in TARGET_COLUMNS
        }
    else:
        # LSTM expects shape (1, 1, features)
        x_lstm = feature_scaled.reshape(1, 1, feature_scaled.shape[1])
        raw    = models.predict(x_lstm, verbose=0)[0]
        preds  = {TARGET_COLUMNS[i]: float(raw[i]) for i in range(len(TARGET_COLUMNS))}

    result = {
        "current_aqi":  float(latest.get("aqi", 0) or 0),
        "timestamp":    str(latest.get("timestamp", "")),
        "model_type":   model_type.upper(),
        **preds,
    }

    print(f"Forecast -> 24h: {preds['aqi_next_24h']:.1f}  "
          f"48h: {preds['aqi_next_48h']:.1f}  72h: {preds['aqi_next_72h']:.1f}")
    return result


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = predict()
    print("\nPrediction result:")
    for k, v in result.items():
        print(f"  {k}: {v}")