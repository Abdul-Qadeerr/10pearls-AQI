import os
import hopsworks
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Hopsworks API Key (.env se load hogi)
HOPSWORKS_API_KEY = os.getenv("hopaworks")
csv_path = "feature_pipeline/data/historical_kandhkot_data.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"⚠️ CSV file nahi mili: {csv_path}. Pehle data fetch wali script chalayein!")

df = pd.read_csv(csv_path)
df['datetime'] = df['datetime'].astype(str)

print(f"🔄 Hopsworks se connect ho raha hai... Rows to upload: {len(df)}")

# Hopsworks Login
project = hopsworks.login(api_key=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

# Kandhkot ke liye NAYA Feature Group
aqi_fg = fs.get_or_create_feature_group(
    name="aqi_kandhkot_fg",
    version=1,
    primary_key=['datetime'],
    description="Historical Weather and Simulated AQI data for Kandhkot"
)

print("🚀 Data Feature Store mein upload ho raha hai...")
aqi_fg.insert(df)
print("Kandhkot's Backfill data has beensuccessfully upload.")