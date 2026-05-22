import os
import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Path ko sahi karne ke liye hum check karte hain (.env agar bahar hai to wahan se load karein)
if os.path.exists("../.env"):
    load_dotenv("../.env")
else:
    load_dotenv()

# Hopsworks key check karein
api_key = os.getenv("hopaworks") or os.getenv("HOPSWORKS_API_KEY")

if api_key:
    os.environ["HOPSWORKS_API_KEY"] = api_key
    print("✅ .env file se API Key successfully load ho gayi hai!")
else:
    print("⚠️ .env mein key nahi mili. Hopsworks abhi aapse terminal par manual key maangega.")

# 1. Kandhkot ka data generate karein (Memory mein)
print("🔄 Kandhkot ka historical data generate ho raha hai...")
CITY_NAME = "Kandhkot"

all_records = []
for i in range(5):  # 5 days of data
    dt = datetime.now() - timedelta(days=i)
    for h in range(24):
        record_dt = dt - timedelta(hours=h)
        simulated_aqi = int(np.random.uniform(100, 250))
        all_records.append({
            "datetime": record_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "city": CITY_NAME,
            "temperature": float(np.random.uniform(25, 42)),
            "humidity": float(np.random.uniform(40, 80)),
            "pressure": float(np.random.uniform(1000, 1015)),
            "wind_speed": float(np.random.uniform(1, 8)),
            "aqi": simulated_aqi
        })

df = pd.DataFrame(all_records)
print(f"✅ Data ready! Rows: {len(df)}")

print(f"🔄 Hopsworks se connect ho raha hai...")
# 2. Hopsworks Login
project = hopsworks.login()
fs = project.get_feature_store()

# 3. Kandhkot Feature Group
aqi_fg = fs.get_or_create_feature_group(
    name="aqi_kandhkot_fg",
    version=1,
    primary_key=['datetime'],
    description="Historical Weather and Simulated AQI data for Kandhkot"
)

print("🚀 Data Feature Store mein upload ho raha hai... Isme 1 minute lag sakta hai...")
aqi_fg.insert(df)
print("🎉 Mubarak ho! Kandhkot ka Phase 3 Backfill data successfully upload ho gaya hai.")
