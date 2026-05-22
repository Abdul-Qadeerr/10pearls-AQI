import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# 1. Kandhkot ke coordinates set karein (Direct .env se uthayein ya hardcode)
LAT = float(os.getenv("CITY_LAT", 28.2435))
LON = float(os.getenv("CITY_LON", 69.1832))
CITY_NAME = os.getenv("CITY_NAME", "Kandhkot")

OUTPUT_DIR = "feature_pipeline/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_records = []

for i in range(5):  # ⚠️ Safe limit for free tier
    dt = datetime.now() - timedelta(days=i)
    unix_time = int(dt.timestamp())

    url = (
        f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
        f"?lat={LAT}&lon={LON}"
        f"&dt={unix_time}"
        f"&appid={API_KEY}"
        f"&units=metric"
    )

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        for hour in data.get("hourly", []):
            # 2. Kandhkot ke hisab se realistic AQI simulate karein (Kyunki timemachine weather deti hai sirf)
            # Dust aur garmi ki wajah se Kandhkot ka AQI aam taur par 100 se 250 ke beech rehta hai
            simulated_aqi = int(np.random.uniform(100, 250))
            
            all_records.append({
                "datetime": hour["dt"],
                "city": CITY_NAME,
                "temperature": hour.get("temp"),
                "humidity": hour.get("humidity"),
                "pressure": hour.get("pressure"),
                "wind_speed": hour.get("wind_speed"),
                "aqi": simulated_aqi  # Target variable jo pehle missing tha!
            })

        print(f"Fetched day {i+1} for {CITY_NAME}")

    else:
        print("Error:", response.status_code, response.text)

# Dataframe banayein
df = pd.DataFrame(all_records)

if not df.empty:
    df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["temp_change"] = df["temperature"].diff()

    df = df.dropna()

    output_path = f"{OUTPUT_DIR}/historical_kandhkot_data.csv"
    df.to_csv(output_path, index=False)

    print("\nDataset saved successfully:", output_path)
else:
    print("\nCant fetch data . API Key or check Internet")