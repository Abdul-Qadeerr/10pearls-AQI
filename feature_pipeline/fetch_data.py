import os
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

AQICN_KEY = os.getenv("AQICN_API_KEY")
OW_KEY    = os.getenv("OPENWEATHER_API_KEY")
CITY_NAME = os.getenv("CITY_NAME", "Karachi")
LAT       = float(os.getenv("CITY_LAT", 24.8607))
LON       = float(os.getenv("CITY_LON", 67.0011))


def fetch_aqicn():
    url  = "https://api.waqi.info/feed/geo:{};{}/?token={}".format(LAT, LON, AQICN_KEY)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "ok":
        raise ValueError("AQICN error: {}".format(data.get("data")))
    feed = data["data"]
    iaqi = feed.get("iaqi", {})
    result = {
        "aqi":  feed.get("aqi"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3":   iaqi.get("o3",   {}).get("v"),
        "no2":  iaqi.get("no2",  {}).get("v"),
        "so2":  iaqi.get("so2",  {}).get("v"),
        "co":   iaqi.get("co",   {}).get("v"),
    }
    print("[AQICN] AQI={}  PM2.5={}  PM10={}".format(result["aqi"], result["pm25"], result["pm10"]))
    return result


def fetch_openweather():
    url  = "https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid={}&units=metric".format(LAT, LON, OW_KEY)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    result = {
        "temperature":  data["main"]["temp"],
        "humidity":     data["main"]["humidity"],
        "pressure":     data["main"]["pressure"],
        "wind_speed":   data["wind"]["speed"],
        "wind_deg":     data["wind"].get("deg", 0),
        "weather_desc": data["weather"][0]["description"],
        "clouds":       data["clouds"]["all"],
    }
    print("[OW] Temp={}C  Humidity={}%  Wind={}m/s".format(result["temperature"], result["humidity"], result["wind_speed"]))
    return result


def fetch_all():
    now        = datetime.now(timezone.utc)
    aqicn_data = fetch_aqicn()
    ow_data    = fetch_openweather()
    record = {
        "timestamp": now.isoformat(),
        "city":      CITY_NAME,
        **aqicn_data,
        **ow_data,
    }
    print("Record collected at {}".format(now.strftime("%Y-%m-%d %H:%M UTC")))
    return record


if __name__ == "__main__":
    import json
    print(json.dumps(fetch_all(), indent=2))