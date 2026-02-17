import requests
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
AIR_POLLUTION_API = os.getenv("AIR_POLLUTION_API")
MONGO_URI = os.getenv("MONGO_URI")  # FIX: load from .env, not hardcoded

# Coordinates (Karachi)
LAT, LON = 24.8607, 67.0011

# MongoDB connection
client = MongoClient(MONGO_URI)
db = client["aqi_database"]
features_col = db["historical_pollutants"]


def fetch_pollution_data(start_date, end_date):
    """Fetch pollution data from OpenWeather API."""
    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
        f"lat={LAT}&lon={LON}&start={start_unix}&end={end_unix}&appid={AIR_POLLUTION_API}"
    )

    response = requests.get(url, timeout=30)
    data = response.json()

    if "list" not in data:
        print("No data found")
        return pd.DataFrame()

    records = []
    for entry in data["list"]:
        dt = datetime.utcfromtimestamp(entry["dt"])  # FIX: use UTC explicitly
        main = entry["main"]
        components = entry["components"]
        records.append({
            "timestamp": dt,
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "co": components.get("co"),
            "no2": components.get("no2"),
            "so2": components.get("so2"),
            "o3": components.get("o3"),
            "nh3": components.get("nh3"),
            "aqi": main.get("aqi")
        })

    df = pd.DataFrame(records)
    return df


def store_data_in_mongo(df):
    """Store pollution data in MongoDB, avoiding duplicates."""
    if not df.empty:
        inserted = 0
        for record in df.to_dict("records"):
            if features_col.count_documents({"timestamp": record["timestamp"]}, limit=1) == 0:
                features_col.insert_one(record)
                inserted += 1
        print(f"{inserted} new records inserted into MongoDB")
    else:
        print("No data to insert")


if __name__ == "__main__":
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)

    df = fetch_pollution_data(start_date, end_date)
    print(df.head())
    store_data_in_mongo(df)