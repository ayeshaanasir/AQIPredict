import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
from pymongo import MongoClient
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
AIR_POLLUTION_API = os.getenv("AIR_POLLUTION_API")
MONGO_URI = os.getenv("MONGO_URI")

# Coordinates & timezone
LAT, LON = 24.8607, 67.0011  # Karachi
TIMEZONE = "Asia/Karachi"


def connect_to_mongodb():
    """Connect to MongoDB and return database"""
    try:
        client = MongoClient(MONGO_URI)
        db = client["aqi_database"]
        db.command('ping')
        logger.info("✓ Connected to MongoDB")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


def fetch_pollution_data(start_date, end_date):
    """Fetch pollution data from OpenWeather API"""
    try:
        start_unix = int(start_date.timestamp())
        end_unix = int(end_date.timestamp())

        url = (
            f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
            f"lat={LAT}&lon={LON}&start={start_unix}&end={end_unix}&appid={AIR_POLLUTION_API}"
        )

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "list" not in data:
            logger.warning("No pollutant data found in API response")
            return pd.DataFrame()

        records = []
        for entry in data["list"]:
            dt = datetime.fromtimestamp(entry["dt"], tz=timezone.utc).replace(tzinfo=None)
            main = entry["main"]
            components = entry["components"]
            records.append({
                "timestamp": dt,
                "pm2_5": float(components.get("pm2_5", 0)),
                "pm10": float(components.get("pm10", 0)),
                "co": float(components.get("co", 0)),
                "no2": float(components.get("no2", 0)),
                "so2": float(components.get("so2", 0)),
                "o3": float(components.get("o3", 0)),
                "nh3": float(components.get("nh3", 0)),
                "aqi": int(main.get("aqi", 0))
            })

        df = pd.DataFrame(records)
        logger.info(f"✓ Fetched {len(df)} pollution records")
        return df

    except Exception as e:
        logger.error(f"Error fetching pollution data: {e}")
        return pd.DataFrame()


def fetch_weather_data(start_date, end_date):
    """Fetch weather data from Open-Meteo API"""
    try:
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        weather_url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={LAT}&longitude={LON}"
            f"&start_date={start_date_str}&end_date={end_date_str}"
            "&hourly=temperature_2m,relative_humidity_2m,"
            "pressure_msl,windspeed_10m,winddirection_10m,precipitation"
            f"&timezone={TIMEZONE}"
        )

        response = requests.get(weather_url, timeout=30)
        response.raise_for_status()

        data = response.json()
        weather_df = pd.DataFrame(data["hourly"])

        weather_df.rename(columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "pressure_msl": "pressure",
            "windspeed_10m": "windspeed",
            "winddirection_10m": "winddirection",
            "precipitation": "precipitation"
        }, inplace=True)

        weather_df["time"] = pd.to_datetime(weather_df["time"]).dt.tz_localize(None)

        weather_cols = ['temperature', 'humidity', 'pressure', 'windspeed', 'winddirection', 'precipitation']
        for col in weather_cols:
            weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce').fillna(0.0).astype(float)

        logger.info(f"✓ Fetched {len(weather_df)} weather records")
        return weather_df

    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return pd.DataFrame()


def create_features(pollution_df, weather_df):
    """Merge data and create engineered features"""
    try:
        pollution_df["timestamp"] = pd.to_datetime(pollution_df["timestamp"]).dt.tz_localize(None)
        weather_df["time"] = pd.to_datetime(weather_df["time"]).dt.tz_localize(None)

        merged_df = pd.merge(
            pollution_df,
            weather_df,
            left_on="timestamp",
            right_on="time",
            how="inner"
        )

        if merged_df.empty:
            logger.warning("Merge resulted in empty dataframe")
            return merged_df

        merged_df.drop(columns=["time"], inplace=True)

        # Time-based features
        merged_df["hour"] = merged_df["timestamp"].dt.hour
        merged_df["day"] = merged_df["timestamp"].dt.day
        merged_df["month"] = merged_df["timestamp"].dt.month
        merged_df["day_of_week"] = merged_df["timestamp"].dt.dayofweek
        merged_df["is_weekend"] = merged_df["day_of_week"].isin([5, 6]).astype(int)

        # Cyclical encoding
        merged_df["hour_sin"] = np.sin(2 * np.pi * merged_df["hour"] / 24)
        merged_df["hour_cos"] = np.cos(2 * np.pi * merged_df["hour"] / 24)
        merged_df["month_sin"] = np.sin(2 * np.pi * merged_df["month"] / 12)
        merged_df["month_cos"] = np.cos(2 * np.pi * merged_df["month"] / 12)

        # AQI-derived features
        merged_df["aqi_change"] = merged_df["aqi"].diff().fillna(0.0)

        default_aqi = float(merged_df["aqi"].iloc[0]) if len(merged_df) > 0 else 3.0
        merged_df["aqi_lag_1"] = merged_df["aqi"].shift(1).fillna(default_aqi).astype(float)
        merged_df["aqi_lag_3"] = merged_df["aqi"].shift(3).fillna(default_aqi).astype(float)
        merged_df["aqi_lag_6"] = merged_df["aqi"].shift(6).fillna(default_aqi).astype(float)
        merged_df["aqi_lag_24"] = merged_df["aqi"].shift(24).fillna(default_aqi).astype(float)

        # Rolling statistics
        merged_df["aqi_rolling_mean_24"] = merged_df["aqi"].rolling(window=24, min_periods=1).mean()
        merged_df["aqi_rolling_std_24"] = merged_df["aqi"].rolling(window=24, min_periods=1).std().fillna(0.0)
        merged_df["aqi_rolling_max_24"] = merged_df["aqi"].rolling(window=24, min_periods=1).max()
        merged_df["aqi_rolling_min_24"] = merged_df["aqi"].rolling(window=24, min_periods=1).min()

        # Interaction features
        merged_df["pm_ratio"] = merged_df["pm2_5"] / (merged_df["pm10"] + 1e-6)
        merged_df["temp_humidity"] = merged_df["temperature"] * merged_df["humidity"]
        merged_df["wind_pm25"] = merged_df["windspeed"] * merged_df["pm2_5"]
        merged_df["pressure_temp"] = merged_df["pressure"] * merged_df["temperature"]

        # Enforce data types
        float_cols = [
            'aqi_change', 'aqi_lag_1', 'aqi_lag_3', 'aqi_lag_6', 'aqi_lag_24',
            'aqi_rolling_mean_24', 'aqi_rolling_std_24', 'aqi_rolling_max_24',
            'aqi_rolling_min_24', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'pm_ratio', 'temp_humidity', 'wind_pm25', 'pressure_temp',
            'pm2_5', 'pm10', 'co', 'no2', 'so2', 'o3', 'nh3',
            'temperature', 'humidity', 'pressure', 'windspeed', 'winddirection', 'precipitation'
        ]
        for col in float_cols:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].astype(float)

        int_cols = ['hour', 'day', 'month', 'day_of_week', 'is_weekend', 'aqi']
        for col in int_cols:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].astype(int)

        logger.info(f"✓ Created {len(merged_df)} feature records with {len(merged_df.columns)} features")
        return merged_df

    except Exception as e:
        logger.error(f"Error creating features: {e}")
        raise


def save_to_mongodb(db, df):
    """Save features to MongoDB — handles duplicates"""
    try:
        if df.isnull().any().any():
            logger.warning("Data contains null values, filling with defaults")
            df = df.fillna(0.0)

        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

        features_col = db["merged_features"]
        features_col.create_index("timestamp", unique=True)

        records = df.to_dict("records")
        inserted_count = 0
        duplicate_count = 0
        error_count = 0

        for record in records:
            try:
                features_col.insert_one(record)
                inserted_count += 1
            except Exception as e:
                if "duplicate key error" in str(e).lower():
                    duplicate_count += 1
                    features_col.update_one(
                        {"timestamp": record["timestamp"]},
                        {"$set": record},
                        upsert=True
                    )
                else:
                    error_count += 1
                    logger.debug(f"Error inserting record: {e}")

        logger.info(f"✓ MongoDB save complete:")
        logger.info(f"  - New records inserted: {inserted_count}")
        logger.info(f"  - Duplicates updated: {duplicate_count}")
        if error_count > 0:
            logger.warning(f"  - Errors: {error_count}")

        total = features_col.count_documents({})
        logger.info(f"  - Total records in database: {total}")

    except Exception as e:
        logger.error(f"Error saving to MongoDB: {e}")
        raise


def run_feature_pipeline(start_date=None, end_date=None):
    """Main pipeline execution"""
    try:
        if end_date is None:
            end_date = datetime.now(timezone.utc).replace(tzinfo=None)
        if start_date is None:
            start_date = end_date - timedelta(days=7)

        logger.info(f"{'='*60}")
        logger.info(f"Running Feature Pipeline: {start_date.date()} to {end_date.date()}")
        logger.info(f"{'='*60}")

        pollution_df = fetch_pollution_data(start_date, end_date)
        weather_df = fetch_weather_data(start_date, end_date)

        if pollution_df.empty or weather_df.empty:
            logger.error("No data fetched. Exiting.")
            return None

        features_df = create_features(pollution_df, weather_df)

        if features_df.empty:
            logger.error("Feature creation resulted in empty dataframe. Exiting.")
            return None

        db = connect_to_mongodb()
        save_to_mongodb(db, features_df)

        logger.info("✓ Feature pipeline completed successfully!")
        return features_df

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    # FIX: was "ArgfumentParser" (typo) — corrected to ArgumentParser
    parser = argparse.ArgumentParser(description='Run AQI Feature Pipeline')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    end = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None

    run_feature_pipeline(start, end)