import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
from pymongo import MongoClient
import certifi
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
AIR_POLLUTION_API = os.getenv("AIR_POLLUTION_API")
MONGO_URI = os.getenv("MONGO_URI")

LAT, LON = 24.8607, 67.0011
TIMEZONE = "Asia/Karachi"


# ─────────────────────────────────────────────────────────────
# US EPA AQI Calculator
# ─────────────────────────────────────────────────────────────

def _linear(Cp, BPhi, BPlo, Ihi, Ilo):
    return round(((Ihi - Ilo) / (BPhi - BPlo)) * (Cp - BPlo) + Ilo)


def _pm25_to_aqi(c: float) -> int:
    c = round(float(c), 1)
    bp = [
        (0.0,   12.0,   0,  50), (12.1,  35.4,  51, 100),
        (35.5,  55.4, 101, 150), (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300), (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for BPlo, BPhi, Ilo, Ihi in bp:
        if BPlo <= c <= BPhi:
            return _linear(c, BPhi, BPlo, Ihi, Ilo)
    return 500


def _pm10_to_aqi(c: float) -> int:
    c = int(float(c))
    bp = [
        (0,   54,   0,  50), (55,  154,  51, 100),
        (155, 254, 101, 150), (255, 354, 151, 200),
        (355, 424, 201, 300), (425, 504, 301, 400),
        (505, 604, 401, 500),
    ]
    for BPlo, BPhi, Ilo, Ihi in bp:
        if BPlo <= c <= BPhi:
            return _linear(c, BPhi, BPlo, Ihi, Ilo)
    return 500


def _no2_to_aqi(c_ugm3: float) -> int:
    c = round(float(c_ugm3) / 1.88)   # µg/m³ → ppb
    bp = [
        (0,   53,   0,  50), (54,  100,  51, 100),
        (101, 360, 101, 150), (361, 649, 151, 200),
        (650, 1249, 201, 300), (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ]
    for BPlo, BPhi, Ilo, Ihi in bp:
        if BPlo <= c <= BPhi:
            return _linear(c, BPhi, BPlo, Ihi, Ilo)
    return 500


def _o3_to_aqi(c_ugm3: float) -> int:
    c = round(float(c_ugm3) * 0.02445 * (1000 / 48), 3)  # µg/m³ → ppb
    bp = [
        (0,  54,   0,  50), (55,  70,  51, 100),
        (71, 85, 101, 150), (86, 105, 151, 200),
        (106, 200, 201, 300),
    ]
    for BPlo, BPhi, Ilo, Ihi in bp:
        if BPlo <= c <= BPhi:
            return _linear(c, BPhi, BPlo, Ihi, Ilo)
    return 300


def calculate_us_aqi(pm2_5=None, pm10=None, o3=None, no2=None) -> int:
    """
    Calculate US EPA AQI (0-500) from raw µg/m³ concentrations.
    Returns the highest sub-index (worst pollutant wins).
    """
    subs = []
    if pm2_5 is not None and pm2_5 >= 0: subs.append(_pm25_to_aqi(pm2_5))
    if pm10  is not None and pm10  >= 0: subs.append(_pm10_to_aqi(pm10))
    if no2   is not None and no2   >= 0: subs.append(_no2_to_aqi(no2))
    if o3    is not None and o3    >= 0: subs.append(_o3_to_aqi(o3))
    return max(subs) if subs else 0


# ─────────────────────────────────────────────────────────────
# MongoDB
# ─────────────────────────────────────────────────────────────

def connect_to_mongodb():
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        db = client["aqi_database"]
        db.command('ping')
        logger.info("✓ Connected to MongoDB")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


# ─────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────

def fetch_pollution_data(start_date, end_date) -> pd.DataFrame:
    """
    Fetch pollution data and compute REAL US EPA AQI (0-500).
    The OpenWeather API's main.aqi field (1-5 European index) is IGNORED.
    """
    try:
        start_unix = int(start_date.timestamp())
        end_unix   = int(end_date.timestamp())

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
            dt         = datetime.fromtimestamp(entry["dt"], tz=timezone.utc).replace(tzinfo=None)
            components = entry["components"]

            pm2_5 = float(components.get("pm2_5", 0))
            pm10  = float(components.get("pm10",  0))
            o3    = float(components.get("o3",    0))
            no2   = float(components.get("no2",   0))

            # ✅ Real US EPA AQI — NOT the API's 1-5 field
            real_aqi = calculate_us_aqi(pm2_5=pm2_5, pm10=pm10, o3=o3, no2=no2)

            records.append({
                "timestamp": dt,
                "pm2_5": pm2_5,
                "pm10":  pm10,
                "co":    float(components.get("co",  0)),
                "no2":   no2,
                "so2":   float(components.get("so2", 0)),
                "o3":    o3,
                "nh3":   float(components.get("nh3", 0)),
                "aqi":   real_aqi,   # ✅ 0-500 US EPA scale
            })

        df = pd.DataFrame(records)
        logger.info(
            f"✓ Fetched {len(df)} pollution records | "
            f"AQI range: {df['aqi'].min()}-{df['aqi'].max()} (mean: {df['aqi'].mean():.1f})"
        )
        return df

    except Exception as e:
        logger.error(f"Error fetching pollution data: {e}")
        return pd.DataFrame()


def fetch_weather_data(start_date, end_date) -> pd.DataFrame:
    """Fetch hourly weather data from Open-Meteo archive API."""
    try:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str   = end_date.strftime("%Y-%m-%d")

        url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={LAT}&longitude={LON}"
            f"&start_date={start_str}&end_date={end_str}"
            "&hourly=temperature_2m,relative_humidity_2m,"
            "pressure_msl,windspeed_10m,winddirection_10m,precipitation"
            f"&timezone={TIMEZONE}"
        )

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data["hourly"])
        df.rename(columns={
            "temperature_2m":      "temperature",
            "relative_humidity_2m": "humidity",
            "pressure_msl":        "pressure",
            "windspeed_10m":       "windspeed",
            "winddirection_10m":   "winddirection",
            "precipitation":       "precipitation",
        }, inplace=True)

        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

        for col in ["temperature", "humidity", "pressure", "windspeed", "winddirection", "precipitation"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

        logger.info(f"✓ Fetched {len(df)} weather records")
        return df

    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────

def create_features(pollution_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge pollution + weather data and engineer model features."""
    try:
        pollution_df["timestamp"] = pd.to_datetime(pollution_df["timestamp"]).dt.tz_localize(None)
        weather_df["time"]        = pd.to_datetime(weather_df["time"]).dt.tz_localize(None)

        merged = pd.merge(
            pollution_df, weather_df,
            left_on="timestamp", right_on="time",
            how="inner"
        )

        if merged.empty:
            logger.warning("Merge resulted in empty dataframe — check timestamp alignment")
            return merged

        merged.drop(columns=["time"], inplace=True)

        # ── Time-based features ───────────────────────────────────────────
        merged["hour"]       = merged["timestamp"].dt.hour
        merged["day"]        = merged["timestamp"].dt.day
        merged["month"]      = merged["timestamp"].dt.month
        merged["day_of_week"] = merged["timestamp"].dt.dayofweek
        merged["is_weekend"] = merged["day_of_week"].isin([5, 6]).astype(int)

        # Cyclical encoding so the model sees periodicity
        merged["hour_sin"]  = np.sin(2 * np.pi * merged["hour"]  / 24)
        merged["hour_cos"]  = np.cos(2 * np.pi * merged["hour"]  / 24)
        merged["month_sin"] = np.sin(2 * np.pi * merged["month"] / 12)
        merged["month_cos"] = np.cos(2 * np.pi * merged["month"] / 12)

        # ── AQI-derived features ──────────────────────────────────────────
        # These work on the real 0-500 AQI values now
        merged["aqi_change"] = merged["aqi"].diff().fillna(0.0)

        default_aqi = float(merged["aqi"].iloc[0]) if len(merged) > 0 else 100.0
        merged["aqi_lag_1"]  = merged["aqi"].shift(1).fillna(default_aqi).astype(float)
        merged["aqi_lag_3"]  = merged["aqi"].shift(3).fillna(default_aqi).astype(float)
        merged["aqi_lag_6"]  = merged["aqi"].shift(6).fillna(default_aqi).astype(float)
        merged["aqi_lag_24"] = merged["aqi"].shift(24).fillna(default_aqi).astype(float)

        merged["aqi_rolling_mean_24"] = merged["aqi"].rolling(window=24, min_periods=1).mean()
        merged["aqi_rolling_std_24"]  = merged["aqi"].rolling(window=24, min_periods=1).std().fillna(0.0)
        merged["aqi_rolling_max_24"]  = merged["aqi"].rolling(window=24, min_periods=1).max()
        merged["aqi_rolling_min_24"]  = merged["aqi"].rolling(window=24, min_periods=1).min()

        # AQI category label (useful for classification later)
        merged["aqi_category"] = pd.cut(
            merged["aqi"],
            bins=[-1, 50, 100, 150, 200, 300, 500],
            labels=["Good", "Moderate", "USG", "Unhealthy", "Very Unhealthy", "Hazardous"]
        ).astype(str)

        # ── Interaction features ──────────────────────────────────────────
        merged["pm_ratio"]     = merged["pm2_5"] / (merged["pm10"] + 1e-6)
        merged["temp_humidity"] = merged["temperature"] * merged["humidity"]
        merged["wind_pm25"]    = merged["windspeed"] * merged["pm2_5"]
        merged["pressure_temp"] = merged["pressure"] * merged["temperature"]

        # ── Enforce dtypes ────────────────────────────────────────────────
        float_cols = [
            "aqi_change", "aqi_lag_1", "aqi_lag_3", "aqi_lag_6", "aqi_lag_24",
            "aqi_rolling_mean_24", "aqi_rolling_std_24", "aqi_rolling_max_24", "aqi_rolling_min_24",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "pm_ratio", "temp_humidity", "wind_pm25", "pressure_temp",
            "pm2_5", "pm10", "co", "no2", "so2", "o3", "nh3",
            "temperature", "humidity", "pressure", "windspeed", "winddirection", "precipitation",
        ]
        for col in float_cols:
            if col in merged.columns:
                merged[col] = merged[col].astype(float)

        int_cols = ["hour", "day", "month", "day_of_week", "is_weekend", "aqi"]
        for col in int_cols:
            if col in merged.columns:
                merged[col] = merged[col].astype(int)

        logger.info(
            f"✓ Created {len(merged)} feature records | "
            f"{len(merged.columns)} features | "
            f"AQI range: {merged['aqi'].min()}-{merged['aqi'].max()}"
        )
        return merged

    except Exception as e:
        logger.error(f"Error creating features: {e}")
        raise


# ─────────────────────────────────────────────────────────────
# MongoDB persistence
# ─────────────────────────────────────────────────────────────

def save_to_mongodb(db, df: pd.DataFrame) -> None:
    """Upsert feature records to MongoDB (handles duplicates cleanly)."""
    try:
        if df.isnull().any().any():
            logger.warning("Null values detected — filling with 0")
            df = df.fillna(0.0)

        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

        features_col = db["merged_features"]
        features_col.create_index("timestamp", unique=True)

        records = df.to_dict("records")
        inserted = duplicates = errors = 0

        for record in records:
            record.pop("_id", None)
            try:
                features_col.insert_one(record)
                inserted += 1
            except Exception as e:
                if "duplicate key error" in str(e).lower():
                    duplicates += 1
                    update_data = {k: v for k, v in record.items() if k != "_id"}
                    features_col.update_one(
                        {"timestamp": record["timestamp"]},
                        {"$set": update_data},
                    )
                else:
                    errors += 1
                    logger.debug(f"Insert error: {e}")

        total = features_col.count_documents({})
        logger.info(
            f"✓ MongoDB save — inserted: {inserted} | updated: {duplicates} | "
            f"errors: {errors} | total: {total}"
        )

    except Exception as e:
        logger.error(f"Error saving to MongoDB: {e}")
        raise


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def run_feature_pipeline(start_date=None, end_date=None) -> pd.DataFrame | None:
    """
    Execute the full feature pipeline:
    1. Fetch raw weather + pollution data
    2. Compute US EPA AQI and engineer features
    3. Store features in MongoDB
    """
    try:
        if end_date is None:
            end_date = datetime.now(timezone.utc).replace(tzinfo=None)
        if start_date is None:
            start_date = end_date - timedelta(days=7)

        logger.info("=" * 60)
        logger.info(f"Feature Pipeline: {start_date.date()} → {end_date.date()}")
        logger.info("=" * 60)

        pollution_df = fetch_pollution_data(start_date, end_date)
        weather_df   = fetch_weather_data(start_date, end_date)

        if pollution_df.empty or weather_df.empty:
            logger.error("No data fetched — aborting")
            return None

        features_df = create_features(pollution_df, weather_df)

        if features_df.empty:
            logger.error("Feature creation yielded empty dataframe — aborting")
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

    parser = argparse.ArgumentParser(description="Run AQI Feature Pipeline")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date",   type=str, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    end   = datetime.strptime(args.end_date,   "%Y-%m-%d") if args.end_date   else None

    run_feature_pipeline(start, end)
