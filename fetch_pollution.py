import requests
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import certifi
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API key from .env
load_dotenv()
AIR_POLLUTION_API = os.getenv("AIR_POLLUTION_API")
MONGO_URI = os.getenv("MONGO_URI")

# Coordinates (Karachi)
LAT, LON = 24.8607, 67.0011

# MongoDB connection
client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client["aqi_database"]
features_col = db["historical_pollutants"]


# ─────────────────────────────────────────────────────────────
# US EPA AQI Calculator (replaces the OpenWeather 1-5 index)
# ─────────────────────────────────────────────────────────────

def _linear(Cp, BPhi, BPlo, Ihi, Ilo):
    """EPA linear interpolation formula."""
    return round(((Ihi - Ilo) / (BPhi - BPlo)) * (Cp - BPlo) + Ilo)


def _pm25_to_aqi(c: float) -> int:
    """Convert PM2.5 (µg/m³, 24-h average) to US EPA AQI sub-index."""
    c = round(float(c), 1)
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201,  300),
        (250.5, 350.4, 301,  400),
        (350.5, 500.4, 401,  500),
    ]
    for BPlo, BPhi, Ilo, Ihi in breakpoints:
        if BPlo <= c <= BPhi:
            return _linear(c, BPhi, BPlo, Ihi, Ilo)
    return 500


def _pm10_to_aqi(c: float) -> int:
    """Convert PM10 (µg/m³, 24-h average) to US EPA AQI sub-index."""
    c = int(float(c))
    breakpoints = [
        (0,   54,   0,  50),
        (55,  154,  51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ]
    for BPlo, BPhi, Ilo, Ihi in breakpoints:
        if BPlo <= c <= BPhi:
            return _linear(c, BPhi, BPlo, Ihi, Ilo)
    return 500


def _o3_to_aqi(c_ppb: float) -> int:
    """Convert O3 (ppb, 8-h average) to US EPA AQI sub-index."""
    c = round(float(c_ppb), 3)
    breakpoints = [
        (0,    54,   0,  50),
        (55,   70,  51, 100),
        (71,   85, 101, 150),
        (86,  105, 151, 200),
        (106, 200, 201, 300),
    ]
    for BPlo, BPhi, Ilo, Ihi in breakpoints:
        if BPlo <= c <= BPhi:
            return _linear(c, BPhi, BPlo, Ihi, Ilo)
    return 300


def _no2_to_aqi(c_ppb: float) -> int:
    """Convert NO2 (ppb, 1-h average) to US EPA AQI sub-index."""
    c = round(float(c_ppb))
    breakpoints = [
        (0,    53,   0,  50),
        (54,  100,  51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ]
    for BPlo, BPhi, Ilo, Ihi in breakpoints:
        if BPlo <= c <= BPhi:
            return _linear(c, BPhi, BPlo, Ihi, Ilo)
    return 500


def calculate_us_aqi(pm2_5=None, pm10=None, o3_ugm3=None, no2_ugm3=None) -> int:
    """
    Calculate US EPA AQI (0-500 scale) from raw pollutant concentrations.

    OpenWeatherMap returns concentrations in µg/m³.
    O3 and NO2 are converted from µg/m³ → ppb before applying breakpoints.

    Returns the highest sub-index (worst pollutant determines overall AQI).
    """
    sub_indices = []

    if pm2_5 is not None and pm2_5 >= 0:
        sub_indices.append(_pm25_to_aqi(pm2_5))

    if pm10 is not None and pm10 >= 0:
        sub_indices.append(_pm10_to_aqi(pm10))

    if o3_ugm3 is not None and o3_ugm3 >= 0:
        # µg/m³ → ppb  (molecular weight O3 = 48 g/mol, at 25°C)
        o3_ppb = o3_ugm3 * (1000 / 48) * (0.02445)  # simplified conversion
        sub_indices.append(_o3_to_aqi(o3_ppb))

    if no2_ugm3 is not None and no2_ugm3 >= 0:
        # µg/m³ → ppb  (molecular weight NO2 = 46 g/mol)
        no2_ppb = no2_ugm3 / 1.88
        sub_indices.append(_no2_to_aqi(no2_ppb))

    if not sub_indices:
        return 0

    return max(sub_indices)


# ─────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────

def fetch_pollution_data(start_date, end_date) -> pd.DataFrame:
    """
    Fetch hourly pollution data from OpenWeather Air Pollution History API
    and compute real US EPA AQI (0-500) from raw pollutant concentrations.

    NOTE: The API's main.aqi field returns a European 1-5 index — we IGNORE it
    and calculate the proper US EPA AQI from the component concentrations instead.
    """
    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history?"
        f"lat={LAT}&lon={LON}&start={start_unix}&end={end_unix}&appid={AIR_POLLUTION_API}"
    )

    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        logger.error(f"Failed to fetch pollution data: HTTP {response.status_code}")
        return pd.DataFrame()

    data = response.json()
    if "list" not in data:
        logger.warning("No data found in API response")
        return pd.DataFrame()

    records = []
    for entry in data["list"]:
        dt = datetime.utcfromtimestamp(entry["dt"])
        components = entry["components"]

        pm2_5 = components.get("pm2_5")
        pm10  = components.get("pm10")
        o3    = components.get("o3")
        no2   = components.get("no2")

        # ✅ Calculate REAL US EPA AQI (0-500) — NOT the API's 1-5 European index
        real_aqi = calculate_us_aqi(
            pm2_5=pm2_5,
            pm10=pm10,
            o3_ugm3=o3,
            no2_ugm3=no2,
        )

        records.append({
            "timestamp": dt,
            "pm2_5":  float(pm2_5)  if pm2_5  is not None else 0.0,
            "pm10":   float(pm10)   if pm10   is not None else 0.0,
            "co":     float(components.get("co",  0)),
            "no2":    float(no2)    if no2    is not None else 0.0,
            "so2":    float(components.get("so2", 0)),
            "o3":     float(o3)     if o3     is not None else 0.0,
            "nh3":    float(components.get("nh3", 0)),
            "aqi":    real_aqi,   # ✅ 0-500 scale (US EPA AQI)
        })

    df = pd.DataFrame(records)
    logger.info(f"✓ Fetched {len(df)} pollution records | AQI range: {df['aqi'].min()}-{df['aqi'].max()}")
    return df


def store_data_in_mongo(df: pd.DataFrame) -> None:
    """Store pollution data in MongoDB, avoiding duplicates."""
    if df.empty:
        logger.warning("No data to insert")
        return

    inserted = 0
    for record in df.to_dict("records"):
        if features_col.count_documents({"timestamp": record["timestamp"]}, limit=1) == 0:
            features_col.insert_one(record)
            inserted += 1

    logger.info(f"✓ {inserted} new records inserted into MongoDB (historical_pollutants)")


if __name__ == "__main__":
    end_date   = datetime.utcnow()
    start_date = end_date - timedelta(days=7)

    logger.info(f"Fetching pollution data from {start_date.date()} to {end_date.date()} ...")
    df = fetch_pollution_data(start_date, end_date)

    if not df.empty:
        print(df[["timestamp", "pm2_5", "pm10", "aqi"]].head(10))
        print(f"\nAQI stats — min: {df['aqi'].min()}, max: {df['aqi'].max()}, mean: {df['aqi'].mean():.1f}")
        store_data_in_mongo(df)
    else:
        logger.error("No data fetched")
