"""
prediction_pipeline.py
-----------------------
Loads the trained model from the MongoDB Model Registry,
fetches the 72-hour weather forecast, builds the feature matrix,
and generates REAL US EPA AQI (0-500) predictions.
Saves predictions back to MongoDB under the 'predictions' collection.
"""

import pandas as pd
import numpy as np
import joblib
import logging
import os
import pytz
import requests
import certifi
from bson import ObjectId
import gridfs
import io
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
MONGO_URI         = os.getenv("MONGO_URI")
AIR_POLLUTION_API = os.getenv("AIR_POLLUTION_API")

LAT, LON  = 24.8607, 67.0011
TIMEZONE  = "Asia/Karachi"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# MongoDB helpers
# ─────────────────────────────────────────────────────────────

def connect_to_mongodb():
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client["aqi_database"]
    db.command("ping")
    logger.info("✓ Connected to MongoDB")
    return db


def load_active_model(db):
    """Fetch the active model metadata from the MongoDB Model Registry."""
    doc = db["model_registry"].find_one({"is_active": True}, sort=[("created_at", -1)])
    if doc is None:
        raise RuntimeError("No active model found. Run training_pipeline.py first.")
    return doc


def load_model_and_scaler(model_doc):
    db = connect_to_mongodb()
    fs = gridfs.GridFS(db)

    # Load model from MongoDB GridFS
    model_file = fs.get(ObjectId(model_doc["model_file_id"]))
    model = joblib.load(io.BytesIO(model_file.read()))

    # Load scaler from MongoDB GridFS
    scaler = None
    if model_doc.get("scaler_file_id"):
        scaler_file = fs.get(ObjectId(model_doc["scaler_file_id"]))
        scaler = joblib.load(io.BytesIO(scaler_file.read()))

    logger.info(f"✓ Loaded model: {model_doc['model_name']} from MongoDB")
    return model, scaler, model_doc["features"]


# ─────────────────────────────────────────────────────────────
# Fetch future weather forecast
# ─────────────────────────────────────────────────────────────

def fetch_forecast_weather(hours: int = 72) -> pd.DataFrame:
    """Fetch hourly weather forecast from Open-Meteo (free, no API key)."""
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,relative_humidity_2m,"
        "pressure_msl,windspeed_10m,winddirection_10m,precipitation_probability"
        f"&timezone={TIMEZONE}&forecast_days=4"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df.rename(columns={
        "temperature_2m":          "temperature",
        "relative_humidity_2m":    "humidity",
        "pressure_msl":            "pressure",
        "windspeed_10m":           "windspeed",
        "winddirection_10m":       "winddirection",
        "precipitation_probability": "precipitation",
    }, inplace=True)

    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

    karachi_tz = pytz.timezone("Asia/Karachi")
    now_pkt = datetime.now(pytz.utc).astimezone(karachi_tz).replace(tzinfo=None)
    df = df[df["time"] >= now_pkt].head(hours).reset_index(drop=True)

    logger.info(f"✓ Fetched {len(df)} forecast weather rows")
    return df


# ─────────────────────────────────────────────────────────────
# Fetch latest historical AQI for lag features
# ─────────────────────────────────────────────────────────────

def fetch_latest_aqi_history(db, lookback_hours: int = 48) -> pd.DataFrame:
    """Pull recent AQI records from the feature store for constructing lag features."""
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
    cursor = db["merged_features"].find(
        {"timestamp": {"$gte": cutoff}},
        sort=[("timestamp", 1)]
    )
    df = pd.DataFrame(list(cursor))

    if df.empty:
        logger.warning("No recent AQI history found — lag features will use defaults")
        return df

    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # Sanity check: ensure stored data is real AQI, not the old 1-5 index
    if not df.empty and "aqi" in df.columns and df["aqi"].max() <= 5:
        logger.error(
            "⚠️  Stored AQI values are still in 1-5 range! "
            "Re-run backfill_data.py after fixing fetch_pollution.py / feature_pipeline.py."
        )

    logger.info(f"✓ Loaded {len(df)} historical rows (AQI range: {df['aqi'].min()}-{df['aqi'].max()})")
    return df


# ─────────────────────────────────────────────────────────────
# Build prediction feature frame
# ─────────────────────────────────────────────────────────────

def build_forecast_features(
    forecast_weather: pd.DataFrame,
    history_df: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Construct the feature matrix for future timestamps.
    Uses forecast weather + lag/rolling features extrapolated from recent history.
    """
    # Default seed AQI — use real values from history if available
    if not history_df.empty and "aqi" in history_df.columns:
        last_aqi_values = list(history_df["aqi"].values[-24:])
        default_pollutants = {
            "pm2_5": float(history_df["pm2_5"].mean()),
            "pm10":  float(history_df["pm10"].mean()),
            "co":    float(history_df["co"].mean()),
            "no2":   float(history_df["no2"].mean()),
            "so2":   float(history_df["so2"].mean()),
            "o3":    float(history_df["o3"].mean()),
            "nh3":   float(history_df["nh3"].mean()),
        }
    else:
        # Reasonable Karachi defaults in US EPA AQI context
        last_aqi_values = [150] * 24   # ← realistic default (was 3 in old code)
        default_pollutants = {
            "pm2_5": 55.0, "pm10": 80.0, "co": 500.0,
            "no2": 30.0,   "so2": 10.0,  "o3": 60.0, "nh3": 5.0,
        }

    rows = []

    for _, row in forecast_weather.iterrows():
        ts = row["time"]

        current_aqi = last_aqi_values[-1] if last_aqi_values else 150

        def safe_lag(n):
            idx = len(last_aqi_values) - n
            return float(last_aqi_values[idx]) if idx >= 0 else float(current_aqi)

        window  = np.array(last_aqi_values[-24:], dtype=float)
        hour    = ts.hour
        month   = ts.month
        dow     = ts.dayofweek

        feat = {
            "timestamp": ts,
            **default_pollutants,
            # Weather (real forecast values)
            "temperature":  float(row["temperature"]),
            "humidity":     float(row["humidity"]),
            "pressure":     float(row["pressure"]),
            "windspeed":    float(row["windspeed"]),
            "winddirection": float(row["winddirection"]),
            "precipitation": float(row["precipitation"]),
            # Time
            "hour":       hour,
            "day":        ts.day,
            "month":      month,
            "day_of_week": dow,
            "is_weekend": int(dow in [5, 6]),
            "hour_sin":   np.sin(2 * np.pi * hour  / 24),
            "hour_cos":   np.cos(2 * np.pi * hour  / 24),
            "month_sin":  np.sin(2 * np.pi * month / 12),
            "month_cos":  np.cos(2 * np.pi * month / 12),
            # AQI lag / rolling (seeded from real 0-500 history)
            "aqi_change":          float(window[-1] - window[-2]) if len(window) >= 2 else 0.0,
            "aqi_lag_1":           safe_lag(1),
            "aqi_lag_3":           safe_lag(3),
            "aqi_lag_6":           safe_lag(6),
            "aqi_lag_24":          safe_lag(24),
            "aqi_rolling_mean_24": float(np.mean(window)),
            "aqi_rolling_std_24":  float(np.std(window)) if len(window) > 1 else 0.0,
            "aqi_rolling_max_24":  float(np.max(window)),
            "aqi_rolling_min_24":  float(np.min(window)),
        }

        # Interaction features
        feat["pm_ratio"]     = feat["pm2_5"] / (feat["pm10"] + 1e-6)
        feat["temp_humidity"] = feat["temperature"] * feat["humidity"]
        feat["wind_pm25"]    = feat["windspeed"] * feat["pm2_5"]
        feat["pressure_temp"] = feat["pressure"] * feat["temperature"]

        rows.append(feat)
        last_aqi_values.append(current_aqi)   # shift window for next step

    df = pd.DataFrame(rows)

    # Align to exactly the columns the model was trained on
    for col in feature_cols:
        if col not in df.columns:
            logger.warning(f"Missing feature '{col}' — filling with 0")
            df[col] = 0.0

    return df


# ─────────────────────────────────────────────────────────────
# AQI category helper (US EPA scale)
# ─────────────────────────────────────────────────────────────

def aqi_to_category(aqi: int) -> str:
    if aqi <= 50:   return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else:            return "Hazardous"


# ─────────────────────────────────────────────────────────────
# Save predictions
# ─────────────────────────────────────────────────────────────

def save_predictions(db, predictions_df: pd.DataFrame) -> None:
    """Upsert predictions into MongoDB 'predictions' collection."""
    col = db["predictions"]
    col.create_index("timestamp", unique=True)

    now = datetime.utcnow()
    inserted = updated = 0

    for record in predictions_df.to_dict("records"):
        record["created_at"] = now
        record.pop("_id", None)  # remove _id before any operation
        
        try:
            col.insert_one(record)
            inserted += 1
        except Exception:
            # Remove _id again in case insert_one added it
            record.pop("_id", None)
            update_data = {k: v for k, v in record.items() if k != "_id"}
            col.update_one(
                {"timestamp": record["timestamp"]},
                {"$set": update_data},
                upsert=True,
            )
            updated += 1

    logger.info(f"✓ Predictions saved — inserted: {inserted} | updated: {updated}")

# ─────────────────────────────────────────────────────────────
# Main inference pipeline
# ─────────────────────────────────────────────────────────────

def run_inference_pipeline() -> pd.DataFrame:
    """
    Full inference pipeline:
    1. Load model from registry
    2. Fetch 72-hour weather forecast
    3. Build feature matrix (with real 0-500 AQI lag features)
    4. Predict and clip to valid US EPA AQI range (0-500)
    5. Save predictions to MongoDB
    """
    logger.info("=" * 60)
    logger.info("AQI INFERENCE PIPELINE (US EPA 0-500 scale)")
    logger.info("=" * 60)

    db = connect_to_mongodb()

    # Load model
    model_doc = load_active_model(db)
    model, scaler, feature_cols = load_model_and_scaler(model_doc)

    # Fetch data
    forecast_weather = fetch_forecast_weather(hours=72)
    history_df       = fetch_latest_aqi_history(db, lookback_hours=48)

    # Build feature matrix
    X_df       = build_forecast_features(forecast_weather, history_df, feature_cols)
    timestamps = X_df["timestamp"].copy()

    X = X_df[feature_cols].values

    if scaler is not None:
        X = scaler.transform(X)

    # Predict — clip to valid US EPA AQI range (0-500)
    raw_preds = model.predict(X)
    aqi_preds = np.clip(np.round(raw_preds), 0, 500).astype(int)   # ✅ 0-500, NOT 1-5

    # Build results DataFrame
    results = pd.DataFrame({
        "timestamp":     timestamps.values,
        "predicted_aqi": aqi_preds,
        "aqi_category":  [aqi_to_category(v) for v in aqi_preds],
    })

    # Attach forecast weather columns for the dashboard
    for col in ["temperature", "humidity", "pressure", "windspeed", "winddirection", "precipitation"]:
        if col in X_df.columns:
            results[col] = X_df[col].values

    save_predictions(db, results)

    logger.info(f"\n✓ Inference complete — {len(results)} predictions")
    logger.info(f"  AQI range: {aqi_preds.min()} – {aqi_preds.max()} (mean: {aqi_preds.mean():.1f})")
    logger.info(f"  Categories: {results['aqi_category'].value_counts().to_dict()}")

    return results


if __name__ == "__main__":
    run_inference_pipeline()
