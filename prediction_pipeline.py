"""
prediction_pipeline.py
-----------------------
Loads the trained model from the Model Registry (MongoDB),
fetches the latest features, and generates a 72-hour AQI forecast.
Saves predictions back to MongoDB under the 'predictions' collection.

Run this AFTER training_pipeline.py has saved a model.
"""

import pandas as pd
import numpy as np
import joblib
import logging
import os
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
AIR_POLLUTION_API = os.getenv("AIR_POLLUTION_API")

LAT, LON = 24.8607, 67.0011
TIMEZONE = "Asia/Karachi"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. MongoDB helpers
# ─────────────────────────────────────────────────────────────

def connect_to_mongodb():
    client = MongoClient(MONGO_URI)
    db = client["aqi_database"]
    db.command('ping')
    logger.info("✓ Connected to MongoDB")
    return db


def load_active_model(db):
    """Fetch the active model metadata from MongoDB Model Registry."""
    models_col = db["model_registry"]
    model_doc = models_col.find_one({"is_active": True}, sort=[("created_at", -1)])
    if model_doc is None:
        raise RuntimeError("No active model found in registry. Run training_pipeline.py first.")
    return model_doc


def load_model_and_scaler(model_doc):
    """Load the saved sklearn model and scaler from disk."""
    model_path = model_doc["model_path"]
    scaler_path = model_doc.get("scaler_path")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Make sure the models/ directory is present and training has been run."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None

    logger.info(f"✓ Loaded model: {model_doc['model_name']} from {model_path}")
    return model, scaler, model_doc["features"]


# ─────────────────────────────────────────────────────────────
# 2. Fetch future weather (forecast)
# ─────────────────────────────────────────────────────────────

def fetch_forecast_weather(hours: int = 72) -> pd.DataFrame:
    """
    Fetch hourly weather forecast from Open-Meteo (free, no key needed).
    Returns a DataFrame indexed by future hourly timestamps.
    """
    forecast_url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,relative_humidity_2m,"
        "pressure_msl,windspeed_10m,winddirection_10m,precipitation_probability"
        f"&timezone={TIMEZONE}"
        f"&forecast_days=4"          # 4 days covers 72 h comfortably
    )

    response = requests.get(forecast_url, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df.rename(columns={
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "pressure_msl": "pressure",
        "windspeed_10m": "windspeed",
        "winddirection_10m": "winddirection",
        "precipitation_probability": "precipitation",
    }, inplace=True)

    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)

    # Keep only the next `hours` rows
    now = datetime.utcnow()
    df = df[df["time"] >= now].head(hours).reset_index(drop=True)

    logger.info(f"✓ Fetched {len(df)} forecast weather rows")
    return df


# ─────────────────────────────────────────────────────────────
# 3. Fetch latest historical AQI (for lag features)
# ─────────────────────────────────────────────────────────────

def fetch_latest_aqi_history(db, lookback_hours: int = 48) -> pd.DataFrame:
    """
    Pull the most recent AQI records from the feature store
    so we can construct lag features for the forecast window.
    """
    features_col = db["merged_features"]
    cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)

    cursor = features_col.find(
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
    logger.info(f"✓ Loaded {len(df)} historical rows for lag feature construction")
    return df


# ─────────────────────────────────────────────────────────────
# 4. Build prediction feature frame
# ─────────────────────────────────────────────────────────────

def build_forecast_features(
    forecast_weather: pd.DataFrame,
    history_df: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Construct the feature matrix that the model expects for future timestamps.
    Uses forecast weather + extrapolated lag/rolling features from history.
    """
    # Seed AQI sequence from the last known values (for rolling & lag features)
    if not history_df.empty and "aqi" in history_df.columns:
        last_aqi_values = list(history_df["aqi"].values[-24:])
    else:
        last_aqi_values = [3] * 24      # default to "moderate"

    rows = []

    for i, row in forecast_weather.iterrows():
        ts = row["time"]

        # Current best-guess AQI is the last value we know
        current_aqi = last_aqi_values[-1] if last_aqi_values else 3

        # Time features
        hour = ts.hour
        day = ts.day
        month = ts.month
        dow = ts.dayofweek

        # Lag features
        def safe_lag(n):
            idx = len(last_aqi_values) - n
            return float(last_aqi_values[idx]) if idx >= 0 else float(current_aqi)

        aqi_lag_1 = safe_lag(1)
        aqi_lag_3 = safe_lag(3)
        aqi_lag_6 = safe_lag(6)
        aqi_lag_24 = safe_lag(24)

        # Rolling from last 24 values
        window = last_aqi_values[-24:]
        arr = np.array(window, dtype=float)
        aqi_rolling_mean_24 = float(np.mean(arr))
        aqi_rolling_std_24 = float(np.std(arr)) if len(arr) > 1 else 0.0
        aqi_rolling_max_24 = float(np.max(arr))
        aqi_rolling_min_24 = float(np.min(arr))
        aqi_change = float(arr[-1] - arr[-2]) if len(arr) >= 2 else 0.0

        feat = {
            "timestamp": ts,
            # Pollutants — use rolling mean from history as proxy (no future data)
            "pm2_5": float(history_df["pm2_5"].mean()) if not history_df.empty else 20.0,
            "pm10": float(history_df["pm10"].mean()) if not history_df.empty else 30.0,
            "co": float(history_df["co"].mean()) if not history_df.empty else 200.0,
            "no2": float(history_df["no2"].mean()) if not history_df.empty else 10.0,
            "so2": float(history_df["so2"].mean()) if not history_df.empty else 5.0,
            "o3": float(history_df["o3"].mean()) if not history_df.empty else 60.0,
            "nh3": float(history_df["nh3"].mean()) if not history_df.empty else 2.0,
            # Weather
            "temperature": float(row["temperature"]),
            "humidity": float(row["humidity"]),
            "pressure": float(row["pressure"]),
            "windspeed": float(row["windspeed"]),
            "winddirection": float(row["winddirection"]),
            "precipitation": float(row["precipitation"]),
            # Time
            "hour": hour,
            "day": day,
            "month": month,
            "day_of_week": dow,
            "is_weekend": int(dow in [5, 6]),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            # AQI-derived
            "aqi_change": aqi_change,
            "aqi_lag_1": aqi_lag_1,
            "aqi_lag_3": aqi_lag_3,
            "aqi_lag_6": aqi_lag_6,
            "aqi_lag_24": aqi_lag_24,
            "aqi_rolling_mean_24": aqi_rolling_mean_24,
            "aqi_rolling_std_24": aqi_rolling_std_24,
            "aqi_rolling_max_24": aqi_rolling_max_24,
            "aqi_rolling_min_24": aqi_rolling_min_24,
        }

        # Interaction features
        feat["pm_ratio"] = feat["pm2_5"] / (feat["pm10"] + 1e-6)
        feat["temp_humidity"] = feat["temperature"] * feat["humidity"]
        feat["wind_pm25"] = feat["windspeed"] * feat["pm2_5"]
        feat["pressure_temp"] = feat["pressure"] * feat["temperature"]

        rows.append(feat)

        # Append current_aqi to rolling window (so next iteration shifts correctly)
        last_aqi_values.append(current_aqi)

    df = pd.DataFrame(rows)

    # Keep only the feature columns the model was trained on
    available_cols = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_cols)
    if missing:
        logger.warning(f"Missing feature columns (filling with 0): {missing}")
        for col in missing:
            df[col] = 0.0

    return df


# ─────────────────────────────────────────────────────────────
# 5. Save predictions
# ─────────────────────────────────────────────────────────────

def save_predictions(db, predictions_df: pd.DataFrame) -> None:
    """Upsert predictions into MongoDB 'predictions' collection."""
    preds_col = db["predictions"]
    preds_col.create_index("timestamp", unique=True)

    now = datetime.utcnow()
    inserted = 0
    updated = 0

    for record in predictions_df.to_dict("records"):
        record["created_at"] = now
        try:
            preds_col.insert_one(record)
            inserted += 1
        except Exception:
            preds_col.update_one(
                {"timestamp": record["timestamp"]},
                {"$set": record},
                upsert=True
            )
            updated += 1

    logger.info(f"✓ Predictions saved — inserted: {inserted}, updated: {updated}")


# ─────────────────────────────────────────────────────────────
# 6. Main pipeline
# ─────────────────────────────────────────────────────────────

def run_inference_pipeline() -> pd.DataFrame:
    """
    Full inference pipeline:
    1. Load model from registry
    2. Fetch 72-hour weather forecast
    3. Build feature matrix
    4. Predict AQI
    5. Save predictions to MongoDB
    """
    logger.info("=" * 60)
    logger.info("AQI INFERENCE PIPELINE")
    logger.info("=" * 60)

    db = connect_to_mongodb()

    # Load model
    model_doc = load_active_model(db)
    model, scaler, feature_cols = load_model_and_scaler(model_doc)

    # Fetch data
    forecast_weather = fetch_forecast_weather(hours=72)
    history_df = fetch_latest_aqi_history(db, lookback_hours=48)

    # Build features
    X_df = build_forecast_features(forecast_weather, history_df, feature_cols)
    timestamps = X_df["timestamp"].copy()

    X = X_df[feature_cols].values

    # Scale if scaler exists
    if scaler is not None:
        X = scaler.transform(X)

    # Predict
    raw_preds = model.predict(X)

    # Clip to valid AQI range [1, 5] and round
    clipped = np.clip(np.round(raw_preds), 1, 5).astype(int)

    # Build results DataFrame (include weather columns for app.py)
    results = pd.DataFrame({
        "timestamp": timestamps.values,
        "predicted_aqi": clipped,
    })

    # Add weather columns back for the dashboard
    weather_cols = ["temperature", "humidity", "pressure", "windspeed", "winddirection", "precipitation"]
    for col in weather_cols:
        if col in X_df.columns:
            results[col] = X_df[col].values

    # Save
    save_predictions(db, results)

    logger.info(f"✓ Inference pipeline complete — {len(results)} predictions generated")
    logger.info(f"  AQI range: {clipped.min()} – {clipped.max()}")
    return results


if __name__ == "__main__":
    run_inference_pipeline()