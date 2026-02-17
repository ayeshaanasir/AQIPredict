import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fetch_pollution import fetch_pollution_data
from fetch_weather import fetch_weather_data


def merge_data(pollution_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pollution and weather dataframes on their timestamp columns.
    Returns a merged dataframe with engineered features.
    """
    if pollution_df.empty or weather_df.empty:
        print("One or both dataframes are empty — cannot merge.")
        return pd.DataFrame()

    # Ensure both timestamp columns are timezone-naive datetime64
    pollution_df["timestamp"] = pd.to_datetime(pollution_df["timestamp"]).dt.tz_localize(None)
    weather_df["time"] = pd.to_datetime(weather_df["time"]).dt.tz_localize(None)

    # Inner merge
    merged_df = pd.merge(
        pollution_df,
        weather_df,
        left_on="timestamp",
        right_on="time",
        how="inner"
    )

    # Drop redundant time column
    merged_df.drop(columns=["time"], inplace=True)

    if merged_df.empty:
        print("Merge produced no rows — check timestamp alignment.")
        return merged_df

    # ── Time-based features ──────────────────────────────────────────
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

    # ── AQI-derived features ─────────────────────────────────────────
    merged_df["aqi_change"] = merged_df["aqi"].diff().fillna(0.0)

    default_aqi = float(merged_df["aqi"].iloc[0]) if len(merged_df) > 0 else 3.0
    merged_df["aqi_lag_1"] = merged_df["aqi"].shift(1).fillna(default_aqi).astype(float)
    merged_df["aqi_lag_3"] = merged_df["aqi"].shift(3).fillna(default_aqi).astype(float)
    merged_df["aqi_lag_6"] = merged_df["aqi"].shift(6).fillna(default_aqi).astype(float)
    merged_df["aqi_lag_24"] = merged_df["aqi"].shift(24).fillna(default_aqi).astype(float)

    # Rolling stats (24-hour window)
    merged_df["aqi_rolling_mean_24"] = merged_df["aqi"].rolling(window=24, min_periods=1).mean()
    merged_df["aqi_rolling_std_24"] = merged_df["aqi"].rolling(window=24, min_periods=1).std().fillna(0.0)
    merged_df["aqi_rolling_max_24"] = merged_df["aqi"].rolling(window=24, min_periods=1).max()
    merged_df["aqi_rolling_min_24"] = merged_df["aqi"].rolling(window=24, min_periods=1).min()

    # ── Interaction features ─────────────────────────────────────────
    merged_df["pm_ratio"] = merged_df["pm2_5"] / (merged_df["pm10"] + 1e-6)
    merged_df["temp_humidity"] = merged_df["temperature"] * merged_df["humidity"]
    merged_df["wind_pm25"] = merged_df["windspeed"] * merged_df["pm2_5"]
    merged_df["pressure_temp"] = merged_df["pressure"] * merged_df["temperature"]

    print(f"✓ Merged dataframe: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    return merged_df


if __name__ == "__main__":
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)

    print("Fetching pollution data...")
    p_df = fetch_pollution_data(start_date, end_date)

    print("Fetching weather data...")
    w_df = fetch_weather_data(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

    print("Merging data...")
    merged = merge_data(p_df, w_df)

    if not merged.empty:
        print(merged.head())
        print(f"\nColumns: {list(merged.columns)}")