import requests
import pandas as pd
from datetime import datetime, timedelta

# Coordinates & timezone
LAT, LON = 24.8607, 67.0011
TIMEZONE = "Asia/Karachi"


def fetch_weather_data(start_date, end_date):
    """
    Fetch hourly historical weather data between start_date and end_date.
    start_date, end_date: str in YYYY-MM-DD format
    """
    weather_url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,"
        "pressure_msl,windspeed_10m,winddirection_10m,precipitation"
        f"&timezone={TIMEZONE}"
    )

    response = requests.get(weather_url, timeout=30)
    if response.status_code != 200:
        print("Failed to fetch weather data:", response.status_code)
        return pd.DataFrame()

    data = response.json()

    # Convert to DataFrame
    weather_df = pd.DataFrame(data["hourly"])

    # Rename columns
    weather_df.rename(columns={
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "pressure_msl": "pressure",
        "windspeed_10m": "windspeed",
        "winddirection_10m": "winddirection",
        "precipitation": "precipitation"
    }, inplace=True)

    # Convert time column to datetime (timezone-naive)
    weather_df["time"] = pd.to_datetime(weather_df["time"]).dt.tz_localize(None)

    return weather_df


if __name__ == "__main__":
    end_date_dt = datetime.utcnow()
    start_date_dt = end_date_dt - timedelta(days=7)

    end_date = end_date_dt.strftime("%Y-%m-%d")
    start_date = start_date_dt.strftime("%Y-%m-%d")

    print(f"Fetching weather data from {start_date} to {end_date}...")
    df = fetch_weather_data(start_date, end_date)

    if not df.empty:
        print(df.head())
        print(f"Total records fetched: {len(df)}")
    else:
        print("No data fetched")