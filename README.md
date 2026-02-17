# 🌍 Karachi AQI Predictor

A fully automated Air Quality Index (AQI) prediction system for Karachi, Pakistan.
Predicts AQI for the **next 72 hours** using real weather + pollution data and machine learning.

## 🗂️ Files
- `feature_pipeline.py` — Fetches & engineers features (runs every hour)
- `training_pipeline.py` — Trains ML models (runs daily)
- `prediction_pipeline.py` — Generates 72h AQI forecast
- `backfill_data.py` — One-time historical data backfill
- `app.py` — Streamlit dashboard
- `AQI_eda.ipynb` — Exploratory Data Analysis

## 🚀 Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
python backfill_data.py --months 3
python training_pipeline.py
python prediction_pipeline.py
python -m streamlit run app.py
```

## ⚙️ Automated Pipelines (GitHub Actions)
- **Every hour** → feature pipeline + predictions
- **Every day** → retrains ML model

## 📡 Data Sources
- [OpenWeather Air Pollution API](https://openweathermap.org/api/air-pollution)
- [Open-Meteo Forecast API](https://open-meteo.com)
