import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import certifi

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

st.set_page_config(
    page_title="Karachi AQI Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# US EPA AQI category helper (0-500 scale)
# ─────────────────────────────────────────────────────────────

def get_aqi_category(aqi_value):
    try:
        val = int(float(str(aqi_value)))
    except (TypeError, ValueError, OverflowError):
        return "Unknown", "#cccccc", "No data available"
    if val < 0:
        return "Unknown", "#cccccc", "No data available"
    elif val <= 50:
        return "Good", "#00e400", "Air quality is satisfactory — enjoy outdoor activities."
    elif val <= 100:
        return "Moderate", "#ffff00", "Air quality is acceptable; sensitive individuals should reduce prolonged outdoor exertion."
    elif val <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00", "Members of sensitive groups may experience health effects."
    elif val <= 200:
        return "Unhealthy", "#ff0000", "Everyone may begin to experience health effects."
    elif val <= 300:
        return "Very Unhealthy", "#8f3f97", "Health alert: everyone may experience more serious health effects."
    else:
        return "Hazardous", "#7e0023", "🚨 Health emergency! Stay indoors and keep windows closed."


def aqi_text_color(aqi_value: int) -> str:
    return "black" if aqi_value <= 100 else "white"


def safe_int(val, default=0) -> int:
    try:
        return max(default, int(float(str(val))))
    except (TypeError, ValueError, OverflowError):
        return default


# ─────────────────────────────────────────────────────────────
# Hazardous AQI alert banners
# ─────────────────────────────────────────────────────────────

def show_aqi_alerts(predictions_df: pd.DataFrame) -> None:
    if predictions_df.empty or "predicted_aqi" not in predictions_df.columns:
        return
    max_aqi = safe_int(predictions_df["predicted_aqi"].max())
    if max_aqi > 300:
        st.error(f"🚨 **HAZARDOUS AQI ALERT** — Peak forecast AQI: **{max_aqi}**  \nOutdoor activity is dangerous for everyone. Stay indoors, seal windows, and use air purifiers.")
    elif max_aqi > 200:
        st.error(f"⛔ **VERY UNHEALTHY AIR QUALITY** — Peak forecast AQI: **{max_aqi}**  \nEveryone should avoid prolonged outdoor activity. Sensitive groups must stay indoors.")
    elif max_aqi > 150:
        st.warning(f"⚠️ **UNHEALTHY AIR QUALITY** — Peak forecast AQI: **{max_aqi}**  \nChildren, elderly, and people with respiratory/heart conditions should limit outdoor exertion.")
    elif max_aqi > 100:
        st.warning(f"⚠️ **UNHEALTHY FOR SENSITIVE GROUPS** — Peak forecast AQI: **{max_aqi}**  \nSensitive individuals should reduce prolonged outdoor activity.")


# ─────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────

def metric_card(label: str, value):
    st.markdown(f"""
        <div style="background-color:#1a1a2e;border:1px solid #4a4a6a;border-radius:12px;
                    padding:20px 15px;text-align:center;margin:4px 0px;">
            <p style="color:#a0a0c0;font-size:13px;font-weight:600;margin:0 0 8px 0;
                      text-transform:uppercase;letter-spacing:0.5px;">{label}</p>
            <p style="color:#ffffff;font-size:32px;font-weight:700;margin:0;line-height:1.2;">{value}</p>
        </div>
    """, unsafe_allow_html=True)


def dark_chart(fig, height=400):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    return fig


COLOR_MAP = {
    "Good":                           "#00e400",
    "Moderate":                       "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy":                      "#ff0000",
    "Very Unhealthy":                 "#8f3f97",
    "Hazardous":                      "#7e0023",
    "Unknown":                        "#cccccc",
}

MODEL_COLORS = {
    "Random Forest":       "#4da6ff",
    "Gradient Boosting":   "#ff7043",
    "XGBoost":             "#66bb6a",
    "Ridge Regression":    "#ab47bc",
    "Lasso Regression":    "#ffca28",
    "Neural Network (MLP)":"#26c6da",
}


# ─────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_database():
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        db = client["aqi_database"]
        db.command("ping")
        return db
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        st.stop()


@st.cache_data(ttl=900)
def load_predictions() -> pd.DataFrame:
    try:
        db = get_database()
        cursor = db["predictions"].find({}, sort=[("timestamp", 1)]).limit(72)
        df = pd.DataFrame(list(cursor))
        if df.empty:
            return pd.DataFrame()
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["predicted_aqi"] = df["predicted_aqi"].apply(lambda x: safe_int(x, default=0))
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["aqi_category"] = df["predicted_aqi"].apply(lambda x: get_aqi_category(x)[0])
        return df
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_historical_data(days: int = 7) -> pd.DataFrame:
    try:
        db = get_database()
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        cursor = db["merged_features"].find({"timestamp": {"$gte": cutoff}}, sort=[("timestamp", 1)])
        df = pd.DataFrame(list(cursor))
        if df.empty:
            return pd.DataFrame()
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["aqi"] = df["aqi"].apply(lambda x: safe_int(x, default=0))
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_model_info():
    try:
        db = get_database()
        return db["model_registry"].find_one({"is_active": True}, sort=[("created_at", -1)])
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_all_model_results() -> pd.DataFrame:
    """
    Fetch ALL model entries from model_registry (active + inactive)
    so we can show every model trained in the last run side by side.
    The training pipeline saves one document per model per run,
    marking only the best as is_active=True.
    """
    try:
        db = get_database()
        # Get the most recent training session timestamp from the active model
        active = db["model_registry"].find_one({"is_active": True}, sort=[("created_at", -1)])
        if not active:
            return pd.DataFrame()

        # Fetch all models created within 10 minutes of that session
        # (all models trained in the same run share roughly the same created_at)
        session_time = active["created_at"]
        from datetime import timedelta as td
        window_start = session_time - td(minutes=10)
        window_end   = session_time + td(minutes=10)

        cursor = db["model_registry"].find({
            "created_at": {"$gte": window_start, "$lte": window_end}
        })
        docs = list(cursor)

        # Fallback: if only one doc found, get the last 10 entries
        if len(docs) <= 1:
            cursor = db["model_registry"].find({}, sort=[("created_at", -1)]).limit(10)
            docs = list(cursor)

        if not docs:
            return pd.DataFrame()

        rows = []
        for doc in docs:
            m = doc.get("metrics", {})
            rows.append({
                "Model":      doc.get("model_name", "Unknown"),
                "Test RMSE":  round(float(m.get("test_rmse",  0)), 3),
                "Test MAE":   round(float(m.get("test_mae",   0)), 3),
                "Test R²":    round(float(m.get("test_r2",    0)), 4),
                "Train RMSE": round(float(m.get("train_rmse", 0)), 3),
                "Train MAE":  round(float(m.get("train_mae",  0)), 3),
                "Train R²":   round(float(m.get("train_r2",   0)), 4),
                "N Train":    int(m.get("n_train", m.get("n_samples_train", 0))),
                "N Test":     int(m.get("n_test",  m.get("n_samples_test",  0))),
                "N Features": int(m.get("n_features", 0)),
                "Is Best":    bool(doc.get("is_active", False)),
                "Trained At": doc.get("created_at", ""),
            })

        return pd.DataFrame(rows).sort_values("Test RMSE").reset_index(drop=True)

    except Exception as e:
        st.error(f"Error loading model results: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# Model comparison helpers
# ─────────────────────────────────────────────────────────────

MODEL_DESCRIPTIONS = {
    "Random Forest":
        "Ensemble of 200 decision trees trained via bagging (bootstrap aggregating). "
        "Each tree sees a random subset of features and data. Final prediction = average of all trees. "
        "Handles non-linear patterns, outliers, and feature interactions. Robust to overfitting.",
    "Gradient Boosting":
        "Sequential ensemble: each tree corrects the residual errors of the previous one. "
        "Uses 200 estimators with learning_rate=0.05 and max_depth=5. "
        "Strong on tabular data; can overfit if not tuned carefully.",
    "XGBoost":
        "Optimised gradient boosting with L1/L2 regularisation built in. "
        "200 estimators, max_depth=7, learning_rate=0.05. "
        "Faster training than sklearn GBM and often achieves better accuracy.",
    "Ridge Regression":
        "Linear model with L2 regularisation (alpha=1.0). Shrinks coefficients toward zero "
        "but never to exactly zero. Fast, interpretable baseline. Works well when AQI "
        "has approximately linear relationships with features.",
    "Lasso Regression":
        "Linear model with L1 regularisation (alpha=0.1, max_iter=5000). Performs automatic "
        "feature selection by shrinking some coefficients to exactly zero. "
        "Useful for identifying the most important predictors.",
    "Neural Network (MLP)":
        "Multi-layer perceptron with hidden layers (128→64→32 neurons), ReLU activations, "
        "and early stopping. Captures complex non-linear patterns but requires more data "
        "and is slower to train than tree-based models.",
}

def get_model_description(name: str) -> str:
    for key, desc in MODEL_DESCRIPTIONS.items():
        if key.lower() in name.lower():
            return desc
    return "ML model trained on 40+ AQI features with chronological train/test split."


def get_model_color(name: str) -> str:
    for key, color in MODEL_COLORS.items():
        if key.lower() in name.lower():
            return color
    return "#4da6ff"

def hex_to_rgba(hex_color: str, alpha: float = 0.4) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────────────────────
# Main dashboard
# ─────────────────────────────────────────────────────────────

def main():
    st.title("🌍 Karachi Air Quality Index (AQI) Predictor")
    st.markdown("Real-time AQI monitoring and 3-day forecast — **US EPA AQI scale (0–500)**")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_Pakistan.svg/320px-Flag_of_Pakistan.svg.png",
            width=100,
        )
        st.header("📍 Location")
        st.info("**Karachi, Pakistan**  \nLat: 24.8607°N  \nLon: 67.0011°E")

        st.header("ℹ️ About")
        st.markdown(
            "Predicts AQI using:\n"
            "- 🌡️ Weather (Open-Meteo)\n"
            "- 💨 Pollution (OpenWeather)\n"
            "- 🤖 ML (multiple models)\n"
            "- 📊 40+ engineered features"
        )

        model_info = load_model_info()
        if model_info:
            st.header("🤖 Model Info")
            metric_card("Model",     model_info["model_name"])
            metric_card("Test RMSE", f"{model_info['metrics']['test_rmse']:.2f}")
            metric_card("Test R²",   f"{model_info['metrics']['test_r2']:.4f}")
            metric_card("AQI Scale", model_info.get("aqi_scale", "US EPA 0-500"))
            st.caption(f"Last trained: {model_info['created_at'].strftime('%Y-%m-%d %H:%M')}")

        st.header("🔄 Controls")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.header("🎨 US EPA AQI Scale")
        for bg, fg, label in [
            ("#00e400", "black", "0–50   — Good"),
            ("#ffff00", "black", "51–100 — Moderate"),
            ("#ff7e00", "white", "101–150 — Unhealthy (Sensitive)"),
            ("#ff0000", "white", "151–200 — Unhealthy"),
            ("#8f3f97", "white", "201–300 — Very Unhealthy"),
            ("#7e0023", "white", "301–500 — Hazardous"),
        ]:
            st.markdown(
                f"<div style='background:{bg};padding:5px;margin:2px;border-radius:3px;"
                f"color:{fg};'><b>{label}</b></div>",
                unsafe_allow_html=True,
            )

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading data ..."):
        predictions_df = load_predictions()
        historical_df  = load_historical_data(days=7)
        all_models_df  = load_all_model_results()

    if predictions_df.empty:
        st.warning("⚠️ No predictions available. Run the inference pipeline first.")
        st.code("python prediction_pipeline.py", language="bash")
        st.stop()

    show_aqi_alerts(predictions_df)

    # ── Current status ────────────────────────────────────────────────────────
    current_aqi = safe_int(predictions_df.iloc[0]["predicted_aqi"])
    category, color, description = get_aqi_category(current_aqi)
    txt_color = aqi_text_color(current_aqi)

    st.header("📊 Current Status")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Current AQI", current_aqi)
    with c2:
        st.markdown(
            f"<div style='background-color:{color};padding:20px;border-radius:12px;"
            f"text-align:center;margin:4px 0;'>"
            f"<h3 style='margin:0;color:{txt_color};'>{category}</h3></div>",
            unsafe_allow_html=True,
        )
    with c3:
        avg_24h = predictions_df.head(24)["predicted_aqi"].mean() if len(predictions_df) >= 24 else None
        metric_card("24h Average", f"{avg_24h:.0f}" if avg_24h is not None else "N/A")
    with c4:
        metric_card("72h Peak", safe_int(predictions_df["predicted_aqi"].max()))

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"**Health Advisory:** {description}")

    with st.expander("ℹ️ US EPA AQI Health Guidelines"):
        st.markdown("""
        | AQI Range | Category | Recommendation |
        |-----------|----------|----------------|
        | 0–50 | Good | Enjoy outdoor activities |
        | 51–100 | Moderate | Consider reducing prolonged outdoor exertion |
        | 101–150 | Unhealthy for Sensitive Groups | Limit prolonged outdoor exertion |
        | 151–200 | Unhealthy | Reduce prolonged or heavy outdoor exertion |
        | 201–300 | Very Unhealthy | Avoid prolonged outdoor exertion |
        | 301–500 | Hazardous | Stay indoors; keep windows closed |
        """)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 3-Day Forecast",
        "📊 Historical Trends",
        "🌡️ Weather Impact",
        "🤖 Model Comparison",
        "📋 Data Table",
        "📉 Analytics",
    ])

    # ── TAB 1: 3-Day Forecast ─────────────────────────────────────────────────
    with tab1:
        st.subheader("72-Hour AQI Forecast (US EPA scale)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions_df["timestamp"], y=predictions_df["predicted_aqi"],
            mode="lines+markers", name="Predicted AQI",
            line=dict(color="#4da6ff", width=3), marker=dict(size=8),
        ))
        for y_val, col, lbl in [
            (50, "green", "Good"), (100, "yellow", "Moderate"),
            (150, "orange", "USG"), (200, "red", "Unhealthy"),
            (300, "purple", "Very Unhealthy"),
        ]:
            fig.add_hline(y=y_val, line_dash="dash", line_color=col,
                          annotation_text=lbl, annotation_position="right")
        fig.update_layout(
            yaxis=dict(range=[0, max(150, safe_int(predictions_df["predicted_aqi"].max()) + 20)]),
            yaxis_title="AQI (US EPA 0-500)", hovermode="x unified",
        )
        st.plotly_chart(dark_chart(fig, 500), use_container_width=True)

        if len(predictions_df) >= 24:
            st.subheader("Next 24 Hours — Hourly Breakdown")
            h = predictions_df.head(24).copy()
            h["category"] = h["predicted_aqi"].apply(lambda x: get_aqi_category(x)[0])
            fig2 = px.bar(h, x="timestamp", y="predicted_aqi",
                          color="category", color_discrete_map=COLOR_MAP,
                          labels={"predicted_aqi": "AQI"})
            st.plotly_chart(dark_chart(fig2), use_container_width=True)

        st.subheader("Day-by-Day Summary")
        predictions_df["date"] = predictions_df["timestamp"].dt.date
        daily = (
            predictions_df.groupby("date")["predicted_aqi"]
            .agg(["min", "max", "mean"]).round(0).astype(int).reset_index()
        )
        daily.columns = ["date", "Min AQI", "Max AQI", "Avg AQI"]
        for _, row in daily.iterrows():
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f"<div style='background-color:#1a1a2e;border:1px solid #4a4a6a;"
                    f"border-radius:12px;padding:20px;text-align:center;'>"
                    f"<p style='color:#a0a0c0;font-size:12px;margin:0 0 6px 0;'>📅 DATE</p>"
                    f"<p style='color:#fff;font-size:18px;font-weight:700;margin:0;'>{row['date']}</p></div>",
                    unsafe_allow_html=True,
                )
            with c2: metric_card("Min AQI", row["Min AQI"])
            with c3: metric_card("Max AQI", row["Max AQI"])
            with c4: metric_card("Avg AQI", row["Avg AQI"])

    # ── TAB 2: Historical Trends ───────────────────────────────────────────────
    with tab2:
        st.subheader("Historical AQI Trends (Last 7 Days)")
        if not historical_df.empty:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=historical_df["timestamp"], y=historical_df["aqi"],
                mode="lines", name="Actual AQI",
                line=dict(color="#2ca02c", width=2),
                fill="tozeroy", fillcolor="rgba(44,160,44,0.1)",
            ))
            for y_val, col, lbl in [
                (50, "green", "Good"), (100, "yellow", "Moderate"),
                (150, "orange", "USG"), (200, "red", "Unhealthy"),
                (300, "purple", "Very Unhealthy"),
            ]:
                fig3.add_hline(y=y_val, line_dash="dot", line_color=col,
                               annotation_text=lbl, annotation_position="right")
            fig3.update_layout(yaxis_title="AQI (US EPA 0-500)")
            st.plotly_chart(dark_chart(fig3), use_container_width=True)

            st.subheader("7-Day Statistics")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: metric_card("Average",    f"{historical_df['aqi'].mean():.0f}")
            with c2: metric_card("Maximum",    int(historical_df["aqi"].max()))
            with c3: metric_card("Minimum",    int(historical_df["aqi"].min()))
            with c4: metric_card("Std Dev",    f"{historical_df['aqi'].std():.1f}")
            with c5:
                mv = historical_df["aqi"].mode()
                metric_card("Most Common", int(mv[0]) if len(mv) > 0 else "N/A")

            fig4 = px.histogram(historical_df, x="aqi", nbins=20,
                                title="AQI Frequency Distribution",
                                labels={"aqi": "AQI (0-500)"})
            st.plotly_chart(dark_chart(fig4, 300), use_container_width=True)
        else:
            st.info("No historical data available.")

    # ── TAB 3: Weather Impact ─────────────────────────────────────────────────
    with tab3:
        st.subheader("Weather Impact on AQI")
        has_weather = all(c in predictions_df.columns for c in ["temperature", "humidity", "windspeed", "pressure"])
        if has_weather:
            c1, c2 = st.columns(2)
            with c1:
                fig5 = px.scatter(predictions_df, x="temperature", y="predicted_aqi",
                                  color="predicted_aqi", title="Temperature vs AQI",
                                  color_continuous_scale="RdYlGn_r", trendline="lowess",
                                  labels={"predicted_aqi": "AQI"})
                st.plotly_chart(dark_chart(fig5), use_container_width=True)
            with c2:
                fig6 = px.scatter(predictions_df, x="humidity", y="predicted_aqi",
                                  color="predicted_aqi", title="Humidity vs AQI",
                                  color_continuous_scale="RdYlGn_r", trendline="lowess",
                                  labels={"predicted_aqi": "AQI"})
                st.plotly_chart(dark_chart(fig6), use_container_width=True)

            fig7 = px.line(predictions_df.head(48), x="timestamp", y="windspeed",
                           title="Wind Speed Forecast (48h)",
                           labels={"windspeed": "Wind Speed (m/s)"})
            st.plotly_chart(dark_chart(fig7, 300), use_container_width=True)

            st.subheader("Current Weather Conditions")
            cw = predictions_df.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            with c1: metric_card("🌡️ Temperature", f"{float(cw['temperature']):.1f}°C")
            with c2: metric_card("💧 Humidity",    f"{float(cw['humidity']):.0f}%")
            with c3: metric_card("💨 Wind Speed",  f"{float(cw['windspeed']):.1f} m/s")
            with c4: metric_card("🔽 Pressure",    f"{float(cw['pressure']):.0f} hPa")
        else:
            st.info("Weather columns not available. Re-run the inference pipeline.")

    # ── TAB 4: MODEL COMPARISON ───────────────────────────────────────────────
    with tab4:
        st.subheader("🤖 All Models — Training Results & Comparison")
        st.markdown("Every model trained in the pipeline is shown below, evaluated on a **chronological 80/20 train/test split** (no shuffling — simulates real forecasting).")

        if all_models_df.empty:
            st.warning("No model data found in MongoDB. Run `python training_pipeline.py` first.")
        else:
            # ── Best model banner ─────────────────────────────────────────────
            # Show Best Model banner
best_rows = all_models_df[all_models_df["Is Best"] == True]
if not best_rows.empty:
    best = best_rows.iloc[0]
    mc = get_model_color(best["Model"])
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#0d2137,#1F4E79);"
        f"border:2px solid {mc};padding:20px;border-radius:12px;margin-bottom:16px;'>"
        f"<h3 style='color:white;margin:0;'>🏆 Best Model Selected: {best['Model']}</h3></div>",
        unsafe_allow_html=True,
    )

            # ── Individual model cards ────────────────────────────────────────
            st.markdown("### 📋 Individual Model Results")

            cols_per_row = 2
            models_list  = all_models_df.to_dict("records")

            for i in range(0, len(models_list), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j >= len(models_list):
                        break
                    row  = models_list[i + j]
                    mc   = get_model_color(row["Model"])
                    desc = get_model_description(row["Model"])
                    is_best = row["Is Best"]
                    border  = f"2px solid {mc}" if is_best else "1px solid #4a4a6a"
                    badge   = "&nbsp; 🏆 BEST" if is_best else ""

                    # Overfitting indicator
                    rmse_gap = row["Train RMSE"] - row["Test RMSE"]  # negative = overfit
                    if abs(rmse_gap) < 2:
                        fit_label = "✅ Good fit"
                        fit_color = "#66bb6a"
                    elif rmse_gap < 0:
                        fit_label = "⚠️ Slight overfit"
                        fit_color = "#ffca28"
                    else:
                        fit_label = "✅ Generalises well"
                        fit_color = "#66bb6a"

                    with cols[j]:
                        st.markdown(
                            f"""<div style='background-color:#1a1a2e;border:{border};
                                border-radius:12px;padding:18px;margin:6px 0;'>
                                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>
                                  <h4 style='color:{mc};margin:0;'>{row['Model']}{badge}</h4>
                                  <span style='background:{fit_color}22;color:{fit_color};
                                    border:1px solid {fit_color};border-radius:20px;
                                    padding:2px 10px;font-size:11px;font-weight:600;'>{fit_label}</span>
                                </div>
                                <p style='color:#888;font-size:12px;margin:0 0 14px 0;line-height:1.5;'>{desc}</p>

                                <p style='color:#a0a0c0;font-size:10px;font-weight:700;
                                   letter-spacing:1px;margin:0 0 6px 0;'>TEST SET (unseen data)</p>
                                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;'>
                                  <div style='background:#0d2137;border-radius:8px;padding:10px;text-align:center;'>
                                    <p style='color:#a0a0c0;font-size:10px;margin:0;'>RMSE</p>
                                    <p style='color:white;font-size:22px;font-weight:700;margin:0;'>{row['Test RMSE']}</p>
                                  </div>
                                  <div style='background:#0d2137;border-radius:8px;padding:10px;text-align:center;'>
                                    <p style='color:#a0a0c0;font-size:10px;margin:0;'>MAE</p>
                                    <p style='color:white;font-size:22px;font-weight:700;margin:0;'>{row['Test MAE']}</p>
                                  </div>
                                  <div style='background:#0d2137;border-radius:8px;padding:10px;text-align:center;'>
                                    <p style='color:#a0a0c0;font-size:10px;margin:0;'>R²</p>
                                    <p style='color:white;font-size:22px;font-weight:700;margin:0;'>{row['Test R²']}</p>
                                  </div>
                                </div>

                                <p style='color:#555;font-size:10px;font-weight:700;
                                   letter-spacing:1px;margin:0 0 6px 0;'>TRAIN SET</p>
                                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;'>
                                  <div style='background:#111827;border-radius:8px;padding:8px;text-align:center;'>
                                    <p style='color:#555;font-size:10px;margin:0;'>RMSE</p>
                                    <p style='color:#aaa;font-size:17px;font-weight:600;margin:0;'>{row['Train RMSE']}</p>
                                  </div>
                                  <div style='background:#111827;border-radius:8px;padding:8px;text-align:center;'>
                                    <p style='color:#555;font-size:10px;margin:0;'>MAE</p>
                                    <p style='color:#aaa;font-size:17px;font-weight:600;margin:0;'>{row['Train MAE']}</p>
                                  </div>
                                  <div style='background:#111827;border-radius:8px;padding:8px;text-align:center;'>
                                    <p style='color:#555;font-size:10px;margin:0;'>R²</p>
                                    <p style='color:#aaa;font-size:17px;font-weight:600;margin:0;'>{row['Train R²']}</p>
                                  </div>
                                </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Bar charts ────────────────────────────────────────────────────
            st.markdown("### 📊 Side-by-Side Metric Charts")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Test RMSE** — lower is better")
                df_s = all_models_df.sort_values("Test RMSE")
                fig_rmse = go.Figure()
                fig_rmse.add_trace(go.Bar(
                    name="Test RMSE",
                    x=df_s["Model"], y=df_s["Test RMSE"],
                    marker_color=[get_model_color(m) for m in df_s["Model"]],
                    text=df_s["Test RMSE"], textposition="outside",
                ))
                fig_rmse.add_trace(go.Bar(
                    name="Train RMSE",
                    x=df_s["Model"], y=df_s["Train RMSE"],
                    marker_color=[hex_to_rgba(get_model_color(m), 0.4) for m in df_s["Model"]],
                    text=df_s["Train RMSE"], textposition="outside",
                ))
                fig_rmse.update_layout(barmode="group", yaxis_title="RMSE",
                                       legend=dict(orientation="h"))
                st.plotly_chart(dark_chart(fig_rmse, 380), use_container_width=True)

            with c2:
                st.markdown("**Test R²** — higher is better")
                df_s2 = all_models_df.sort_values("Test R²", ascending=False)
                fig_r2 = go.Figure()
                fig_r2.add_trace(go.Bar(
                    name="Test R²",
                    x=df_s2["Model"], y=df_s2["Test R²"],
                    marker_color=[get_model_color(m) for m in df_s2["Model"]],
                    text=df_s2["Test R²"], textposition="outside",
                ))
                fig_r2.add_trace(go.Bar(
                    name="Train R²",
                    x=df_s2["Model"], y=df_s2["Train R²"],
                    marker_color=[hex_to_rgba(get_model_color(m), 0.4) for m in df_s2["Model"]],
                    text=df_s2["Train R²"], textposition="outside",
                ))
                fig_r2.update_layout(barmode="group", yaxis_title="R²",
                                     yaxis=dict(range=[0, 1.1]),
                                     legend=dict(orientation="h"))
                st.plotly_chart(dark_chart(fig_r2, 380), use_container_width=True)

            # MAE bar
            st.markdown("**Test MAE** — lower is better")
            df_mae = all_models_df.sort_values("Test MAE")
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Bar(
                name="Test MAE",
                x=df_mae["Model"], y=df_mae["Test MAE"],
                marker_color=[hex_to_rgba(get_model_color(m), 0.4) for m in df_mae["Model"]],
                text=df_mae["Test MAE"], textposition="outside",
            ))
            fig_mae.add_trace(go.Bar(
                name="Train MAE",
                x=df_mae["Model"], y=df_mae["Train MAE"],
                marker_color=[hex_to_rgba(get_model_color(m), 0.4) for m in df_mae["Model"]],
                text=df_mae["Train MAE"], textposition="outside",
            ))
            fig_mae.update_layout(barmode="group", yaxis_title="MAE",
                                  legend=dict(orientation="h"))
            st.plotly_chart(dark_chart(fig_mae, 360), use_container_width=True)

            # Radar chart
            st.markdown("**Overall Radar — Normalised Performance (higher = better on all axes)**")
            max_rmse = all_models_df["Test RMSE"].max() or 1
            max_mae  = all_models_df["Test MAE"].max()  or 1
            fig_radar = go.Figure()
            cats = ["R²", "Inv RMSE", "Inv MAE"]
            for _, row in all_models_df.iterrows():
                vals = [
                    float(row["Test R²"]),
                    1 - float(row["Test RMSE"]) / max_rmse,
                    1 - float(row["Test MAE"])  / max_mae,
                ]
                vals += [vals[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=cats + [cats[0]],
                    fill="toself", name=row["Model"],
                    line=dict(color=get_model_color(row["Model"])),
                    opacity=0.65,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(dark_chart(fig_radar, 450), use_container_width=True)

            # ── Full metrics table ────────────────────────────────────────────
            st.markdown("### 📄 Complete Metrics Table")
            table_df = all_models_df.copy()
            table_df["Selected"] = table_df["Is Best"].apply(lambda x: "🏆 Best" if x else "")
            show_cols = ["Model", "Test RMSE", "Test MAE", "Test R²",
                         "Train RMSE", "Train MAE", "Train R²",
                         "N Train", "N Test", "N Features", "Selected"]
            show_cols = [c for c in show_cols if c in table_df.columns]
            st.dataframe(table_df[show_cols], use_container_width=True, hide_index=True)

            # ── How models are selected explainer ─────────────────────────────
            with st.expander("ℹ️ How does the pipeline select the best model?"):
                st.markdown("""
                **Selection criterion:** The model with the **lowest Test RMSE** is automatically
                marked `is_active=True` in MongoDB and used for all future predictions.

                **Why RMSE?** Root Mean Squared Error penalises large errors more than MAE.
                In AQI prediction, a single large miss — predicting *Good* when air is *Hazardous* —
                is far more dangerous than many small misses, so RMSE is the right metric to optimise.

                **Train/Test split:** Data is split **chronologically** (80% train / 20% test).
                No shuffling is used. This mirrors real-world deployment where the model always
                predicts future data it has never seen.

                **Metrics explained:**
                | Metric | Meaning | Good value |
                |--------|---------|------------|
                | RMSE | Root Mean Squared Error in AQI units | As low as possible |
                | MAE | Mean Absolute Error in AQI units | As low as possible |
                | R² | % of AQI variance explained (0–1) | Closer to 1.0 |
                | Train vs Test gap | If Train RMSE << Test RMSE, model is overfitting | Should be small |
                """)

    # ── TAB 5: Data Table ─────────────────────────────────────────────────────
    with tab5:
        st.subheader("Detailed Predictions (US EPA AQI 0-500)")
        ddf = predictions_df.copy()
        ddf["timestamp"] = ddf["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        col_map = {
            "timestamp": "Time", "predicted_aqi": "AQI (0-500)",
            "aqi_category": "Category", "temperature": "Temp (°C)",
            "humidity": "Humidity (%)", "windspeed": "Wind (m/s)",
            "pressure": "Pressure (hPa)",
        }
        display_cols = [c for c in col_map if c in ddf.columns]
        ddf = ddf[display_cols].rename(columns=col_map)
        for c in ["Temp (°C)", "Wind (m/s)"]:
            if c in ddf.columns:
                ddf[c] = pd.to_numeric(ddf[c], errors="coerce").round(1)
        for c in ["Humidity (%)", "Pressure (hPa)"]:
            if c in ddf.columns:
                ddf[c] = pd.to_numeric(ddf[c], errors="coerce").round(0)
        st.dataframe(ddf, use_container_width=True, height=600, hide_index=True)
        st.download_button("📥 Download CSV", ddf.to_csv(index=False),
                           f"aqi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           "text/csv", use_container_width=True)

    # ── TAB 6: Analytics ──────────────────────────────────────────────────────
    with tab6:
        st.subheader("Advanced Analytics")

        preds_copy = predictions_df.copy()
        preds_copy["hour"] = preds_copy["timestamp"].dt.hour
        hp = (preds_copy.groupby("hour")["predicted_aqi"]
              .agg(["mean", "std"]).reset_index().fillna(0))

        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=hp["hour"], y=hp["mean"],
                                  mode="lines+markers", name="Avg AQI",
                                  line=dict(color="#4da6ff")))
        fig8.add_trace(go.Scatter(x=hp["hour"], y=hp["mean"] + hp["std"],
                                  mode="lines", line=dict(width=0), showlegend=False))
        fig8.add_trace(go.Scatter(x=hp["hour"], y=hp["mean"] - hp["std"],
                                  mode="lines", line=dict(width=0),
                                  fillcolor="rgba(77,166,255,0.2)", fill="tonexty",
                                  name="±1 Std Dev"))
        fig8.update_layout(xaxis_title="Hour of Day", yaxis_title="AQI (US EPA 0-500)")
        st.plotly_chart(dark_chart(fig8), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            cc = predictions_df["predicted_aqi"].apply(lambda x: get_aqi_category(x)[0]).value_counts()
            if not cc.empty:
                fig9 = px.pie(values=cc.values, names=cc.index,
                              title="Forecast Distribution (72h)",
                              color=cc.index, color_discrete_map=COLOR_MAP)
                fig9.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig9, use_container_width=True)
        with c2:
            if not historical_df.empty and "aqi" in historical_df.columns:
                hc = historical_df["aqi"].apply(lambda x: get_aqi_category(x)[0]).value_counts()
                if not hc.empty:
                    fig10 = px.pie(values=hc.values, names=hc.index,
                                   title="Historical Distribution (7 days)",
                                   color=hc.index, color_discrete_map=COLOR_MAP)
                    fig10.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                    st.plotly_chart(fig10, use_container_width=True)
            else:
                st.info("No historical data available for chart.")

        shap_path = "models/shap_feature_importance.png"
        if os.path.exists(shap_path):
            st.subheader("🔍 SHAP Feature Importance")
            st.image(shap_path, caption="Features with the largest impact on AQI predictions")
        else:
            st.info("SHAP importance plot not yet generated — will appear here after retraining.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: st.caption("📊 OpenWeather API + Open-Meteo API")
    with c2: st.caption(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with c3: st.caption("🤖 Powered by ML | AQI scale: US EPA 0-500")


if __name__ == "__main__":
    main()
