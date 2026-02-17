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
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────
# US EPA AQI category helper (0-500 scale)
# ─────────────────────────────────────────────────────────────

def get_aqi_category(aqi_value):
    """Map US EPA AQI (0-500) to (label, color, description). Type-safe."""
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
    """Safely convert any MongoDB/numpy value to a plain Python int."""
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
        st.error(
            f"🚨 **HAZARDOUS AQI ALERT** — Peak forecast AQI: **{max_aqi}**  \n"
            "Outdoor activity is dangerous for everyone. Stay indoors, seal windows, and use air purifiers."
        )
    elif max_aqi > 200:
        st.error(
            f"⛔ **VERY UNHEALTHY AIR QUALITY** — Peak forecast AQI: **{max_aqi}**  \n"
            "Everyone should avoid prolonged outdoor activity. Sensitive groups must stay indoors."
        )
    elif max_aqi > 150:
        st.warning(
            f"⚠️ **UNHEALTHY AIR QUALITY** — Peak forecast AQI: **{max_aqi}**  \n"
            "Children, elderly, and people with respiratory/heart conditions should limit outdoor exertion."
        )
    elif max_aqi > 100:
        st.warning(
            f"⚠️ **UNHEALTHY FOR SENSITIVE GROUPS** — Peak forecast AQI: **{max_aqi}**  \n"
            "Sensitive individuals should reduce prolonged outdoor activity."
        )


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
        # Type-safe conversion — handles numpy int64, float64, strings, None
        df["predicted_aqi"] = df["predicted_aqi"].apply(lambda x: safe_int(x, default=0))
        df = df.sort_values("timestamp").reset_index(drop=True)
        # Pre-compute category column
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
        cursor = db["merged_features"].find(
            {"timestamp": {"$gte": cutoff}},
            sort=[("timestamp", 1)],
        )
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

    if predictions_df.empty:
        st.warning("⚠️ No predictions available. Run the inference pipeline first.")
        st.code("python prediction_pipeline.py", language="bash")
        st.stop()

    # ── Alert banners ─────────────────────────────────────────────────────────
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 3-Day Forecast",
        "📊 Historical Trends",
        "🌡️ Weather Impact",
        "📋 Data Table",
        "📉 Analytics",
    ])

    # ── TAB 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("72-Hour AQI Forecast (US EPA scale)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions_df["timestamp"],
            y=predictions_df["predicted_aqi"],
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

    # ── TAB 2 ─────────────────────────────────────────────────────────────────
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

    # ── TAB 3 ─────────────────────────────────────────────────────────────────
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

    # ── TAB 4 ─────────────────────────────────────────────────────────────────
    with tab4:
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

    # ── TAB 5 ─────────────────────────────────────────────────────────────────
    with tab5:
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
