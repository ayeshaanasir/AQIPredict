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

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

# ── 100% inline styles — nothing Streamlit can override ─────────────────────
def metric_card(label, value):
    st.markdown(f"""
        <div style="background-color:#1a1a2e;border:1px solid #4a4a6a;border-radius:12px;padding:20px 15px;text-align:center;margin:4px 0px;">
            <p style="color:#a0a0c0;font-size:13px;font-weight:600;margin:0 0 8px 0;text-transform:uppercase;letter-spacing:0.5px;">{label}</p>
            <p style="color:#ffffff;font-size:32px;font-weight:700;margin:0;line-height:1.2;">{value}</p>
        </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_database():
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        db = client["aqi_database"]
        db.command('ping')
        return db
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        st.stop()


def get_aqi_category(aqi_value):
    if aqi_value == 1:   return "Good",      "#00e400", "Air quality is satisfactory"
    elif aqi_value == 2: return "Fair",      "#ffff00", "Air quality is acceptable"
    elif aqi_value == 3: return "Moderate",  "#ff7e00", "Sensitive groups should limit outdoor activity"
    elif aqi_value == 4: return "Poor",      "#ff0000", "Everyone may experience health effects"
    elif aqi_value == 5: return "Very Poor", "#8f3f97", "Health alert: everyone may experience serious effects"
    else:                return "Unknown",   "#cccccc", "No data available"


@st.cache_data(ttl=900)
def load_predictions():
    try:
        db = get_database()
        cursor = db["predictions"].find({}, sort=[("created_at", -1)]).limit(72)
        df = pd.DataFrame(list(cursor))
        if df.empty: return pd.DataFrame()
        if '_id' in df.columns: df = df.drop('_id', axis=1)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_historical_data(days=7):
    try:
        db = get_database()
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        cursor = db["merged_features"].find({"timestamp": {"$gte": cutoff}}, sort=[("timestamp", 1)])
        df = pd.DataFrame(list(cursor))
        if df.empty: return pd.DataFrame()
        if '_id' in df.columns: df = df.drop('_id', axis=1)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')
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


def dark_chart(fig, height=400):
    fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    return fig


def main():
    st.title("🌍 Karachi Air Quality Index (AQI) Predictor")
    st.markdown("Real-time AQI monitoring and 3-day forecast powered by Machine Learning")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_Pakistan.svg/320px-Flag_of_Pakistan.svg.png", width=100)
        st.header("📍 Location")
        st.info("**Karachi, Pakistan**  \nLat: 24.8607°N  \nLon: 67.0011°E")
        st.header("ℹ️ About")
        st.markdown("Predicts AQI using:\n- 🌡️ Weather\n- 💨 Pollution\n- 🤖 ML\n- 📊 40+ features")

        model_info = load_model_info()
        if model_info:
            st.header("🤖 Model Info")
            metric_card("Model", model_info['model_name'])
            metric_card("Test RMSE", f"{model_info['metrics']['test_rmse']:.4f}")
            metric_card("Test R²", f"{model_info['metrics']['test_r2']:.4f}")
            st.caption(f"Last trained: {model_info['created_at'].strftime('%Y-%m-%d %H:%M')}")

        st.header("🔄 Controls")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.header("🎨 AQI Scale")
        for color, label in [("#00e400","1 - Good"), ("#ffff00","2 - Fair"), ("#ff7e00","3 - Moderate"), ("#ff0000","4 - Poor"), ("#8f3f97","5 - Very Poor")]:
            txt = "black" if label in ["1 - Good","2 - Fair"] else "white"
            st.markdown(f"<div style='background:{color};padding:5px;margin:2px;border-radius:3px;color:{txt};'><b>{label}</b></div>", unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading data..."):
        predictions_df = load_predictions()
        historical_df  = load_historical_data(days=7)

    if predictions_df.empty:
        st.warning("⚠️ No predictions available. Run the inference pipeline first.")
        st.code("python prediction_pipeline.py", language="bash")
        st.stop()

    current_aqi = int(predictions_df.iloc[0]['predicted_aqi'])
    category, color, description = get_aqi_category(current_aqi)

    # ── Current Status ────────────────────────────────────────────────────────
    st.header("📊 Current Status")
    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card("Current AQI", current_aqi)
    with col2:
        st.markdown(f"""
        <div style='background-color:{color};padding:20px;border-radius:12px;text-align:center;margin:4px 0;'>
        <h2 style='margin:0;color:{"white" if current_aqi >= 3 else "black"};'>{category}</h2>
        </div>""", unsafe_allow_html=True)
    with col3:
        avg_24h = predictions_df.head(24)['predicted_aqi'].mean() if len(predictions_df) >= 24 else None
        metric_card("24h Average", f"{avg_24h:.1f}" if avg_24h else "N/A")
    with col4:
        metric_card("72h Peak", int(predictions_df['predicted_aqi'].max()))

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"**Health Advisory:** {description}")

    with st.expander("ℹ️ AQI Health Guidelines"):
        st.markdown("""
        | AQI | Category | Recommendations |
        |-----|----------|-----------------|
        | 1 | Good | Enjoy outdoor activities |
        | 2 | Fair | Sensitive people limit outdoor exertion |
        | 3 | Moderate | Limit prolonged outdoor exertion |
        | 4 | Poor | Avoid prolonged outdoor exertion |
        | 5 | Very Poor | Everyone avoid all outdoor exertion |
        """)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 3-Day Forecast","📊 Historical Trends","🌡️ Weather Impact","📋 Data Table","📉 Analytics"])

    # TAB 1 ───────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("3-Day AQI Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=predictions_df['timestamp'], y=predictions_df['predicted_aqi'],
            mode='lines+markers', name='Predicted AQI', line=dict(color='#4da6ff', width=3), marker=dict(size=8)))
        for y, c, lbl in [(1.5,"green","Good"),(2.5,"yellow","Fair"),(3.5,"orange","Moderate"),(4.5,"red","Poor")]:
            fig.add_hline(y=y, line_dash="dash", line_color=c, annotation_text=lbl, annotation_position="right")
        fig.update_layout(yaxis=dict(range=[0,6]), hovermode='x unified')
        st.plotly_chart(dark_chart(fig, 500), use_container_width=True)

        if len(predictions_df) >= 24:
            st.subheader("Next 24 Hours - Hourly Breakdown")
            h = predictions_df.head(24).copy()
            h['category'] = h['predicted_aqi'].apply(lambda x: get_aqi_category(x)[0])
            fig2 = px.bar(h, x='timestamp', y='predicted_aqi', color='category',
                color_discrete_map={'Good':'#00e400','Fair':'#ffff00','Moderate':'#ff7e00','Poor':'#ff0000','Very Poor':'#8f3f97'})
            st.plotly_chart(dark_chart(fig2), use_container_width=True)

        st.subheader("Day-by-Day Summary")
        predictions_df['date'] = predictions_df['timestamp'].dt.date
        daily = predictions_df.groupby('date').agg({'predicted_aqi':['min','max','mean']}).round(1)
        daily.columns = ['Min AQI','Max AQI','Avg AQI']
        daily = daily.reset_index()
        for _, row in daily.iterrows():
            c1,c2,c3,c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div style="background-color:#1a1a2e;border:1px solid #4a4a6a;border-radius:12px;padding:20px;text-align:center;">
                    <p style="color:#a0a0c0;font-size:12px;font-weight:600;margin:0 0 6px 0;">📅 DATE</p>
                    <p style="color:#ffffff;font-size:18px;font-weight:700;margin:0;">{str(row['date'])}</p></div>""", unsafe_allow_html=True)
            with c2: metric_card("Min AQI", row['Min AQI'])
            with c3: metric_card("Max AQI", row['Max AQI'])
            with c4: metric_card("Avg AQI", row['Avg AQI'])

    # TAB 2 ───────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Historical AQI Trends (Last 7 Days)")
        if not historical_df.empty:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=historical_df['timestamp'], y=historical_df['aqi'],
                mode='lines', name='Actual AQI', line=dict(color='#2ca02c', width=2),
                fill='tozeroy', fillcolor='rgba(44,160,44,0.1)'))
            st.plotly_chart(dark_chart(fig3), use_container_width=True)

            st.subheader("7-Day Statistics")
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: metric_card("Average",     f"{historical_df['aqi'].mean():.2f}")
            with c2: metric_card("Maximum",     int(historical_df['aqi'].max()))
            with c3: metric_card("Minimum",     int(historical_df['aqi'].min()))
            with c4: metric_card("Std Dev",     f"{historical_df['aqi'].std():.2f}")
            with c5:
                mv = historical_df['aqi'].mode()
                metric_card("Most Common", int(mv[0]) if len(mv) > 0 else "N/A")

            fig4 = px.histogram(historical_df, x='aqi', nbins=5, title="AQI Frequency Distribution")
            st.plotly_chart(dark_chart(fig4, 300), use_container_width=True)
        else:
            st.info("No historical data available.")

    # TAB 3 ───────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Weather Impact on AQI")
        c1, c2 = st.columns(2)
        with c1:
            fig5 = px.scatter(predictions_df, x='temperature', y='predicted_aqi', color='predicted_aqi',
                title="Temperature vs AQI", color_continuous_scale='RdYlGn_r', trendline="lowess")
            st.plotly_chart(dark_chart(fig5), use_container_width=True)
        with c2:
            fig6 = px.scatter(predictions_df, x='humidity', y='predicted_aqi', color='predicted_aqi',
                title="Humidity vs AQI", color_continuous_scale='RdYlGn_r', trendline="lowess")
            st.plotly_chart(dark_chart(fig6), use_container_width=True)

        fig7 = px.line(predictions_df.head(48), x='timestamp', y='windspeed', title="Wind Speed Forecast (48h)")
        st.plotly_chart(dark_chart(fig7, 300), use_container_width=True)

        st.subheader("Current Weather Conditions")
        c1,c2,c3,c4 = st.columns(4)
        cw = predictions_df.iloc[0]
        with c1: metric_card("🌡️ Temperature", f"{cw['temperature']:.1f}°C")
        with c2: metric_card("💧 Humidity",    f"{cw['humidity']:.0f}%")
        with c3: metric_card("💨 Wind Speed",  f"{cw['windspeed']:.1f} m/s")
        with c4: metric_card("🔽 Pressure",    f"{cw['pressure']:.0f} hPa")

    # TAB 4 ───────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Detailed Predictions")
        ddf = predictions_df.copy()
        ddf['timestamp'] = ddf['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        ddf['category']  = ddf['predicted_aqi'].apply(lambda x: get_aqi_category(x)[0])
        cols = {'timestamp':'Time','predicted_aqi':'AQI','category':'Category',
                'temperature':'Temp (°C)','humidity':'Humidity (%)','windspeed':'Wind (m/s)','pressure':'Pressure (hPa)'}
        ddf = ddf[list(cols.keys())].rename(columns=cols)
        for c in ['Temp (°C)','Wind (m/s)']: ddf[c] = ddf[c].round(1)
        for c in ['Humidity (%)','Pressure (hPa)']: ddf[c] = ddf[c].round(0)
        st.dataframe(ddf, use_container_width=True, height=600, hide_index=True)
        st.download_button("📥 Download CSV", ddf.to_csv(index=False),
            f"aqi_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)

    # TAB 5 ───────────────────────────────────────────────────────────────────
    with tab5:
        st.subheader("Advanced Analytics")
        if not predictions_df.empty:
            predictions_df['hour'] = predictions_df['timestamp'].dt.hour
            hp = predictions_df.groupby('hour')['predicted_aqi'].agg(['mean','std']).reset_index()
            fig8 = go.Figure()
            fig8.add_trace(go.Scatter(x=hp['hour'], y=hp['mean'], mode='lines+markers', name='Avg AQI', line=dict(color='#4da6ff')))
            fig8.add_trace(go.Scatter(x=hp['hour'], y=hp['mean']+hp['std'], mode='lines', line=dict(width=0), showlegend=False))
            fig8.add_trace(go.Scatter(x=hp['hour'], y=hp['mean']-hp['std'], mode='lines', line=dict(width=0),
                fillcolor='rgba(77,166,255,0.2)', fill='tonexty', showlegend=False))
            fig8.update_layout(xaxis_title="Hour of Day", yaxis_title="AQI")
            st.plotly_chart(dark_chart(fig8), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                cc = predictions_df['predicted_aqi'].apply(lambda x: get_aqi_category(x)[0]).value_counts()
                fig9 = px.pie(values=cc.values, names=cc.index, title="Forecast (72h)",
                    color=cc.index, color_discrete_map={'Good':'#00e400','Fair':'#ffff00','Moderate':'#ff7e00','Poor':'#ff0000','Very Poor':'#8f3f97'})
                fig9.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig9, use_container_width=True)
            with c2:
                if not historical_df.empty:
                    hc = historical_df['aqi'].apply(lambda x: get_aqi_category(x)[0]).value_counts()
                    fig10 = px.pie(values=hc.values, names=hc.index, title="Historical (7 days)",
                        color=hc.index, color_discrete_map={'Good':'#00e400','Fair':'#ffff00','Moderate':'#ff7e00','Poor':'#ff0000','Very Poor':'#8f3f97'})
                    fig10.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig10, use_container_width=True)

    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    with c1: st.caption("📊 OpenWeather API + Open-Meteo API")
    with c2: st.caption(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with c3: st.caption("🤖 Powered by Machine Learning")


if __name__ == "__main__":
    main()
