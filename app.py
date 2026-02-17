import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os
import certifi

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

# Page config
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .metric-card {
        background-color: #1e1e2e;
        border: 1px solid #444;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-label {
        color: #aaaaaa;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

def metric_card(label, value):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
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
    if aqi_value == 1:
        return "Good", "#00e400", "Air quality is satisfactory"
    elif aqi_value == 2:
        return "Fair", "#ffff00", "Air quality is acceptable"
    elif aqi_value == 3:
        return "Moderate", "#ff7e00", "Sensitive groups should limit outdoor activity"
    elif aqi_value == 4:
        return "Poor", "#ff0000", "Everyone may experience health effects"
    elif aqi_value == 5:
        return "Very Poor", "#8f3f97", "Health alert: everyone may experience serious effects"
    else:
        return "Unknown", "#cccccc", "No data available"


@st.cache_data(ttl=900)
def load_predictions():
    try:
        db = get_database()
        predictions_col = db["predictions"]
        cursor = predictions_col.find({}, sort=[("created_at", -1)]).limit(72)
        df = pd.DataFrame(list(cursor))
        if df.empty:
            return pd.DataFrame()
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        return df
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def load_historical_data(days=7):
    try:
        db = get_database()
        features_col = db["merged_features"]
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        cursor = features_col.find({"timestamp": {"$gte": cutoff}}, sort=[("timestamp", 1)])
        df = pd.DataFrame(list(cursor))
        if df.empty:
            return pd.DataFrame()
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        return df
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_model_info():
    try:
        db = get_database()
        models_col = db["model_registry"]
        return models_col.find_one({"is_active": True}, sort=[("created_at", -1)])
    except Exception:
        return None


def main():
    st.title("🌍 Karachi Air Quality Index (AQI) Predictor")
    st.markdown("Real-time AQI monitoring and 3-day forecast powered by Machine Learning")

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_Pakistan.svg/320px-Flag_of_Pakistan.svg.png", width=100)
        st.header("📍 Location")
        st.info("**Karachi, Pakistan**  \nLat: 24.8607°N  \nLon: 67.0011°E")

        st.header("ℹ️ About")
        st.markdown("""
        This application predicts Air Quality Index (AQI) for Karachi using:
        - 🌡️ Weather data
        - 💨 Pollution levels
        - 🤖 Machine Learning
        - 📊 40+ engineered features
        """)

        model_info = load_model_info()
        if model_info:
            st.header("🤖 Model Info")
            metric_card("Model Type", model_info['model_name'])
            metric_card("Test RMSE", f"{model_info['metrics']['test_rmse']:.4f}")
            metric_card("Test R²", f"{model_info['metrics']['test_r2']:.4f}")
            st.caption(f"Last trained: {model_info['created_at'].strftime('%Y-%m-%d %H:%M')}")

        st.header("🔄 Controls")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.header("🎨 AQI Scale")
        st.markdown("""
        <div style='background-color:#00e400;padding:5px;margin:2px;border-radius:3px;'><b>1 - Good</b></div>
        <div style='background-color:#ffff00;padding:5px;margin:2px;border-radius:3px;'><b>2 - Fair</b></div>
        <div style='background-color:#ff7e00;padding:5px;margin:2px;border-radius:3px;color:white;'><b>3 - Moderate</b></div>
        <div style='background-color:#ff0000;padding:5px;margin:2px;border-radius:3px;color:white;'><b>4 - Poor</b></div>
        <div style='background-color:#8f3f97;padding:5px;margin:2px;border-radius:3px;color:white;'><b>5 - Very Poor</b></div>
        """, unsafe_allow_html=True)

    with st.spinner('Loading data...'):
        predictions_df = load_predictions()
        historical_df = load_historical_data(days=7)

    if predictions_df.empty:
        st.warning("⚠️ No predictions available. Please run the inference pipeline first.")
        st.code("python prediction_pipeline.py", language="bash")
        st.stop()

    current_aqi = int(predictions_df.iloc[0]['predicted_aqi'])
    category, color, description = get_aqi_category(current_aqi)

    st.header("📊 Current Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Current AQI", current_aqi)
    with col2:
        st.markdown(f"""
        <div style='background-color:{color};padding:20px;border-radius:10px;text-align:center;'>
        <h3 style='margin:0;color:{"white" if current_aqi >= 3 else "black"};'>{category}</h3>
        </div>""", unsafe_allow_html=True)
    with col3:
        if len(predictions_df) >= 24:
            avg_24h = predictions_df.head(24)['predicted_aqi'].mean()
            metric_card("24h Average", f"{avg_24h:.1f}")
        else:
            metric_card("24h Average", "N/A")
    with col4:
        metric_card("72h Peak", int(predictions_df['predicted_aqi'].max()))

    st.info(f"**Health Advisory:** {description}")

    with st.expander("ℹ️ AQI Health Guidelines & Recommendations"):
        st.markdown("""
        | AQI | Category | Health Impact | Recommendations |
        |-----|----------|---------------|-----------------|
        | 1 | Good | Air quality is satisfactory | Enjoy outdoor activities |
        | 2 | Fair | Acceptable for most people | Sensitive people consider limiting outdoor exertion |
        | 3 | Moderate | Sensitive groups may experience effects | Limit prolonged outdoor exertion |
        | 4 | Poor | Everyone may begin to experience effects | Avoid prolonged outdoor exertion |
        | 5 | Very Poor | Health alert | Everyone should avoid all outdoor exertion |
        """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 3-Day Forecast", "📊 Historical Trends",
        "🌡️ Weather Impact", "📋 Data Table", "📉 Analytics"
    ])

    # TAB 1: Forecast
    with tab1:
        st.subheader("3-Day AQI Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions_df['timestamp'], y=predictions_df['predicted_aqi'],
            mode='lines+markers', name='Predicted AQI',
            line=dict(color='#1f77b4', width=3), marker=dict(size=8),
            hovertemplate='<b>Time</b>: %{x}<br><b>AQI</b>: %{y}<extra></extra>'
        ))
        fig.add_hline(y=1.5, line_dash="dash", line_color="green", annotation_text="Good", annotation_position="right")
        fig.add_hline(y=2.5, line_dash="dash", line_color="yellow", annotation_text="Fair", annotation_position="right")
        fig.add_hline(y=3.5, line_dash="dash", line_color="orange", annotation_text="Moderate", annotation_position="right")
        fig.add_hline(y=4.5, line_dash="dash", line_color="red", annotation_text="Poor", annotation_position="right")
        fig.update_layout(title="AQI Forecast (Next 72 Hours)", xaxis_title="Time",
                          yaxis_title="AQI Value", yaxis=dict(range=[0, 6]),
                          hovermode='x unified', height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Next 24 Hours - Hourly Breakdown")
        if len(predictions_df) >= 24:
            hourly_df = predictions_df.head(24).copy()
            hourly_df['category'] = hourly_df['predicted_aqi'].apply(lambda x: get_aqi_category(x)[0])
            fig2 = px.bar(hourly_df, x='timestamp', y='predicted_aqi', color='category',
                          color_discrete_map={'Good': '#00e400', 'Fair': '#ffff00',
                                              'Moderate': '#ff7e00', 'Poor': '#ff0000', 'Very Poor': '#8f3f97'},
                          title="Next 24 Hours - AQI by Hour",
                          labels={'predicted_aqi': 'AQI', 'timestamp': 'Time'})
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Day-by-Day Summary")
        predictions_df['date'] = predictions_df['timestamp'].dt.date
        daily_summary = predictions_df.groupby('date').agg({'predicted_aqi': ['min', 'max', 'mean']}).round(1)
        daily_summary.columns = ['Min AQI', 'Max AQI', 'Avg AQI']
        daily_summary = daily_summary.reset_index()

        for idx, row in daily_summary.iterrows():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # FIX: convert date to string
                st.markdown(f"**📅 {str(row['date'])}**")
            with col2:
                metric_card("Min AQI", float(row['Min AQI']))
            with col3:
                metric_card("Max AQI", float(row['Max AQI']))
            with col4:
                metric_card("Avg AQI", float(row['Avg AQI']))

    # TAB 2: Historical Trends
    with tab2:
        st.subheader("Historical AQI Trends (Last 7 Days)")
        if not historical_df.empty:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=historical_df['timestamp'], y=historical_df['aqi'],
                mode='lines', name='Actual AQI',
                line=dict(color='#2ca02c', width=2),
                fill='tozeroy', fillcolor='rgba(44, 160, 44, 0.1)'
            ))
            fig3.update_layout(title="Historical AQI (Last 7 Days)", xaxis_title="Date",
                               yaxis_title="AQI Value", hovermode='x unified', height=400)
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("7-Day Statistics")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                metric_card("Average", f"{historical_df['aqi'].mean():.2f}")
            with col2:
                metric_card("Maximum", int(historical_df['aqi'].max()))
            with col3:
                metric_card("Minimum", int(historical_df['aqi'].min()))
            with col4:
                metric_card("Std Dev", f"{historical_df['aqi'].std():.2f}")
            with col5:
                mode_val = historical_df['aqi'].mode()
                metric_card("Most Common", int(mode_val[0]) if len(mode_val) > 0 else "N/A")

            fig4 = px.histogram(historical_df, x='aqi', nbins=5,
                                title="AQI Frequency Distribution",
                                labels={'aqi': 'AQI Value', 'count': 'Frequency'})
            fig4.update_layout(height=300)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No historical data available.")

    # TAB 3: Weather Impact
    with tab3:
        st.subheader("Weather Impact on AQI")
        if not predictions_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig5 = px.scatter(predictions_df, x='temperature', y='predicted_aqi',
                                  color='predicted_aqi', title="Temperature vs Predicted AQI",
                                  labels={'temperature': 'Temperature (°C)', 'predicted_aqi': 'AQI'},
                                  color_continuous_scale='RdYlGn_r', trendline="lowess")
                fig5.update_layout(height=400)
                st.plotly_chart(fig5, use_container_width=True)
            with col2:
                fig6 = px.scatter(predictions_df, x='humidity', y='predicted_aqi',
                                  color='predicted_aqi', title="Humidity vs Predicted AQI",
                                  labels={'humidity': 'Humidity (%)', 'predicted_aqi': 'AQI'},
                                  color_continuous_scale='RdYlGn_r', trendline="lowess")
                fig6.update_layout(height=400)
                st.plotly_chart(fig6, use_container_width=True)

            fig7 = px.line(predictions_df.head(48), x='timestamp', y='windspeed',
                           title="Wind Speed Forecast (48 Hours)",
                           labels={'windspeed': 'Wind Speed (m/s)', 'timestamp': 'Time'})
            fig7.update_layout(height=300)
            st.plotly_chart(fig7, use_container_width=True)

            st.subheader("Current Weather Conditions")
            col1, col2, col3, col4 = st.columns(4)
            cw = predictions_df.iloc[0]
            with col1:
                metric_card("🌡️ Temperature", f"{cw['temperature']:.1f}°C")
            with col2:
                metric_card("💧 Humidity", f"{cw['humidity']:.0f}%")
            with col3:
                metric_card("💨 Wind Speed", f"{cw['windspeed']:.1f} m/s")
            with col4:
                metric_card("🔽 Pressure", f"{cw['pressure']:.0f} hPa")

    # TAB 4: Data Table
    with tab4:
        st.subheader("Detailed Predictions")
        display_df = predictions_df.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['category'] = display_df['predicted_aqi'].apply(lambda x: get_aqi_category(x)[0])
        display_columns = {
            'timestamp': 'Time', 'predicted_aqi': 'AQI', 'category': 'Category',
            'temperature': 'Temp (°C)', 'humidity': 'Humidity (%)',
            'windspeed': 'Wind (m/s)', 'pressure': 'Pressure (hPa)'
        }
        display_df = display_df[list(display_columns.keys())].rename(columns=display_columns)
        display_df['Temp (°C)'] = display_df['Temp (°C)'].round(1)
        display_df['Humidity (%)'] = display_df['Humidity (%)'].round(0)
        display_df['Wind (m/s)'] = display_df['Wind (m/s)'].round(1)
        display_df['Pressure (hPa)'] = display_df['Pressure (hPa)'].round(0)
        st.dataframe(display_df, use_container_width=True, height=600, hide_index=True)
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV", data=csv,
            file_name=f"aqi_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", use_container_width=True
        )

    # TAB 5: Analytics
    with tab5:
        st.subheader("Advanced Analytics")
        if not predictions_df.empty:
            predictions_df['hour'] = predictions_df['timestamp'].dt.hour
            hourly_pattern = predictions_df.groupby('hour')['predicted_aqi'].agg(['mean', 'std']).reset_index()
            fig8 = go.Figure()
            fig8.add_trace(go.Scatter(x=hourly_pattern['hour'], y=hourly_pattern['mean'],
                                      mode='lines+markers', name='Average AQI', line=dict(color='blue')))
            fig8.add_trace(go.Scatter(x=hourly_pattern['hour'],
                                      y=hourly_pattern['mean'] + hourly_pattern['std'],
                                      mode='lines', line=dict(width=0), showlegend=False))
            fig8.add_trace(go.Scatter(x=hourly_pattern['hour'],
                                      y=hourly_pattern['mean'] - hourly_pattern['std'],
                                      mode='lines', line=dict(width=0),
                                      fillcolor='rgba(68,68,68,0.3)', fill='tonexty', showlegend=False))
            fig8.update_layout(title="Average AQI by Hour of Day",
                               xaxis_title="Hour of Day", yaxis_title="AQI Value", height=400)
            st.plotly_chart(fig8, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Forecast AQI Distribution")
                category_counts = predictions_df['predicted_aqi'].apply(
                    lambda x: get_aqi_category(x)[0]).value_counts()
                fig9 = px.pie(values=category_counts.values, names=category_counts.index,
                              title="AQI Category Distribution (72h)",
                              color=category_counts.index,
                              color_discrete_map={'Good': '#00e400', 'Fair': '#ffff00',
                                                  'Moderate': '#ff7e00', 'Poor': '#ff0000', 'Very Poor': '#8f3f97'})
                st.plotly_chart(fig9, use_container_width=True)
            with col2:
                if not historical_df.empty:
                    st.markdown("#### Historical AQI Distribution")
                    hist_counts = historical_df['aqi'].apply(lambda x: get_aqi_category(x)[0]).value_counts()
                    fig10 = px.pie(values=hist_counts.values, names=hist_counts.index,
                                   title="AQI Category Distribution (7 days)",
                                   color=hist_counts.index,
                                   color_discrete_map={'Good': '#00e400', 'Fair': '#ffff00',
                                                       'Moderate': '#ff7e00', 'Poor': '#ff0000', 'Very Poor': '#8f3f97'})
                    st.plotly_chart(fig10, use_container_width=True)
        else:
            st.info("Insufficient data for analytics")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("📊 Data sources: OpenWeather API, Open-Meteo API")
    with col2:
        st.caption(f"🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col3:
        st.caption("🤖 Powered by Machine Learning")


if __name__ == "__main__":
    main()
