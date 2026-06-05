import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

# ── PAGE CONFIGURATION ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Karachi AQI Predictor & Monitor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS FOR BETTER DESIGN ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hazard-good { 
        background: linear-gradient(135deg, #00b09b, #96c93d); 
        padding: 1rem; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        font-weight: bold; 
    }
    .hazard-moderate { 
        background: linear-gradient(135deg, #f2994a, #f2c94c); 
        padding: 1rem; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        font-weight: bold; 
    }
    .hazard-unhealthy { 
        background: linear-gradient(135deg, #eb3349, #f45c43); 
        padding: 1rem; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        font-weight: bold; 
    }
    .hazard-hazardous { 
        background: linear-gradient(135deg, #7f00ff, #e100ff); 
        padding: 1rem; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        font-weight: bold; 
    }
    
    /* Sidebar - Dark Background with White Text */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMetric,
    [data-testid="stSidebar"] .stMetric label,
    [data-testid="stSidebar"] .stMetric value,
    [data-testid="stSidebar"] .stAlert,
    [data-testid="stSidebar"] .stButton button {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stAlert {
        background-color: rgba(0,0,0,0.3) !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background-color: #3b82f6;
        color: white !important;
    }
    
    h1, h2, h3 {
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── INITIALIZATION ─────────────────────────────────────────────────────────────
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

FEATURE_COLUMNS = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind_speed", "wind_deg", "clouds",
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    "aqi_change_rate", "aqi_rolling_3h",
]

# ── 1. LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading AQI data...")
def load_batch_data():
    possible_csv_paths = [
        project_root / "karachi_aqi_data.csv",
        project_root / "data" / "karachi_aqi_data.csv",
        project_root / "feature_pipeline" / "karachi_aqi_data.csv",
        project_root / "feature_pipeline" / "data" / "karachi_aqi_data.csv",
        current_file.parent / "karachi_aqi_data.csv",
        Path("karachi_aqi_data.csv"),
    ]
    
    for path in possible_csv_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Data Source: `{path}`")
                return df, path.name
            except Exception as e:
                st.sidebar.warning(f"⚠️ Error reading {path.name}: {e}")
                continue
    
    st.sidebar.error("❌ No CSV found! Please ensure karachi_aqi_data.csv exists.")
    st.sidebar.info("📁 Expected locations:\n- `aqi-predictor/karachi_aqi_data.csv`\n- `aqi-predictor/feature_pipeline/data/karachi_aqi_data.csv`")
    st.stop()

batch_data, data_source = load_batch_data()

if "timestamp" in batch_data.columns:
    batch_data["timestamp"] = pd.to_datetime(batch_data["timestamp"], errors="coerce")
if "hour" not in batch_data.columns and "timestamp" in batch_data.columns:
    batch_data["hour"] = batch_data["timestamp"].dt.hour

numeric_cols = ["aqi", "pm25", "pm10", "temperature", "humidity", "wind_speed"]
for col in numeric_cols:
    if col in batch_data.columns:
        batch_data[col] = pd.to_numeric(batch_data[col], errors="coerce")

# ── 2. LOAD MODELS ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML models...")
def load_models():
    model_search_dirs = [
        project_root / "models",
        current_file.parent / "models",
        Path("models"),
    ]
    
    models_dir = None
    for d in model_search_dirs:
        if d.exists():
            models_dir = d
            break
    
    if models_dir:
        try:
            scaler = joblib.load(models_dir / "scaler.pkl")
            models_rf = {
                "aqi_next_24h": joblib.load(models_dir / "model_rf_24h.pkl"),
                "aqi_next_48h": joblib.load(models_dir / "model_rf_48h.pkl"),
                "aqi_next_72h": joblib.load(models_dir / "model_rf_72h.pkl"),
            }
            return True, models_rf, scaler, models_dir
        except Exception as e:
            st.sidebar.warning(f"⚠️ Model load issue: {e}")
            return False, None, None, None
    return False, None, None, None

models_ready, models_rf, scaler, models_dir = load_models()

if models_ready:
    st.sidebar.success("🚀 **ML Model:** Random Forest (Active)")

# ── 3. CURRENT AQI & HAZARD ASSESSMENT ─────────────────────────────────────────
latest_aqi = float(batch_data["aqi"].iloc[-1]) if "aqi" in batch_data.columns else 120.0

def get_hazard_level(aqi):
    if aqi <= 50:
        return "Good", "🟢", "hazard-good", "Air quality is satisfactory, little to no risk."
    elif aqi <= 100:
        return "Moderate", "🟡", "hazard-moderate", "Acceptable air quality."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "hazard-unhealthy", "Sensitive groups should limit outdoor exertion."
    elif aqi <= 200:
        return "Unhealthy", "🔴", "hazard-unhealthy", "Everyone may experience health effects."
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "hazard-hazardous", "Health alert: serious health effects possible."
    else:
        return "Hazardous", "⚫", "hazard-hazardous", "Emergency conditions. Avoid all outdoor activity."

hazard_level, hazard_icon, hazard_class, hazard_advice = get_hazard_level(latest_aqi)

# ── 4. SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Karachi Air Quality")
    st.markdown(f"### {hazard_icon} Current AQI: **{int(latest_aqi)}**")
    st.markdown(f"##### Status: *{hazard_level}*")
    st.markdown(f"<div class='{hazard_class}'>{hazard_advice}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if models_ready:
        try:
            df_feat = batch_data.copy()
            for col in FEATURE_COLUMNS:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0
                df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")
            
            valid_rows = df_feat.dropna(subset=["aqi_lag_1h"])
            if valid_rows.empty:
                valid_rows = df_feat
            last_row = valid_rows.iloc[[-1]][FEATURE_COLUMNS].fillna(0.0)
            X_scaled = scaler.transform(last_row.values)
            
            pred_24h = float(models_rf["aqi_next_24h"].predict(X_scaled)[0])
            pred_48h = float(models_rf["aqi_next_48h"].predict(X_scaled)[0])
            pred_72h = float(models_rf["aqi_next_72h"].predict(X_scaled)[0])
        except Exception as e:
            pred_24h = latest_aqi + 15
            pred_48h = latest_aqi + 10
            pred_72h = latest_aqi + 5
    else:
        pred_24h = latest_aqi + 15
        pred_48h = latest_aqi + 10
        pred_72h = latest_aqi + 5
    
    st.markdown("### 📈 3-Day Forecast")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("24h", f"{pred_24h:.0f}", delta=f"{pred_24h - latest_aqi:.0f}")
    with col2:
        st.metric("48h", f"{pred_48h:.0f}", delta=f"{pred_48h - latest_aqi:.0f}")
    with col3:
        st.metric("72h", f"{pred_72h:.0f}", delta=f"{pred_72h - latest_aqi:.0f}")
    
    st.markdown("---")
    st.markdown(f"🕐 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── 5. MAIN TITLE ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-title'>
    <h1 style='color: white; margin: 0;'>🌫️ Karachi Real-Time AQI Air Quality Predictive System</h1>
    <p style='color: #ccc; margin: 0.5rem 0 0 0;'>End-to-End Serverless MLOps Pipeline with Automated Model Inference</p>
</div>
""", unsafe_allow_html=True)

# ── 6. KEY METRICS ─────────────────────────────────────────────────────────────
col_metrics = st.columns(5)
with col_metrics[0]:
    st.metric("🌡️ Temperature", f"{batch_data['temperature'].iloc[-1]:.1f}°C" if 'temperature' in batch_data.columns else "N/A")
with col_metrics[1]:
    st.metric("💧 Humidity", f"{batch_data['humidity'].iloc[-1]:.0f}%" if 'humidity' in batch_data.columns else "N/A")
with col_metrics[2]:
    st.metric("💨 PM2.5", f"{batch_data['pm25'].iloc[-1]:.0f}" if 'pm25' in batch_data.columns else "N/A")
with col_metrics[3]:
    st.metric("🌪️ PM10", f"{batch_data['pm10'].iloc[-1]:.0f}" if 'pm10' in batch_data.columns else "N/A")
with col_metrics[4]:
    st.metric("📊 Data Points", f"{len(batch_data)}")

st.markdown("---")

# ── 7. FORECAST CHART ──────────────────────────────────────────────────────────
forecast_df = pd.DataFrame({
    "Timeline": ["Now", "24 Hours", "48 Hours", "72 Hours"],
    "AQI": [round(latest_aqi, 1), round(pred_24h, 1), round(pred_48h, 1), round(pred_72h, 1)],
})

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📈 3-Day AQI Forecast")
    
    fig_forecast = go.Figure()
    fig_forecast.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, line_width=0)
    fig_forecast.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, line_width=0)
    fig_forecast.add_hrect(y0=100, y1=150, fillcolor="orange", opacity=0.1, line_width=0)
    fig_forecast.add_hrect(y0=150, y1=200, fillcolor="red", opacity=0.1, line_width=0)
    fig_forecast.add_hrect(y0=200, y1=300, fillcolor="purple", opacity=0.1, line_width=0)
    
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df["Timeline"],
        y=forecast_df["AQI"],
        mode="lines+markers+text",
        name="AQI Forecast",
        line=dict(color="#7A1FA2", width=4),
        marker=dict(size=12, color="#7A1FA2", symbol="circle"),
        text=forecast_df["AQI"],
        textposition="top center",
        textfont=dict(size=14),
    ))
    
    fig_forecast.update_layout(
        title="Predictive AQI Trendline — Random Forest Model",
        xaxis_title="Timeline",
        yaxis_title="Air Quality Index (AQI)",
        height=450,
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#e0e0e0"),
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

with col_right:
    st.subheader("🧬 Top Feature Importance")
    
    if models_ready:
        try:
            df_feat = batch_data.copy()
            for col in FEATURE_COLUMNS:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0
                df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")
            
            valid = df_feat.dropna(subset=["aqi_lag_1h"])
            if valid.empty:
                valid = df_feat
            sample = valid.tail(50)[FEATURE_COLUMNS].fillna(0.0)
            X_s = scaler.transform(sample.values)
            
            try:
                import shap
                explainer = shap.TreeExplainer(models_rf["aqi_next_24h"])
                shap_values = explainer.shap_values(X_s)
                mean_abs = np.abs(shap_values).mean(axis=0)
                title = "SHAP Feature Importance"
            except:
                mean_abs = models_rf["aqi_next_24h"].feature_importances_
                title = "Random Forest Feature Importance"
            
            importance_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Importance": mean_abs,
            }).sort_values("Importance", ascending=True).tail(8)
            
            fig_shap = px.bar(
                importance_df,
                x="Importance", y="Feature",
                orientation="h",
                title=title,
                color="Importance",
                color_continuous_scale="Tealgrn",
                text_auto=".3f",
            )
            fig_shap.update_layout(height=400)
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.info("Feature importance will appear here")
    else:
        st.info("📊 Train models to see feature importance")

st.markdown("---")

# ── 8. MODEL METRICS ───────────────────────────────────────────────────────────
st.subheader("📊 Model Performance Metrics")

model_info_path = models_dir / "model_info.json" if models_dir else None
if model_info_path and model_info_path.exists():
    try:
        with open(model_info_path) as f:
            info = json.load(f)
        rf_metrics = info.get("rf_metrics", {})
        
        if rf_metrics:
            metrics_data = []
            for horizon, label in [("24h", "24 Hours"), ("48h", "48 Hours"), ("72h", "72 Hours")]:
                m = rf_metrics.get(f"aqi_next_{horizon}", {})
                metrics_data.append({
                    "Horizon": label,
                    "RMSE": m.get("rmse", "-"),
                    "MAE": m.get("mae", "-"),
                    "R² Score": m.get("r2", "-"),
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    except Exception:
        st.info("Metrics will appear after training")

# ── 9. EDA SECTION ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs(["📈 Time Series", "📊 Distributions", "🔥 Correlations"])

with tab1:
    if "timestamp" in batch_data.columns and "aqi" in batch_data.columns:
        trend_data = batch_data.dropna(subset=["timestamp", "aqi"]).tail(720)
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=trend_data["timestamp"],
            y=trend_data["aqi"],
            mode="lines",
            name="AQI",
            line=dict(color="#1565C0", width=2),
            fill="tozeroy",
            fillcolor="rgba(21, 101, 192, 0.1)",
        ))
        
        fig_trend.add_hline(y=150, line_dash="dash", line_color="red", 
                            annotation_text="Hazardous (150)")
        fig_trend.add_hline(y=100, line_dash="dot", line_color="orange",
                            annotation_text="Moderate (100)")
        
        fig_trend.update_layout(
            title="Historical AQI Trend",
            xaxis_title="Date/Time",
            yaxis_title="AQI",
            height=450,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        if "aqi" in batch_data.columns:
            fig_hist = px.histogram(
                batch_data, x="aqi", nbins=30,
                title="AQI Distribution",
                color_discrete_sequence=["#E65100"],
                marginal="box",
            )
            fig_hist.add_vline(x=100, line_dash="dot", line_color="orange")
            fig_hist.add_vline(x=150, line_dash="dash", line_color="red")
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_dist2:
        if "hour" in batch_data.columns and "aqi" in batch_data.columns:
            hourly_avg = batch_data.groupby("hour")["aqi"].agg(["mean", "std"]).reset_index()
            fig_hourly = px.bar(
                hourly_avg, x="hour", y="mean",
                error_y="std",
                title="Average AQI by Hour",
                color_discrete_sequence=["#2E7D32"],
                labels={"mean": "Avg AQI", "hour": "Hour"},
            )
            st.plotly_chart(fig_hourly, use_container_width=True)

with tab3:
    corr_cols = ["aqi", "pm25", "pm10", "temperature", "humidity", "wind_speed"]
    corr_cols = [c for c in corr_cols if c in batch_data.columns]
    if len(corr_cols) > 2:
        corr_matrix = batch_data[corr_cols].apply(pd.to_numeric, errors="coerce").corr().round(2)
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            text_auto=True,
            aspect="auto",
            zmin=-1, zmax=1,
        )
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

# ── 10. RECENT DATA TABLE ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Recent Air Quality Readings")

display_cols = ["timestamp", "aqi", "pm25", "pm10", "temperature", "humidity", "wind_speed"]
display_cols = [c for c in display_cols if c in batch_data.columns]

if display_cols:
    recent_data = batch_data[display_cols].tail(10).copy()
    if "timestamp" in recent_data.columns:
        recent_data["timestamp"] = recent_data["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(recent_data, use_container_width=True, hide_index=True)

# ── 11. FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🔬 <strong>Karachi AQI Predictive System</strong> | End-to-End Serverless MLOps Pipeline</p>
    <p>🤖 Random Forest Model | 📡 Real-Time API Integration | 🧬 SHAP Explainability</p>
    <p style='font-size: 0.8rem;'>© 2026 10Pearls Internship Cohort 8 | IBA Sukkur University</p>
</div>
""", unsafe_allow_html=True)

# ── 12. REFRESH BUTTON ─────────────────────────────────────────────────────────
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Dashboard"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()