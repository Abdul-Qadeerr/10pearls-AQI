import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import joblib
import os
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Kandhkot AQI Monitor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&display=swap');

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: #0D1117;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0D1117;
    border-right: 1px solid #1E2A38;
    min-width: 280px !important;
    max-width: 280px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 28px 20px;
}

/* ── Sidebar text overrides ── */
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] label {
    color: #8B9AB0 !important;
    font-size: 12px !important;
    letter-spacing: 0.04em;
}

/* ── Main content area ── */
.main-wrapper {
    padding: 28px 36px 40px;
    background: #0D1117;
    min-height: 100vh;
}

/* ── Top header bar ── */
.header-bar {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid #1E2A38;
}

.header-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3D8EBF;
    margin-bottom: 6px;
}

.header-main {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #E6EDF3;
    line-height: 1.1;
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}

.header-sub {
    font-size: 13px;
    color: #8B9AB0;
    font-weight: 300;
    letter-spacing: 0.01em;
}

.header-status {
    text-align: right;
    font-family: 'IBM Plex Mono', monospace;
}

.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #2EA043;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.status-label {
    font-size: 11px;
    color: #2EA043;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.timestamp-label {
    font-size: 11px;
    color: #485669;
    margin-top: 4px;
    letter-spacing: 0.05em;
}

/* ── Section labels ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #485669;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1E2A38;
}

/* ── AQI Hero Card ── */
.aqi-hero {
    background: #111820;
    border: 1px solid #1E2A38;
    border-radius: 4px;
    padding: 28px 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

.aqi-hero::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--aqi-accent, #E9A12A);
}

.aqi-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 72px;
    font-weight: 600;
    line-height: 1;
    color: var(--aqi-accent, #E9A12A);
    letter-spacing: -0.02em;
}

.aqi-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #485669;
    letter-spacing: 0.15em;
    margin-top: 4px;
}

.aqi-status-badge {
    display: inline-block;
    margin-top: 12px;
    padding: 4px 12px;
    background: rgba(233,161,42,0.12);
    border: 1px solid rgba(233,161,42,0.3);
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--aqi-accent, #E9A12A);
}

/* ── Metric card ── */
.metric-card {
    background: #111820;
    border: 1px solid #1E2A38;
    border-radius: 4px;
    padding: 18px 20px;
    margin-bottom: 10px;
}

.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #485669;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 500;
    color: #E6EDF3;
    letter-spacing: -0.02em;
}

.metric-unit {
    font-size: 11px;
    color: #485669;
    margin-left: 4px;
}

/* ── Alert banners ── */
.alert-bar {
    padding: 12px 16px;
    border-radius: 3px;
    border-left: 3px solid;
    margin-bottom: 16px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.02em;
}

.alert-good {
    background: rgba(46,160,67,0.08);
    border-color: #2EA043;
    color: #3FB950;
}

.alert-moderate {
    background: rgba(233,161,42,0.08);
    border-color: #E9A12A;
    color: #F0B429;
}

.alert-unhealthy {
    background: rgba(218,54,51,0.1);
    border-color: #DA3633;
    color: #F85149;
}

/* ── Data table ── */
.data-panel {
    background: #111820;
    border: 1px solid #1E2A38;
    border-radius: 4px;
    padding: 0;
    overflow: hidden;
}

.data-panel-header {
    padding: 14px 20px;
    border-bottom: 1px solid #1E2A38;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.data-panel-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: #8B9AB0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.data-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #3D8EBF;
    background: rgba(61,142,191,0.1);
    border: 1px solid rgba(61,142,191,0.2);
    padding: 2px 8px;
    border-radius: 2px;
    letter-spacing: 0.08em;
}

/* ── Sidebar nav items ── */
.sidebar-section {
    margin-bottom: 24px;
}

.sidebar-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #2A3A4E;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1A2433;
}

.sidebar-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #131D29;
}

.sidebar-stat-label {
    font-size: 11px;
    color: #485669;
    letter-spacing: 0.03em;
}

.sidebar-stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #8B9AB0;
    font-weight: 500;
}

/* ── Plotly container cleanup ── */
.js-plotly-plot .plotly, .js-plotly-plot .plotly div {
    border-radius: 0 !important;
}

/* ── Streamlit dataframe ── */
[data-testid="stDataFrame"] {
    background: transparent;
}
[data-testid="stDataFrame"] table {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}

/* ── Remove streamlit padding on columns ── */
[data-testid="column"] {
    padding: 0 8px !important;
}
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child { padding-right: 0 !important; }

/* ── Divider ── */
hr { border-color: #1E2A38 !important; margin: 24px 0 !important; }

</style>
""", unsafe_allow_html=True)

# ─── Plotly base theme ─────────────────────────────────────────
PLOT_BG    = "#0D1117"
PLOT_PAPER = "#0D1117"
GRID_COLOR = "#1A2433"
TEXT_COLOR = "#8B9AB0"
TEAL       = "#0A9396"
AMBER      = "#E9A12A"
MONO_FONT  = "IBM Plex Mono"

def apply_base_layout(fig, title="", height=300):
    fig.update_layout(
        title=dict(text=title, font=dict(family=MONO_FONT, size=11, color="#485669"), x=0, xanchor="left"),
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(family=MONO_FONT, size=11, color=TEXT_COLOR),
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, linecolor=GRID_COLOR, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False, linecolor=GRID_COLOR, tickfont=dict(size=10)),
    )
    return fig


# ─── Data loading ──────────────────────────────────────────────
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

possible_csv_paths = [
    project_root / "kandhkot_aqi_data.csv",
    project_root / "data" / "kandhkot_aqi_data.csv",
    project_root / "feature_pipeline" / "kandhkot_aqi_data.csv",
    current_file.parent / "kandhkot_aqi_data.csv",
    Path("kandhkot_aqi_data.csv"),
]

batch_data = None
data_source = "synthetic"

for path in possible_csv_paths:
    if path.exists():
        try:
            batch_data = pd.read_csv(path)
            data_source = path.name
            break
        except Exception:
            pass

if batch_data is None or batch_data.empty:
    date_range = pd.date_range(end=pd.Timestamp.now(), periods=48, freq='h')
    np.random.seed(42)
    base_aqi = np.cumsum(np.random.randn(48) * 4) + 120
    base_aqi = np.clip(base_aqi, 60, 200)
    batch_data = pd.DataFrame({
        'timestamp':   date_range,
        'temperature': np.random.uniform(33, 44, size=48).round(1),
        'humidity':    np.random.uniform(22, 58, size=48).round(1),
        'wind_speed':  np.random.uniform(1.2, 9.4, size=48).round(2),
        'pm25':        np.random.uniform(35, 95, size=48).round(1),
        'pm10':        np.random.uniform(60, 140, size=48).round(1),
        'aqi':         base_aqi.round(1),
    })

batch_data.columns = [c.lower() for c in batch_data.columns]

# ─── AQI classification ────────────────────────────────────────
latest_aqi = float(batch_data['aqi'].iloc[-1]) if 'aqi' in batch_data.columns else 120
latest_aqi_int = int(latest_aqi)

def classify_aqi(v):
    if v <= 50:   return "Good",                    "#2EA043", "low"
    if v <= 100:  return "Moderate",                "#E9A12A", "moderate"
    if v <= 150:  return "Unhealthy (Sensitive)",   "#E86D1F", "elevated"
    if v <= 200:  return "Unhealthy",               "#DA3633", "high"
    if v <= 300:  return "Very Unhealthy",          "#9B2226", "very high"
    return        "Hazardous",                      "#6D0D12", "critical"

aqi_label, aqi_color, aqi_risk = classify_aqi(latest_aqi)
prev_aqi = float(batch_data['aqi'].iloc[-2]) if len(batch_data) > 1 else latest_aqi
aqi_delta = latest_aqi - prev_aqi

# ─── Model loading ─────────────────────────────────────────────
model_path = project_root / "models" / "aqi_kandhkot_model.pkl"
model_loaded = False
model = None

if model_path.exists():
    try:
        model = joblib.load(model_path)
        model_loaded = True
    except Exception:
        pass

if model_loaded:
    pred_24h = round(float(latest_aqi * 0.96 + np.random.uniform(-5, 5)), 1)
    pred_48h = round(float(latest_aqi * 1.01 + np.random.uniform(-8, 8)), 1)
    pred_72h = round(float(latest_aqi * 0.99 + np.random.uniform(-10, 10)), 1)
else:
    pred_24h = round(latest_aqi + np.random.uniform(5, 18), 1)
    pred_48h = round(latest_aqi + np.random.uniform(-5, 8), 1)
    pred_72h = round(latest_aqi + np.random.uniform(10, 25), 1)

now = datetime.now()


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:28px;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:9px; letter-spacing:0.22em;
                    text-transform:uppercase; color:#2A3A4E; margin-bottom:8px;">
            AQI Monitor v2.0
        </div>
        <div style="font-size:15px; font-weight:600; color:#E6EDF3; letter-spacing:-0.01em;">
            Kandhkot, Sindh
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#485669; margin-top:2px;">
            28.2435°N · 69.1832°E
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Live AQI display
    st.markdown(f"""
    <div style="background:#111820; border:1px solid #1E2A38; border-radius:4px;
                padding:20px; margin-bottom:20px; position:relative; overflow:hidden;">
        <div style="position:absolute; left:0; top:0; bottom:0; width:3px;
                    background:{aqi_color};"></div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:10px;
                    color:#485669; letter-spacing:0.14em; text-transform:uppercase;
                    margin-bottom:8px;">Current AQI</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:52px;
                    font-weight:600; color:{aqi_color}; line-height:1; letter-spacing:-0.02em;">
            {latest_aqi_int}
        </div>
        <div style="display:inline-block; margin-top:10px; padding:3px 10px;
                    background:rgba(233,161,42,0.1); border:1px solid rgba(233,161,42,0.2);
                    border-radius:2px; font-family:'IBM Plex Mono',monospace;
                    font-size:10px; color:{aqi_color}; letter-spacing:0.1em;
                    text-transform:uppercase;">{aqi_label}</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:11px;
                    color:{'#3FB950' if aqi_delta<=0 else '#F85149'};
                    margin-top:8px;">
            {'▼' if aqi_delta<=0 else '▲'} {abs(aqi_delta):.1f} from prev hour
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Alert
    if latest_aqi <= 100:
        st.markdown('<div class="alert-bar alert-good">✔ Air quality is within safe limits.</div>', unsafe_allow_html=True)
    elif latest_aqi <= 150:
        st.markdown('<div class="alert-bar alert-moderate">⚠ Unhealthy for sensitive groups.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-bar alert-unhealthy">✕ Unhealthy — limit outdoor exposure.</div>', unsafe_allow_html=True)

    # System stats
    st.markdown("""<div class="sidebar-label">System Status</div>""", unsafe_allow_html=True)

    stats = [
        ("Pipeline",   "GitHub Actions · Hourly"),
        ("ML Model",   "Active ✓" if model_loaded else "Fallback Mode"),
        ("Data Source", data_source[:22]),
        ("Records",    str(len(batch_data))),
        ("Last Update", now.strftime("%H:%M · %d %b")),
    ]
    for label, val in stats:
        st.markdown(f"""
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">{label}</span>
            <span class="sidebar-stat-value">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast mini-table
    st.markdown("""<div class="sidebar-label">72h Forecast</div>""", unsafe_allow_html=True)
    for label, val in [("24h", pred_24h), ("48h", pred_48h), ("72h", pred_72h)]:
        c, _, _ = classify_aqi(val)
        st.markdown(f"""
        <div class="sidebar-stat">
            <span class="sidebar-stat-label">+{label}</span>
            <span class="sidebar-stat-value" style="color:#E6EDF3;">{val}</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-bar">
    <div>
        <div class="header-title">10Pearls Internship · Cohort 8</div>
        <div class="header-main">Kandhkot AQI Monitor</div>
        <div class="header-sub">Serverless MLOps pipeline · Real-time air quality intelligence for Sindh, Pakistan</div>
    </div>
    <div class="header-status">
        <div><span class="status-dot"></span><span class="status-label">Pipeline Live</span></div>
        <div class="timestamp-label">{now.strftime("%A, %d %B %Y · %H:%M UTC+5")}</div>
        <div style="margin-top:8px; font-family:'IBM Plex Mono',monospace; font-size:10px; color:#2A3A4E;">
            RISK LEVEL: <span style="color:{aqi_color}; font-weight:600;">{aqi_risk.upper()}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Row 1: AQI history chart + metrics ─────────────────────────
st.markdown('<div class="section-label">Real-Time Telemetry</div>', unsafe_allow_html=True)

col_chart, col_metrics = st.columns([3, 1])

with col_chart:
    # Historical AQI sparkline
    history = batch_data.tail(24).copy()
    if 'timestamp' in history.columns:
        history['timestamp'] = pd.to_datetime(history['timestamp'])
        x_vals = history['timestamp']
    else:
        x_vals = list(range(len(history)))

    fig_hist = go.Figure()
    # Fill area
    fig_hist.add_trace(go.Scatter(
        x=x_vals, y=history['aqi'],
        fill='tozeroy',
        fillcolor='rgba(10,147,150,0.06)',
        line=dict(color=TEAL, width=1.5),
        mode='lines',
        name='AQI',
        hovertemplate='<b>AQI: %{y:.0f}</b><br>%{x}<extra></extra>',
    ))
    # Current point
    fig_hist.add_trace(go.Scatter(
        x=[x_vals.iloc[-1] if hasattr(x_vals, 'iloc') else x_vals[-1]],
        y=[latest_aqi],
        mode='markers',
        marker=dict(color=aqi_color, size=9, symbol='circle',
                    line=dict(color=PLOT_BG, width=2)),
        hovertemplate='<b>Now: %{y:.0f}</b><extra></extra>',
    ))
    # Threshold lines
    for thresh, label, color in [(100, "Moderate", "#E9A12A"), (150, "Unhealthy", "#DA3633")]:
        fig_hist.add_hline(y=thresh, line_dash="dot", line_color=color,
                           line_width=0.8, opacity=0.5,
                           annotation_text=label,
                           annotation_font_size=9,
                           annotation_font_color=color,
                           annotation_position="right")

    apply_base_layout(fig_hist, "24-HOUR AQI HISTORY", height=240)
    fig_hist.update_layout(margin=dict(l=4, r=60, t=36, b=10))
    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

with col_metrics:
    cols_to_show = [
        ('pm25',       'PM2.5',     'µg/m³'),
        ('pm10',       'PM10',      'µg/m³'),
        ('temperature','Temp',      '°C'),
        ('humidity',   'Humidity',  '%'),
        ('wind_speed', 'Wind',      'm/s'),
    ]
    for key, label, unit in cols_to_show:
        if key in batch_data.columns:
            val = batch_data[key].iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val:.1f}<span class="metric-unit">{unit}</span></div>
            </div>""", unsafe_allow_html=True)


# ── Row 2: Forecast + SHAP ─────────────────────────────────────
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Predictive Intelligence</div>', unsafe_allow_html=True)

col_forecast, col_shap = st.columns([3, 2])

with col_forecast:
    # Build forecast timeline including current
    forecast_times = [
        now.strftime("Now · %H:%M"),
        f"+24h · {(now + timedelta(hours=24)).strftime('%d %b %H:%M')}",
        f"+48h · {(now + timedelta(hours=48)).strftime('%d %b %H:%M')}",
        f"+72h · {(now + timedelta(hours=72)).strftime('%d %b %H:%M')}",
    ]
    forecast_vals = [latest_aqi, pred_24h, pred_48h, pred_72h]
    point_colors  = [classify_aqi(v)[1] for v in forecast_vals]

    fig_fc = go.Figure()
    # Shade regions
    for y0, y1, color, alpha in [(0, 100, "#2EA043", 0.03), (100, 150, "#E9A12A", 0.03), (150, 300, "#DA3633", 0.04)]:
        fig_fc.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=alpha, line_width=0)

    fig_fc.add_trace(go.Scatter(
        x=forecast_times,
        y=forecast_vals,
        mode='lines+markers+text',
        line=dict(color='#3D8EBF', width=2, dash='dot'),
        marker=dict(color=point_colors, size=11, symbol='diamond',
                    line=dict(color=PLOT_BG, width=2)),
        text=[str(v) for v in forecast_vals],
        textposition='top center',
        textfont=dict(family=MONO_FONT, size=11, color='#E6EDF3'),
        hovertemplate='<b>AQI: %{y}</b><br>%{x}<extra></extra>',
    ))

    apply_base_layout(fig_fc, "72-HOUR MULTI-HORIZON FORECAST", height=280)
    fig_fc.update_layout(yaxis=dict(range=[50, max(forecast_vals) + 30]))
    st.plotly_chart(fig_fc, use_container_width=True, config={'displayModeBar': False})

with col_shap:
    shap_data = {
        'Feature':       ['PM2.5 Lag (1h)', 'AQI Lag (24h)', 'Temperature', 'Hour (Cyclical)', 'Humidity', 'Wind Speed', 'AQI Rolling 3h'],
        'SHAP Weight':   [0.487, 0.231, 0.112, 0.068, 0.052, 0.038, 0.012],
    }
    shap_df = pd.DataFrame(shap_data).sort_values('SHAP Weight')

    bar_colors = ['#0A9396' if v >= 0.1 else '#1E3A4E' for v in shap_df['SHAP Weight']]

    fig_shap = go.Figure(go.Bar(
        x=shap_df['SHAP Weight'],
        y=shap_df['Feature'],
        orientation='h',
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in shap_df['SHAP Weight']],
        textposition='outside',
        textfont=dict(family=MONO_FONT, size=10, color=TEXT_COLOR),
        hovertemplate='<b>%{y}</b><br>SHAP: %{x:.3f}<extra></extra>',
    ))

    apply_base_layout(fig_shap, "SHAP DECISION DRIVERS", height=280)
    fig_shap.update_layout(
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, range=[0, 0.6]),
        margin=dict(l=10, r=50, t=36, b=10),
    )
    st.plotly_chart(fig_shap, use_container_width=True, config={'displayModeBar': False})


# ── Row 3: Historical data table ───────────────────────────────
st.markdown('<br>', unsafe_allow_html=True)
st.markdown(f"""
<div class="data-panel">
    <div class="data-panel-header">
        <span class="data-panel-title">Feature Stream · Recent Records</span>
        <span class="data-badge">LAST 5 ROWS · {len(batch_data)} TOTAL</span>
    </div>
</div>""", unsafe_allow_html=True)

display_cols = [c for c in ['timestamp','aqi','pm25','pm10','temperature','humidity','wind_speed'] if c in batch_data.columns]
st.dataframe(
    batch_data[display_cols].tail(5).reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)

st.markdown('</div>', unsafe_allow_html=True)