# 🌫️ AQI Predictor - Serverless MLOps Pipeline for Air Quality Forecasting

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat&logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Hopsworks](https://img.shields.io/badge/Hopsworks-FF6B00?style=flat&logo=apache&logoColor=white)](https://www.hopsworks.ai/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen)](https://karachi-aqi-monitor.streamlit.app/)

> **🏆 10Pearls Internship Program | Cohort 8 | Abdul Qadeer | Sukkur IBA University**

An **end-to-end serverless Machine Learning pipeline** that predicts the Air Quality Index (AQI) for **Karachi, Sindh, Pakistan** for the next 3 days. The system automates data ingestion, feature storage, model training, and deployment using a **production-grade MLOps architecture**.

---

## 🔗 Quick Links

| Link | URL |
|------|-----|
| 🚀 **Live Dashboard** | [karachi-aqi-monitor.streamlit.app](https://karachi-aqi-monitor.streamlit.app/) |
| 📂 **GitHub Repository** | [github.com/Abdul-Qadeerr/10pearls-AQI](https://github.com/Abdul-Qadeerr/10pearls-AQI) |
| 📊 **Hopsworks Feature Store** | `aqi_predictor_10shine` |

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🤖 **Automated Data Pipeline** | Hourly data fetching from AQICN + OpenWeather APIs |
| 💾 **Serverless Feature Store** | Production-grade feature management via Hopsworks |
| 🧠 **Multi-Model Evaluation** | Random Forest + LSTM with automatic best-model selection |
| ⚙️ **CI/CD Automation** | GitHub Actions — hourly feature update + daily retraining |
| 📊 **Interactive Dashboard** | Streamlit UI with real-time AQI, EDA charts, and hazard alerts |
| 🔬 **Model Explainability** | SHAP TreeExplainer plots for feature importance |
| 📧 **Email Alerts** | Automatic email notifications for hazardous AQI levels |
| 📈 **EDA Section** | Historical trend, distribution, hourly pattern, correlation heatmap |
| 💾 **Fallback Cache** | Local CSV backup when APIs are unavailable |

---

## 🛠️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AQI PREDICTIVE SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │  AQICN API   │   │ OpenWeather  │   │  Hopsworks   │        │
│  │  (Hourly)    │   │     API      │   │ Feature Store│        │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                  │                  │                 │
│         └────────┬─────────┘                  │                 │
│                  │                            │                 │
│                  ▼                            │                 │
│     ┌─────────────────────┐                  │                 │
│     │   Feature Pipeline  │──────────────────┘                 │
│     │   (GitHub Actions)  │                                     │
│     └──────────┬──────────┘                                     │
│                │                                                │
│                ▼                                                │
│     ┌─────────────────────┐                                     │
│     │  Training Pipeline  │                                     │
│     │  (Daily Retraining) │                                     │
│     └──────────┬──────────┘                                     │
│                │                                                │
│                ▼                                                │
│     ┌─────────────────────┐                                     │
│     │ Streamlit Dashboard │                                     │
│     │  (Real-time UI)     │                                     │
│     └─────────────────────┘                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Hourly Trigger** → GitHub Actions cron job activates
2. **Data Ingestion** → AQICN + OpenWeather APIs fetch raw data
3. **Feature Engineering** → 27 features generated (cyclical encoding, lag features)
4. **Feature Store** → Data uploaded to Hopsworks (fallback: local CSV)
5. **Daily Training** → Random Forest + LSTM models trained and evaluated
6. **Model Registry** → Best model saved to `models/` folder
7. **Dashboard** → Streamlit loads model and shows predictions + EDA

---

## 📂 Project Structure

```
aqi-predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml      # Hourly data ingestion
│   └── training_pipeline.yml     # Daily model retraining
├── feature_pipeline/
│   ├── fetch_data.py             # API integration
│   ├── compute_features.py       # 27-feature engineering
│   ├── upload_to_hopsworks.py    # Feature store upload
│   └── backfill_pipeline.py      # 90-day historical backfill
├── training_pipeline/
│   └── train.py                  # RF + LSTM training
├── dashboard/
│   └── app.py                    # Streamlit UI
├── models/
│   ├── model_rf_24h.pkl          # Random Forest — 24h forecast
│   ├── model_rf_48h.pkl          # Random Forest — 48h forecast
│   ├── model_rf_72h.pkl          # Random Forest — 72h forecast
│   ├── scaler.pkl                # Feature scaler
│   └── model_info.json           # Model metrics (RMSE, MAE, R²)
├── data/
│   └── karachi_aqi_data.csv      # 90-day historical cache
├── requirements.txt
└── README.md
```

---

## 📊 Tech Stack

| Component | Technology |
|-----------|------------|
| **Data APIs** | AQICN API, OpenWeatherMap API |
| **Feature Store** | Hopsworks (Free Tier) |
| **ML Frameworks** | Scikit-Learn, TensorFlow-CPU |
| **Orchestration** | GitHub Actions (ubuntu-latest) |
| **Frontend** | Streamlit + Plotly |
| **Explainability** | SHAP (TreeExplainer) |
| **Email Alerts** | yagmail |

---

## 📊 Model Performance

| Model | Horizon | RMSE | MAE | R² |
|-------|---------|------|-----|----|
| Random Forest | 24h | 18.62 | 14.1 | 0.31 |
| Random Forest | 48h | 23.63 | 18.2 | -0.10 |
| Random Forest | 72h | 25.10 | 19.8 | -0.21 |
| LSTM | 24h | 71.72 | 58.4 | -9.30 |
| LSTM | 48h | 73.65 | 60.1 | -9.64 |
| LSTM | 72h | 75.09 | 61.3 | -9.84 |

**Selected Model:** Random Forest (Avg RMSE: 22.45 vs LSTM: 73.48)

> **Note on R²:** Negative R² for 48h and 72h horizons is expected with synthetic backfill data.
> With real historical API data, these metrics will improve significantly.

---


## 🛠️ Setup Instructions

### Prerequisites

- Python 3.10
- Hopsworks Account (Free Tier)
- AQICN API Key — [Get here](https://aqicn.org/data-platform/token/)
- OpenWeather API Key — [Get here](https://openweathermap.org/api)

### Environment Variables

Create a `.env` file in the project root:

```env
AQICN_TOKEN=your_aqicn_token
OPENWEATHER_API_KEY=your_openweather_key
HOPSWORKS_API_KEY=your_hopsworks_key
CITY_NAME=Karachi
CITY_LAT=24.8607
CITY_LON=67.0011


### Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run backfill (one time — generates 90 days historical data)
python -m feature_pipeline.backfill_pipeline

# Train models
python -m training_pipeline.train

# Launch dashboard locally
streamlit run dashboard/app.py

# Run feature pipeline manually
python -m feature_pipeline.upload_to_hopsworks
```

---

## 🔮 Replication for Other Cities

To adapt for other cities (Hyderabad, Sukkur, Lahore):

1. Update coordinates in `.env` file:

```env
CITY_NAME=Lahore
CITY_LAT=31.5204
CITY_LON=74.3587
```

2. Run backfill pipeline for new location
3. Retrain models
4. Dashboard auto-updates with new data

The pipeline is fully parameterized — just change `CITY_NAME`, `CITY_LAT`, and `CITY_LON`.

---

## 👤 Author

**Abdul Qadeer**

- 10Pearls Internship Program — Cohort 8
- Sukkur IBA University
- [GitHub Profile](https://github.com/Abdul-Qadeerr)

---

## 📄 License

This project is part of the **10Pearls Shine Internship Program (Cohort 8)**.