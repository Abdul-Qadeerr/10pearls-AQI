# Pearls AQI Predictor 🌫️

End-to-end serverless ML pipeline that predicts Air Quality Index (AQI) for the next **3 days**.

## Stack
| Component | Tool |
|---|---|
| Data APIs | AQICN, OpenWeather |
| Feature Store | Hopsworks (free tier) |
| ML Models | Random Forest, Ridge, TensorFlow |
| CI/CD | GitHub Actions |
| Dashboard | Streamlit + Flask |
| Explainability | SHAP |

## Setup

```bash
git clone <your-repo>
cd aqi-predictor
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env .env.local             # fill in your keys
```

## Run feature pipeline manually
```bash
python -m feature_pipeline.fetch_data
python -m feature_pipeline.compute_features
python -m feature_pipeline.upload_to_hopsworks
```

## Project Structure
```
feature_pipeline/    ← hourly data fetch + feature engineering
training_pipeline/   ← daily model training
inference/           ← load model + predict next 3 days
dashboard/           ← Streamlit UI + Flask API
```
