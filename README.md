---

# AQI Predictor 🌫️

An end-to-end serverless Machine Learning pipeline that predicts the Air Quality Index (AQI) for the next **3 days**. The project automates data ingestion, feature storage, model training, and deployment using a modular architecture.

---

## 🎯 Features

* **Automated Data Pipeline:** Hourly data fetching from AQICN and OpenWeather APIs.
* **Serverless Feature Store:** Centralized feature management using Hopsworks.
* **Multi-Model Evaluation:** Includes Random Forest, Ridge Regression, and TensorFlow models.
* **Automated CI/CD:** GitHub Actions trigger regular feature updates and daily model re-training.
* **Interactive Dashboard:** Streamlit UI combined with a Flask API backend for real-time predictions.
* **Model Explainability:** SHAP integration to understand which features (like humidity, wind speed, or PM2.5) are driving the predictions.

---

## 🏗️ System Architecture

The project is split into three decoupled pipelines following MLOps best practices:

```
                  ┌────────────────────────┐
                  │  AQICN & OpenWeather   │ (Hourly Fetch)
                  └───────────┬────────────┘
                              ▼
┌──────────────────────────────────────────────────────────┐
│ 1. Feature Pipeline (GitHub Actions)                     │
│    └─► Fetch Data ──► Compute Features ──► Hopsworks FS  │
└─────────────────────────────┬────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────┐
│ 2. Training Pipeline (Daily Re-train)                    │
│    └─► Pull Features ──► Train (RF/TensorFlow) ──► Save   │
└─────────────────────────────┬────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────┐
│ 3. Inference & Dashboard (Streamlit + Flask)             │
│    └─► Load Model ──► Batch Predict ──► Display 3-Day AQI│
└──────────────────────────────────────────────────────────┘

```

---

## 🛠️ Tech Stack

| Component | Tool / Framework |
| --- | --- |
| **Data APIs** | AQICN API, OpenWeatherMap API |
| **Feature Store** | Hopsworks (Free Tier) |
| **ML Frameworks** | Scikit-Learn (Random Forest, Ridge), TensorFlow |
| **Orchestration / CI-CD** | GitHub Actions |
| **Backend API** | Flask |
| **Frontend UI** | Streamlit |
| **Model Interpretability** | SHAP (SHapley Additive exPlanations) |

---

## 📁 Project Structure

```text
├── .github/workflows/        # GitHub Actions for automated execution
├── feature_pipeline/         # Hourly data fetch + feature engineering
│   ├── fetch_data.py
│   ├── compute_features.py
│   └── upload_to_hopsworks.py
├── training_pipeline/        # Daily model training & evaluation
│   └── train.py
├── inference/                # Loads the latest model & predicts next 3 days
│   └── predict.py
├── dashboard/                # Streamlit UI + Flask API
│   ├── app.py                # Streamlit interface
│   └── api.py                # Flask backend services
├── .env.example              # Template for environment variables
├── requirements.txt          # Project dependencies
└── README.md

```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+ or Python 3.10
* Hopsworks Account (Free Tier)
* API Keys for AQICN and OpenWeather

### Setup Instructions

1. **Clone the repository:**
```bash
git clone <your-repo-link>
cd aqi-predictor

```



```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

```

3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



```

4. **Environment Variables:**
   `cp .env.example .env` (ya `.env.local`) karein aur apni API keys add karein:
   ```env
   HOPSWORKS_API_KEY=your_hopsworks_key
   AQICN_API_KEY=your_aqicn_key
   OPENWEATHER_API_KEY=your_openweather_key

```

---

## 🏃‍♂️ Running the Project

### 1. Run Feature Pipeline Manually

To fetch data, process features, and insert them into the Hopsworks Feature Store:

```bash
python -m feature_pipeline.fetch_data
python -m feature_pipeline.compute_features
python -m feature_pipeline.upload_to_hopsworks

```

### 2. Run Training Pipeline

To train the models using features from Hopsworks:

```bash
python -m training_pipeline.train

```

### 3. Launch the Dashboard & API

Start the Flask backend first (if detached) or run the Streamlit app:

```bash
# To run the Streamlit UI
streamlit run dashboard/app.py

```

---

## 🤖 Model Performance & Explainability

* Describe briefly here which model performs best (e.g., "Random Forest achieved the lowest MAE of X.XX").
* SHAP plots are saved in the `inference/` or `dashboard/` directory to show feature importance.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---