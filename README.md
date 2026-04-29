---

# **Pearls AQI Predictor**

AI-Powered Air Quality Forecasting System (3-Day Prediction)

---

## **Project Overview**

Pearls AQI Predictor is an end-to-end machine learning system that forecasts the **Air Quality Index (AQI)** for the next 3 days using a serverless architecture.

The project demonstrates a complete pipeline including **data collection, preprocessing, feature engineering, model training, and deployment** with a web-based dashboard for real-time predictions.

---

## **Key Features**

### **Feature Engineering Pipeline**

* Fetches real-time data from external APIs (AQICN, OpenWeather)
* Generates time-based features (hour, day, month)
* Creates derived features such as AQI change rate
* Stores processed data in a feature store (Hopsworks or Vertex AI)

### **Historical Data Backfill**

* Processes historical data for model training
* Builds structured datasets for evaluation

### **Model Training Pipeline**

* Implements multiple models:

  * Random Forest
  * Ridge Regression
  * TensorFlow models
* Evaluates models using RMSE, MAE, and R²
* Stores trained models in a model registry

### **Automated Pipelines (CI/CD)**

* Feature pipeline runs hourly
* Training pipeline runs daily
* Uses Apache Airflow or GitHub Actions

### **Web Dashboard**

* Built using Streamlit and Flask
* Displays:

  * Real-time AQI predictions
  * 3-day forecast
  * Interactive visualizations

### **Advanced Analytics**

* Exploratory Data Analysis (EDA)
* Feature importance using SHAP or LIME
* AQI alerts for hazardous conditions

---

## **Tech Stack**

* Python
* Scikit-learn
* TensorFlow
* Hopsworks / Vertex AI
* Apache Airflow / GitHub Actions
* Streamlit / Flask
* AQICN / OpenWeather APIs
* SHAP
* Git

---

## **Project Workflow**

1. Extract data from APIs
2. Transform data through cleaning and feature engineering
3. Load processed data into a feature store
4. Train and evaluate machine learning models
5. Deploy predictions through a web dashboard

---

## Project Architecture

flowchart LR

A[External APIs\n(AQICN / OpenWeather)] --> B[Data Extraction]

B --> C[Feature Engineering Pipeline]
C --> D[Feature Store\n(Hopsworks / Vertex AI)]

D --> E[Training Pipeline]
E --> F[Model Training\n(Random Forest / Ridge / TensorFlow)]
F --> G[Model Registry]

D --> H[Prediction Pipeline]
G --> H

H --> I[Real-Time Predictions]

I --> J[Web Dashboard\n(Streamlit / Flask)]

subgraph Automation
K[Airflow / GitHub Actions]
end

K --> C
K --> E
---

## **Example Use Case**

A user can view predicted AQI levels for the next three days and take precautionary measures if hazardous conditions are expected.

---

## **Future Improvements**

* Add more advanced deep learning models
* Improve prediction accuracy with additional features
* Deploy on cloud infrastructure for scalability

---

## **Author**

Abdul Qadeer
BS Computer Science – Sukkur IBA University

---
