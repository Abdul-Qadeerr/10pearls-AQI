import os
import joblib
import pandas as pd
import numpy as np

# Path to the saved model
MODEL_PATH = "models/aqi_kandhkot_model.pkl"

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}. Please run the training pipeline first.")
    
    print("📥 Loading trained Random Forest model...")
    return joblib.load(MODEL_PATH)

def predict_current_aqi(temperature, humidity, pressure, wind_speed):
    # Load model instance
    model = load_trained_model()
    
    # Structure input features exactly in the order the model was trained on
    features = ["temperature", "humidity", "pressure", "wind_speed"]
    input_data = pd.DataFrame([[temperature, humidity, pressure, wind_speed]], columns=features)
    
    # Generate prediction
    print("🔮 Calculating AQI prediction...")
    predicted_aqi = model.predict(input_data)[0]
    
    return predicted_aqi

if __name__ == "__main__":
    print("--- 🌫️ Kandhkot AQI Inference System ---")
    
    # Sample real-time weather parameters for testing
    current_temp = 38.5     # in °C
    current_humid = 55.0    # in %
    current_press = 1008.0  # in hPa
    current_wind = 4.2      # in m/s
    
    print(f"\n📋 Input Features:")
    print(f"   🌡️ Temperature: {current_temp}°C")
    print(f"   💧 Humidity: {current_humid}%")
    print(f"   🎈 Pressure: {current_press} hPa")
    print(f"   💨 Wind Speed: {current_wind} m/s\n")
    
    try:
        prediction = predict_current_aqi(current_temp, current_humid, current_press, current_wind)
        print(f"🎯 Predicted AQI for Kandhkot: {prediction:.2f}")
        
        # AQI Classification Thresholds
        if prediction <= 50:
            print("🟢 Category: Good")
        elif prediction <= 100:
            print("🟡 Category: Moderate")
        elif prediction <= 150:
            print("🟠 Category: Unhealthy for Sensitive Groups")
        else:
            print("🔴 Category: Hazardous / Severe Pollution")
            
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")