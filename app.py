import streamlit as st
import pandas as pd
import joblib
from utils import fetch_weather_data, process_data
from sklearn.metrics import accuracy_score

st.title("Weather Forecast ML App")

# 1. Data Connection
st.header("1. Data Connection")
data_option = st.radio("Select data source:", ["CSV Upload", "Open-Meteo API"])

df = None
if data_option == "CSV Upload":
    uploaded_file = st.file_uploader("Upload your weather_daily.csv", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
else:
    if st.button("Get data from Open-Meteo"):
        raw = fetch_weather_data(38.7223, -9.1333, "2023-01-01", "2024-12-31")
        df = process_data(raw)
        st.success("Data successfully downloaded!")

# 2. Model Training
st.header("2. Model Training")
if st.button("Train Model"):
    try:
  
        model = joblib.load("weather_model.pkl")
        best_acc = joblib.load("best_acc.pkl")  

        st.info("Model loaded successfully.")
        st.metric(label="Best Model Accuracy", value=f"{best_acc * 100:.2f} %") 
    except:
        st.error("Model file not found. Please run train.py first!")

# 3. 7-Day Forecast
st.header("3. 7-Day Precipitation Forecast")
if st.button("Make Prediction"):
    try:
        model = joblib.load("weather_model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("features.pkl")
        
        # Get fresh 7-day forecast
        forecast_raw = fetch_weather_data(38.7223, -9.1333, None, None, is_archive=False)
        forecast_df = process_data(forecast_raw)
        
        # Predict probability
        X_new = forecast_df[features]
        X_new_scaled = scaler.transform(X_new)
        probs = model.predict_proba(X_new_scaled)[:, 1]
        
        # Create results table
        res = pd.DataFrame({
            "Date": forecast_df.index.date,
            "Probability (%)": (probs * 100).round(1),
            "Result": ["Rain expected" if p > 0.5 else "No rain" for p in probs]
        })
        
        st.table(res)
        
    except Exception as e:
        st.error(f"Error: {e}. Please train the model first!")