# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import pandas as pd
from utils import fetch_weather_data, process_data

st.title("Weather Forecast ML")

try:
    model = joblib.load("weather_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
except:
    st.error("Please run train.py first!")
    st.stop()

if st.button("Get 7-day Forecast"):
    forecast_raw = fetch_weather_data(38.7223, -9.1333, None, None, is_archive=False)
    df_forecast = process_data(forecast_raw)
    
    X_new = df_forecast[features]
    X_new_scaled = scaler.transform(X_new)
    
    probs = model.predict_proba(X_new_scaled)[:, 1]
    
    res = pd.DataFrame({
        "Date": df_forecast.index.date,
        "Probability (%)": (probs * 100).round(1),
        "Result": ["Rain expected" if p > 0.5 else "No rain" for p in probs]
    })
    
    st.table(res)