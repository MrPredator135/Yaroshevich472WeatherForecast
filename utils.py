import requests
import pandas as pd

DAILY_VARS = [
    "precipitation_sum", "rain_sum", "temperature_2m_max", 
    "temperature_2m_min", "wind_speed_10m_max", 
    "shortwave_radiation_sum", "sunshine_duration"
]

def fetch_weather_data(lat, lon, start_date, end_date, is_archive=True):
    if is_archive:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_date, "end_date": end_date,
            "daily": ",".join(DAILY_VARS), "timezone": "UTC"
        }
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "forecast_days": 7, "daily": ",".join(DAILY_VARS), "timezone": "UTC"
        }
    
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def process_data(json_data):
    df = pd.DataFrame(json_data["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df = df.set_index("date").drop(columns=["time"])
    
    if "precipitation_sum" in df.columns:
        df["target"] = (df["precipitation_sum"] > 0).astype(int)
    
    return df