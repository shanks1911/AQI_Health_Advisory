import requests
import pandas as pd
from datetime import datetime, timedelta

def get_weather_history(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Fetches historical hourly weather data from the Open-Meteo Archive API
    for a specific date range.
    """
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m'
    }

    print(f"Fetching historical weather from {start_date} to {end_date}...")
    try:
        response = requests.get(weather_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during historical weather request: {e}")
        return None

def get_recent_weather(latitude: float, longitude: float, past_days: int = 2) -> pd.DataFrame | None:
    """
    Fetches recent past weather data using the Open-Meteo Forecast API.
    """
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m',
        'past_days': past_days
    }
    
    print(f"Fetching recent weather for the last {past_days} days...")
    try:
        response = requests.get(forecast_url, params=params)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during recent weather request: {e}")
        return None

def get_weather_forecast(latitude: float, longitude: float, forecast_days: int = 5) -> pd.DataFrame | None:
    """Fetches future hourly weather forecast data from the Open-Meteo Forecast API."""
    forecast_url = "https://api.open-Meteo.com/v1/forecast"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m',
        'forecast_days': forecast_days
    }
    print(f"Fetching {forecast_days}-day weather forecast...")
    try:
        response = requests.get(forecast_url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during weather forecast request: {e}")
        return None