import requests # type: ignore
import os
from dotenv import load_dotenv # type: ignore
import streamlit as st

load_dotenv()

def get_api_key(key_name):
    """Get API key from Streamlit secrets or environment variables"""
    try:
        # Streamlit secrets (for Streamlit Cloud)
        return st.secrets[key_name]
    except:
        # Fall back
        return os.getenv(key_name)
    
# API_KEY = os.getenv("MAPS_KEY")
API_KEY = get_api_key("MAPS_KEY")

if not API_KEY:
    raise ValueError("API key not found. Make sure you have a .env file with MAPS_KEY set.")

def get_coords_from_city(city_name: str) -> tuple | None:
    """Converts a city name to latitude and longitude."""
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={city_name}&key={API_KEY}"
    try:
        response = requests.get(geocode_url)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'OK':
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def get_air_quality(latitude: float, longitude: float) -> dict | None:
    """Fetches air quality data and returns it as a dictionary."""
    air_quality_url = f"https://airquality.googleapis.com/v1/currentConditions:lookup?key={API_KEY}"
    payload = {
        "location": {"latitude": latitude, "longitude": longitude},
        "extraComputations": ["HEALTH_RECOMMENDATIONS", "POLLUTANT_CONCENTRATION", "LOCAL_AQI"]
    }
    try:
        response = requests.post(air_quality_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None