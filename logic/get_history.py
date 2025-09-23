import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MAPS_KEY")

def get_air_quality_history(latitude: float, longitude: float, days_back: int = 30) -> dict | None:
    history_url = f"https://airquality.googleapis.com/v1/history:lookup?key={API_KEY}"
    total_hours = 24 * days_back
    all_hours_info = []
    page_token = None
    while True:
        payload = {"hours": total_hours, "location": {"latitude": latitude, "longitude": longitude}}
        if page_token:
            payload['pageToken'] = page_token
        try:
            response = requests.post(history_url, json=payload)
            response.raise_for_status()
            data = response.json()
            all_hours_info.extend(data.get('hoursInfo', []))
            page_token = data.get('nextPageToken')
            if not page_token:
                break
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None
    return {"hoursInfo": all_hours_info}

def main():
    """Main function to run the history data fetching and RETURN the data."""
    print("--- Fetching historical data ---")
    test_lat = 19.2183
    test_lon = 72.9781
    
    history_data = get_air_quality_history(test_lat, test_lon, days_back=30)
    
    if history_data:
        print("✅ Data fetched successfully.")
        return history_data # <-- Key change: Return the data
    else:
        print("--- Failed to fetch data from the API ---")
        return None

if __name__ == "__main__":
    main()