import requests
import os
import json
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

def get_api_key(key_name):
    """Get API key from Streamlit secrets or environment variables"""
    try:
        return st.secrets[key_name]
    except:
        return os.getenv(key_name)

API_KEY = get_api_key("MAPS_KEY")


# ══════════════════════════════════════════════════════════════════════════════
#  EXISTING FUNCTION — unchanged
#  Google Air Quality History API (hard cap: 30 days / 720 hours)
# ══════════════════════════════════════════════════════════════════════════════

def get_air_quality_history(latitude: float, longitude: float, days_back: int = 30) -> dict | None:
    """
    Fetches up to 30 days of hourly AQI history from the Google Air Quality API.
    Returns a dict with key 'hoursInfo' containing a list of hourly records.
    This is the primary data source — Google data takes priority in the merge.
    """
    history_url = f"https://airquality.googleapis.com/v1/history:lookup?key={API_KEY}"

    # Google API hard-caps at 720 hours regardless of what we pass
    capped_days = min(days_back, 30)
    total_hours = 24 * capped_days

    all_hours_info = []
    page_token = None

    while True:
        payload = {
            "hours": total_hours,
            "location": {"latitude": latitude, "longitude": longitude}
        }
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
            print(f"Google AQI history error: {e}")
            return None

    return {"hoursInfo": all_hours_info}


# ══════════════════════════════════════════════════════════════════════════════
#  NEW FUNCTION — OpenAQ v3 API (Issue 5)
#  Free, no API key required, supports years of historical data
#  Provides PM2.5 and PM10 measurements by nearest sensor to given coordinates
# ══════════════════════════════════════════════════════════════════════════════

def get_openaq_history(
    latitude: float,
    longitude: float,
    days_back: int = 150,
    radius_meters: int = 25000
) -> pd.DataFrame | None:
    """
    Fetches historical hourly PM2.5 and PM10 data from the OpenAQ v3 API
    for the nearest sensor(s) within radius_meters of the given coordinates.

    OpenAQ is free, requires no API key, and provides data going back years —
    making it the ideal supplement to the 30-day Google API cap.

    The returned DataFrame is indexed by UTC timestamp and has columns:
        pm25  — µg/m³ (NaN where unavailable)
        pm10  — µg/m³ (NaN where unavailable)

    These are then used by merge_aqi_sources() to compute a supplementary
    AQI estimate for the pre-Google period (days 31–180).

    Parameters
    ──────────
    latitude, longitude : location to search around
    days_back           : how many days of history to fetch (up to ~365+)
    radius_meters       : search radius for nearest OpenAQ sensor

    Returns
    ───────
    pd.DataFrame indexed by UTC datetime, or None on failure
    """
    BASE_URL = "https://api.openaq.org/v3"

    # ── Step 1: Find nearest location IDs within radius ───────────────────────
    try:
        loc_resp = requests.get(
            f"{BASE_URL}/locations",
            params={
                "coordinates": f"{latitude},{longitude}",
                "radius":      radius_meters,
                "parameters":  "pm25,pm10",
                "limit":       5,          # Take up to 5 nearest stations
                "order_by":    "distance",
            },
            timeout=15,
        )
        loc_resp.raise_for_status()
        locations = loc_resp.json().get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"OpenAQ location search error: {e}")
        return None

    if not locations:
        print(f"No OpenAQ sensors found within {radius_meters}m of ({latitude}, {longitude}). "
              f"Try increasing radius_meters.")
        return None

    location_ids = [loc["id"] for loc in locations]
    print(f"Found {len(location_ids)} OpenAQ station(s): {location_ids}")

    # ── Step 2: Fetch measurements for each location ──────────────────────────
    date_to   = datetime.utcnow()
    date_from = date_to - timedelta(days=days_back)

    all_records = []

    for loc_id in location_ids:
        page = 1
        while True:
            try:
                meas_resp = requests.get(
                    f"{BASE_URL}/locations/{loc_id}/measurements",
                    params={
                        "date_from": date_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "date_to":   date_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "parameters": "pm25,pm10",
                        "limit":      1000,
                        "page":       page,
                    },
                    timeout=20,
                )
                meas_resp.raise_for_status()
                meas_data = meas_resp.json()
                results   = meas_data.get("results", [])

                if not results:
                    break

                for r in results:
                    param = r.get("parameter", "")
                    value = r.get("value")
                    ts_raw = r.get("date", {}).get("utc")
                    if param in ("pm25", "pm10") and value is not None and ts_raw:
                        all_records.append({
                            "timestamp": ts_raw,
                            "parameter": param,
                            "value":     float(value),
                        })

                # Respect pagination
                meta      = meas_data.get("meta", {})
                total     = meta.get("found", 0)
                fetched   = page * 1000
                if fetched >= total:
                    break
                page += 1

            except requests.exceptions.RequestException as e:
                print(f"OpenAQ measurements error (loc {loc_id}, page {page}): {e}")
                break

    if not all_records:
        print("OpenAQ returned no measurements for the requested period.")
        return None

    # ── Step 3: Pivot into hourly pm25 / pm10 columns ─────────────────────────
    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Round to the nearest hour so rows from multiple sensors align
    df["timestamp"] = df["timestamp"].dt.floor("H")

    # Average across sensors for the same hour + parameter
    df = (
        df.groupby(["timestamp", "parameter"])["value"]
        .mean()
        .unstack(level="parameter")
        .rename_axis(None, axis=1)
    )

    # Ensure both columns exist even if one parameter was missing
    for col in ("pm25", "pm10"):
        if col not in df.columns:
            df[col] = float("nan")

    df = df[["pm25", "pm10"]].sort_index()
    print(f"OpenAQ: {len(df)} hourly records fetched "
          f"({df.index.min()} → {df.index.max()})")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  NEW FUNCTION — PM-to-AQI conversion (EPA formula)
#  Needed to convert OpenAQ raw µg/m³ values into a comparable AQI number
# ══════════════════════════════════════════════════════════════════════════════

def _pm25_to_aqi(pm25: float) -> float | None:
    """
    Converts PM2.5 concentration (µg/m³) to US EPA AQI using the standard
    linear interpolation formula.

    Breakpoint table: EPA AQI Technical Assistance Document (2024 revision).
    Returns None for negative or clearly invalid readings.
    """
    if pm25 < 0:
        return None

    # (C_low, C_high, AQI_low, AQI_high)
    breakpoints = [
        (0.0,    9.0,    0,   50),
        (9.1,   35.4,   51,  100),
        (35.5,  55.4,  101,  150),
        (55.5, 125.4,  151,  200),
        (125.5, 225.4, 201,  300),
        (225.5, 325.4, 301,  500),
    ]

    for c_lo, c_hi, aqi_lo, aqi_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            aqi = ((aqi_hi - aqi_lo) / (c_hi - c_lo)) * (pm25 - c_lo) + aqi_lo
            return round(aqi, 1)

    # Above 325.4 µg/m³ — cap at 500
    return 500.0


def _openaq_df_to_aqi_dict(openaq_df: pd.DataFrame) -> dict:
    """
    Converts an OpenAQ DataFrame (pm25, pm10 columns, UTC DatetimeIndex)
    into the same {'hoursInfo': [...]} dict format that get_air_quality_history()
    returns, so build_master_dataframe() in app.py can consume both sources
    identically without any changes.

    AQI is derived from PM2.5 where available, falling back to a rough PM10
    estimate when PM2.5 is missing.
    """
    hours_info = []

    for ts, row in openaq_df.iterrows():
        aqi = None

        if pd.notna(row.get("pm25")):
            aqi = _pm25_to_aqi(row["pm25"])
        elif pd.notna(row.get("pm10")):
            # Rough PM10 → AQI approximation (less accurate, last resort)
            aqi = _pm25_to_aqi(row["pm10"] * 0.6)

        if aqi is None:
            continue

        entry = {
            "dateTime": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "indexes":  [{"aqi": aqi}],
            "pollutants": [],
        }

        if pd.notna(row.get("pm25")):
            entry["pollutants"].append({
                "code":          "pm25",
                "concentration": {"value": row["pm25"], "units": "µg/m³"},
            })
        if pd.notna(row.get("pm10")):
            entry["pollutants"].append({
                "code":          "pm10",
                "concentration": {"value": row["pm10"], "units": "µg/m³"},
            })

        hours_info.append(entry)

    return {"hoursInfo": hours_info}


# ══════════════════════════════════════════════════════════════════════════════
#  NEW FUNCTION — Extended history merger (Issue 5)
#  Called by app.py instead of get_air_quality_history() when days_back > 30
# ══════════════════════════════════════════════════════════════════════════════

def get_extended_aqi_history(
    latitude: float,
    longitude: float,
    days_back: int = 150,
) -> dict | None:
    """
    Returns up to `days_back` days of hourly AQI history by combining:
        1. Google Air Quality API  — most recent 30 days  (authoritative, full AQI)
        2. OpenAQ v3 API           — older period beyond 30 days  (PM2.5/PM10 → AQI)

    The two sources are merged with Google data taking priority: if both sources
    have a record for the same hour, the Google record is kept.

    The returned dict has the same {'hoursInfo': [...]} schema as
    get_air_quality_history(), so all downstream code (build_master_dataframe
    in app.py) works without modification.

    Parameters
    ──────────
    days_back : total days of history wanted. Values ≤ 30 use Google only.
                Values > 30 fetch 30 days from Google + (days_back - 30) from OpenAQ.

    Returns
    ───────
    dict {'hoursInfo': [...]} or None if both sources fail
    """

    # ── 1. Always fetch the most recent 30 days from Google ───────────────────
    print(f"Fetching last 30 days from Google Air Quality API…")
    google_dict = get_air_quality_history(latitude, longitude, days_back=30)

    if days_back <= 30:
        # No extension needed — return Google data as-is
        return google_dict

    # ── 2. Fetch older period from OpenAQ ─────────────────────────────────────
    openaq_days = days_back - 30
    print(f"Fetching {openaq_days} additional days from OpenAQ…")
    openaq_df = get_openaq_history(latitude, longitude, days_back=days_back)

    if openaq_df is None:
        print("OpenAQ fetch failed — falling back to Google-only data (30 days).")
        return google_dict

    # ── 3. Convert OpenAQ → hoursInfo dict format ─────────────────────────────
    openaq_dict = _openaq_df_to_aqi_dict(openaq_df)

    # ── 4. Build a set of timestamps already covered by Google ────────────────
    google_timestamps = set()
    if google_dict:
        for entry in google_dict.get("hoursInfo", []):
            ts = entry.get("dateTime")
            if ts:
                google_timestamps.add(ts)

    # ── 5. Merge: keep all Google records + OpenAQ records not in Google window
    merged_hours = []

    # All Google records (authoritative, keep as-is)
    if google_dict:
        merged_hours.extend(google_dict.get("hoursInfo", []))

    # OpenAQ records only for timestamps NOT already covered by Google
    openaq_added = 0
    for entry in openaq_dict.get("hoursInfo", []):
        if entry.get("dateTime") not in google_timestamps:
            merged_hours.append(entry)
            openaq_added += 1

    # Sort chronologically
    merged_hours.sort(key=lambda x: x.get("dateTime", ""))

    print(f"Merged dataset: {len(google_timestamps)} Google records + "
          f"{openaq_added} OpenAQ records = {len(merged_hours)} total hourly records")

    return {"hoursInfo": merged_hours}


# ══════════════════════════════════════════════════════════════════════════════
#  ORIGINAL STANDALONE ENTRY POINT — unchanged
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Main function to run the history data fetching and RETURN the data."""
    print("--- Fetching historical data ---")
    test_lat = 19.2183
    test_lon = 72.9781

    history_data = get_air_quality_history(test_lat, test_lon, days_back=30)

    if history_data:
        print("✅ Data fetched successfully.")
        return history_data
    else:
        print("--- Failed to fetch data from the API ---")
        return None


if __name__ == "__main__":
    # To test the extended fetcher:
    # python -m logic.get_history
    print("--- Testing extended AQI history fetch ---")
    test_lat = 19.2183
    test_lon = 72.9781

    extended = get_extended_aqi_history(test_lat, test_lon, days_back=150)
    if extended:
        hours = extended.get("hoursInfo", [])
        print(f"Total records: {len(hours)}")
        if hours:
            print(f"Earliest: {hours[0]['dateTime']}")
            print(f"Latest:   {hours[-1]['dateTime']}")
    else:
        print("Failed.")