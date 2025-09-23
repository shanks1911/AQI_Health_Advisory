import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings

from logic.get_history import get_air_quality_history
from logic.get_weather import get_weather_forecast

def make_predictions(lat: float, lon: float, model_path='aqi_model_robust.pkl') -> pd.DataFrame | None:
    """
    Loads a trained model, makes future AQI predictions with improved error handling.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return None

    try:
        model = joblib.load(model_path)
        print(f"Loaded model: {type(model).__name__}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # --- 1. Fetch Forecast Data ---
    future_weather_df = get_weather_forecast(lat, lon, forecast_days=5)
    if future_weather_df is None:
        print("Could not fetch weather forecast. Aborting.")
        return None

    # --- 2. Fetch Recent AQI History ---
    recent_aqi_dict = get_air_quality_history(lat, lon, days_back=3)
    if recent_aqi_dict is None:
        print("Could not fetch recent AQI history. Aborting.")
        return None

    # --- 3. Prepare DataFrames for Prediction ---
    aqi_records = []
    for entry in recent_aqi_dict.get('hoursInfo', []):
        record = {
            'date': entry.get('dateTime'), 
            'aqi': entry.get('indexes', [{}])[0].get('aqi')
        }
        aqi_records.append(record)
    
    recent_aqi_df = pd.DataFrame(aqi_records)
    
    # Handle empty or invalid AQI data
    if recent_aqi_df.empty or recent_aqi_df['aqi'].isna().all():
        print("No valid AQI data found in history. Cannot make predictions.")
        return None
    
    recent_aqi_df['date'] = pd.to_datetime(recent_aqi_df['date'])
    recent_aqi_df.set_index('date', inplace=True)
    
    # Ensure timezone consistency
    if not future_weather_df.index.tz:
        future_weather_df = future_weather_df.tz_localize('UTC')
    
    # Combine data
    combined_df = pd.concat([recent_aqi_df, future_weather_df], axis=1)

    # --- 4. Get Model Features with Better Error Handling ---
    try:
        if hasattr(model, 'feature_names_in_'):
            model_features = list(model.feature_names_in_)
        elif hasattr(model, 'feature_name_'):
            model_features = list(model.feature_name_())
        elif hasattr(model, 'booster_'):  # For LightGBM
            model_features = list(model.booster_.feature_name())
        else:
            # Fallback to expected features
            model_features = [
                'hour', 'day_of_week', 'month', 'aqi_lag_1hr', 'temp_lag_1hr',
                'aqi_lag_24hr', 'temp_lag_24hr', 'aqi_rolling_avg_24hr',
                'temperature_2m', 'relative_humidity_2m', 'precipitation',
                'wind_speed_10m', 'wind_direction_10m'
            ]
            print(f"Warning: Using fallback features: {model_features}")
            
    except Exception as e:
        print(f"Error determining model features: {e}")
        return None

    print(f"Model expects {len(model_features)} features: {model_features}")

    # --- 5. Feature Engineering with Error Handling ---
    prediction_timestamps = future_weather_df.index
    
    for current_ts in prediction_timestamps:
        try:
            # Time-based features
            combined_df.loc[current_ts, 'hour'] = current_ts.hour
            combined_df.loc[current_ts, 'day_of_week'] = current_ts.dayofweek
            combined_df.loc[current_ts, 'month'] = current_ts.month

            # Create lag features
            combined_df['aqi_lag_1hr'] = combined_df['aqi'].shift(1)
            combined_df['temp_lag_1hr'] = combined_df['temperature_2m'].shift(1)
            combined_df['aqi_lag_24hr'] = combined_df['aqi'].shift(24)
            combined_df['temp_lag_24hr'] = combined_df['temperature_2m'].shift(24)
            combined_df['aqi_rolling_avg_24hr'] = combined_df['aqi'].shift(1).rolling(window=24, min_periods=1).mean()

            # Prepare features for this timestamp
            features_for_prediction = combined_df.loc[[current_ts]][model_features].copy()
            
            # Handle missing values more robustly
            # For lag features, use forward fill then backward fill
            for col in model_features:
                if col in features_for_prediction.columns:
                    if features_for_prediction[col].isna().any():
                        if 'lag' in col or 'rolling' in col:
                            # Use the most recent available value for lag features
                            last_valid = combined_df[col].dropna()
                            if not last_valid.empty:
                                features_for_prediction[col] = features_for_prediction[col].fillna(last_valid.iloc[-1])
                            else:
                                features_for_prediction[col] = features_for_prediction[col].fillna(0)
                        else:
                            features_for_prediction[col] = features_for_prediction[col].fillna(0)
                else:
                    # Add missing feature with default value
                    features_for_prediction[col] = 0
            
            # Ensure all expected features are present
            missing_features = set(model_features) - set(features_for_prediction.columns)
            for missing_feat in missing_features:
                features_for_prediction[missing_feat] = 0
                print(f"Warning: Added missing feature '{missing_feat}' with value 0")
            
            # Reorder columns to match model expectations
            features_for_prediction = features_for_prediction[model_features]
            
            # Validate feature data types and values
            features_for_prediction = features_for_prediction.astype(float)
            
            # Check for any remaining NaN or infinite values
            if features_for_prediction.isna().any().any():
                print(f"Warning: NaN values found, filling with 0")
                features_for_prediction = features_for_prediction.fillna(0)
            
            if np.isinf(features_for_prediction.values).any():
                print(f"Warning: Infinite values found, replacing with 0")
                features_for_prediction = features_for_prediction.replace([np.inf, -np.inf], 0)
            
            # Make prediction with additional error handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    prediction = model.predict(features_for_prediction)[0]
                    
                    # Validate prediction
                    if np.isnan(prediction) or np.isinf(prediction):
                        print(f"Invalid prediction for {current_ts}, using fallback")
                        prediction = combined_df['aqi'].dropna().mean()  # Use historical average as fallback
                    
                    combined_df.loc[current_ts, 'aqi'] = max(0, prediction)  # Ensure non-negative AQI
                    
                except Exception as pred_error:
                    print(f"Prediction error at {current_ts}: {pred_error}")
                    # Use fallback prediction
                    fallback_aqi = combined_df['aqi'].dropna().mean()
                    combined_df.loc[current_ts, 'aqi'] = fallback_aqi
                    print(f"Used fallback AQI: {fallback_aqi}")

        except Exception as ts_error:
            print(f"Error processing timestamp {current_ts}: {ts_error}")
            continue

    # --- 6. Finalize Results ---
    predictions_df = combined_df.loc[prediction_timestamps][['aqi']].copy()
    predictions_df.rename(columns={'aqi': 'Predicted_AQI'}, inplace=True)
    predictions_df['Predicted_AQI'] = predictions_df['Predicted_AQI'].round(1)
    
    # Final validation
    if predictions_df.empty or predictions_df['Predicted_AQI'].isna().all():
        print("Error: No valid predictions generated")
        return None
    
    print(f"Generated {len(predictions_df)} predictions successfully")
    return predictions_df

if __name__ == "__main__":
    print("--- Running Standalone AQI Forecast ---")
    thane_lat = 19.2183
    thane_lon = 72.9781
    
    forecast = make_predictions(thane_lat, thane_lon)
    
    if forecast is not None:
        print("\n--- AQI Forecast ---")
        print(forecast)
    else:
        print("Failed to generate forecast")