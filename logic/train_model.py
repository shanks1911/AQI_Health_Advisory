import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import os
import joblib
import numpy as np

def train_robust_models(df: pd.DataFrame) -> dict | None:
    """
    Trains and evaluates robust models with consistent feature handling.
    """
    if df is None or df.empty:
        print("Error: DataFrame is empty or None.")
        return None

    # --- 1. Define Features and Target with Better Validation ---
    core_features = [
        'hour', 'day_of_week', 'month', 'aqi_lag_1hr', 'temp_lag_1hr',
        'aqi_lag_24hr', 'temp_lag_24hr', 'aqi_rolling_avg_24hr',
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'wind_speed_10m', 'wind_direction_10m'
    ]
    
    # Check which features actually exist in the data
    existing_features = [f for f in core_features if f in df.columns]
    print(f"Available features: {existing_features}")
    
    if len(existing_features) < 5:  # Minimum feature requirement
        print(f"Warning: Only {len(existing_features)} features available. May affect model quality.")
    
    # Prepare feature matrix and target
    X = df[existing_features].copy()
    y = df['aqi'].copy()
    
    # Handle missing values and data quality
    print(f"Original data shape: {X.shape}")
    print(f"Missing values per feature:")
    for col in X.columns:
        missing_count = X[col].isna().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} ({missing_count/len(X)*100:.1f}%)")
    
    # Fill missing values with appropriate strategies
    for col in X.columns:
        if X[col].isna().any():
            if 'lag' in col or 'rolling' in col:
                # For lag features, use forward fill then mean
                X[col] = X[col].fillna(method='ffill').fillna(X[col].mean())
            else:
                # For other features, use mean
                X[col] = X[col].fillna(X[col].mean())
    
    # Remove any remaining NaN values
    initial_len = len(X)
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    if len(X) < initial_len:
        print(f"Removed {initial_len - len(X)} rows with missing values")
    
    print(f"Final training data shape: {X.shape}")
    
    if len(X) < 100:  # Minimum data requirement
        print("Error: Insufficient data for training (need at least 100 samples)")
        return None

    # --- 2. Time-Series Split ---
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # --- 3. Train Both Models with Error Handling ---
    models_results = {}
    
    # Train LightGBM
    try:
        print("Training LightGBM...")
        lgbm = lgb.LGBMRegressor(
            objective='regression_l1', 
            n_estimators=500, 
            learning_rate=0.05,
            num_leaves=10, 
            reg_lambda=1.0, 
            random_state=42, 
            n_jobs=-1,
            verbosity=-1  # Suppress warnings
        )
        lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        lgbm_preds = lgbm.predict(X_test)
        
        # Validate predictions
        if np.isnan(lgbm_preds).any() or np.isinf(lgbm_preds).any():
            print("Warning: LightGBM produced invalid predictions")
            lgbm_r2 = -999  # Mark as invalid
            lgbm_mae = 999
        else:
            lgbm_r2 = r2_score(y_test, lgbm_preds)
            lgbm_mae = mean_absolute_error(y_test, lgbm_preds)
        
        models_results['lgbm'] = {
            'model': lgbm,
            'name': 'LightGBM',
            'r2': lgbm_r2,
            'mae': lgbm_mae,
            'predictions': lgbm_preds
        }
        print(f"LightGBM R²: {lgbm_r2:.4f}, MAE: {lgbm_mae:.2f}")
        
    except Exception as e:
        print(f"LightGBM training failed: {e}")
        models_results['lgbm'] = {'r2': -999, 'mae': 999}

    # Train Ridge Regression
    try:
        print("Training Ridge Regression...")
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        ridge_preds = ridge.predict(X_test)
        
        # Validate predictions
        if np.isnan(ridge_preds).any() or np.isinf(ridge_preds).any():
            print("Warning: Ridge produced invalid predictions")
            ridge_r2 = -999
            ridge_mae = 999
        else:
            ridge_r2 = r2_score(y_test, ridge_preds)
            ridge_mae = mean_absolute_error(y_test, ridge_preds)
        
        models_results['ridge'] = {
            'model': ridge,
            'name': 'Ridge Regression',
            'r2': ridge_r2,
            'mae': ridge_mae,
            'predictions': ridge_preds
        }
        print(f"Ridge R²: {ridge_r2:.4f}, MAE: {ridge_mae:.2f}")
        
    except Exception as e:
        print(f"Ridge training failed: {e}")
        models_results['ridge'] = {'r2': -999, 'mae': 999}

    # --- 4. Select and Return the Best Model ---
    if not models_results or all(result['r2'] < 0 for result in models_results.values()):
        print("Error: All models failed to train successfully")
        return None
    
    # Select best model based on R² score
    best_key = max(models_results.keys(), key=lambda k: models_results[k]['r2'])
    best_result = models_results[best_key]
    
    if best_result['r2'] < 0.1:  # Minimum acceptable performance
        print(f"Warning: Best model has low R² score: {best_result['r2']:.4f}")
    
    # Save the best model with metadata
    best_model = best_result['model']
    
    # Store feature names in model for later use
    if hasattr(best_model, 'feature_names_in_'):
        # scikit-learn models
        pass  # Already stored
    elif hasattr(best_model, 'booster_'):
        # LightGBM - store feature names
        best_model.feature_names_in_ = existing_features
    
    try:
        joblib.dump(best_model, 'aqi_model_robust.pkl')
        print(f"✅ Best model ({best_result['name']}) saved as 'aqi_model_robust.pkl'")
    except Exception as e:
        print(f"Error saving model: {e}")
        return None
    
    # Return the results
    return {
        "model": best_model,
        "model_name": best_result['name'],
        "mae": best_result['mae'],
        "r2": best_result['r2'],
        "features_used": existing_features,
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }

if __name__ == "__main__":
    print("--- Running Robust Model Training ---")
    dataset_path = 'aqi_weather_training_dataset.csv'
    if os.path.exists(dataset_path):
        df_from_file = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
        results = train_robust_models(df_from_file)
        if results:
            print("\n--- Final Model Results ---")
            print(f"Best Model: {results['model_name']}")
            print(f"R-squared: {results['r2']:.4f}")
            print(f"MAE: {results['mae']:.2f}")
            print(f"Features: {len(results['features_used'])}")
            print(f"Training samples: {results['training_samples']}")
    else:
        print("Dataset not found. Please run build_master_dataset.py first.")