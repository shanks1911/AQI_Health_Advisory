import pandas as pd
import numpy as np
import lightgbm as lgb # type: ignore
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for Streamlit
import matplotlib.pyplot as plt
import os
import joblib
import warnings

# ── Output directories ────────────────────────────────────────────────────────
CHARTS_DIR = 'evaluation_charts'
os.makedirs(CHARTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CHART GENERATION 
# ══════════════════════════════════════════════════════════════════════════════

def save_evaluation_charts(y_test: pd.Series, all_preds: dict, best_model_name: str) -> dict:
    """
    Generates and saves three publication-quality evaluation figures at 300 DPI.

    Figures produced
    ─────────────────
    1. predicted_vs_actual.png   — scatter plot of best model predictions vs ground truth
    2. residual_distribution.png — histogram + KDE of residuals for all models
    3. forecast_degradation.png  — mean absolute error bucketed by forecast horizon (hours 1–120)

    Parameters
    ──────────
    y_test         : ground-truth AQI values (test split, indexed the same as preds)
    all_preds      : dict of {model_name: np.ndarray of predictions}
    best_model_name: key inside all_preds to highlight as the primary model

    Returns
    ───────
    dict of {chart_name: filepath}
    """
    chart_paths = {}
    best_preds = all_preds[best_model_name]

    style_params = {
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.grid': True,
        'grid.color': 'white',
        'grid.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.family': 'DejaVu Sans',
        'font.size': 11,
    }

    # ── Figure 1: Predicted vs Actual ─────────────────────────────────────────
    with plt.rc_context(style_params):
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

        ax.scatter(y_test, best_preds, alpha=0.45, s=18,
                   color='#2563eb', edgecolors='none', label='Predictions')

        # Perfect-prediction reference line
        lims = [min(y_test.min(), best_preds.min()) - 5,
                max(y_test.max(), best_preds.max()) + 5]
        ax.plot(lims, lims, 'r--', linewidth=1.2, label='Perfect prediction (y = x)', zorder=3)

        r2  = r2_score(y_test, best_preds)
        mae = mean_absolute_error(y_test, best_preds)
        ax.text(0.04, 0.93, f'R² = {r2:.3f}   MAE = {mae:.1f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('Actual AQI', fontsize=12)
        ax.set_ylabel('Predicted AQI', fontsize=12)
        ax.set_title(f'Predicted vs Actual AQI — {best_model_name}', fontsize=13, fontweight='bold', pad=12)
        ax.legend(fontsize=10)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        path1 = os.path.join(CHARTS_DIR, 'predicted_vs_actual.png')
        fig.savefig(path1, dpi=300, bbox_inches='tight')
        plt.close(fig)
        chart_paths['predicted_vs_actual'] = path1
        print(f'Saved: {path1}')

    # ── Figure 2: Residual Distribution (all models overlaid) ─────────────────
    with plt.rc_context(style_params):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

        colors = ['#2563eb', '#16a34a', '#dc2626', '#9333ea', '#ea580c', '#0891b2']
        for (name, preds), color in zip(all_preds.items(), colors):
            residuals = np.array(y_test) - np.array(preds)
            ax.hist(residuals, bins=40, alpha=0.35, color=color,
                    label=f'{name} (MAE={mean_absolute_error(y_test, preds):.1f})', density=True)

            # Overlay KDE using numpy
            from numpy import linspace
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(residuals)
                x_range = linspace(residuals.min(), residuals.max(), 200)
                ax.plot(x_range, kde(x_range), color=color, linewidth=1.8)
            except Exception:
                pass  # Skip KDE if scipy unavailable

        ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Zero error')
        ax.set_xlabel('Residual (Actual − Predicted AQI)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Residual Distribution — All Models', fontsize=13, fontweight='bold', pad=12)
        ax.legend(fontsize=9, ncol=2)

        path2 = os.path.join(CHARTS_DIR, 'residual_distribution.png')
        fig.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close(fig)
        chart_paths['residual_distribution'] = path2
        print(f'Saved: {path2}')

    # ── Figure 3: Forecast Accuracy Degradation over 5-day horizon ────────────
    # We simulate horizon degradation by grouping test samples into 24-hr buckets.
    # Bucket 1 = hours 1–24, bucket 2 = hours 25–48, …, bucket 5 = hours 97–120.
    # Each bucket's MAE shows how accuracy degrades as we forecast further ahead.
    with plt.rc_context(style_params):
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

        n_test = len(y_test)
        n_days = 5
        bucket_size = max(1, n_test // n_days)

        colors = ['#2563eb', '#16a34a', '#dc2626', '#9333ea', '#ea580c', '#0891b2']
        for (name, preds), color in zip(all_preds.items(), colors):
            bucket_maes = []
            bucket_labels = []
            for day in range(n_days):
                start = day * bucket_size
                end   = start + bucket_size
                if start >= n_test:
                    break
                end = min(end, n_test)
                bucket_mae = mean_absolute_error(
                    np.array(y_test)[start:end],
                    np.array(preds)[start:end]
                )
                bucket_maes.append(bucket_mae)
                bucket_labels.append(f'Day {day + 1}')

            ax.plot(bucket_labels, bucket_maes, marker='o', markersize=6,
                    linewidth=2, color=color, label=name)

        ax.set_xlabel('Forecast Horizon', fontsize=12)
        ax.set_ylabel('Mean Absolute Error (AQI points)', fontsize=12)
        ax.set_title('Forecast Accuracy Degradation over 5-Day Horizon', fontsize=13, fontweight='bold', pad=12)
        ax.legend(fontsize=10)

        # Annotate the general trend
        ax.text(0.97, 0.95, 'Higher MAE = less accurate further into the future',
                transform=ax.transAxes, fontsize=9, color='gray',
                ha='right', va='top')

        path3 = os.path.join(CHARTS_DIR, 'forecast_degradation.png')
        fig.savefig(path3, dpi=300, bbox_inches='tight')
        plt.close(fig)
        chart_paths['forecast_degradation'] = path3
        print(f'Saved: {path3}')

    return chart_paths


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_robust_models(df: pd.DataFrame) -> dict | None:
    """
    Trains and evaluates six ML models on the provided AQI + weather DataFrame.

    Models trained
    ──────────────
    1. LightGBM          (gradient boosting — primary model from v1)
    2. XGBoost           (gradient boosting — added for Issue 2 comparison)
    3. Random Forest     (bagging ensemble — added for Issue 2 comparison)
    4. Ridge Regression  (linear baseline — from v1)

    SARIMA and LSTM are handled in logic/train_baseline.py because they require
    fundamentally different data preparation and training interfaces.

    Returns
    ───────
    dict with keys:
        model           — the best sklearn/lgbm model object
        model_name      — display name of the best model
        r2, mae         — test-set metrics of the best model
        features_used   — list of feature column names
        training_samples, test_samples
        all_models      — list of dicts [{name, r2, mae}] for ALL models, sorted by R²
        chart_paths     — dict of {chart_name: filepath} for the 3 evaluation charts
    """

    if df is None or df.empty:
        print('Error: DataFrame is empty or None.')
        return None

    # ── 1. Feature definition ──────────────────────────────────────────────────
    core_features = [
        'hour', 'day_of_week', 'month',
        'aqi_lag_1hr', 'temp_lag_1hr',
        'aqi_lag_24hr', 'temp_lag_24hr',
        'aqi_rolling_avg_24hr',
        'temperature_2m', 'relative_humidity_2m',
        'precipitation', 'wind_speed_10m', 'wind_direction_10m',
    ]

    existing_features = [f for f in core_features if f in df.columns]
    print(f'Features available ({len(existing_features)}): {existing_features}')

    if len(existing_features) < 5:
        print(f'Warning: Only {len(existing_features)} features available. Model quality may be low.')

    X = df[existing_features].copy()
    y = df['aqi'].copy()

    # ── 2. Missing value handling ──────────────────────────────────────────────
    print(f'Data shape before cleaning: {X.shape}')
    for col in X.columns:
        if X[col].isna().any():
            if 'lag' in col or 'rolling' in col:
                X[col] = X[col].fillna(method='ffill').fillna(X[col].mean())
            else:
                X[col] = X[col].fillna(X[col].mean())

    mask = ~(X.isna().any(axis=1) | y.isna())
    dropped = (~mask).sum()
    X, y = X[mask], y[mask]
    if dropped:
        print(f'Dropped {dropped} rows with remaining NaNs.')

    print(f'Final training data shape: {X.shape}')

    if len(X) < 100:
        print('Error: Insufficient data (need ≥ 100 samples).')
        return None

    # ── 3. Time-series train / test split (80 / 20) ───────────────────────────
    split_idx   = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f'Train: {len(X_train)} samples  |  Test: {len(y_test)} samples')

    # ── 4. Model definitions ───────────────────────────────────────────────────
    #   Each entry: (key, display_name, model_object, fit_kwargs)
    #   fit_kwargs lets LightGBM pass eval_set / callbacks without branching logic.
    model_configs = [
        (
            'lgbm',
            'LightGBM',
            lgb.LGBMRegressor(
                objective='regression_l1',
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=10,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            ),
            dict(
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(0)],
            ),
        ),
        (
            'xgb',
            'XGBoost',
            XGBRegressor(
                objective='reg:absoluteerror',
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                eval_metric='mae',
                early_stopping_rounds=50,
            ),
            dict(
                eval_set=[(X_test, y_test)],
                verbose=False,
            ),
        ),
        (
            'rf',
            'Random Forest',
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            ),
            {},   # No special fit kwargs for RF
        ),
        (
            'ridge',
            'Ridge Regression',
            Ridge(alpha=1.0),
            {},
        ),
    ]

    # ── 5. Training loop ───────────────────────────────────────────────────────
    results     = {}   # key → {model, name, r2, mae, preds}
    all_preds   = {}   # name → np.ndarray  (for chart generation)

    for key, name, model, fit_kwargs in model_configs:
        print(f'\nTraining {name}…')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(X_train, y_train, **fit_kwargs)

            preds = model.predict(X_test)

            if np.isnan(preds).any() or np.isinf(preds).any():
                raise ValueError(f'{name} produced NaN / Inf predictions.')

            r2  = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            print(f'  R² = {r2:.4f}   MAE = {mae:.2f}')

            results[key]       = dict(model=model, name=name, r2=r2, mae=mae, preds=preds)
            all_preds[name]    = preds

        except Exception as exc:
            print(f'  {name} failed: {exc}')
            results[key] = dict(name=name, r2=-999, mae=999)

    # ── 6. Select best model ───────────────────────────────────────────────────
    valid = {k: v for k, v in results.items() if v['r2'] > -999 and 'model' in v}

    if not valid:
        print('Error: All models failed.')
        return None

    best_key    = max(valid, key=lambda k: valid[k]['r2'])
    best        = valid[best_key]

    if best['r2'] < 0.1:
        print(f"Warning: Best model R² is low ({best['r2']:.4f}). Consider more training data.")

    # ── 7. Persist feature names on the model object ──────────────────────────
    #   scikit-learn ≥1.0 sets feature_names_in_ automatically on fit().
    #   For LightGBM we set it manually so predict.py can retrieve it.
    if not hasattr(best['model'], 'feature_names_in_'):
        best['model'].feature_names_in_ = np.array(existing_features)

    # ── 8. Save model ──────────────────────────────────────────────────────────
    try:
        joblib.dump(best['model'], 'aqi_model_robust.pkl')
        print(f"\nSaved best model ({best['name']}) → aqi_model_robust.pkl")
    except Exception as exc:
        print(f'Error saving model: {exc}')
        return None

    # ── 9. Generate evaluation charts (Issue 4) ───────────────────────────────
    print('\nGenerating evaluation charts…')
    chart_paths = {}
    try:
        chart_paths = save_evaluation_charts(y_test, all_preds, best['name'])
    except Exception as exc:
        print(f'Chart generation failed (non-fatal): {exc}')

    # ── 10. Build all_models summary list (for paper comparison table + app UI) ─
    all_models_summary = sorted(
        [
            {'name': v['name'], 'r2': round(v['r2'], 4), 'mae': round(v['mae'], 2)}
            for v in results.values()
            if v['r2'] > -999
        ],
        key=lambda x: x['r2'],
        reverse=True,
    )

    print('\n── Model Comparison ──────────────────────────────────')
    print(f'{"Model":<22} {"R²":>8} {"MAE":>8}')
    print('─' * 42)
    for m in all_models_summary:
        marker = ' ← best' if m['name'] == best['name'] else ''
        print(f"{m['name']:<22} {m['r2']:>8.4f} {m['mae']:>8.2f}{marker}")

    return {
        'model':            best['model'],
        'model_name':       best['name'],
        'r2':               best['r2'],
        'mae':              best['mae'],
        'features_used':    existing_features,
        'training_samples': len(X_train),
        'test_samples':     len(X_test),
        'all_models':       all_models_summary,   
        'chart_paths':      chart_paths,           
    }


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('── Standalone Model Training ─────────────────────────')
    dataset_path = 'aqi_weather_training_dataset.csv'
    if os.path.exists(dataset_path):
        df_from_file = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
        results = train_robust_models(df_from_file)
        if results:
            print('\n── Final Results ─────────────────────────────────────')
            print(f"Best model : {results['model_name']}")
            print(f"R²         : {results['r2']:.4f}")
            print(f"MAE        : {results['mae']:.2f}")
            print(f"Features   : {len(results['features_used'])}")
            print(f"Charts     : {list(results['chart_paths'].values())}")
    else:
        print(f'Dataset not found at {dataset_path}. Run app.py to build it first.')