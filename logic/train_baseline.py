"""
logic/train_baseline.py
────────────────────────────────────────────────────────────────────────────────
SARIMA and LSTM baselines for the AQI forecasting paper (Issue 2).

These models are kept separate from train_model.py because:
  - SARIMA (statsmodels) is a univariate time-series model — it uses AQI history
    only, no weather features.
  - LSTM (PyTorch) requires sequence-shaped tensors, a training loop, and GPU
    handling — fundamentally different from the sklearn/lgbm interface.

Both use the SAME 80/20 time-series split as train_model.py so results are
directly comparable in the paper's model comparison table.

Usage (standalone, run once offline to get paper metrics):
    python -m logic.train_baseline  aqi_weather_training_dataset.csv

Requirements (add to requirements.txt if not already present):
    statsmodels>=0.14.0
    torch>=2.0.0          # or use tensorflow/keras if preferred
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHARTS_DIR = 'evaluation_charts'
os.makedirs(CHARTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _time_split(series: pd.Series, ratio: float = 0.8):
    """80/20 time-series split — same rule as train_model.py."""
    idx = int(len(series) * ratio)
    return series.iloc[:idx], series.iloc[idx:]


def _print_metrics(name: str, y_true, y_pred):
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f'  {name:<28}  R² = {r2:.4f}   MAE = {mae:.2f}')
    return r2, mae


# ══════════════════════════════════════════════════════════════════════════════
#  SARIMA BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def train_sarima(aqi_series: pd.Series) -> dict:
    """
    Fits a SARIMA(1,1,1)(1,0,1)[24] model on the training split of aqi_series
    and evaluates one-step-ahead forecasts on the test split.

    SARIMA serves as the classical time-series baseline in the paper — it uses
    NO weather features, demonstrating the value that exogenous variables add.

    Order choice:
        (p=1, d=1, q=1)  — parsimonious ARIMA; AQI series is typically I(1)
        (P=1, D=0, Q=1, s=24) — captures the 24-hour seasonal AQI cycle
    You can swap in auto_arima from pmdarima for data-driven order selection.

    Returns
    ───────
    dict: {name, r2, mae, predictions (np.ndarray), order, seasonal_order}
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print('statsmodels not installed. Run: pip install statsmodels')
        return {'name': 'SARIMA', 'r2': None, 'mae': None}

    train, test = _time_split(aqi_series)

    order          = (1, 1, 1)
    seasonal_order = (1, 0, 1, 24)   # 24-hour daily seasonality

    print(f'\nFitting SARIMA{order}×{seasonal_order}…  '
          f'(train={len(train)}, test={len(test)})')

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False)

        # One-step-ahead rolling forecast on the test window
        preds = []
        history = list(train)
        for obs in test:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fc_model = SARIMAX(
                    history,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fc_fit   = fc_model.filter(fit.params)
                forecast = fc_fit.forecast(steps=1)
            preds.append(float(forecast.iloc[0]))
            history.append(obs)

        preds = np.array(preds)
        r2, mae = _print_metrics('SARIMA', test.values, preds)

        return {
            'name':           'SARIMA',
            'r2':             round(r2, 4),
            'mae':            round(mae, 2),
            'predictions':    preds,
            'order':          order,
            'seasonal_order': seasonal_order,
        }

    except Exception as exc:
        print(f'  SARIMA failed: {exc}')
        return {'name': 'SARIMA', 'r2': None, 'mae': None}


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def _make_sequences(data: np.ndarray, look_back: int = 24):
    """
    Converts a 1-D time series into (X, y) pairs for sequence modelling.
    Each X[i] is the window data[i : i+look_back], y[i] = data[i+look_back].
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i: i + look_back])
        y.append(data[i + look_back])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm(df: pd.DataFrame, look_back: int = 24, epochs: int = 30) -> dict:
    """
    Trains a single-layer LSTM on the AQI + weather feature matrix.

    Architecture
    ─────────────
        Input  →  LSTM(64 hidden units)  →  Linear(1)
    Loss: MAE (L1Loss), consistent with LightGBM's objective for fair comparison.
    Optimizer: Adam, lr=1e-3.
    Sequence length: 24 hours (look_back).

    The model uses the same 13 features as train_model.py, reshaped into
    (batch, sequence_length, n_features) tensors.

    Returns
    ───────
    dict: {name, r2, mae, predictions (np.ndarray)}
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print('PyTorch not installed. Run: pip install torch')
        return {'name': 'LSTM', 'r2': None, 'mae': None}

    core_features = [
        'hour', 'day_of_week', 'month',
        'aqi_lag_1hr', 'temp_lag_1hr',
        'aqi_lag_24hr', 'temp_lag_24hr',
        'aqi_rolling_avg_24hr',
        'temperature_2m', 'relative_humidity_2m',
        'precipitation', 'wind_speed_10m', 'wind_direction_10m',
    ]
    features = [f for f in core_features if f in df.columns]

    X_df = df[features].fillna(0).values.astype(np.float32)
    y_all = df['aqi'].fillna(0).values.astype(np.float32)

    # Normalise features (zero-mean, unit-variance) — important for LSTMs
    X_mean, X_std = X_df.mean(axis=0), X_df.std(axis=0) + 1e-8
    y_mean, y_std = y_all.mean(), y_all.std() + 1e-8
    X_norm = (X_df - X_mean) / X_std
    y_norm = (y_all - y_mean) / y_std

    # Build sequences
    Xs, ys = [], []
    for i in range(len(X_norm) - look_back):
        Xs.append(X_norm[i: i + look_back])
        ys.append(y_norm[i + look_back])
    Xs = np.array(Xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    split = int(len(Xs) * 0.8)
    X_train_t = torch.tensor(Xs[:split])
    y_train_t = torch.tensor(ys[:split])
    X_test_t  = torch.tensor(Xs[split:])
    y_test_t  = torch.tensor(ys[split:])
    y_test_raw = y_all[split + look_back:]  # un-normalised ground truth

    print(f'\nTraining LSTM…  '
          f'(train seqs={len(X_train_t)}, test seqs={len(X_test_t)}, '
          f'features={len(features)}, look_back={look_back}, epochs={epochs})')

    # ── Model definition ───────────────────────────────────────────────────────
    class AQILSTMModel(nn.Module):
        def __init__(self, n_features, hidden=64):
            super().__init__()
            self.lstm   = nn.LSTM(n_features, hidden, batch_first=True)
            self.linear = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.linear(out[:, -1, :]).squeeze(-1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = AQILSTMModel(n_features=len(features)).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=64,
        shuffle=False,   # Keep time order
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        avg = epoch_loss / len(X_train_t)
        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch + 1:>3}/{epochs}  train MAE (norm) = {avg:.4f}')

    # ── Evaluation ─────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        preds_norm = model(X_test_t.to(device)).cpu().numpy()

    # De-normalise predictions back to AQI scale
    preds_raw = preds_norm * y_std + y_mean
    preds_raw = np.clip(preds_raw, 0, None)   # AQI can't be negative

    # Align lengths (y_test_raw already trimmed by look_back above)
    min_len   = min(len(preds_raw), len(y_test_raw))
    preds_raw = preds_raw[:min_len]
    y_eval    = y_test_raw[:min_len]

    r2, mae = _print_metrics('LSTM', y_eval, preds_raw)

    return {
        'name':        'LSTM',
        'r2':          round(r2, 4),
        'mae':         round(mae, 2),
        'predictions': preds_raw,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_all_baselines(df: pd.DataFrame) -> list[dict]:
    """
    Runs SARIMA and LSTM baselines and returns a list of result dicts,
    each compatible with the all_models list format from train_model.py.

    Combine with train_model.py results for the full 6-model comparison table:

        from logic.train_model   import train_robust_models
        from logic.train_baseline import run_all_baselines

        ml_results       = train_robust_models(df)
        baseline_results = run_all_baselines(df)

        all_models = ml_results['all_models'] + baseline_results
        all_models_sorted = sorted(all_models, key=lambda x: x['r2'] or -999, reverse=True)
    """
    print('\n══ Baseline Models ══════════════════════════════════')

    results = []

    # SARIMA — needs only the AQI column
    sarima = train_sarima(df['aqi'].dropna())
    results.append({'name': sarima['name'], 'r2': sarima['r2'], 'mae': sarima['mae']})

    # LSTM — needs full feature DataFrame
    lstm = train_lstm(df)
    results.append({'name': lstm['name'], 'r2': lstm['r2'], 'mae': lstm['mae']})

    print('\n── Baseline Summary ──────────────────────────────────')
    print(f'{"Model":<22} {"R²":>8} {"MAE":>8}')
    print('─' * 42)
    for r in results:
        r2_str  = f"{r['r2']:.4f}" if r['r2'] is not None else '   N/A'
        mae_str = f"{r['mae']:.2f}"  if r['mae'] is not None else '   N/A'
        print(f"{r['name']:<22} {r2_str:>8} {mae_str:>8}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'aqi_weather_training_dataset.csv'

    if not os.path.exists(dataset_path):
        print(f'Dataset not found: {dataset_path}')
        print('Run the Streamlit app first to generate training data, '
              'then pass the CSV path as the first argument.')
        sys.exit(1)

    print(f'Loading dataset: {dataset_path}')
    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    print(f'Shape: {df.shape}')

    run_all_baselines(df)