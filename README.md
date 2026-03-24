# 🌍 Air Quality Index Forecasting Dashboard

<div align="center">

![AQI Dashboard](https://img.shields.io/badge/AQI-Forecasting%20Dashboard-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSJjdXJyZW50Q29sb3IiLz4KPC9zdmc+)
![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49+-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Real-time AQI monitoring and intelligent 5-day forecasting powered by machine learning**

[🚀 Getting Started](#️-installation) •
[✨ Features](#-features) •
[🤖 AI Chat](#-ai-health-advisor) •
[🔬 ML Pipeline](#-machine-learning-pipeline)

</div>

---

## 📋 Overview

The **Air Quality Index (AQI) Forecasting Dashboard** is a comprehensive web application that provides real-time air quality monitoring and intelligent 5-day forecasting for any city worldwide. Built with Streamlit and powered by an ensemble of machine learning models, the dashboard helps users make informed decisions about outdoor activities and health precautions.

### 🎯 Key Highlights

- **Real-time AQI Data**: Instant air quality readings for any city via Google Air Quality API
- **5-Day ML Forecasting**: Predictions from four models — LightGBM, XGBoost, Random Forest, and Ridge Regression — with automatic best-model selection
- **Extended Training Data**: Up to 150 days of historical AQI from Google Air Quality API + OpenAQ (vs. the 30-day API cap alone)
- **Model Evaluation Charts**: Publication-quality figures — predicted vs. actual, residual distribution, and forecast accuracy degradation
- **AI Health Advisor**: Personalized health recommendations powered by Google Gemini AI
- **Local Timezone Support**: Automatic timezone detection and conversion
- **Health Profile Integration**: Customized advice for specific health conditions

### 🔗 Live App: https://aqihealthadvisory.streamlit.app/

---

## ✨ Features

### 🌡️ Current Air Quality Monitoring
- Universal AQI readings with health category labels
- Pollutant concentration levels (PM2.5, PM10, CO, NO2, O3)
- Dominant pollutant identification
- Health recommendations for the general population

### 📈 Intelligent Forecasting
- 5-day hourly predictions using the best-performing ML model
- Full model comparison table (R², MAE) for all four trained models
- Automatic timezone conversion to local city time
- Downloadable forecast data as CSV

### 📊 Evaluation Charts (300 DPI)
- **Predicted vs Actual AQI** — scatter plot with perfect-prediction reference line
- **Residual Distribution** — overlaid histogram + KDE for all models
- **Forecast Accuracy Degradation** — per-day MAE across the 5-day horizon

### 🤖 AI Health Advisor
- Personalized health chatbot powered by Gemini AI
- Health condition profiling (Asthma, COPD, Heart Disease, etc.)
- Contextual advice based on current and forecasted AQI
- Quick action buttons for common questions
- Chat history export as text file

---

## 🛠️ Installation

### Prerequisites
- Python 3.12 or higher
- Google Maps API Key (Geocoding + Air Quality APIs enabled)
- Gemini API Key
- OpenAQ API Key (free — for extended training data beyond 30 days)

### 1. Clone the Repository
```bash
git clone https://github.com/shanks1911/aqi-forecasting-dashboard.git
cd aqi-forecasting-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
Or with Poetry:
```bash
poetry install
```

### 3. Environment Setup
Create a `.env` file in the project root:
```env
MAPS_KEY=your_google_maps_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAQ_API_KEY=your_openaq_api_key_here
```

### 4. API Key Setup

#### Google Maps & Air Quality API
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Enable **Geocoding API** and **Air Quality API**
3. Create an API key and add it to `.env` as `MAPS_KEY`

#### Gemini AI API
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Generate an API key and add it to `.env` as `GEMINI_API_KEY`

#### OpenAQ API
1. Visit [openaq.org](https://openaq.org) and sign up for a free account
2. Generate an API key and add it to `.env` as `OPENAQ_API_KEY`
3. This unlocks up to 150 days of historical AQI data (vs. Google's 30-day cap), giving the ML models significantly more training data

### 5. Run the Application
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

---

## 🏗️ Project Structure

```
aqi-forecasting-dashboard/
├── app.py                      # Main Streamlit application
├── logic/
│   ├── get_data.py            # City geocoding & current AQI (Google API)
│   ├── get_history.py         # Historical AQI — Google + OpenAQ merge
│   ├── get_weather.py         # Weather data from OpenMeteo
│   ├── train_model.py         # ML model training — 4 models + chart export
│   ├── train_baseline.py      # SARIMA & LSTM baselines (offline, for research)
│   └── predict.py             # Iterative 5-day forecast generation
├── evaluation_charts/          # Auto-generated 300 DPI evaluation figures
├── .env                        # Environment variables (create this)
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 📊 Data Sources & APIs

### 🌍 Google Air Quality API
- Real-time AQI for 100+ countries
- Historical AQI up to 30 days back
- Pollutant concentrations and health recommendations
- Universal AQI index with category classifications

### 🟦 OpenAQ v3 API
- Free historical PM2.5 / PM10 sensor data going back years
- Used to supplement Google's 30-day cap — extends training window to 150 days
- PM2.5 converted to AQI using the standard EPA breakpoint formula
- No additional cost; requires a free API key

### 🌤️ OpenMeteo Weather API
- Historical weather from the archive API
- Real-time and 5-day forecast weather
- Hourly: temperature, humidity, precipitation, wind speed/direction

### 🤖 Google Gemini AI
- Gemini 2.5 Flash for health consultations
- Context-aware responses using current AQI + 5-day forecast
- Personalized advice for user-specified health conditions
- Streaming responses for real-time interaction

---

## 🤖 AI Health Advisor

The integrated chatbot provides personalized advice based on the user's health profile and live AQI data.

### Health Conditions Supported
- **Respiratory**: Asthma, COPD, Bronchitis, Allergies
- **Cardiovascular**: Heart Disease, Hypertension
- **Metabolic**: Diabetes
- **Special Groups**: Pregnancy, Elderly (65+), Children
- **Custom**: Any user-defined condition

### Chat Features
- Quick action buttons (safe to exercise? stay indoors? precautions?)
- Streaming real-time responses
- Full context: current AQI + 5-day forecast injected into every session
- Export chat history as a text file

---

## 🔬 Machine Learning Pipeline

### Data Collection
- Google Air Quality API: most recent 30 days (authoritative, full AQI index)
- OpenAQ API: older period up to 150 days total (PM2.5/PM10 → AQI via EPA formula)
- OpenMeteo: matching hourly weather for the full date range
- All sources merged into a single hourly DataFrame

### Feature Engineering
| Category | Features |
|---|---|
| Temporal | Hour of day, day of week, month |
| Lag features | AQI lag 1 hr, AQI lag 24 hr, temperature lag 1 hr, temperature lag 24 hr |
| Rolling stats | 24-hour AQI rolling average |
| Weather | Temperature, humidity, precipitation, wind speed, wind direction |

### Models Trained
| Model | Notes |
|---|---|
| LightGBM | Gradient boosting with early stopping |
| XGBoost | Gradient boosting with early stopping |
| Random Forest | Bagging ensemble, 300 trees |
| Ridge Regression | Linear baseline |
| SARIMA *(offline)* | Classical time-series baseline — run via `logic/train_baseline.py` |
| LSTM *(offline)* | Deep learning baseline — run via `logic/train_baseline.py` |

The best model (highest R² on the 20% test split) is automatically saved and used for forecasting.

### Validation
- 80/20 time-series split (no shuffling — respects temporal order)
- Metrics: R², MAE
- Three evaluation charts exported at 300 DPI to `evaluation_charts/`

---

## 🚀 Usage Guide

1. **Enter a city** in the sidebar (e.g. Mumbai, London, New York)
2. **Click "Generate 5-Day AQI Forecast"** — fetches data, trains models, generates charts
3. **Explore the three tabs:**
   - 📊 **Current AQI** — live readings, model comparison table, evaluation charts
   - 📈 **5-Day Forecast** — hourly forecast chart, summary, CSV download
   - 🤖 **Health Chat** — set your health profile and start a conversation

---

## 🔧 Configuration

### Timezone
Automatically detected from city coordinates using `timezonefinder`. All forecast timestamps display in the local timezone.

### Model Persistence
The best model is saved as `aqi_model_robust.pkl` after each forecast generation run. Subsequent forecasts load this file directly unless re-trained.

### API Rate Limits
- **Google APIs**: Free tier with generous quotas (monitor in Cloud Console)
- **OpenAQ**: Free tier, rate-limited — the app pages through results gracefully
- **OpenMeteo**: Free with fair-use limits
- **Gemini AI**: Free tier available; upgrade for higher throughput

---

## 🙏 Acknowledgments

**APIs & Services:** Google Cloud (Air Quality + Geocoding), OpenAQ, OpenMeteo, Google AI (Gemini)

**Libraries:** Streamlit, LightGBM, XGBoost, scikit-learn, Pandas, NumPy, Matplotlib, Plotly, SciPy

---

<div align="center">

**Made with ❤️ by [Sanket Dangle](https://github.com/shanks1911)**

![Footer](https://img.shields.io/badge/Built%20with-Python%20%7C%20Streamlit%20%7C%20AI-blue?style=for-the-badge)

</div>