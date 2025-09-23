# 🌍 Air Quality Index Forecasting Dashboard

<div align="center">

![AQI Dashboard](https://img.shields.io/badge/AQI-Forecasting%20Dashboard-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSJjdXJyZW50Q29sb3IiLz4KPC9zdmc+)
![Python](https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49+-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Real-time AQI monitoring and intelligent 5-day forecasting powered by machine learning**

[🚀 Getting Started](#-getting-started) •
[✨ Features](#-features) •
[🛠️ Installation](#️-installation) •
[🤖 AI Chat](#-ai-health-advisor)

</div>

---

## 📋 Overview

The **Air Quality Index (AQI) Forecasting Dashboard** is a comprehensive web application that provides real-time air quality monitoring and intelligent 5-day forecasting for any city worldwide. Built with Streamlit and powered by advanced machine learning models, this dashboard helps users make informed decisions about outdoor activities and health precautions.

### 🎯 Key Highlights

- **Real-time AQI Data**: Instant air quality readings for any city
- **5-Day ML Forecasting**: Advanced machine learning predictions using LightGBM and Ridge Regression
- **AI Health Advisor**: Personalized health recommendations powered by Google's Gemini AI
- **Local Timezone Support**: Automatic timezone detection and conversion
- **Interactive Visualizations**: Clean, professional charts and data displays
- **Health Profile Integration**: Customized advice for specific health conditions

### 🔗 Link: https://aqihealthadvisory.streamlit.app/

---

## ✨ Features

### 🌡️ **Current Air Quality Monitoring**
- **Universal AQI readings** with color-coded categories
- **Pollutant concentration levels** (PM2.5, PM10, CO, NO2, O3)
- **Dominant pollutant identification**
- **Health recommendations** for general population

### 📈 **Intelligent Forecasting**
- **5-day hourly predictions** using ML models
- **Model performance metrics** (R², MAE, Accuracy)
- **Automatic timezone conversion** to local time
- **Downloadable forecast data** in CSV format

### 🤖 **AI Health Advisor**
- **Personalized health chatbot** powered by Gemini AI
- **Health condition profiling** (Asthma, COPD, Heart Disease, etc.)
- **Contextual advice** based on current and forecasted AQI
- **Quick action buttons** for common questions
- **Chat history export** functionality

### 🎨 **User Experience**
- **Clean, professional interface** with intuitive navigation
- **Responsive design** for all screen sizes
- **Auto-loading city data** when location changes
- **Progress tracking** for model training and forecasting
- **Comprehensive error handling**

---

## 🛠️ Installation

### Prerequisites
- Python 3.12 or higher
- Google Maps API Key
- Gemini API Key (for health chatbot)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/aqi-forecasting-dashboard.git
cd aqi-forecasting-dashboard
```

### 2. Install Dependencies

### 3. Environment Setup
Create a `.env` file in the project root:
```env
MAPS_KEY=your_google_maps_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. API Key Setup

#### Google Maps & Air Quality API
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the following APIs:
   - Geocoding API
   - Air Quality API
3. Create an API key and add it to your `.env` file

#### Gemini AI API
1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Generate an API key
3. Add it to your `.env` file

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
│   ├── get_data.py            # City geocoding & current AQI data
│   ├── get_history.py         # Historical AQI data fetching
│   ├── get_weather.py         # Weather data from OpenMeteo
│   ├── train_model.py         # ML model training pipeline
│   └── predict.py             # Forecast generation
├── .env                       # Environment variables (create this)
├── .gitignore                 # Git ignore patterns
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

---

## 📊 Data Sources & APIs

### 🌍 **Google Air Quality API**
- Real-time AQI data for 100+ countries
- Historical air quality data (up to 30 days)
- Pollutant concentrations and health recommendations
- Universal AQI index with category classifications

### 🌤️ **OpenMeteo Weather API**
- Historical weather data from archives
- Real-time weather conditions
- 5-day weather forecasts
- Hourly data including temperature, humidity, wind speed

### 🤖 **Google Gemini AI**
- Advanced language model for health consultations
- Context-aware responses based on AQI data
- Personalized recommendations for health conditions
- Streaming responses for real-time interaction

---

## 🤖 AI Health Advisor

The integrated health chatbot provides personalized advice based on:

### Health Conditions Supported:
- ✅ **Respiratory**: Asthma, COPD, Bronchitis, Allergies
- ✅ **Cardiovascular**: Heart Disease, Hypertension
- ✅ **Metabolic**: Diabetes
- ✅ **Special Groups**: Pregnancy, Elderly (65+), Children
- ✅ **Custom Conditions**: User-defined health concerns

### Chat Features:
- **Quick Action Buttons**: Common questions with one-click
- **Streaming Responses**: Real-time AI responses
- **Context Awareness**: Includes current AQI and forecast data
- **Chat History**: Export conversations as text files
- **Health Guidelines**: Built-in AQI reference guide

---

## 🔬 Machine Learning Pipeline

### Model Training Process:
1. **Data Collection**: Fetches 30 days of historical AQI and weather data
2. **Feature Engineering**: Creates lag features, rolling averages, and time-based features
3. **Model Training**: Trains both LightGBM and Ridge Regression models
4. **Model Selection**: Automatically selects the best-performing model
5. **Model Persistence**: Saves the trained model for future predictions

### Features Used:
- **Temporal**: Hour of day, day of week, month
- **Lag Features**: 1-hour and 24-hour AQI and temperature lags
- **Weather**: Temperature, humidity, precipitation, wind speed/direction
- **Rolling Statistics**: 24-hour moving averages

### Model Performance:
- **R² Score**: Typically achieves 0.65-0.80 accuracy
- **MAE**: Mean Absolute Error usually < 10 AQI points
- **Robust Validation**: Time-series split for realistic performance estimation

---

## 🚀 Usage Guide

### Quick Start:
1. **Enter a City**: Type any city name in the sidebar
2. **Generate Forecast**: Click "Generate 5-Day AQI Forecast"
3. **Explore Data**: Navigate through the three main tabs:
   - 📊 Current AQI
   - 📈 5-Day Forecast
   - 🤖 Health Chat

### Health Chatbot Usage:
1. **Set Health Profile**: Select your health conditions
2. **Start New Chat**: Initialize the AI advisor
3. **Ask Questions**: Use quick buttons or type custom questions
4. **Get Personalized Advice**: Receive contextual health recommendations

---

## 🔧 Configuration

### Timezone Settings
The app automatically detects location-based timezones using coordinates. All forecast times are displayed in the local timezone of the selected city.

### Model Configuration
Models are automatically trained and saved as `aqi_model_robust.pkl`. The system will:
- Try LightGBM first (usually best performance)
- Fall back to Ridge Regression if needed
- Provide model performance metrics

### API Rate Limits
- **Google APIs**: Generous free tiers (check your quotas)
- **OpenMeteo**: Free with fair usage limits
- **Gemini AI**: Free tier available with usage limits

---

## 🙏 Acknowledgments

### APIs & Services:
- **Google Cloud**: Air Quality API and Geocoding services
- **OpenMeteo**: Free weather API with excellent coverage
- **Google AI**: Gemini AI for intelligent health consultations

### Libraries & Frameworks:
- **Streamlit**: Amazing framework for data apps
- **LightGBM**: High-performance gradient boosting
- **scikit-learn**: Machine learning toolkit
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation and analysis

---

<div align="center">

**Made with ❤️ by [Sanket Dangle](https://github.com/your-username)**

![Footer](https://img.shields.io/badge/Built%20with-Python%20%7C%20Streamlit%20%7C%20AI-blue?style=for-the-badge)

</div>