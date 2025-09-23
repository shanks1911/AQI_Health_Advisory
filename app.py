import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px
import pytz
from timezonefinder import TimezoneFinder
from google import genai
from dotenv import load_dotenv

from logic.get_data import get_coords_from_city, get_air_quality
from logic.get_history import get_air_quality_history
from logic.get_weather import get_weather_history, get_recent_weather, get_weather_forecast
from logic.train_model import train_robust_models
from logic.predict import make_predictions

load_dotenv()

def get_api_key(key_name):
    """Get API key from Streamlit secrets or environment variables"""
    try:
        # Streamlit secrets (for Streamlit Cloud)
        return st.secrets[key_name]
    except:
        # Fall back
        return os.getenv(key_name)

# Initialize Gemini client
@st.cache_resource
def init_gemini_client():
    """Initialize Gemini client (cached to avoid recreating)"""
    try:
        # api_key = os.getenv("GEMINI_API_KEY")
        api_key = get_api_key("GEMINI_API_KEY")
        if not api_key:
            return None
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {e}")
        return None

def build_master_dataframe(aqi_data_dict: dict, historical_weather: pd.DataFrame, recent_weather: pd.DataFrame) -> pd.DataFrame | None:
    # This function is complete and requires no changes.
    if aqi_data_dict is None or historical_weather is None or recent_weather is None: return None
    full_weather_df = pd.concat([historical_weather, recent_weather])
    full_weather_df = full_weather_df[~full_weather_df.index.duplicated(keep='first')]
    aqi_records = []
    for entry in aqi_data_dict.get('hoursInfo', []):
        record = {'date': entry.get('dateTime')}
        indexes = entry.get('indexes', [])
        if indexes: record['aqi'] = indexes[0].get('aqi')
        for pollutant in entry.get('pollutants', []):
            code, concentration = pollutant.get('code'), pollutant.get('concentration', {}).get('value')
            if code and concentration is not None: record[f'{code}_concentration'] = concentration
        aqi_records.append(record)
    aqi_df = pd.DataFrame(aqi_records)
    aqi_df['date'] = pd.to_datetime(aqi_df['date'])
    aqi_df.set_index('date', inplace=True)
    if not full_weather_df.index.tz: full_weather_df = full_weather_df.tz_localize('UTC')
    master_df = pd.merge(aqi_df, full_weather_df, left_index=True, right_index=True, how='inner')
    master_df.sort_index(inplace=True)
    weather_cols_to_check = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m']
    for col in weather_cols_to_check:
        if col in master_df.columns: master_df[col] = master_df[col].replace(0, np.nan)
    master_df.dropna(subset=weather_cols_to_check, inplace=True)
    master_df.fillna(0, inplace=True)
    master_df['hour'] = master_df.index.hour
    master_df['day_of_week'] = master_df.index.dayofweek
    master_df['month'] = master_df.index.month
    for lag in [1, 24]:
        master_df[f'aqi_lag_{lag}hr'] = master_df['aqi'].shift(lag)
        if 'pm25_concentration' in master_df.columns: master_df[f'pm25_lag_{lag}hr'] = master_df['pm25_concentration'].shift(lag)
        if 'temperature_2m' in master_df.columns: master_df[f'temp_lag_{lag}hr'] = master_df['temperature_2m'].shift(lag)
    master_df['aqi_rolling_avg_24hr'] = master_df['aqi'].shift(1).rolling(window=24).mean()
    master_df.dropna(inplace=True)
    return master_df

def get_aqi_category(aqi_value):
    """Return AQI category based on value"""
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def convert_forecast_to_local_time(forecast_df: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
    """Convert forecast DataFrame from UTC to local timezone based on coordinates."""
    if forecast_df is None or forecast_df.empty:
        return forecast_df
    
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=latitude, lng=longitude)
    
    if timezone_str is None:
        return forecast_df
    
    local_df = forecast_df.copy()
    
    if local_df.index.tz is None:
        local_df = local_df.tz_localize('UTC')
    
    local_tz = pytz.timezone(timezone_str)
    local_df = local_df.tz_convert(local_tz)
    
    return local_df

def format_forecast_summary(forecast_df: pd.DataFrame) -> str:
    """Create a concise forecast summary for the chatbot"""
    if forecast_df is None or forecast_df.empty:
        return "No forecast data available."
    
    avg_aqi = forecast_df['Predicted_AQI'].mean()
    max_aqi = forecast_df['Predicted_AQI'].max()
    min_aqi = forecast_df['Predicted_AQI'].min()
    
    avg_category = get_aqi_category(avg_aqi)
    max_category = get_aqi_category(max_aqi)
    
    forecast_df_copy = forecast_df.copy()
    forecast_df_copy['date'] = forecast_df_copy.index.date
    daily_avg = forecast_df_copy.groupby('date')['Predicted_AQI'].mean()
    
    summary = f"""
5-Day AQI Forecast Summary:
- Average AQI: {avg_aqi:.0f} ({avg_category})
- Peak AQI: {max_aqi:.0f} ({max_category})
- Best AQI: {min_aqi:.0f} ({get_aqi_category(min_aqi)})

Daily Averages:
"""
    for date, aqi in daily_avg.head().items():
        cat = get_aqi_category(aqi)
        summary += f"- {date}: {aqi:.0f} ({cat})\n"
    
    return summary.strip()

def initialize_chat_session(client, city: str, current_aqi: dict, forecast_summary: str, health_conditions: list):
    """Initialize a new chat session with health context"""
    
    # Build context
    current_context = ""
    if current_aqi:
        current_context = f"""
CURRENT AIR QUALITY IN {city.upper()}:
- AQI Level: {current_aqi['aqi']} ({current_aqi['category']})
- Dominant Pollutant: {current_aqi.get('dominantPollutant', 'Unknown')}
- General Health Advice: {current_aqi.get('health_recommendations', {}).get('generalPopulation', 'Follow standard precautions')}
"""
    
    forecast_context = ""
    if forecast_summary and forecast_summary != "No forecast data available.":
        forecast_context = f"""
UPCOMING FORECAST:
{forecast_summary}
"""
    
    conditions_context = ""
    if health_conditions:
        conditions_list = ", ".join(health_conditions)
        conditions_context = f"""
USER'S HEALTH CONCERNS: {conditions_list}
Pay special attention to how air quality affects these conditions.
"""
    
    system_prompt = f"""You are an AI health advisor specializing in air quality and its impact on human health. Provide personalized, actionable advice based on current and forecasted air quality conditions.

{current_context}

{forecast_context}

{conditions_context}

INSTRUCTIONS:
1. Provide specific, actionable advice based on the user's health conditions and air quality data
2. Explain how current and forecasted air quality levels affect their health conditions
3. Give practical recommendations for outdoor activities, indoor air quality, medications, and protective measures
4. Be empathetic and supportive while being medically responsible
5. Always recommend consulting healthcare providers for serious concerns
6. Use clear, non-technical language while being scientifically accurate
7. Structure responses with: Current Situation, Health Impact, Recommendations, When to Seek Help

HEALTH CONDITION KNOWLEDGE:
- Asthma: Air pollution triggers attacks, especially PM2.5 and ozone
- COPD: Poor air quality worsens breathing difficulties
- Heart Disease: PM2.5 increases cardiovascular event risk
- Diabetes: Air pollution affects blood sugar control
- Pregnancy: Pollution exposure affects fetal development
- Elderly: Increased sensitivity to all pollutants
- Children: Developing lungs are more susceptible

Remember: This is advisory information, not a substitute for professional medical advice.

Please acknowledge you understand this context and are ready to provide health advice for this user in {city}."""

    try:
        chat = client.chats.create(model="gemini-2.0-flash-exp")
        response = chat.send_message(system_prompt)
        return chat
    except Exception as e:
        st.error(f"Error initializing chat: {e}")
        return None

def render_health_chatbot():
    """Render the health advisory chatbot section"""
    
    st.divider()
    st.header("🤖 AQI Health Advisory Chatbot")
    st.write("Get personalized health advice based on your city's air quality conditions.")
    
    # Check if Gemini is available
    client = init_gemini_client()
    if not client:
        st.error("Health chatbot is unavailable. Please set GEMINI_API_KEY in your .env file.")
        return
    
    # Get city from main app
    current_city = st.session_state.get('city', '')
    if not current_city:
        st.warning("Please select a city in the sidebar above to use the health chatbot.")
        return
    
    # Health conditions input
    st.subheader("Your Health Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Select any health conditions:**")
        health_conditions = []
        if st.checkbox("Asthma", key="health_asthma"):
            health_conditions.append("asthma")
        if st.checkbox("COPD", key="health_copd"):
            health_conditions.append("copd")
        if st.checkbox("Heart Disease", key="health_heart"):
            health_conditions.append("heart_disease")
        if st.checkbox("Diabetes", key="health_diabetes"):
            health_conditions.append("diabetes")
        if st.checkbox("Pregnancy", key="health_pregnancy"):
            health_conditions.append("pregnancy")
    
    with col2:
        st.write("**Age/Special Groups:**")
        if st.checkbox("Elderly (65+)", key="health_elderly"):
            health_conditions.append("elderly")
        if st.checkbox("Children", key="health_children"):
            health_conditions.append("children")
        if st.checkbox("Allergies", key="health_allergies"):
            health_conditions.append("allergies")
        if st.checkbox("Bronchitis", key="health_bronchitis"):
            health_conditions.append("bronchitis")
    
    # Other conditions
    other_condition = st.text_input("Other health conditions:", key="health_other")
    if other_condition.strip():
        health_conditions.append(other_condition.strip())
    
    # Chat session management
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_forecast = st.checkbox("Include forecast data", value=True, key="include_forecast_chat")
    
    with col2:
        if st.button("🚀 Start New Chat", type="primary"):
            # Prepare data for chat initialization
            current_aqi = None
            if 'current_data' in st.session_state:
                current_data = st.session_state['current_data']
                main_aqi = current_data['indexes'][0]
                current_aqi = {
                    'aqi': main_aqi['aqi'],
                    'category': main_aqi['category'],
                    'dominantPollutant': main_aqi.get('dominantPollutant', 'Unknown'),
                    'health_recommendations': current_data.get('healthRecommendations', {})
                }
            
            # Get forecast summary
            forecast_summary = ""
            if include_forecast and 'forecast' in st.session_state:
                forecast_df = st.session_state['forecast']
                if forecast_df is not None and not forecast_df.empty:
                    # Convert to local time for summary
                    lat, lon = st.session_state.get('coordinates', (0, 0))
                    forecast_df_local = convert_forecast_to_local_time(forecast_df, lat, lon)
                    forecast_summary = format_forecast_summary(forecast_df_local)
            
            # Initialize chat
            with st.spinner("Initializing your personal health advisor..."):
                chat = initialize_chat_session(client, current_city, current_aqi, forecast_summary, health_conditions)
                if chat:
                    st.session_state['health_chat'] = chat
                    st.session_state['chat_city'] = current_city
                    st.session_state['chat_conditions'] = health_conditions
                    st.session_state['chat_history'] = []
                    st.success(f"Health advisor ready for {current_city}!")
                    st.rerun()
                else:
                    st.error("Failed to start chat session.")
    
    with col3:
        if st.button("🔄 Reset Chat"):
            # Clear chat session
            for key in ['health_chat', 'chat_city', 'chat_conditions', 'chat_history']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Chat session reset!")
            st.rerun()
    
    # Display chat status
    if 'health_chat' in st.session_state:
        st.success(f"✅ Active chat session for {st.session_state.get('chat_city', current_city)}")
        if st.session_state.get('chat_conditions'):
            st.info(f"Health profile: {', '.join(st.session_state['chat_conditions'])}")
    else:
        st.info("Click 'Start New Chat' to begin getting personalized health advice.")
        return
    
    # Quick action buttons
    st.subheader("Quick Questions")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("🏃‍♂️ Safe to exercise?", key="quick_exercise"):
            st.session_state['pending_question'] = "Is it safe for me to exercise outside today given the current air quality and my health conditions?"
    
    with quick_col2:
        if st.button("🏠 Stay indoors?", key="quick_indoors"):
            st.session_state['pending_question'] = "Should I avoid going outside today? What's the best time to go out if I need to?"
    
    with quick_col3:
        if st.button("⚠️ Precautions needed?", key="quick_precautions"):
            st.session_state['pending_question'] = "What precautions should I take today and this week given the air quality forecast?"
    
    with quick_col4:
        if st.button("💊 Medication advice?", key="quick_medication"):
            st.session_state['pending_question'] = "Should I consider adjusting my medications or treatment based on the current air quality?"
    
    # Chat interface
    st.subheader("💬 Chat with Your Health Advisor")
    
    # Display chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        for i, (user_msg, bot_msg, timestamp) in enumerate(st.session_state['chat_history']):
            st.write(f"**You ({timestamp}):**")
            st.write(user_msg)
            st.write("**🏥 Health Advisor:**")
            st.info(bot_msg)
            st.divider()
    
    # Handle pending questions from quick buttons
    if 'pending_question' in st.session_state:
        question = st.session_state['pending_question']
        del st.session_state['pending_question']
        
        timestamp = datetime.now().strftime("%H:%M")
        
        # Add user message to display immediately
        st.write(f"**You ({timestamp}):**")
        st.write(question)
        st.write("**🏥 Health Advisor:**")
        
        # Get streaming response
        with st.spinner("Analyzing your health profile and air quality conditions..."):
            try:
                chat = st.session_state['health_chat']
                response_placeholder = st.empty()
                
                # Stream the response
                response_stream = chat.send_message_stream(question)
                full_response = ""
                
                for chunk in response_stream:
                    full_response += chunk.text
                    response_placeholder.info(full_response + "▋")  # Typing indicator
                
                response_placeholder.info(full_response)  # Final response
                
                # Add to history
                st.session_state['chat_history'].append((question, full_response, timestamp))
                st.rerun()
                
            except Exception as e:
                st.error(f"Error getting response: {e}")
    
    # Text input for custom questions
    user_question = st.text_area(
        "Ask your health question:",
        placeholder="e.g., I have a cough today. Is it safe to take my daily walk?",
        height=80,
        key="user_question_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col2:
        send_button = st.button("Send 💬", type="primary")
    
    # Handle user input
    if send_button and user_question.strip():
        timestamp = datetime.now().strftime("%H:%M")
        
        # Add user message immediately
        st.write(f"**You ({timestamp}):**")
        st.write(user_question)
        st.write("**🏥 Health Advisor:**")
        
        # Get streaming response
        try:
            chat = st.session_state['health_chat']
            response_placeholder = st.empty()
            
            # Stream the response
            response_stream = chat.send_message_stream(user_question)
            full_response = ""
            
            for chunk in response_stream:
                full_response += chunk.text
                response_placeholder.info(full_response + "▋")  # Typing indicator
            
            response_placeholder.info(full_response)  # Final response
            
            # Add to history and clear input
            st.session_state['chat_history'].append((user_question, full_response, timestamp))
            st.session_state['user_question_input'] = ""  # Clear the text area
            st.rerun()
            
        except Exception as e:
            st.error(f"Error getting response: {e}")
    
    # Chat controls
    st.subheader("Chat Controls")
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        if st.button("🗑️ Clear History"):
            st.session_state['chat_history'] = []
            st.rerun()
    
    with control_col2:
        if st.button("💾 Export Chat") and st.session_state.get('chat_history'):
            chat_text = f"AQI Health Chat - {current_city}\n"
            chat_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            chat_text += f"Health Conditions: {', '.join(st.session_state.get('chat_conditions', []))}\n"
            chat_text += "="*50 + "\n\n"
            
            for user_msg, bot_msg, timestamp in st.session_state['chat_history']:
                chat_text += f"[{timestamp}] You: {user_msg}\n\n"
                chat_text += f"Health Advisor: {bot_msg}\n\n"
                chat_text += "-"*30 + "\n\n"
            
            st.download_button(
                "Download Chat History",
                chat_text,
                file_name=f"health_chat_{current_city}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain"
            )
    
    with control_col3:
        if st.button("ℹ️ Health Guidelines"):
            with st.expander("AQI Health Guidelines", expanded=True):
                st.markdown("""
                **🟢 Good (0-50):** Safe for all activities
                
                **🟡 Moderate (51-100):** Sensitive people should limit prolonged outdoor exertion
                
                **🟠 Unhealthy for Sensitive Groups (101-150):** Respiratory/heart patients, children, elderly should limit outdoor activities
                
                **🔴 Unhealthy (151-200):** Everyone should limit outdoor exertion
                
                **🟣 Very Unhealthy (201-300):** Everyone should avoid outdoor activities
                
                **🟤 Hazardous (301+):** Stay indoors, keep windows closed
                """)

# Page Configuration
st.set_page_config(
    page_title="AQI Forecasting Dashboard", 
    page_icon="🌍", 
    layout="wide"
)

# Header
st.title("🌍 Air Quality Index Forecasting Dashboard")
st.write("Real-time AQI monitoring and intelligent 5-day forecasting powered by machine learning")
st.divider()

# Sidebar for Location Input
with st.sidebar:
    st.header("Location Settings")
    city = st.text_input("Enter city name:", value="Thane")
    
    # Auto-fetch current data when city changes
    if city and (city != st.session_state.get('last_city', '')):
        st.session_state['last_city'] = city
        with st.spinner(f"Fetching data for {city}..."):
            coordinates = get_coords_from_city(city)
            if coordinates:
                st.session_state.clear()
                st.session_state['last_city'] = city
                st.session_state['coordinates'] = coordinates
                st.session_state['current_data'] = get_air_quality(coordinates[0], coordinates[1])
                st.session_state['city'] = city
                st.success(f"Data loaded for {city}")
            else:
                st.error(f"Could not find coordinates for '{city}'.")
                st.session_state.clear()
    
    st.divider()
    
    # Forecast Generation
    st.header("Generate Forecast")
    
    if 'current_data' in st.session_state:
        if st.button("Generate 5-Day AQI Forecast", type="primary"):
            lat, lon = st.session_state['coordinates']
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Building comprehensive forecast..."):
                try:
                    # Fetch historical data
                    status_text.text("Fetching historical AQI data...")
                    progress_bar.progress(20)
                    days_total, days_recent_gap = 30, 2
                    end_date_hist = (datetime.now() - timedelta(days=days_recent_gap)).strftime('%Y-%m-%d')
                    start_date_hist = (datetime.now() - timedelta(days=days_total)).strftime('%Y-%m-%d')
                    
                    aqi_history_data = get_air_quality_history(lat, lon, days_back=days_total)
                    progress_bar.progress(40)
                    
                    # Fetch weather data
                    status_text.text("Fetching weather data...")
                    historical_weather_data = get_weather_history(lat, lon, start_date=start_date_hist, end_date=end_date_hist)
                    recent_weather_data = get_recent_weather(lat, lon, past_days=days_recent_gap)
                    progress_bar.progress(60)
                    
                    # Build dataset and train model
                    status_text.text("Training AI model...")
                    if aqi_history_data and historical_weather_data is not None and recent_weather_data is not None:
                        final_df = build_master_dataframe(aqi_history_data, historical_weather_data, recent_weather_data)
                        if final_df is not None and not final_df.empty:
                            st.session_state['training_results'] = train_robust_models(final_df)
                            progress_bar.progress(80)
                            
                            # Generate forecast
                            status_text.text("Generating forecast...")
                            st.session_state['forecast'] = make_predictions(lat, lon)
                            progress_bar.progress(100)
                            
                            status_text.text("Forecast complete!")
                        else:
                            st.error("Could not generate dataset for training.")
                    else:
                        st.error("Failed to fetch required data.")
                        
                except Exception as e:
                    st.error(f"Error during forecast generation: {str(e)}")
    else:
        st.info("Enter a city name above to start")

# Main Content Area - Using tabs for better organization
if 'current_data' in st.session_state:
    tab1, tab2, tab3 = st.tabs(["📊 Current AQI", "📈 5-Day Forecast", "🤖 Health Chat"])
    
    with tab1:
        # Current Conditions
        current_data = st.session_state['current_data']
        
        st.header(f"Current Air Quality in {st.session_state['city']}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            main_aqi = current_data['indexes'][0]
            aqi_value = main_aqi['aqi']
            st.metric("Universal AQI", aqi_value, help=f"Category: {main_aqi['category']}")
        
        with col2:
            st.metric("Dominant Pollutant", main_aqi.get('dominantPollutant', 'N/A'))
            
            # Show pollutant levels
            if 'pollutants' in current_data:
                st.write("**Pollutant Levels:**")
                for pollutant in current_data['pollutants'][:3]:
                    concentration = pollutant.get('concentration', {})
                    st.write(f"{pollutant.get('displayName', 'Unknown')}: {concentration.get('value', 'N/A')} {concentration.get('units', '')}")
        
        with col3:
            # Health recommendations
            recommendations = current_data.get('healthRecommendations', {})
            st.write("**Health Recommendations:**")
            st.info(recommendations.get('generalPopulation', 'No specific recommendations available.'))
        
        # Model Performance
        if 'training_results' in st.session_state:
            st.divider()
            st.header("AI Model Performance")
            results = st.session_state['training_results']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model Type", results['model_name'])
            with col2:
                st.metric("R² Score", f"{results['r2']:.3f}")
            with col3:
                st.metric("MAE", f"{results['mae']:.2f}")
            with col4:
                accuracy_percent = results['r2'] * 100
                st.metric("Accuracy", f"{accuracy_percent:.1f}%")
    
    with tab2:
        # Forecast Section
        if 'forecast' in st.session_state:
            st.header("5-Day AQI Forecast")
            forecast_df = st.session_state['forecast']
            
            # Convert to local time
            lat, lon = st.session_state['coordinates']
            forecast_df_local = convert_forecast_to_local_time(forecast_df, lat, lon)
            
            if forecast_df_local is not None and not forecast_df_local.empty:
                # Get timezone info for display
                tf = TimezoneFinder()
                timezone_str = tf.timezone_at(lat=lat, lng=lon)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader("Forecast Chart")
                    forecast_display = forecast_df_local.copy()
                    forecast_display.index = forecast_display.index.strftime('%m/%d %H:%M')
                    st.line_chart(forecast_display['Predicted_AQI'], height=300)
                
                with col2:
                    st.subheader("Forecast Summary")
                    
                    # Show timezone info
                    if timezone_str:
                        st.info(f"**Timezone:** {timezone_str}")
                    
                    # Show forecast range
                    start_time = forecast_df_local.index[0].strftime('%b %d, %Y %H:%M')
                    end_time = forecast_df_local.index[-1].strftime('%b %d, %Y %H:%M')
                    st.write(f"**Forecast Period:**")
                    st.write(f"From: {start_time}")
                    st.write(f"To: {end_time}")
                    
                    st.divider()
                    
                    # AQI statistics
                    avg_aqi = forecast_df_local['Predicted_AQI'].mean()
                    max_aqi = forecast_df_local['Predicted_AQI'].max()
                    min_aqi = forecast_df_local['Predicted_AQI'].min()
                    
                    st.metric("Average AQI", f"{avg_aqi:.0f}")
                    st.write(f"Category: {get_aqi_category(avg_aqi)}")
                    
                    st.metric("Peak AQI", f"{max_aqi:.0f}")
                    st.write(f"Category: {get_aqi_category(max_aqi)}")
                    
                    st.metric("Best AQI", f"{min_aqi:.0f}")
                    st.write(f"Category: {get_aqi_category(min_aqi)}")
                
                # Detailed forecast table
                st.subheader("Detailed Forecast Data")
                
                display_df = forecast_df_local.copy()
                display_df['AQI Category'] = display_df['Predicted_AQI'].apply(get_aqi_category)
                
                # Format the datetime index for better display in the table
                display_df.index = display_df.index.strftime('%Y-%m-%d %H:%M %Z')
                display_df = display_df.round(1)
                
                st.dataframe(display_df, width='stretch')
                
                # Download option
                csv = display_df.to_csv()
                st.download_button(
                    label="Download Forecast Data",
                    data=csv,
                    file_name=f"aqi_forecast_{st.session_state['city']}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )
            else:
                st.error("Could not generate the forecast. Please try again.")
        else:
            st.info("Generate a forecast in the sidebar to view predictions.")
    
    with tab3:
        # Health Advisory Chatbot
        render_health_chatbot()

else:
    # No current data available
    st.info("Enter a city name in the sidebar to start using the AQI Forecasting Dashboard.")

# Footer
st.divider()
st.write("Air Quality forecasting powered by machine learning • Built with Streamlit")
st.write("Data sources: OpenMeteo, Google Air Quality APIs • Health advice powered by Gemini AI")