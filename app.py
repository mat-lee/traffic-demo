import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import folium
from streamlit_folium import st_folium
import xgboost

# Load the pre-trained models
@st.cache_resource
def load_models():
    clf = joblib.load("clf.pkl")      # Classifier
    tfidf = joblib.load("tfidf.pkl")  # TFIDF Vectorizer
    pca = joblib.load("pca.pkl")      # Truncated SVD
    sc = joblib.load("sc.pkl")        # Standard Scaler
    le = joblib.load("le.pkl")        # Label Encoder
    return clf, tfidf, pca, sc, le

clf, tfidf, pca, sc, le = load_models()
n_components = 1

# -------------------------------
# Preprocessing function (same as original)
# -------------------------------
def preprocess_input(lat, lng, tmp, description, humidity, pressure, visibility, wind_speed, precipitation, 
                     amenity, bump, crossing, give_way, junction, no_exit, railway, roundabout, station, stop,
                     traffic_calming, traffic_signal, turning_loop, accident_duration, month, year,
                     wind, hour, day_of_week, distance, accident_angle, wind_angle, is_highway, weather_is_fair, 
                     weather_is_cloudy, weather_is_windy, weather_is_foggy, 
                     weather_is_raining, weather_is_thundering, weather_is_snowy, weather_is_hail, 
                     weather_is_smoky, weather_is_tornado, weather_is_light, weather_is_heavy):
    """
    Converts user inputs into a feature vector for prediction.
    """

    dtype_mapping = {
        # 'Source': 'float64',
        'Start_Lat': 'float64',
        'Start_Lng': 'float64',
        'Distance(mi)': 'float64',
        'Temperature(F)': 'float64',
        'Wind_Chill(F)': 'float64',
        'Humidity(%)': 'float64',
        'Pressure(in)': 'float64',
        'Visibility(mi)': 'float64',
        'Wind_Speed(mph)': 'float64',
        'Precipitation(in)': 'float64',
        'Amenity': 'bool',
        'Bump': 'bool',
        'Crossing': 'bool',
        'Give_Way': 'bool',
        'Junction': 'bool',
        'No_Exit': 'bool',
        'Railway': 'bool',
        'Roundabout': 'bool',
        'Station': 'bool',
        'Stop': 'bool',
        'Traffic_Calming': 'bool',
        'Traffic_Signal': 'bool',
        'Turning_Loop': 'bool',
        'Accident_Duration': 'float64',
        'Weather_Timestamp_Delta': 'float64',
        'Hour': 'int32',
        'DayOfWeek': 'int32',
        'Month': 'int32',
        'Year': 'int32',
        'Is_Morning_Rush': 'int64',
        'Is_Evening_Rush': 'int64',
        'Is_Weekend': 'int64',
        'Is_Wind_Calm': 'int64',
        'Is_Wind_Variable': 'int64',
        'Wind_Angle': 'float64',
        'Wind_Accident_Alignment': 'float64',
        'desc_svd_0': 'float64',
        # 'desc_svd_1': 'float64',
        # 'desc_svd_2': 'float64',
        # 'desc_svd_3': 'float64',
        # 'desc_svd_4': 'float64',
        # 'desc_svd_5': 'float64',
        # 'desc_svd_6': 'float64',
        # 'desc_svd_7': 'float64',
        'Is_Highway': 'int64',
        'Weather_Is_Fair': 'int64',
        'Weather_Is_Cloudy': 'int64',
        'Weather_Is_Windy': 'int64',
        'Weather_Is_Foggy': 'int64',
        'Weather_Is_Raining': 'int64',
        'Weather_Is_Thundering': 'int64',
        'Weather_Is_Snowy': 'int64',
        'Weather_Is_Hail': 'int64',
        'Weather_Is_Smoky': 'int64',
        'Weather_Is_Tornado': 'int64',
        'Weather_Is_Light': 'int64',
        'Weather_Is_Heavy': 'int64'
        }

    wind_chill = 35.74 + 0.6215 * tmp - 35.75 * (wind_speed ** 0.16) + 0.4275 * tmp * (wind_speed ** 0.16)

    # Numeric features
    data = pd.DataFrame({
        # "Source": [1],
        "Start_Lat": [lat], 
        "Start_Lng": [lng],
        "Distance(mi)": [distance],
        "Temperature(F)": [tmp],
        "Wind_Chill(F)": [wind_chill],
        "Humidity(%)": [humidity],
        "Pressure(in)": [pressure],
        "Visibility(mi)": [visibility],
        "Wind_Speed(mph)": [wind_speed],
        "Precipitation(in)": [precipitation],
        "Amenity": [1 if amenity else 0],
        "Bump": [1 if bump else 0],
        "Crossing": [1 if crossing else 0],
        "Give_Way": [1 if give_way else 0],
        "Junction": [1 if junction else 0],
        "No_Exit": [1 if no_exit else 0],
        "Railway": [1 if railway else 0],
        "Roundabout": [1 if roundabout else 0],
        "Station": [1 if station else 0],
        "Stop": [1 if stop else 0],
        "Traffic_Calming": [1 if traffic_calming else 0],
        "Traffic_Signal": [1 if traffic_signal else 0],
        "Turning_Loop": [1 if turning_loop else 0],
        "Weather_Is_Fair": [1 if weather_is_fair else 0],
        "Weather_Is_Cloudy": [1 if weather_is_cloudy else 0],
        "Weather_Is_Windy": [1 if weather_is_windy else 0],
        "Weather_Is_Foggy": [1 if weather_is_foggy else 0],
        "Weather_Is_Raining": [1 if weather_is_raining else 0],
        "Weather_Is_Thundering": [1 if weather_is_thundering else 0],
        "Weather_Is_Snowy": [1 if weather_is_snowy else 0],
        "Weather_Is_Hail": [1 if weather_is_hail else 0],
        "Weather_Is_Smoky": [1 if weather_is_smoky else 0],
        "Weather_Is_Tornado": [1 if weather_is_tornado else 0],
        "Weather_Is_Light": [1 if weather_is_light else 0],
        "Weather_Is_Heavy": [1 if weather_is_heavy else 0],
        "Accident_Duration": [accident_duration * 60],  # Convert minutes to seconds
        "Weather_Timestamp_Delta": [0],
        "Hour": [hour],
        "DayOfWeek": [day_of_week],
        "Month": [month],
        "Year": [year],  
    })

    # Derived features
    data['Is_Morning_Rush'] = data['Hour'].between(7, 9).astype(int)
    data['Is_Evening_Rush'] = data['Hour'].between(17, 19).astype(int)
    data['Is_Weekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
    data['Is_Wind_Calm'] = 1 if wind == 'Calm' else 0
    data['Is_Wind_Variable'] = 1 if wind == 'Variable' else 0
    data['Wind_Angle'] = wind_angle
    data['Wind_Accident_Alignment'] = abs(wind_angle - accident_angle)

    # Clean and preprocess description
    description = clean_text(description)
    tfidf_matrix = tfidf.transform([description])
    pca_matrix = pca.transform(tfidf_matrix)
    feature_names = [f"desc_svd_{i}" for i in range(n_components)]
    desc_df = pd.DataFrame(pca_matrix, columns=feature_names)

    # Merge description features
    data = pd.merge(data, desc_df, left_index=True, right_index=True)

    data['Is_Highway'] = int(is_highway)

    for col in data.columns:
        if col in dtype_mapping:
            data[col] = data[col].astype(dtype_mapping[col])

    data = sc.transform(data)

    return data

def clean_text(text):
    """Cleans and preprocesses the text description."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

##### Streamlit UI
st.set_page_config(page_title="Traffic Accident Severity Predictor", layout="wide")

st.title("🚗 Traffic Accident Severity Predictor")
st.markdown("*Click on the map to select location, adjust parameters to see real-time predictions*")

# Create columns for layout
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.subheader("📍 Location & Basic Info")
    
    # Interactive US Map
    st.write("**Click on map to select location:**")
    
    if 'latitude' not in st.session_state:
        st.session_state.latitude = 39.8283  # Center of US
        st.session_state.longitude = -98.5795
    
    m = folium.Map(
        location=[st.session_state.latitude, st.session_state.longitude], 
        zoom_start=3,
        width=400,
        height=300
    )
    
    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude],
        popup=f"Lat: {st.session_state.latitude:.4f}, Lng: {st.session_state.longitude:.4f}",
        tooltip="Selected Location"
    ).add_to(m)
    
    map_data = st_folium(m, key="map", width=400, height=300)
    
    if map_data['last_clicked'] is not None:
        st.session_state.latitude = map_data['last_clicked']['lat']
        st.session_state.longitude = map_data['last_clicked']['lng']
        st.rerun()
    
    st.info(f"📍 Selected: {st.session_state.latitude:.4f}, {st.session_state.longitude:.4f}")
    
    col1a, col1b = st.columns(2)
    with col1a:
        description = st.text_area("Description", "Multi-vehicle collision", height=80)
        distance = st.slider("Distance (mi)", 0.0, 10.0, 1.5, key="dist")
        hour = st.slider("Hour", 0, 23, 14, key="hr")
        
    with col1b:
        accident_duration = st.slider("Duration (min)", 0, 120, 30, key="dur")
        accident_angle = st.slider("Accident Bearing°", 0, 360, 180, key="acc_ang")
        wind_angle = st.slider("Wind Bearing°", 0, 360, 90, key="wind_ang")

with col2:
    st.subheader("🌤️ Weather & Environment")
    
    col2a, col2b = st.columns(2)
    with col2a:
        temperature = st.number_input("Temp (°F)", value=70.0, step=1.0, key="temp")
        humidity = st.slider("Humidity %", 0, 100, 50, key="humid")
        pressure = st.number_input("Pressure (in)", value=29.92, step=0.1, key="press")
        wind_speed = st.number_input("Wind (mph)", value=5.0, step=1.0, key="wind_spd")
        
    with col2b:
        visibility = st.number_input("Visibility (mi)", value=10.0, step=1.0, key="vis")
        precipitation = st.number_input("Precipitation (in)", value=0.0, step=0.1, key="precip")
        wind = st.selectbox("Wind", ["Calm", "Variable"], key="wind_cond")
        is_highway = st.checkbox("Highway", key="hwy")
    
    # Weather conditions checkboxes
    st.write("**Weather Conditions:**")
    col2_w1, col2_w2, col2_w3 = st.columns(3)
    with col2_w1:
        weather_is_fair = st.checkbox('Fair', key="fair")
        weather_is_cloudy = st.checkbox('Cloudy', key="cloudy")
        weather_is_windy = st.checkbox('Windy', key="windy")
        weather_is_foggy = st.checkbox('Foggy', key="foggy")
    with col2_w2:
        weather_is_raining = st.checkbox('Raining', key="rain")
        weather_is_thundering = st.checkbox('Thunder', key="thunder")
        weather_is_snowy = st.checkbox('Snowy', key="snow")
        weather_is_hail = st.checkbox('Hail', key="hail")
    with col2_w3:
        weather_is_smoky = st.checkbox('Smoky', key="smoke")
        weather_is_tornado = st.checkbox('Tornado', key="tornado")
        weather_is_light = st.checkbox('Light', key="light")
        weather_is_heavy = st.checkbox('Heavy', key="heavy")
    
    # Date/Time in compact format
    col2c, col2d = st.columns(2)
    with col2c:
        day_of_week = st.selectbox("Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], key="dow")
        day_of_week_idx = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(day_of_week)
    with col2d:
        month = st.selectbox("Month", ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], key="mon")
        month_idx = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"].index(month)
        year = st.number_input("Year", value=2020, min_value=2016, max_value=2023, key="yr")
    
    # Road features in compact checkboxes
    st.write("**Nearby Features:**")
    col2e, col2f, col2g = st.columns(3)
    with col2e:
        amenity = st.checkbox('Amenity', key="amen")
        bump = st.checkbox('Speed Bump', key="bump")
        crossing = st.checkbox('Crossing', key="cross")
        give_way = st.checkbox('Give Way', key="give")
        junction = st.checkbox('Junction', key="junc")
    with col2f:
        no_exit = st.checkbox('No Exit', key="no_exit")
        railway = st.checkbox('Railway', key="rail")
        roundabout = st.checkbox('Roundabout', key="round")
        station = st.checkbox('Station', key="stat")
    with col2g:
        stop = st.checkbox('Stop Sign', key="stop")
        traffic_calming = st.checkbox('Traffic Calm', key="calm")
        traffic_signal = st.checkbox('Signal', key="signal")
        turning_loop = st.checkbox('Turn Loop', key="loop")

with col3:
    st.subheader("🎯 Prediction")
    
    # Auto-update prediction (no button needed)
    try:
        X = preprocess_input(
            st.session_state.latitude, st.session_state.longitude, temperature, description, 
            humidity, pressure, visibility, wind_speed, precipitation, 
            amenity, bump, crossing, give_way, junction, no_exit, railway, roundabout, 
            station, stop, traffic_calming, traffic_signal, turning_loop, accident_duration, 
            month_idx, year, wind, hour, day_of_week_idx, distance, accident_angle, wind_angle, is_highway,
            weather_is_fair, weather_is_cloudy, weather_is_windy, weather_is_foggy, 
            weather_is_raining, weather_is_thundering, weather_is_snowy, weather_is_hail, 
            weather_is_smoky, weather_is_tornado, weather_is_light, weather_is_heavy
        )
        
        # Make Prediction
        severity = le.inverse_transform([clf.predict(X)[0]])[0] # Convert to original label
        severity_prob = clf.predict_proba(X)[0]
        
        # Display result with styling
        severity_colors = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
        severity_names = {1: "Minor", 2: "Moderate", 3: "Severe", 4: "Very Severe"}
        
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h2>{severity_colors.get(severity, "❓")} Severity Level</h2>
            <h1 style="color: #ff6b6b;">{severity}</h1>
            <h3>{severity_names.get(severity, "Unknown")}</h3>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        print(e)