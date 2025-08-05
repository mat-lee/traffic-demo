import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re

# Load the pre-trained models
clf = joblib.load("clf.pkl")      # Classifier
tfidf = joblib.load("tfidf.pkl")  # TFIDF Vectorizer
pca = joblib.load("pca.pkl")      # Truncated SVD

n_components = 3

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_input(lat, lng, tmp, description, humidity, pressure, visibility, wind_speed, precipitation, 
                     amenity, bump, crossing, give_way, junction, no_exit, railway, roundabout, station, stop,
                     traffic_calming, traffic_signal, turning_loop, accident_duration, month, year,
                     wind, hour, day_of_week, distance, accident_angle, wind_angle, is_highway):
    """
    Converts user inputs into a feature vector for prediction.
    """

    dtype_mapping = {
        'Source': 'float64',
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
        'desc_svd_1': 'float64',
        'desc_svd_2': 'float64',
        'desc_svd_3': 'float64',
        'desc_svd_4': 'float64',
        'desc_svd_5': 'float64',
        'desc_svd_6': 'float64',
        'desc_svd_7': 'float64',
        'Is_Highway': 'int64'
        }

    wind_chill = 35.74 + 0.6215 * tmp - 35.75 * (wind_speed ** 0.16) + 0.4275 * tmp * (wind_speed ** 0.16)

    # Numeric features
    data = pd.DataFrame({
        "Source": [1],
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
        "Accident_Duration": [accident_duration * 60],  # Convert minutes to seconds
        "Weather_Timestamp_Delta": [0],
        "Hour": [hour],
        "DayOfWeek": [day_of_week],
        "Month": [month],
        "Year": [year],  
        # "Weather": [weather]
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

    pd.set_option('display.max_columns', None)
    # print(data.head())
    # print(description, tfidf_matrix)

    for col in data.columns:
        if col in dtype_mapping:
            data[col] = data[col].astype(dtype_mapping[col])
        else:
            print(f"Warning: Column {col} not found in dtype mapping.")

    return data

def clean_text(text):
    """Cleans and preprocesses the text description."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Traffic Accident Severity Predictor")
st.subheader("Estimates how severe an accident was in terms of effect on traffic flow.")

# Inputs
description = st.text_area("Accident Description", "Multi-vehicle collision with heavy rain")
latitude = st.number_input("Latitude", value=37.7749, step=0.0001)
longitude = st.number_input("Longitude", value=-122.4194, step=0.0001)
distance = st.slider("Distance (miles)", 0.0, 10.0, 1.5)
accident_angle = st.slider("Accident Bearing (degrees)", 0, 360, 180)
wind_angle = st.slider("Wind Bearing (degrees)", 0, 360, 90)
wind_speed = st.number_input("Wind Speed (mph)", value=5.0, step=0.1)
temperature = st.number_input("Temperature (°F)", value=70.0, step=0.1)
humidity = st.number_input("Humidity (%)", value=50, min_value=0, max_value=100)
pressure = st.number_input("Air Pressure (inches)", value=29.92, step=0.01)
visibility = st.number_input("Visibility (miles)", value=10.0, step=0.1)
precipitation = st.number_input("Precipitation (inches)", value=0.0, step=0.1)

# Checkboxes for nearby presence
st.write('Include the nearby presence of a:')
amenity = st.checkbox('Amenity')
bump = st.checkbox('Speed Bump')
crossing = st.checkbox('Crossing')
give_way = st.checkbox('Give Way Sign')
junction = st.checkbox('Junction')
no_exit = st.checkbox('No Exit Sign')
railway = st.checkbox('Railway Crossing')
roundabout = st.checkbox('Roundabout')
station = st.checkbox('Station')
stop = st.checkbox('Stop Sign')
traffic_calming = st.checkbox('Traffic Calming Sign')
traffic_signal = st.checkbox('Traffic Signal')
turning_loop = st.checkbox('Turning Loop')

accident_duration = st.slider("Accident Duration (minutes)", 0, 120, 30)
hour = st.slider("Hour of Day", 0, 23, 14)
day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
day_of_week_idx = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day_of_week)
month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June",
                         "July", "August", "September", "October", "November", "December"])
month_idx = ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"].index(month)
year = st.number_input("Year", value=2023, min_value=2000, max_value=2025)

wind = st.selectbox("Wind Condition", ["Calm", "Variable"])

is_highway = st.checkbox("Highway", value=False)

# Prediction Button
if st.button("Predict Severity"):
    X = preprocess_input(latitude, longitude, temperature, description, humidity, pressure, visibility, wind_speed, precipitation, 
                         amenity, bump, crossing, give_way, junction, no_exit, railway, roundabout, station, stop,
                         traffic_calming, traffic_signal, turning_loop, accident_duration, month_idx, year,
                         wind, hour, day_of_week_idx, distance, accident_angle, wind_angle, is_highway)
    
    # Make Prediction
    severity = clf.predict(X)[0]
    st.success(f"Predicted Severity: **{severity}**")
