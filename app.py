import os
import joblib
import pandas as pd
import streamlit as st

model_path = os.path.join(os.path.dirname(__file__), "energy_model.joblib")
model = joblib.load(model_path)

st.title("Campus Energy Optimizer")

hour = st.slider("Hour", 0, 23, 12)
day = st.slider("Day of Week", 0, 6, 3)
temperature = st.number_input("Temperature (°C)", 10, 45, 28)
humidity = st.number_input("Humidity (%)", 20, 100, 55)
occupancy = st.number_input("Occupancy", 0, 300, 50)
building_type = st.selectbox("Building Type", ["Academic", "Hostel", "Library", "Admin", "Lab"])
is_holiday = st.selectbox("Holiday?", [0, 1])
previous_usage = st.number_input("Previous Usage (kWh)", 0.0, 100.0, 30.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "hour": hour,
        "day_of_week": day,
        "temperature": temperature,
        "humidity": humidity,
        "occupancy": occupancy,
        "building_type": building_type,
        "is_holiday": is_holiday,
        "previous_usage": previous_usage
    }])
    
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Energy Usage: {prediction:.2f} kWh")
