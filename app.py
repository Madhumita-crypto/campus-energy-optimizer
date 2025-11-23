import streamlit as st
import joblib
import numpy as np

model = joblib.load("energy_model.joblib")

st.title("Campus Energy Optimizer")

hour = st.slider("Hour", 0, 23, 12)
day = st.slider("Day of Week", 0, 6, 3)
temperature = st.number_input("Temperature (°C)", 10, 45, 28)
humidity = st.number_input("Humidity (%)", 20, 100, 55)
occupancy = st.number_input("Occupancy", 0, 300, 50)
building_type = st.selectbox("Building Type", ["Academic","Hostel","Library","Admin","Lab"])
is_holiday = st.selectbox("Holiday?", [0,1])
previous_usage = st.number_input("Previous Usage (kWh)", 0.0, 100.0, 30.0)

if st.button("Predict"):
    x = [[hour, day, temperature, humidity, occupancy, building_type, is_holiday, previous_usage]]
    prediction = model.predict(x)[0]
    st.success(f"Predicted Energy Usage: {prediction:.2f} kWh")
