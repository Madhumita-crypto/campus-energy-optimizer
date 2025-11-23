# app.py (UPGRADED)
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Campus Energy Optimizer", layout="centered")

# ---------------------------
# Config - update image path if you move it into your repo
# ---------------------------
APP_IMAGE_PATH = "/mnt/data/e8bb02da-2d3f-47d8-b29e-3eee6c4dd333.png"
# When deploying: put header image in your repo (e.g. assets/header.png)
# and change APP_IMAGE_PATH = "assets/header.png"

# ---------------------------
# Helper: load model (cached)
# ---------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "energy_model.joblib")
model = load_model(MODEL_PATH)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    if os.path.exists(APP_IMAGE_PATH):
        st.image(APP_IMAGE_PATH, use_column_width=True)
    st.title("Campus Energy Optimizer")
    st.markdown(
        """
        **What this does:**  
        Predicts hourly energy usage (kWh) for a campus building using weather, occupancy and previous usage.

        **How to use:**  
        1. Set the time & building details.  
        2. Set environmental & usage inputs.  
        3. Click **Predict**.

        **Model:** RandomForest trained on a campus-simulated dataset.
        """
    )
    st.markdown("---")
    st.subheader("Model details")
    st.write("Algorithm: RandomForestRegressor")
    st.write("Features: hour, day_of_week, temperature, humidity, occupancy, building_type, is_holiday, previous_usage")
    st.markdown("---")
    st.caption("Built by Madhumita • Demo project")

# ---------------------------
# Main header + description
# ---------------------------
st.markdown("""
    <h1 style="text-align: center; margin-bottom: -8px;">🔋 Campus Energy Optimizer</h1>
    <p style="text-align: center; color:#bfc5c9;">
        Estimate energy consumption and get quick efficiency tips for campus buildings.
    </p>
""", unsafe_allow_html=True)

st.markdown("")  # spacing

# ---------------------------
# Inputs grouped visually
# ---------------------------
st.subheader("📅 Timing & Schedule")
col1, col2 = st.columns([2, 1])
with col1:
    hour = st.slider("Hour", 0, 23, 12)
with col2:
    day_of_week = st.select_slider("Day of Week", options=list(range(7)), value=3, format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x] )

holiday = st.selectbox("Holiday?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

st.subheader("🌤 Environment Parameters")
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=28.0, step=0.5)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=55.0, step=0.5)

st.subheader("🏫 Building & Usage Details")
occupancy = st.number_input("Occupancy (people)", min_value=0, max_value=1000, value=50, step=1)
building_type = st.selectbox("Building Type", ["Academic", "Hostel", "Library", "Admin", "Lab"])
previous_usage = st.number_input("Previous Usage (kWh)", min_value=0.0, max_value=1000.0, value=30.0, step=0.5)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------
# Predict button (full width)
# ---------------------------
predict_clicked = st.button("Predict", use_container_width=True)

# ---------------------------
# Prediction logic
# ---------------------------
def make_input_df(hour, day_of_week, temperature, humidity, occupancy, building_type, holiday, previous_usage):
    return pd.DataFrame([{
        "hour": hour,
        "day_of_week": day_of_week,
        "temperature": temperature,
        "humidity": humidity,
        "occupancy": occupancy,
        "building_type": building_type,
        "is_holiday": holiday,
        "previous_usage": previous_usage
    }])

if predict_clicked:
    try:
        input_df = make_input_df(hour, day_of_week, temperature, humidity, occupancy, building_type, holiday, previous_usage)
        prediction = model.predict(input_df)[0]

        # Result card (prettier)
        st.markdown(f"""
            <div style="
                background-color:#0b3b2e;
                padding: 18px;
                border-radius: 10px;
                margin-top: 12px;
            ">
                <h4 style="color:#b6f3d3; margin:0;">Predicted Energy Usage</h4>
                <h2 style="color:white; margin-top:6px;">⚡ {prediction:.2f} kWh</h2>
            </div>
        """, unsafe_allow_html=True)

        # Quick recommendation logic
        tip = ""
        if prediction > 150:
            tip = "⚠️ High usage detected — consider rescheduling heavy equipment, reduce HVAC setpoints, or limit non-essential lighting."
        elif prediction > 90:
            tip = "🔶 Moderate usage — consider minor optimizations: consolidate lab schedules, dim corridors."
        else:
            tip = "✅ Good — energy consumption looks efficient for these inputs."

        st.info(tip)

        # Tiny comparison plot: previous vs predicted
        fig, ax = plt.subplots(figsize=(6,2.2))
        labels = ["Previous (kWh)", "Predicted (kWh)"]
        values = [previous_usage, prediction]
        bars = ax.bar(labels, values)
        ax.set_ylim(0, max(values)*1.4)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(values)*0.03), f"{bar.get_height():.1f}", ha='center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel("kWh")
        st.pyplot(fig)

        # Optional: show raw input and model confidence (R^2-like heuristic)
        with st.expander("Show input details & debug"):
            st.write(input_df.T)
            st.write("Model: RandomForestRegressor (compressed).")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color: #9aa0a6;'>
    Built with ❤️ by Madhumita • Campus Energy Optimizer • Demo project
</p>
""", unsafe_allow_html=True)
