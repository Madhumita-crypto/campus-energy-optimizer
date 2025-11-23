# app.py (FULL FEATURED: UI polish + download report + avg-hour charts + RF/LGBM toggle + subtle CSS)
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from io import StringIO

st.set_page_config(page_title="Campus Energy Optimizer", layout="centered")

# ---------------- CONFIG ----------------
# If testing in Colab before moving to GitHub/Streamlit, these local paths exist:
APP_IMAGE_PATH = "/mnt/data/e8bb02da-2d3f-47d8-b29e-3eee6c4dd333.png"  # move to repo assets on deploy: "assets/header.png"
DATASET_PATH = "/mnt/data/campus_energy_dataset.csv"                   # move to repo if you want auto charts: "data/campus_energy_dataset.csv"

# When deploying to Streamlit Cloud: move the image and dataset files to your repo (assets/, data/) and update these paths accordingly.

# ---------------- CSS THEME (subtle) ----------------
st.markdown(
    """
    <style>
    /* page */
    .css-1aumxhk {padding-top: 1rem;}  /* minor top padding */
    .stButton>button {background-color:#0b7285; color: #fff; border-radius:8px;}
    /* input boxes */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 10px;
    }
    /* result card font */
    .result-card h2 {margin:0; padding:0;}
    /* small */
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Helper: load models safely ----------------
@st.cache_resource
def try_load(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

BASE_DIR = os.path.dirname(__file__)
RF_PATH = os.path.join(BASE_DIR, "energy_model.joblib")
LGBM_PATH = os.path.join(BASE_DIR, "energy_model_lgbm.joblib")  # optional

model_rf = try_load(RF_PATH)
model_lgbm = try_load(LGBM_PATH)

if model_rf is None:
    st.error("Primary model (energy_model.joblib) not found. Please upload it to the repo root.")
    st.stop()

# ---------------- Sidebar ----------------
with st.sidebar:
    if os.path.exists(APP_IMAGE_PATH):
        st.image(APP_IMAGE_PATH, use_column_width=True)
    st.header("Campus Energy Optimizer")
    st.write("Predict hourly energy usage (kWh) for campus buildings. Slide inputs, press Predict.")
    st.markdown("---")
    st.subheader("Model selection")
    model_choice = st.radio("Choose model", ("RandomForest (default)", "LightGBM (if available)"))
    if model_choice == "LightGBM (if available)" and model_lgbm is None:
        st.warning("LightGBM model file not found; RandomForest will be used instead.")
    st.markdown("---")
    st.subheader("Quick tips")
    st.write("- Use realistic values for occupancy & previous usage.")
    st.write("- Toggle model to compare if you add an LGBM model file.")
    st.markdown("---")
    st.caption("Built by Madhumita • Demo project")

# ---------------- Main header ----------------
st.markdown("""
    <h1 style="text-align:center; margin-bottom:2px;">🔋 Campus Energy Optimizer</h1>
    <p style="text-align:center; color:#bfc5c9; margin-top:0;">Estimate kWh for campus buildings & receive quick efficiency tips.</p>
""", unsafe_allow_html=True)
st.write("")  # spacer

# ---------------- Input groups ----------------
st.subheader("📅 Timing & Schedule")
col1, col2 = st.columns([2,1])
with col1:
    hour = st.slider("Hour", 0, 23, 12)
with col2:
    day_of_week = st.select_slider("Day of Week", options=list(range(7)), value=3, format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
holiday = st.selectbox("Holiday?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

st.subheader("🌤 Environment Parameters")
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=28.0, step=0.5)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=55.0, step=0.5)

st.subheader("🏫 Building & Usage Details")
occupancy = st.number_input("Occupancy (people)", min_value=0, max_value=2000, value=50, step=1)
building_type = st.selectbox("Building Type", ["Academic", "Hostel", "Library", "Admin", "Lab"])
previous_usage = st.number_input("Previous Usage (kWh)", min_value=0.0, max_value=5000.0, value=30.0, step=0.5)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("Predict", use_container_width=True)

# ---------------- Utility to build input dataframe ----------------
def build_input_df(hour, day_of_week, temperature, humidity, occupancy, building_type, holiday, previous_usage):
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

# ---------------- Prediction + Response ----------------
if predict_btn:
    input_df = build_input_df(hour, day_of_week, temperature, humidity, occupancy, building_type, holiday, previous_usage)

    # pick model
    use_model = model_rf
    if model_choice == "LightGBM (if available)" and model_lgbm is not None:
        use_model = model_lgbm

    try:
        prediction = use_model.predict(input_df)[0]
    except Exception as e:
        st.exception(f"Prediction error: {e}")
        st.stop()

    # Result card
    st.markdown(f"""
        <div style="background:#073b32; padding:16px; border-radius:10px;">
            <h4 style="color:#b6f3d3; margin:0;">Predicted Energy Usage</h4>
            <h2 style="color:white; margin-top:6px;">⚡ {prediction:.2f} kWh</h2>
        </div>
    """, unsafe_allow_html=True)

    # Recommendations
    if prediction > 150:
        st.warning("⚠️ High usage: consider rescheduling heavy equipment, optimizing HVAC, or staggering lab use.")
    elif prediction > 90:
        st.info("🔶 Moderate usage: consider small optimizations (lighting, consolidation).")
    else:
        st.success("✅ Low usage — looks efficient for these inputs.")

    # Small visualization (altair bar chart)
    comp_df = pd.DataFrame({
        "metric": ["previous_usage", "predicted"],
        "kwh": [previous_usage, prediction]
    })
    chart = alt.Chart(comp_df).mark_bar().encode(
        x=alt.X('metric:N', title='Metric'),
        y=alt.Y('kwh:Q', title='kWh'),
        color=alt.condition(alt.datum.metric == 'predicted', alt.value('#FF7F50'), alt.value('#7A9B76'))
    ).properties(height=240, width=400)
    st.altair_chart(chart, use_container_width=True)

    # Expandable debug / inputs
    with st.expander("View input & model info"):
        st.write(input_df.T)
        st.write(f"Using model: {'LightGBM' if use_model is model_lgbm else 'RandomForest'}")

    # -------- Download report (CSV) --------
    report_df = input_df.copy()
    report_df["predicted_kwh"] = prediction
    csv_str = report_df.to_csv(index=False)
    st.download_button(
        label="📥 Download report (CSV)",
        data=csv_str,
        file_name="energy_prediction_report.csv",
        mime="text/csv"
    )

# ---------------- Average-energy-by-hour chart (from dataset or precomputed CSV) ----------------
st.markdown("---")
st.subheader("📈 Average energy usage by hour (per building type)")
if os.path.exists(os.path.join(BASE_DIR, "avg_hour_building.csv")):
    avg_df = pd.read_csv(os.path.join(BASE_DIR, "avg_hour_building.csv"))
    st.caption("Using precomputed averages (avg_hour_building.csv).")
elif os.path.exists(DATASET_PATH):
    raw = pd.read_csv(DATASET_PATH)
    avg_df = raw.groupby(["building_type","hour"])["energy_usage"].mean().reset_index().rename(columns={"energy_usage":"avg_kwh"})
    st.caption("Computed averages from dataset.")
else:
    avg_df = None
    st.info("No dataset found to compute averages. To enable this chart, add campus_energy_dataset.csv to the repo or upload avg_hour_building.csv.")
    
if avg_df is not None:
    # simple selector
    btype = st.selectbox("Choose building type for the chart", sorted(avg_df["building_type"].unique()))
    chart_df = avg_df[avg_df["building_type"] == btype].sort_values("hour")
    line = alt.Chart(chart_df).mark_line(point=True).encode(
        x=alt.X('hour:Q', title='Hour of day'),
        y=alt.Y('avg_kwh:Q', title='Avg kWh'),
        tooltip=['hour','avg_kwh']
    ).properties(height=300, width=700)
    st.altair_chart(line, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color: #9aa0a6;'>
    Built with ❤️ by Madhumita • Campus Energy Optimizer • Demo project
</p>
""", unsafe_allow_html=True)
