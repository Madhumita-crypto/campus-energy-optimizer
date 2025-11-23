# app.py (CLEANED — NO LightGBM)
import os
import joblib
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Campus Energy Optimizer", layout="centered")

# ---------------- CONFIG (update when you move files into repo) ----------------
# Path to header image you uploaded during testing. When deploying, move the image to your repo (e.g. assets/header.png)
APP_IMAGE_PATH = "/mnt/data/e8bb02da-2d3f-47d8-b29e-3eee6c4dd333.png"

# If you include your dataset or a precomputed averages CSV in the repo, update paths accordingly:
DATASET_PATH = "data/campus_energy_dataset.csv"      # optional (repo)
AVG_CSV_PATH = "avg_hour_building.csv"               # optional (repo)

# ---------------- CSS (subtle) ----------------
st.markdown(
    """
    <style>
    /* small page padding */
    .css-1aumxhk {padding-top: 1rem;}
    /* rounded inputs */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 10px;
    }
    /* button style */
    .stButton>button {background-color:#0b7285; color: #fff; border-radius:8px; padding: 8px 12px;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Helper: load model (cached) ----------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "energy_model.joblib")
if not os.path.exists(MODEL_PATH):
    st.error("Model file energy_model.joblib not found in the repo root. Please upload it and refresh.")
    st.stop()

model = load_model(MODEL_PATH)

# ---------------- Sidebar ----------------
with st.sidebar:
    if os.path.exists(APP_IMAGE_PATH):
        st.image(APP_IMAGE_PATH, use_column_width=True)
    st.title("Campus Energy Optimizer")
    st.markdown(
        """
        Predict hourly energy usage (kWh) for campus buildings using environmental and operational inputs.

        **How to use**
        1. Fill in the inputs.
        2. Click **Predict**.
        3. Download the report if you want to save the result.
        """
    )
    st.markdown("---")
    st.subheader("Model info")
    st.write("Algorithm: RandomForestRegressor")
    st.write("Features: hour, day_of_week, temperature, humidity, occupancy, building_type, is_holiday, previous_usage")
    st.markdown("---")
    st.caption("Built by Madhumita • Demo project")

# ---------------- Header ----------------
st.markdown("""
    <h1 style="text-align:center; margin-bottom:2px;">🔋 Campus Energy Optimizer</h1>
    <p style="text-align:center; color:#bfc5c9; margin-top:0;">Estimate energy consumption & receive quick efficiency tips.</p>
""", unsafe_allow_html=True)

st.write("")  # spacer

# ---------------- Inputs (grouped) ----------------
st.subheader("📅 Timing & Schedule")
col1, col2 = st.columns([2, 1])
with col1:
    hour = st.slider("Hour", 0, 23, 12)
with col2:
    day_of_week = st.select_slider("Day of Week", options=list(range(7)), value=3, format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
holiday = st.selectbox("Holiday?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

st.subheader("🌤 Environment Parameters")
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=28.0, step=0.5)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=55.0, step=0.5)

st.subheader("🏫 Building & Usage Details")
occupancy = st.number_input("Occupancy (people)", min_value=0, max_value=2000, value=50, step=1)
building_type = st.selectbox("Building Type", ["Academic", "Hostel", "Library", "Admin", "Lab"])
previous_usage = st.number_input("Previous Usage (kWh)", min_value=0.0, max_value=5000.0, value=30.0, step=0.5)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- Predict button ----------------
predict_clicked = st.button("Predict", use_container_width=True)

# ---------------- Helper to build input dataframe ----------------
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

# ---------------- Prediction & UI ----------------
if predict_clicked:
    input_df = build_input_df(hour, day_of_week, temperature, humidity, occupancy, building_type, holiday, previous_usage)
    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        st.exception(f"Prediction error: {e}")
        st.stop()

    # pretty result card
    st.markdown(f"""
        <div style="background:#073b32; padding:16px; border-radius:10px;">
            <h4 style="color:#b6f3d3; margin:0;">Predicted Energy Usage</h4>
            <h2 style="color:white; margin-top:6px;">⚡ {prediction:.2f} kWh</h2>
        </div>
    """, unsafe_allow_html=True)

    # recommendation text
    if prediction > 150:
        st.warning("⚠️ High usage detected — consider rescheduling heavy equipment, lowering HVAC setpoints, or limiting non-essential lighting.")
    elif prediction > 90:
        st.info("🔶 Moderate usage — consider small optimizations such as consolidating lab schedules or dimming common-area lights.")
    else:
        st.success("✅ Low usage — looks efficient for these inputs.")

    # small bar chart (previous vs predicted) using altair
    comp_df = pd.DataFrame({
        "metric": ["previous_usage", "predicted"],
        "kwh": [previous_usage, prediction]
    })
    chart = alt.Chart(comp_df).mark_bar().encode(
        x=alt.X('metric:N', title='Metric'),
        y=alt.Y('kwh:Q', title='kWh'),
        color=alt.condition(alt.datum.metric == 'predicted', alt.value('#FF7F50'), alt.value('#7A9B76'))
    ).properties(height=240)
    st.altair_chart(chart, use_container_width=True)

    # show inputs + model info in expander
    with st.expander("View inputs & model"):
        st.write(input_df.T)
        st.write("Model: RandomForestRegressor (compressed)")

    # download report (CSV)
    report_df = input_df.copy()
    report_df["predicted_kwh"] = prediction
    csv_str = report_df.to_csv(index=False)
    st.download_button(
        label="📥 Download report (CSV)",
        data=csv_str,
        file_name="energy_prediction_report.csv",
        mime="text/csv"
    )

# ---------------- Average-energy-by-hour chart ----------------
st.markdown("---")
st.subheader("📈 Average energy usage by hour (per building type)")

avg_df = None
# 1) check for precomputed avg CSV in repo
if os.path.exists(AVG_CSV_PATH):
    try:
        avg_df = pd.read_csv(AVG_CSV_PATH)
        st.caption("Using precomputed averages (avg_hour_building.csv).")
    except Exception:
        avg_df = None

# 2) else, check if dataset exists in repo and compute
if avg_df is None and os.path.exists(DATASET_PATH):
    try:
        raw = pd.read_csv(DATASET_PATH)
        avg_df = raw.groupby(["building_type", "hour"])["energy_usage"].mean().reset_index().rename(columns={"energy_usage": "avg_kwh"})
        st.caption("Computed averages from dataset.")
    except Exception:
        avg_df = None

if avg_df is None:
    st.info("No dataset found to compute averages. To enable the chart, add 'avg_hour_building.csv' or 'data/campus_energy_dataset.csv' to the repo.")
else:
    btype = st.selectbox("Choose building type for the chart", sorted(avg_df["building_type"].unique()))
    chart_df = avg_df[avg_df["building_type"] == btype].sort_values("hour")
    line = alt.Chart(chart_df).mark_line(point=True).encode(
        x=alt.X('hour:Q', title='Hour of day'),
        y=alt.Y('avg_kwh:Q', title='Avg kWh'),
        tooltip=['hour', 'avg_kwh']
    ).properties(height=300)
    st.altair_chart(line, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color: #9aa0a6;'>
    Built with ❤️ by Madhumita • Campus Energy Optimizer • Demo project
</p>
""", unsafe_allow_html=True)
