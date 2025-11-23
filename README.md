**Campus Energy Optimizer**

A Streamlit-based machine learning application that predicts hourly energy consumption (kWh) for different types of campus buildings.
The project demonstrates end-to-end ML deployment — from dataset preparation and model training to UI development and cloud hosting.

**Overview**

Campus Energy Optimizer allows users to input environmental and operational parameters such as:

1) Hour of the day

2) Day of the week

3) Temperature (°C)

4) Humidity (%)

5) Occupancy

6) Building type

7) Holiday indicator

8) Previous energy usage (kWh)

Using these features, the model estimates the expected energy consumption for that hour.
The app also includes **visual analytics, a downloadable report, and hourly average trends per building type**.

**Features**

1) Machine learning model using RandomForestRegressor

2) Automatic preprocessing with sklearn pipelines

3) Compressed .joblib model for fast loading

4) Clean and intuitive UI built with Streamlit

5) Comparison chart (predicted vs previous usage)

6) Hourly average energy usage visualization

7) CSV report download

8) Sidebar with model details and guidance

**Technology Stack**

1) Python

2) Streamlit

3) Scikit-learn

4) Pandas

5) Altair

6) Joblib

**Model Details**

The model uses the following input features:

1) hour

2) day_of_week

3) temperature

4) humidity

5) occupancy

6) building_type

7) is_holiday

8) previous_usage

Algorithm: RandomForestRegressor
Metrics: approx. 0.78 R², ~10.5 MSE
Preprocessing: One-hot encoding + passthrough numerical features
Saved using joblib with compression level 3

**Project Structure**
campus-energy-optimizer
│
├── app.py
├── energy_model.joblib
├── avg_hour_building.csv
├── data/
│     └── campus_energy_dataset.csv
├── assets/
│     └── header.png
└── requirements.txt

**Running the App Locally**

Install requirements:

pip install -r requirements.txt


Run Streamlit:

streamlit run app.py

Retraining the Model (Optional)

If needed, you can retrain and export the RandomForest model:

joblib.dump(model, "energy_model.joblib", compress=3)

Future Enhancements

Additional ML models

More advanced recommendations

Multi-building consumption dashboard

Dataset upload support

Author

**Madhumita Ash
B.Tech IT (2023–2027), GTBIT
Interests: Data Science, Cybersecurity, ML deployment**
