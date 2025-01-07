import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models
xgb_model = joblib.load("hybrid_model.pkl")  # Hybrid NN-XGBoost
nn_model = load_model("nn_model.keras")  # Neural Network

# Title
st.title("Employee Attrition Prediction")

# Sidebar Inputs
st.sidebar.header("Employee Features")
age = st.sidebar.slider("Age", 18, 65, 30)
monthly_income = st.sidebar.number_input("Monthly Income (e.g., 5000)", min_value=1000, max_value=20000, value=5000)
overtime = st.sidebar.selectbox("OverTime (Yes/No)", ["Yes", "No"])
environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)
relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
percent_salary_hike = st.sidebar.slider("Percent Salary Hike (%)", 0, 50, 10)
years_with_curr_manager = st.sidebar.slider("Years with Current Manager", 0, 20, 5)
job_involvement = st.sidebar.slider("Job Involvement (1-4)", 1, 4, 3)

# Convert input into a DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [monthly_income],
    "OverTime": [1 if overtime == "Yes" else 0],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "RelationshipSatisfaction": [relationship_satisfaction],
    "PercentSalaryHike": [percent_salary_hike],
    "YearsWithCurrManager": [years_with_curr_manager],
    "JobInvolvement": [job_involvement],
})

# Preprocess input (if necessary, you can add preprocessing steps here)
input_array = input_data.values.reshape(1, -1)  # Reshape for single sample

# Predictions
try:
    # Neural Network predictions
    nn_predictions = nn_model.predict(input_array, verbose=0).flatten()

    # Combine NN predictions with input data for Hybrid Model
    hybrid_input = np.column_stack((input_array, nn_predictions))

    # Make final prediction using Hybrid Model
    prediction = xgb_model.predict(hybrid_input)

    # Show results
    st.subheader("Prediction Results")
    if prediction[0] == 1:
        st.success("The employee is likely to leave.")
    else:
        st.success("The employee is likely to stay.")
except Exception as e:
    st.error(f"An error occurred: {e}")
