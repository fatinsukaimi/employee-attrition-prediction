import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models and preprocessor
nn_model = load_model("nn_model.keras")
hybrid_model = joblib.load("hybrid_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Title
st.title("Employee Attrition Prediction")

# Reset functionality
if "reset" not in st.session_state:
    st.session_state.reset = False

if st.sidebar.button("Reset Inputs"):
    st.session_state.reset = True
else:
    st.session_state.reset = False

# Helper function to clean and convert numeric inputs
def clean_and_convert_input(input_value):
    try:
        if not input_value:  # If input is empty or None
            return np.nan  # Return NaN to handle missing values gracefully
        cleaned_value = input_value.replace(',', '').replace(' ', '')
        return float(cleaned_value)
    except ValueError:
        st.error(f"Invalid input: {input_value}. Please enter a valid number.")
        return np.nan

# Inputs
age = st.sidebar.slider("Age", 18, 65, 18 if st.session_state.reset else 30)
monthly_income_input = st.sidebar.text_input("Monthly Income (e.g., 5000)", value="" if st.session_state.reset else "5000")
monthly_income = clean_and_convert_input(monthly_income_input)
monthly_rate_input = st.sidebar.text_input("Monthly Rate (e.g., 15000)", value="" if st.session_state.reset else "15000")
monthly_rate = clean_and_convert_input(monthly_rate_input)
overtime = st.sidebar.selectbox("OverTime (Yes/No)", ["Yes", "No"], index=0 if st.session_state.reset else ["Yes", "No"].index("Yes"))
environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 1 if st.session_state.reset else 3)
relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction (1-4)", 1, 4, 1 if st.session_state.reset else 3)
percent_salary_hike = st.sidebar.slider("Percent Salary Hike (%)", 0, 50, 0 if st.session_state.reset else 10)
years_with_curr_manager = st.sidebar.slider("Years with Current Manager", 0, 20, 0 if st.session_state.reset else 5)

# Prepare input data
input_data = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [monthly_income],
    "MonthlyRate": [monthly_rate],
    "OverTime": [1 if overtime == "Yes" else 0],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "RelationshipSatisfaction": [relationship_satisfaction],
    "PercentSalaryHike": [percent_salary_hike],
    "YearsWithCurrManager": [years_with_curr_manager],
})

# Debugging: Display input data
st.write("### Debugging Input Data")
st.write(input_data)

# Process and Predict Button
if st.button("Predict"):
    try:
        # Ensure all inputs are valid
        if input_data.isnull().values.any():
            st.error("Some inputs are missing or invalid. Please fill in all fields correctly.")
        else:
            # Ensure numeric and categorical types
            numeric_columns = preprocessor.transformers[0][2]
            input_data[numeric_columns] = input_data[numeric_columns].astype('float64')

            categorical_columns = preprocessor.transformers[1][2]
            input_data[categorical_columns] = input_data[categorical_columns].astype(str)

            # Preprocess
            input_array = preprocessor.transform(input_data)

            # Predict using Neural Network
            nn_predictions = nn_model.predict(input_array).flatten()

            # Create hybrid features
            input_hybrid = np.column_stack((input_array, nn_predictions))

            # Predict using Hybrid NN-XGBoost
            hybrid_predictions = hybrid_model.predict(input_hybrid)

            # Display predictions
            st.subheader("Prediction Results")
            prediction = "Yes" if hybrid_predictions[0] == 1 else "No"
            st.write(f"Will the employee leave? **{prediction}**")

    except Exception as e:
        st.error(f"Error during processing: {e}")
