import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models and preprocessor
nn_model = load_model("nn_model.keras")
hybrid_model = joblib.load("hybrid_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Title and Subtitle
st.title("Employee Attrition Prediction")
st.markdown("This application predicts employee attrition using a hybrid Neural Network and XGBoost model.")

# Sidebar Inputs
st.sidebar.header("Employee Features")

# Add Reset Prediction button at the top
if "reset_prediction" not in st.session_state:
    st.session_state.reset_prediction = False

if st.sidebar.button("Reset Prediction"):
    st.session_state.reset_prediction = True
else:
    st.session_state.reset_prediction = False

# Inputs in the sidebar
overtime = st.sidebar.selectbox("OverTime (Yes=1, No=0)", ["Yes", "No"])
environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)
relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
monthly_income = st.sidebar.text_input("Monthly Income (e.g., 5000)", value="5000")
years_with_curr_manager = st.sidebar.slider("Years With Current Manager", 0, 20, 5)

# Default values for other features
default_values = {
    "Age": 30,
    "DailyRate": 800,
    "DistanceFromHome": 10,
    "Education": 3,
    "HourlyRate": 50,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobSatisfaction": 3,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 2,
    "PercentSalaryHike": 10,
    "PerformanceRating": 3,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "BusinessTravel": "Travel_Rarely",
    "Department": "Research & Development",
    "EducationField": "Life Sciences",
    "Gender": "Male",
    "JobRole": "Research Scientist",
    "MaritalStatus": "Single",
}

# Combine user inputs and default values
input_data = pd.DataFrame({
    "OverTime": [1 if overtime == "Yes" else 0],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "RelationshipSatisfaction": [relationship_satisfaction],
    "MonthlyIncome": [float(monthly_income)],
    "YearsWithCurrManager": [years_with_curr_manager],
    **{key: [value] for key, value in default_values.items()}
})

# Ensure correct data types
numerical_columns = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
    "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]
categorical_columns = [
    "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"
]

# Convert to appropriate types
input_data[numerical_columns] = input_data[numerical_columns].astype('float64')
input_data[categorical_columns] = input_data[categorical_columns].astype(str)

# Prediction Button
if st.button("Predict"):
    st.session_state.reset_prediction = False
    try:
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
        if not st.session_state.reset_prediction:
            prediction = "Yes" if hybrid_predictions[0] == 1 else "No"
            st.write(f"Will the employee leave? **{prediction}**")

    except Exception as e:
        st.error(f"Error during processing: {e}")
