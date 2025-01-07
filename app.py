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

# Sidebar Inputs
st.sidebar.header("Employee Features")
age = st.sidebar.slider("Age", 18, 65, 30)
monthly_income = st.sidebar.number_input("Monthly Income (e.g., 5000)", min_value=1000, step=500)
overtime = st.sidebar.selectbox("OverTime (Yes/No)", ["Yes", "No"])
environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)
relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
percent_salary_hike = st.sidebar.slider("Percent Salary Hike (%)", 0, 50, 10)
years_with_curr_manager = st.sidebar.slider("Years with Current Manager", 0, 20, 5)
job_involvement = st.sidebar.slider("Job Involvement (1-4)", 1, 4, 3)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)

# Additional inputs for missing columns
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
stock_option_level = st.sidebar.slider("Stock Option Level (0-3)", 0, 3, 0)
hourly_rate = st.sidebar.number_input("Hourly Rate (e.g., 40)", min_value=10, max_value=100, value=40)
daily_rate = st.sidebar.number_input("Daily Rate (e.g., 800)", min_value=100, max_value=2000, value=800)
performance_rating = st.sidebar.slider("Performance Rating (1-4)", 1, 4, 3)
years_in_current_role = st.sidebar.slider("Years in Current Role", 0, 20, 5)
monthly_rate = st.sidebar.number_input("Monthly Rate (e.g., 15000)", min_value=5000, max_value=50000, value=15000)
training_times_last_year = st.sidebar.slider("Training Times Last Year", 0, 10, 3)
business_travel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
distance_from_home = st.sidebar.number_input("Distance from Home (e.g., 10)", min_value=0, max_value=50, value=10)
education_field = st.sidebar.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
years_since_last_promotion = st.sidebar.slider("Years Since Last Promotion", 0, 20, 1)
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10)
num_companies_worked = st.sidebar.slider("Number of Companies Worked", 0, 20, 2)
job_role = st.sidebar.selectbox("Job Role", ["Sales Executive", "Manager", "Research Scientist", "Laboratory Technician", "Other"])
job_level = st.sidebar.slider("Job Level (1-5)", 1, 5, 2)
work_life_balance = st.sidebar.slider("Work-Life Balance (1-4)", 1, 4, 3)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
education = st.sidebar.slider("Education Level (1-5)", 1, 5, 3)

# Convert input into DataFrame
input_data = pd.DataFrame({
    "Age": [int(age)],
    "MonthlyIncome": [float(monthly_income)],
    "OverTime": [1 if overtime == "Yes" else 0],
    "EnvironmentSatisfaction": [int(environment_satisfaction)],
    "RelationshipSatisfaction": [int(relationship_satisfaction)],
    "PercentSalaryHike": [float(percent_salary_hike)],
    "YearsWithCurrManager": [int(years_with_curr_manager)],
    "JobInvolvement": [int(job_involvement)],
    "YearsAtCompany": [int(years_at_company)],
    "JobSatisfaction": [int(job_satisfaction)],
    "MaritalStatus": [marital_status],
    "StockOptionLevel": [int(stock_option_level)],
    "HourlyRate": [float(hourly_rate)],
    "DailyRate": [float(daily_rate)],
    "PerformanceRating": [int(performance_rating)],
    "YearsInCurrentRole": [int(years_in_current_role)],
    "MonthlyRate": [float(monthly_rate)],
    "TrainingTimesLastYear": [int(training_times_last_year)],
    "BusinessTravel": [business_travel],
    "DistanceFromHome": [float(distance_from_home)],
    "EducationField": [education_field],
    "YearsSinceLastPromotion": [int(years_since_last_promotion)],
    "TotalWorkingYears": [int(total_working_years)],
    "NumCompaniesWorked": [int(num_companies_worked)],
    "JobRole": [job_role],
    "JobLevel": [int(job_level)],
    "WorkLifeBalance": [int(work_life_balance)],
    "Gender": [gender],
    "Department": [department],
    "Education": [int(education)],
})

# Debugging: Show input data and its types
st.write("Input Data for Preprocessing:")
st.write(input_data)
st.write("Data Types:")
st.write(input_data.dtypes)

# Preprocess input data
try:
    input_array = preprocessor.transform(input_data)

    # Predict using Neural Network
    nn_predictions = nn_model.predict(input_array).flatten()

    # Create hybrid features
    input_hybrid = np.column_stack((input_array, nn_predictions))

    # Predict using Hybrid NN-XGBoost
    hybrid_predictions = hybrid_model.predict(input_hybrid)

    # Display predictions
    st.subheader("Prediction Results")
    if hybrid_predictions[0] == 1:
        st.write("The employee is likely to leave the company.")
    else:
        st.write("The employee is likely to stay in the company.")

except Exception as e:
    st.error(f"Error during preprocessing: {e}")
