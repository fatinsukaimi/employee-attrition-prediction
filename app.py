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

# Add Reset Prediction button at the top
if "reset_prediction" not in st.session_state:
    st.session_state.reset_prediction = False

if st.sidebar.button("Reset Prediction"):
    st.session_state.reset_prediction = True
else:
    st.session_state.reset_prediction = False

# Helper function to clean and convert numeric inputs
def clean_and_convert_input(input_value):
    try:
        cleaned_value = input_value.replace(',', '').replace(' ', '')
        return float(cleaned_value)
    except ValueError:
        st.error(f"Invalid input: {input_value}. Please enter a valid number.")
        return None

# Inputs
age = st.sidebar.slider("Age", 18, 65, 30)
monthly_income_input = st.sidebar.text_input("Monthly Income (e.g., 5000)", value="5000")
monthly_income = clean_and_convert_input(monthly_income_input)
monthly_rate_input = st.sidebar.text_input("Monthly Rate (e.g., 15000)", value="15000")
monthly_rate = clean_and_convert_input(monthly_rate_input)
overtime = st.sidebar.selectbox("OverTime (Yes/No)", ["Yes", "No"])
environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)
relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
percent_salary_hike = st.sidebar.slider("Percent Salary Hike (%)", 0, 50, 10)
years_with_curr_manager = st.sidebar.slider("Years with Current Manager", 0, 20, 5)
job_involvement = st.sidebar.slider("Job Involvement (1-4)", 1, 4, 3)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
stock_option_level = st.sidebar.slider("Stock Option Level (0-3)", 0, 3, 0)
hourly_rate = st.sidebar.number_input("Hourly Rate (e.g., 40)", min_value=10, max_value=100, value=40)
daily_rate = st.sidebar.number_input("Daily Rate (e.g., 800)", min_value=100, max_value=2000, value=800)
performance_rating = st.sidebar.slider("Performance Rating (1-4)", 1, 4, 3)
years_in_current_role = st.sidebar.slider("Years in Current Role", 0, 20, 5)
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

# Encode categorical variables
marital_status_mapping = {"Single": 0, "Married": 1, "Divorced": 2}
business_travel_mapping = {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
gender_mapping = {"Male": 0, "Female": 1}
department_mapping = {"Sales": 0, "Research & Development": 1, "Human Resources": 2}
education_field_mapping = {"Life Sciences": 0, "Medical": 1, "Marketing": 2, "Technical Degree": 3, "Other": 4}
job_role_mapping = {"Sales Executive": 0, "Manager": 1, "Research Scientist": 2, "Laboratory Technician": 3, "Other": 4}

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
    "JobInvolvement": [job_involvement],
    "YearsAtCompany": [years_at_company],
    "JobSatisfaction": [job_satisfaction],
    "MaritalStatus": [marital_status_mapping[marital_status]],
    "StockOptionLevel": [stock_option_level],
    "HourlyRate": [hourly_rate],
    "DailyRate": [daily_rate],
    "PerformanceRating": [performance_rating],
    "YearsInCurrentRole": [years_in_current_role],
    "TrainingTimesLastYear": [training_times_last_year],
    "BusinessTravel": [business_travel_mapping[business_travel]],
    "DistanceFromHome": [distance_from_home],
    "EducationField": [education_field_mapping[education_field]],
    "YearsSinceLastPromotion": [years_since_last_promotion],
    "TotalWorkingYears": [total_working_years],
    "NumCompaniesWorked": [num_companies_worked],
    "JobRole": [job_role_mapping[job_role]],
    "JobLevel": [job_level],
    "WorkLifeBalance": [work_life_balance],
    "Gender": [gender_mapping[gender]],
    "Department": [department_mapping[department]],
    "Education": [education],
})

# Process and Predict Button
if st.button("Predict"):
    st.session_state.reset_prediction = False
    try:
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
        if not st.session_state.reset_prediction:
            prediction = "Yes" if hybrid_predictions[0] == 1 else "No"
            st.write(f"Will the employee leave? **{prediction}**")

    except Exception as e:
        st.error(f"Error during processing: {e}")
