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

# Reset function
def reset_fields():
    st.session_state.clear()
    st.session_state["age"] = 18
    st.session_state["monthly_income"] = ""
    st.session_state["monthly_rate"] = ""
    st.session_state["overtime"] = "Yes"
    st.session_state["environment_satisfaction"] = 1
    st.session_state["relationship_satisfaction"] = 1
    st.session_state["percent_salary_hike"] = 0
    st.session_state["years_with_curr_manager"] = 0
    st.session_state["job_involvement"] = 1
    st.session_state["years_at_company"] = 0
    st.session_state["job_satisfaction"] = 1
    st.session_state["marital_status"] = "Single"
    st.session_state["stock_option_level"] = 0
    st.session_state["hourly_rate"] = 10
    st.session_state["daily_rate"] = 100
    st.session_state["performance_rating"] = 1
    st.session_state["years_in_current_role"] = 0
    st.session_state["training_times_last_year"] = 0
    st.session_state["business_travel"] = "Travel_Rarely"
    st.session_state["distance_from_home"] = 0
    st.session_state["education_field"] = "Life Sciences"
    st.session_state["years_since_last_promotion"] = 0
    st.session_state["total_working_years"] = 0
    st.session_state["num_companies_worked"] = 0
    st.session_state["job_role"] = "Sales Executive"
    st.session_state["job_level"] = 1
    st.session_state["work_life_balance"] = 1
    st.session_state["gender"] = "Male"
    st.session_state["department"] = "Sales"
    st.session_state["education"] = 1

# Sidebar header and reset button
st.sidebar.header("Employee Features")
if st.sidebar.button("Reset Inputs"):
    reset_fields()

# Inputs
age = st.sidebar.slider("Age", 18, 65, st.session_state.get("age", 18), key="age")
monthly_income = st.sidebar.text_input(
    "Monthly Income (e.g., 5000)", value=st.session_state.get("monthly_income", ""), key="monthly_income"
)
monthly_rate = st.sidebar.text_input(
    "Monthly Rate (e.g., 15000)", value=st.session_state.get("monthly_rate", ""), key="monthly_rate"
)
overtime = st.sidebar.selectbox(
    "OverTime (Yes/No)", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.get("overtime", "Yes")), key="overtime"
)
environment_satisfaction = st.sidebar.slider(
    "Environment Satisfaction (1-4)", 1, 4, st.session_state.get("environment_satisfaction", 1), key="environment_satisfaction"
)
relationship_satisfaction = st.sidebar.slider(
    "Relationship Satisfaction (1-4)", 1, 4, st.session_state.get("relationship_satisfaction", 1), key="relationship_satisfaction"
)
percent_salary_hike = st.sidebar.slider(
    "Percent Salary Hike (%)", 0, 50, st.session_state.get("percent_salary_hike", 0), key="percent_salary_hike"
)
years_with_curr_manager = st.sidebar.slider(
    "Years with Current Manager", 0, 20, st.session_state.get("years_with_curr_manager", 0), key="years_with_curr_manager"
)
job_involvement = st.sidebar.slider(
    "Job Involvement (1-4)", 1, 4, st.session_state.get("job_involvement", 1), key="job_involvement"
)
years_at_company = st.sidebar.slider(
    "Years at Company", 0, 40, st.session_state.get("years_at_company", 0), key="years_at_company"
)
job_satisfaction = st.sidebar.slider(
    "Job Satisfaction (1-4)", 1, 4, st.session_state.get("job_satisfaction", 1), key="job_satisfaction"
)
marital_status = st.sidebar.selectbox(
    "Marital Status", ["Single", "Married", "Divorced"], key="marital_status", index=["Single", "Married", "Divorced"].index(st.session_state.get("marital_status", "Single"))
)
stock_option_level = st.sidebar.slider(
    "Stock Option Level (0-3)", 0, 3, st.session_state.get("stock_option_level", 0), key="stock_option_level"
)
hourly_rate = st.sidebar.number_input(
    "Hourly Rate (e.g., 40)", min_value=10, max_value=100, value=st.session_state.get("hourly_rate", 10), key="hourly_rate"
)
daily_rate = st.sidebar.number_input(
    "Daily Rate (e.g., 800)", min_value=100, max_value=2000, value=st.session_state.get("daily_rate", 100), key="daily_rate"
)
performance_rating = st.sidebar.slider(
    "Performance Rating (1-4)", 1, 4, st.session_state.get("performance_rating", 1), key="performance_rating"
)
years_in_current_role = st.sidebar.slider(
    "Years in Current Role", 0, 20, st.session_state.get("years_in_current_role", 0), key="years_in_current_role"
)
training_times_last_year = st.sidebar.slider(
    "Training Times Last Year", 0, 10, st.session_state.get("training_times_last_year", 0), key="training_times_last_year"
)
business_travel = st.sidebar.selectbox(
    "Business Travel",
    ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
    index=["Travel_Rarely", "Travel_Frequently", "Non-Travel"].index(st.session_state.get("business_travel", "Travel_Rarely")),
    key="business_travel",
)
distance_from_home = st.sidebar.number_input(
    "Distance from Home (e.g., 10)", min_value=0, max_value=50, value=st.session_state.get("distance_from_home", 0), key="distance_from_home"
)
education_field = st.sidebar.selectbox(
    "Education Field",
    ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"],
    index=["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"].index(st.session_state.get("education_field", "Life Sciences")),
    key="education_field",
)
years_since_last_promotion = st.sidebar.slider(
    "Years Since Last Promotion", 0, 20, st.session_state.get("years_since_last_promotion", 0), key="years_since_last_promotion"
)
total_working_years = st.sidebar.slider(
    "Total Working Years", 0, 40, st.session_state.get("total_working_years", 0), key="total_working_years"
)
num_companies_worked = st.sidebar.slider(
    "Number of Companies Worked", 0, 20, st.session_state.get("num_companies_worked", 0), key="num_companies_worked"
)

# Prediction Button
if st.button("Predict"):
    try:
        # Collect all inputs into a DataFrame
        input_data = pd.DataFrame({
            "Age": [age],
            "MonthlyIncome": [monthly_income],
            "MonthlyRate": [monthly_rate],
            "OverTime": [1 if overtime == "Yes" else 0],
            # Add all other features here...
        })

        # Ensure correct types
        input_data = input_data.astype("float64")

        # Preprocess and predict
        processed_input = preprocessor.transform(input_data)
        nn_prediction = nn_model.predict(processed_input).flatten()
        hybrid_input = np.column_stack((processed_input, nn_prediction))
        hybrid_prediction = hybrid_model.predict(hybrid_input)

        # Display the prediction
        st.write(f"Prediction: {'Yes' if hybrid_prediction[0] == 1 else 'No'}")
    except Exception as e:
        st.error(f"Error: {e}")
