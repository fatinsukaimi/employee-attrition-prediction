import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib
from tensorflow.keras.models import load_model

# Load models
xgb_model = joblib.load("hybrid_nn_xgboost_model.pkl")
nn_model = load_model("nn_model.h5")

# Preprocessor (adjust according to your project)
numerical_cols = ['Age', 'DistanceFromHome', 'MonthlyIncome']
categorical_cols = ['JobRole', 'BusinessTravel']
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

# Streamlit App
st.title("Employee Attrition Prediction")

# Sidebar Inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.number_input("Age", min_value=18, max_value=70, value=30)
distance = st.sidebar.number_input("Distance from Home", min_value=1, max_value=30, value=10)
monthly_income = st.sidebar.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
job_role = st.sidebar.selectbox("Job Role", ["Sales", "Research & Development", "HR"])
business_travel = st.sidebar.selectbox("Business Travel", ["Rarely", "Frequently", "None"])

# Create DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'DistanceFromHome': [distance],
    'MonthlyIncome': [monthly_income],
    'JobRole': [job_role],
    'BusinessTravel': [business_travel]
})

# Preprocess Input
X_preprocessed = preprocessor.transform(input_data)

# Make Predictions
nn_prediction = nn_model.predict(X_preprocessed)
hybrid_input = np.column_stack((X_preprocessed, nn_prediction.flatten()))
final_prediction = xgb_model.predict(hybrid_input)

# Display Results
if final_prediction[0] == 1:
    st.error("This employee is likely to leave the company.")
else:
    st.success("This employee is likely to stay.")
