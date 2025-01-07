import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import shap

# Load the preprocessor and models
preprocessor = joblib.load('preprocessor.pkl')
hybrid_model = joblib.load('hybrid_nn_xgboost.pkl')
nn_model = load_model('nn_model.h5')

# Title and Description
st.title("Employee Attrition Prediction")
st.write("Predict the likelihood of an employee leaving the company based on various factors.")

# Input Form
st.sidebar.header("Employee Features")
Age = st.sidebar.slider("Age", 18, 60, 30)
MonthlyIncome = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
OverTime = st.sidebar.selectbox("Works Overtime?", ["Yes", "No"])
EnvironmentSatisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)
RelationshipSatisfaction = st.sidebar.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
PercentSalaryHike = st.sidebar.slider("Percent Salary Hike", 0, 50, 10)
YearsWithCurrManager = st.sidebar.slider("Years with Current Manager", 0, 20, 5)
JobInvolvement = st.sidebar.slider("Job Involvement (1-4)", 1, 4, 3)

# Collect inputs into a DataFrame
input_data = pd.DataFrame({
    'Age': [Age],
    'MonthlyIncome': [MonthlyIncome],
    'OverTime': [OverTime],
    'EnvironmentSatisfaction': [EnvironmentSatisfaction],
    'RelationshipSatisfaction': [RelationshipSatisfaction],
    'PercentSalaryHike': [PercentSalaryHike],
    'YearsWithCurrManager': [YearsWithCurrManager],
    'JobInvolvement': [JobInvolvement]
})

# Preprocess inputs
st.write("## Input Data")
st.write(input_data)
X_processed = preprocessor.transform(input_data)

# Neural Network Prediction
nn_predictions = nn_model.predict(X_processed).flatten()

# Combine NN predictions with preprocessed data
X_hybrid = np.column_stack((X_processed, nn_predictions))

# Hybrid Model Prediction
hybrid_prediction = hybrid_model.predict_proba(X_hybrid)[:, 1]

# Display Prediction
st.write("## Prediction")
attrition_probability = hybrid_prediction[0]
if attrition_probability > 0.5:
    st.error(f"The employee is likely to leave. Probability: {attrition_probability:.2f}")
else:
    st.success(f"The employee is likely to stay. Probability: {attrition_probability:.2f}")

# SHAP Explanation
explainer = shap.TreeExplainer(hybrid_model)
shap_values = explainer.shap_values(X_hybrid)

st.write("## Feature Contribution")
shap.force_plot(explainer.expected_value, shap_values[0, :], input_data, matplotlib=True)
