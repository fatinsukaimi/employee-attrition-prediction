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
st.write("This application predicts employee attrition using the top 5 important features.")

# Input for Top 5 Features
overtime = st.selectbox("OverTime (Yes=1, No=0)", ["Yes", "No"])
environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
monthly_income = st.number_input("Monthly Income (e.g., 5000)", min_value=1000, max_value=20000, step=1000, value=5000)
years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, step=1, value=5)

# Predict Button
if st.button("Predict"):
    try:
        # Prepare input data for the top 5 features
        input_data = pd.DataFrame({
            "OverTime": [1 if overtime == "Yes" else 0],
            "EnvironmentSatisfaction": [environment_satisfaction],
            "RelationshipSatisfaction": [relationship_satisfaction],
            "MonthlyIncome": [monthly_income],
            "YearsWithCurrManager": [years_with_curr_manager],
        })

        # Fill the rest of the columns with default values
        all_columns = [name for transformer in preprocessor.transformers_ for name in transformer[2]]
        for col in all_columns:
            if col not in input_data.columns:
                input_data[col] = 0  # Default value for missing columns

        # Ensure column order matches the preprocessor
        input_data = input_data[all_columns]

        # Debugging: Show input data before processing
        st.write("### Input DataFrame (Before Processing):")
        st.write(input_data)

        # Convert all columns to numeric explicitly
        input_data = input_data.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Preprocess inputs
        input_array = preprocessor.transform(input_data)

        # Predict using Neural Network
        nn_predictions = nn_model.predict(input_array).flatten()

        # Combine features for Hybrid Model
        input_hybrid = np.column_stack((input_array, nn_predictions))

        # Predict using Hybrid NN-XGBoost
        hybrid_predictions = hybrid_model.predict(input_hybrid)
        attrition_probability = hybrid_model.predict_proba(input_hybrid)[:, 1]

        # Display Predictions
        st.subheader("Prediction Results")
        prediction = "Yes" if hybrid_predictions[0] == 1 else "No"
        st.write(f"Will the employee leave? **{prediction}**")
        st.write(f"Probability of Attrition: **{attrition_probability[0]:.2f}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
