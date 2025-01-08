import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models and preprocessor
nn_model = load_model("nn_model.keras")
hybrid_model = joblib.load("hybrid_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Title and Description
st.title("Employee Attrition Prediction")
st.write("This application predicts employee attrition using a hybrid Neural Network and XGBoost model.")

# User Input Form for Top 5 Features
st.write("### Input Employee Features")
overtime = st.selectbox("OverTime (Yes=1, No=0)", ["Yes", "No"])
environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
monthly_income = st.number_input("Monthly Income (e.g., 5000)", min_value=1000, max_value=20000, step=1000, value=5000)
years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, step=1, value=5)

# Predict and Reset Buttons
col1, col2 = st.columns(2)
if "reset_prediction" not in st.session_state:
    st.session_state.reset_prediction = False

with col1:
    if st.button("Predict"):
        st.session_state.reset_prediction = False
        try:
            # Prepare input data for the top 5 features
            input_data = pd.DataFrame({
                "OverTime": [1 if overtime == "Yes" else 0],
                "EnvironmentSatisfaction": [environment_satisfaction],
                "RelationshipSatisfaction": [relationship_satisfaction],
                "MonthlyIncome": [monthly_income],
                "YearsWithCurrManager": [years_with_curr_manager],
            })

            # Dynamically add the "hidden" columns with default values
            all_columns = [name for transformer in preprocessor.transformers_ for name in transformer[2]]
            for col in all_columns:
                if col not in input_data.columns:
                    input_data[col] = 0  # Fill missing columns with 0

            # Ensure column order matches the preprocessor
            input_data = input_data[all_columns]

            # Debugging Logs
            st.write("### Input DataFrame (Before Processing):")
            st.write(input_data)
            st.write("### Data Types:")
            st.write(input_data.dtypes)

            # Convert all columns to numeric types explicitly
            input_data = input_data.astype(float)

            # Debugging Logs
            st.write("### Input DataFrame (After Type Conversion):")
            st.write(input_data)

            # Preprocess inputs
            input_array = preprocessor.transform(input_data)

            # Debugging Logs
            st.write("### Preprocessed Input Array:")
            st.write(input_array)

            # Predict using Neural Network
            nn_predictions = nn_model.predict(input_array).flatten()

            # Debugging Logs
            st.write("### Neural Network Predictions:")
            st.write(nn_predictions)

            # Combine features for Hybrid Model
            input_hybrid = np.column_stack((input_array, nn_predictions))

            # Debugging Logs
            st.write("### Hybrid Input Array:")
            st.write(input_hybrid)

            # Predict using Hybrid NN-XGBoost
            hybrid_predictions = hybrid_model.predict(input_hybrid)
            attrition_probability = hybrid_model.predict_proba(input_hybrid)[:, 1]

            # Display predictions
            st.write("### Prediction Results")
            prediction = "Yes" if hybrid_predictions[0] == 1 else "No"
            st.write(f"Will the employee leave? **{prediction}**")
            st.write(f"Probability of Attrition: **{attrition_probability[0]:.2f}**")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

with col2:
    if st.button("Reset"):
        st.session_state.reset_prediction = True
        st.experimental_rerun()
