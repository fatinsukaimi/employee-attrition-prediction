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

# Top 5 features form
overtime = st.selectbox("OverTime (Yes=1, No=0)", ["Yes", "No"])
environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
monthly_income = st.number_input("Monthly Income (e.g., 5000)", min_value=1000, max_value=20000, step=1000, value=5000)
years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, step=1, value=5)

# Reset and Predict Buttons
col1, col2 = st.columns(2)
if "reset_prediction" not in st.session_state:
    st.session_state.reset_prediction = False

with col1:
    if st.button("Predict"):
        st.session_state.reset_prediction = False
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                "OverTime": [1 if overtime == "Yes" else 0],
                "EnvironmentSatisfaction": [environment_satisfaction],
                "RelationshipSatisfaction": [relationship_satisfaction],
                "MonthlyIncome": [monthly_income],
                "YearsWithCurrManager": [years_with_curr_manager],
            })

            # Preprocess inputs
            input_array = preprocessor.transform(input_data)

            # Predict using Neural Network
            nn_predictions = nn_model.predict(input_array).flatten()

            # Combine features for Hybrid Model
            input_hybrid = np.column_stack((input_array, nn_predictions))

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
