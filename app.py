import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load pre-trained models and preprocessor
try:
    hybrid_model = joblib.load("hybrid_model.pkl")
    nn_model = load_model("nn_model.keras")
    preprocessor = joblib.load("preprocessor.pkl")
except Exception as e:
    st.error(f"Error loading models or preprocessor: {e}")

# Title and Description
st.title("Employee Attrition Prediction")
st.write("This application predicts employee attrition using a hybrid Neural Network and XGBoost model.")

# Input Form for Top Features
overtime = st.selectbox("OverTime (Yes=1, No=0)", [0, 1])
environment_satisfaction = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
monthly_income = st.number_input("Monthly Income (e.g., 5000)", min_value=1000, max_value=20000, step=500, value=5000)
years_with_manager = st.slider("Years With Current Manager", 0, 20, 5)

# Buttons: Predict and Reset
col1, col2 = st.columns(2)

def reset_inputs():
    """Resets all input fields."""
    st.session_state["overtime"] = 0
    st.session_state["environment_satisfaction"] = 3
    st.session_state["relationship_satisfaction"] = 3
    st.session_state["monthly_income"] = 5000
    st.session_state["years_with_manager"] = 5
    st.experimental_rerun()

if col2.button("Reset"):
    reset_inputs()

if col1.button("Predict"):
    try:
        # Combine inputs into a DataFrame
        input_features = pd.DataFrame([[
            overtime,
            environment_satisfaction,
            relationship_satisfaction,
            monthly_income,
            years_with_manager
        ]], columns=["OverTime", "EnvironmentSatisfaction", "RelationshipSatisfaction", "MonthlyIncome", "YearsWithCurrManager"])

        # Debug: Display input DataFrame
        st.write("### Input DataFrame:")
        st.write(input_features)

        # Preprocess inputs
        input_processed = preprocessor.transform(input_features)

        # Debug: Display processed inputs
        st.write("### Preprocessed Input:")
        st.write(input_processed)

        # Predict using Neural Network
        nn_predictions = nn_model.predict(input_processed)

        # Combine NN predictions for hybrid model
        hybrid_input = np.column_stack((input_processed, nn_predictions))

        # Hybrid model predictions
        prediction = hybrid_model.predict(hybrid_input)
        attrition_probability = hybrid_model.predict_proba(hybrid_input)[:, 1]

        # Display results
        st.write("### Prediction Results")
        st.write(f"Will the employee leave the company? **{'Yes' if prediction[0] == 1 else 'No'}**")
        st.write(f"Probability of Attrition: **{attrition_probability[0]:.2f}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
