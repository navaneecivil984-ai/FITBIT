import streamlit as st
import numpy as np
import pickle
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Calorie Burn Predictor", page_icon="🔥")

st.title("🔥 Calorie Burn Prediction")
st.write("Enter workout details to predict calories burned")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_files():
    with open("linear_regression_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("features.json", "r") as f:
        features = json.load(f)

    return model, scaler, features

model, scaler, features = load_files()

# ---------------- USER INPUT ----------------
st.subheader("Enter Inputs")

effective_met = st.number_input("Effective MET", min_value=0.0, value=5.0)
base_met = st.number_input("Base MET", min_value=0.0, value=1.0)
session_duration_hours = st.number_input("Session Duration (hours)", min_value=0.0, value=1.0)
weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=70.0)
bmi = st.number_input("BMI", min_value=0.0, value=24.0)
height_m = st.number_input("Height (m)", min_value=0.0, value=1.7)

# ---------------- PREDICTION ----------------
if st.button("Predict Calories Burned"):

    try:
        # Create input in correct order
        input_data = {
            "effective_met": effective_met,
            "base_met": base_met,
            "session_duration_hours": session_duration_hours,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "height_m": height_m
        }

        # Convert to array using correct feature order
        input_array = np.array([[input_data[col] for col in features]])

        # Scale input
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)

        st.success(f"🔥 Estimated Calories Burned: {prediction[0]:.2f} kcal")

    except Exception as e:
        st.error(f"Error: {e}")