import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Mapping categorical inputs to numerical values
chest_pain_mapping = {'Atypical Angina (ATA)': 0, 'Non-Anginal Pain (NAP)': 1, 'Asymptomatic (ASY)': 2, 'Typical Angina (TA)': 3}
resting_ecg_mapping = {'Normal': 0, 'ST': 1, 'LVH': 2}
exercise_angina_mapping = {'No': 0, 'Yes': 1}
st_slope_mapping = {'Up': 0, 'Flat': 1, 'Down': 2}

# Streamlit UI
st.title("Heart Disease Prediction App")
st.write("Enter the details below to predict the likelihood of heart disease.")

# User input fields
age = st.number_input("Age", min_value=1, max_value=120, step=1)
sex = st.selectbox("Sex", ['Male', 'Female'])
chest_pain = st.selectbox("Chest Pain Type", list(chest_pain_mapping.keys()))
resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, step=1)
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, step=1)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", list(resting_ecg_mapping.keys()))
max_hr = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, step=1)
exercise_angina = st.selectbox("Exercise Induced Angina", list(exercise_angina_mapping.keys()))
oldpeak = st.number_input("Oldpeak (Numeric value measured in depression) <10", min_value=0.0, max_value=10.0, step=0.1)
st_slope = st.selectbox("Slope of the peak exercise (ST)", list(st_slope_mapping.keys()))

# Convert categorical inputs to numerical values
sex = 1 if sex == 'Male' else 0
chest_pain = chest_pain_mapping[chest_pain]
resting_ecg = resting_ecg_mapping[resting_ecg]
exercise_angina = exercise_angina_mapping[exercise_angina]
st_slope = st_slope_mapping[st_slope]

# Prepare input array
input_data = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                        resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]])

# Scale only specified numerical features
columns_to_scale = [0, 3, 4, 7]  # Indices of Age, RestingBP, Cholesterol, MaxHR
input_data[:, columns_to_scale] = scaler.transform(input_data[:, columns_to_scale])

# Predict
if st.button("Predict Heart Disease"):
    prediction = model.predict(input_data)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"
    st.subheader(f"Prediction: {result}")
