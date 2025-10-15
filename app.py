# -*- coding: utf-8 -*-
"""
Heart Disease Prediction Web App
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model 
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
# Prediction function
def heart_pred(input_data):
    arr = np.array(input_data, dtype=float)  # ensure numeric
    arr = arr.reshape(1, -1)
    prediction = loaded_model.predict(arr)
    if prediction[0] == 0:
        return '✅ You are healthy'
    else:
        return '⚠️ You may have heart disease'

# Main function for Streamlit app
def main():
    st.title("💓 Heart Disease Prediction Web App")
    st.write("Enter the details below:")

    # User inputs (based on your dataset features)
    age = st.number_input('Age', min_value=1, max_value=120, step=1)
    sex = st.selectbox('Sex (1 = Male, 0 = Female)', [0, 1])
    cp = st.number_input('Chest Pain Type (0–3)', min_value=0, max_value=3, step=1)
    trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=250, step=1)
    chol = st.number_input('Serum Cholestrol (mg/dl)', min_value=100, max_value=600, step=1)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', [0, 1])
    restecg = st.number_input('Resting ECG Results (0–2)', min_value=0, max_value=2, step=1)
    thalach = st.number_input('Max Heart Rate Achieved', min_value=60, max_value=220, step=1)
    exang = st.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', [0, 1])
    oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=10.0, step=0.1)
    slope = st.number_input('Slope (0–2)', min_value=0, max_value=2, step=1)
    ca = st.number_input('Number of Major Vessels (0–4)', min_value=0, max_value=4, step=1)
    thal = st.number_input('Thal (1 = Normal, 2 = Fixed defect, 3 = Reversible defect)', min_value=1, max_value=3, step=1)

    # Prediction
    diagnosis = ""
    if st.button("🔍 Predict"):
        diagnosis = heart_pred([age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal])
        st.success(diagnosis)

if __name__ == '__main__':
    main()
