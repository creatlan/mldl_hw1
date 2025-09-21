import streamlit as st
import requests

st.title("Diabetes Prediction")

# Ввод данных
pregnancies = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
blood_pressure = st.number_input("Blood Pressure")
skin_thickness = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
pedigree = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

# Кнопка для предсказания
if st.button("Predict"):
    response = requests.post(
        "http://localhost:8000/predict/", 
        json={"Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
              "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi, 
              "DiabetesPedigreeFunction": pedigree, "Age": age}
    )
    prediction = response.json()
    st.write(f"Prediction: {'Diabetic' if prediction['prediction'] == 1 else 'Not Diabetic'}")
