from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path

app = FastAPI()

MODEL_DIR = Path(__file__).resolve().parent / 'saved_models'
model = joblib.load(MODEL_DIR / 'logreg.joblib')
scaler = joblib.load(MODEL_DIR / 'logreg_scaler.joblib')

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict/")
def predict(data: PatientData):
    arr = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure,
                     data.SkinThickness, data.Insulin, data.BMI,
                     data.DiabetesPedigreeFunction, data.Age]])
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)
    return {"prediction": int(prediction[0])}