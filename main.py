from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

model = joblib.load("model.pkl")
app = FastAPI(title="API de Predicción de Préstamos con KNN")

class Cliente(BaseModel):
    edad: int
    ingreso_mensual_cop: float
    historial_credito: float
    monto_prestamo_cop: float

@app.get("/")
def home():
    return {"message": "API de Predicción de Préstamos con KNN"}

@app.post("/predict")
def predict(cliente: Cliente):
    data = pd.DataFrame([cliente.edad, cliente.ingreso_mensual_cop, cliente.historial_credito, cliente.monto_prestamo_cop]).T
    prediction = model.predict(data)
    return {"aprobado": int(prediction[0])}
