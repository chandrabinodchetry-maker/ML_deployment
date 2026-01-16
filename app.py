import joblib
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("bank_marketing_model.pkl")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}
