from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

model = joblib.load("crop_model.pkl")

@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/predict")
def predict(ndvi: float):
    features = pd.DataFrame([{
        'ndvi': ndvi,
        'ndvi_squared': ndvi**2,
        'ndvi_log': np.log(ndvi + 1e-5)
    }])

    pred = model.predict(features)[0]
    return {"prediction": pred}
