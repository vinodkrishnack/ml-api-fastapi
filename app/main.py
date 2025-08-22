# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 11:51:45 2025

@author: Admin
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    prediction = model.predict(arr)
    return {"prediction": int(prediction[0])}