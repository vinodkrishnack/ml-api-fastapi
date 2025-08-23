from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import logging

# 🔹 Create FastAPI app
app = FastAPI()

# 🔹 Add CORS middleware (required to connect with your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Set up logging
logging.basicConfig(level=logging.INFO)

# 🔹 Load your trained model
model = joblib.load("model.pkl")

# 🔹 Define input schema
class InputData(BaseModel):
    features: list

# 🔹 Define prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    logging.info(f"Input: {data.features} → Prediction: {prediction[0]}")
    return {"prediction": int(prediction[0])}
