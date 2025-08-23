from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from collections import defaultdict
import datetime

# Create FastAPI app
app = FastAPI()
from fastapi import FastAPI


@app.get("/")
def read_root():
    return {"message": "FastAPI is running! Use /predict or /metrics endpoints."}

# CORS for frontend hosted on Vercel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ml-frontend.vercel.app"],  # Replace with your actual Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("model.pkl")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Input schema
class InputData(BaseModel):
    features: list

# In-memory logs for metrics
prediction_counts = defaultdict(int)
prediction_log = []

# Prediction route
@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    pred = int(prediction[0])

    # Logging and metrics
    prediction_counts[pred] += 1
    prediction_log.append({
        "timestamp": datetime.datetime.now().isoform
