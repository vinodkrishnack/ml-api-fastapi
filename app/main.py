from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from collections import defaultdict
import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ml-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Load model
model = joblib.load("model.pkl")

# Track predictions
prediction_counts = defaultdict(int)
prediction_log = []

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    pred = int(prediction[0])

    # Log and track
    prediction_counts[pred] += 1
    prediction_log.append({
        "timestamp": datetime.datetime.now().isoformat(),
        "input": data.features,
        "prediction": pred
    })

    logging.info(f"Prediction: {pred} for input: {data.features}")
    return {"prediction": pred}

@app.get("/metrics")
def get_metrics():
    return {
        "total_predictions": sum(prediction_counts.values()),
        "prediction_distribution": prediction_counts,
        "log": prediction_log[-10:]  # Last 10 predictions
    }
