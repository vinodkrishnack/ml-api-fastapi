from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime
from collections import defaultdict
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")

# File to persist metrics
METRICS_FILE = "metrics.json"

# Load metrics if file exists
def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    else:
        return {
            "total_predictions": 0,
            "prediction_distribution": {},
            "log": []
        }

# Save metrics to file
def save_metrics(data):
    with open(METRICS_FILE, "w") as f:
        json.dump(data, f)

metrics = load_metrics()

class InputData(BaseModel):
    features: list

@app.get("/")
def root():
    return {"message": "FastAPI is running! Use /predict or /metrics endpoints."}

@app.post("/predict")
def predict(data: InputData):
    global metrics

    X = np.array(data.features).reshape(1, -1)
    prediction = int(model.predict(X)[0])

    # Update metrics
    metrics["total_predictions"] += 1
    pred_str = str(prediction)
    metrics["prediction_distribution"][pred_str] = metrics["prediction_distribution"].get(pred_str, 0) + 1
    metrics["log"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "input": data.features,
        "prediction": prediction
    })

    # Keep only recent 20
    metrics["log"] = metrics["log"][-20:]

    # Save to file
    save_metrics(metrics)

    return {"prediction": prediction}

@app.get("/metrics")
def get_metrics():
    return metrics
