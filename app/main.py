from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime
from collections import defaultdict

app = FastAPI()

# Allow frontend (Vercel) to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("model.pkl")

# For observability
total_predictions = 0
prediction_distribution = defaultdict(int)
log = []

# Input schema
class InputData(BaseModel):
    features: list

# Root endpoint (for testing)
@app.get("/")
def read_root():
    return {"message": "FastAPI is running! Use /predict or /metrics endpoints."}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    global total_predictions, prediction_distribution, log

    X = np.array(data.features).reshape(1, -1)
    prediction = int(model.predict(X)[0])

    # Update metrics
    total_predictions += 1
    prediction_distribution[str(prediction)] += 1
    log.append({
        "timestamp": datetime.utcnow().isoformat(),
        "input": data.features,
        "prediction": prediction
    })

    # Limit log to last 20 entries
    log = log[-20:]

    return {"prediction": prediction}

# Metrics endpoint
@app.get("/metrics")
def get_metrics():
    return {
        "total_predictions": total_predictions,
        "prediction_distribution": dict(prediction_distribution),
        "log": log
    }
