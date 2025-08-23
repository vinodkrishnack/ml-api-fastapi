from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime

# Load your trained model
model = joblib.load("model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware (allow all origins for development; update for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Vercel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory logs and metrics storage
prediction_logs = []
prediction_counts = {}

# Input schema for prediction
class InputData(BaseModel):
    features: list[float]

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "FastAPI is running! Use /predict or /metrics endpoints."}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    features_array = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features_array)[0]

    # Log prediction with timestamp
    timestamp = datetime.utcnow().isoformat()
    prediction_logs.append({
        "timestamp": timestamp,
        "input": data.features,
        "prediction": int(prediction),
    })

    # Update prediction counts
    prediction_counts[int(prediction)] = prediction_counts.get(int(prediction), 0) + 1

    return {"prediction": int(prediction)}

# Metrics endpoint for observability
@app.get("/metrics")
def get_metrics():
    return {
        "total_predictions": len(prediction_logs),
        "prediction_distribution": prediction_counts,
        "log": prediction_logs[-20:],  # last 20 predictions
    }
