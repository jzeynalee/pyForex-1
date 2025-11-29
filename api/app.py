# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import pandas as pd
# Import your model loaders and feature builders

app = FastAPI(title="Hybrid Trading Brain API")

# Define Input Schema
class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    tick_volume: float

class PredictionRequest(BaseModel):
    candles: List[Candle] # Expecting last 60 candles

# Load models globally on startup
models = {}
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_models():
    # Load models into the 'models' dict
    # models['fusion'] = ...
    print("Models loaded into memory.")

@app.post("/predict")
def predict(payload: PredictionRequest):
    if len(payload.candles) < 60:
        raise HTTPException(status_code=400, detail="Need at least 60 candles")

    # Convert JSON to DataFrame
    df = pd.DataFrame([c.dict() for c in payload.candles])
    
    # Run Inference
    # lstm_vec, vit_vec, yolo_vec = build_features(df, models['lstm'], ...)
    # pred = models['fusion'](...)
    
    # Mock Response
    return {
        "signal": "BUY",
        "confidence": 0.85,
        "market_regime": "VOLATILE_BULLISH"
    }

# Run with: uvicorn api.app:app --reload