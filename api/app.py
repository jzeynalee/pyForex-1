# api/app.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from inference.predictor import HybridPredictor
from pydantic import BaseModel
import pandas as pd
from typing import List

# Global state wrapper
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    ml_models["predictor"] = HybridPredictor()
    print("Models Loaded")
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    tick_volume: float

class PredictionRequest(BaseModel):
    candles: List[Candle]

@app.post("/predict")
def predict(payload: PredictionRequest):
    if len(payload.candles) < 60:
        raise HTTPException(status_code=400, detail="Need at least 60 candles")

    df = pd.DataFrame([c.dict() for c in payload.candles])
    
    # Use the pre-loaded predictor
    probs = ml_models["predictor"].predict(df)
    
    return {
        "probabilities": {
            "buy": float(probs[0]),
            "sell": float(probs[1]),
            "hold": float(probs[2])
        }
    }