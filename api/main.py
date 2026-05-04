"""
FastAPI backend for NYC Taxi Trip Duration Prediction.

Loads the Keras NN model and StandardScaler, exposes /predict and /health endpoints.
"""

from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow import keras


# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "Models"
MODEL_PATH = MODEL_DIR / "taxi_model_NN.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# ── Global holders (populated at startup) ────────────────────────────────────
model = None
scaler = None


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model & scaler once at startup."""
    global model, scaler
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    model = keras.models.load_model(str(MODEL_PATH))
    scaler = joblib.load(str(SCALER_PATH))
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Scaler loaded from {SCALER_PATH}")
    yield  # app runs
    model = None
    scaler = None


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NYC Taxi Trip Duration Predictor",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    "passenger_count",
    "trip_distance",
    "payment_type",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "is_weekend",
]


class TripInput(BaseModel):
    """Seven features expected by the NN model (reduced feature list)."""
    passenger_count: int = Field(..., ge=0, le=9, description="Number of passengers")
    trip_distance: float = Field(..., gt=0, le=12.0, description="Trip distance in miles")
    payment_type: int = Field(..., ge=1, le=5, description="1=Credit, 2=Cash, 3=No charge, 4=Dispute, 5=Unknown")
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup (0-23)")
    pickup_dayofweek: int = Field(..., ge=1, le=7, description="Day of week (1=Sun … 7=Sat)")
    pickup_month: int = Field(..., ge=1, le=12, description="Month of pickup")
    is_weekend: int = Field(..., ge=0, le=1, description="1 if Saturday or Sunday, else 0")


class PredictionResponse(BaseModel):
    trip_duration_seconds: float
    trip_duration_minutes: float


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(trip: TripInput):
    """Return predicted trip duration for one trip."""
    try:
        # Build feature array in the exact order used during training
        features = np.array([[
            trip.passenger_count,
            trip.trip_distance,
            trip.payment_type,
            trip.pickup_hour,
            trip.pickup_dayofweek,
            trip.pickup_month,
            trip.is_weekend,
        ]], dtype=np.float64)

        # Use a DataFrame so sklearn sees the feature names it was fitted with
        import pandas as pd
        features_df = pd.DataFrame(features, columns=FEATURE_ORDER)

        # Scale then predict (model outputs log1p-space)
        features_scaled = scaler.transform(features_df)
        pred_log = model.predict(features_scaled, verbose=0)
        pred_seconds = float(np.expm1(pred_log[0][0]))

        # Clamp to reasonable range
        pred_seconds = max(0.0, pred_seconds)

        return PredictionResponse(
            trip_duration_seconds=round(pred_seconds, 2),
            trip_duration_minutes=round(pred_seconds / 60, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
