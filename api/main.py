"""
FastAPI backend for NYC Taxi Trip Duration Prediction.

Loads both the full Keras NN model and the pruned TFLite model at startup.
The client chooses which model to use via a query parameter.
"""

from pathlib import Path
from contextlib import asynccontextmanager
from enum import Enum

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow import keras
import tensorflow as tf


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERAS_MODEL_PATH = PROJECT_ROOT / "Models" / "taxi_model_NN.keras"
TFLITE_MODEL_PATH = PROJECT_ROOT / "Model_Speedup" / "taxi_model.tflite"
SCALER_PATH = PROJECT_ROOT / "Models" / "scaler.pkl"

# ── Global holders (populated at startup) ────────────────────────────────────
keras_model = None
tflite_model = None
scaler = None


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models & scaler once at startup."""
    global keras_model, tflite_model, scaler

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    scaler = joblib.load(str(SCALER_PATH))
    print(f"Scaler loaded from {SCALER_PATH}")

    if KERAS_MODEL_PATH.exists():
        keras_model = keras.models.load_model(str(KERAS_MODEL_PATH))
        print(f"Keras model loaded from {KERAS_MODEL_PATH}")
    else:
        print(f"Keras model not found at {KERAS_MODEL_PATH}")

    if TFLITE_MODEL_PATH.exists():
        tflite_model = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
        print(f"TFLite model loaded from {TFLITE_MODEL_PATH}")
    else:
        print(f"TFLite model not found at {TFLITE_MODEL_PATH}")

    yield
    keras_model = None
    tflite_model = None
    scaler = None


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NYC Taxi Trip Duration Predictor",
    version="1.1.0",
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


class ModelType(str, Enum):
    keras = "keras"
    tflite = "tflite"


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
    model_used: str


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "keras_model_loaded": keras_model is not None,
        "tflite_model_loaded": tflite_model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    trip: TripInput,
    model_type: ModelType = Query(ModelType.tflite, description="Which model to use: 'keras' or 'tflite'"),
):
    """Return predicted trip duration for one trip."""
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler is not loaded yet.")

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
        features_df = pd.DataFrame(features, columns=FEATURE_ORDER)
        features_scaled = scaler.transform(features_df)

        if model_type == ModelType.tflite:
            if tflite_model is None:
                raise HTTPException(status_code=400, detail="TFLite model is not loaded")
            features_scaled = features_scaled.astype("float32")
            input_details = tflite_model.get_input_details()
            output_details = tflite_model.get_output_details()
            tflite_model.resize_tensor_input(input_details[0]["index"], [len(features_scaled), 7])
            tflite_model.allocate_tensors()
            tflite_model.set_tensor(input_details[0]["index"], features_scaled)
            tflite_model.invoke()
            pred_seconds = float(np.expm1(tflite_model.get_tensor(output_details[0]["index"])[0][0]))
        else:
            if keras_model is None:
                raise HTTPException(status_code=400, detail="Keras model is not loaded")
            pred_log = keras_model(features_scaled, training=False).numpy()
            pred_seconds = float(np.expm1(pred_log[0][0]))

        pred_seconds = max(0.0, pred_seconds)

        return PredictionResponse(
            trip_duration_seconds=round(pred_seconds, 2),
            trip_duration_minutes=round(pred_seconds / 60, 2),
            model_used=model_type.value,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
