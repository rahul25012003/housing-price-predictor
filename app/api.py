"""
FastAPI Prediction Endpoint
============================
REST API for housing price prediction.

Run: uvicorn app.api:app --reload --port 8000
"""

import os, sys
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

app = FastAPI(
    title="Indian Housing Price Predictor API",
    description="Predict house prices in Indian metro cities",
    version="1.0.0"
)

# load model + transformers at startup
try:
    model = joblib.load(config.BEST_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    encoders = joblib.load(config.ENCODER_PATH)
    print("Model and transformers loaded successfully")
except FileNotFoundError as e:
    print(f"[ERROR] Could not load model artifacts: {e}")
    print("Run the training pipeline first!")
    model = None


class HouseInput(BaseModel):
    city: str = Field(..., example="Mumbai")
    area_sqft: int = Field(..., ge=200, le=10000, example=1200)
    bedrooms: int = Field(..., ge=1, le=6, example=3)
    bathrooms: int = Field(..., ge=1, le=5, example=2)
    property_type: str = Field(..., example="Apartment")
    furnishing: str = Field(..., example="Semi-Furnished")
    age_years: int = Field(..., ge=0, le=50, example=5)
    floor: int = Field(..., ge=0, le=30, example=4)
    parking: int = Field(..., ge=0, le=3, example=1)


class PredictionResponse(BaseModel):
    predicted_price_lakhs: float
    lower_bound: float
    upper_bound: float
    input_summary: str


@app.get("/")
def root():
    return {"message": "Housing Price Predictor API", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: HouseInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    try:
        # derive features
        room_density = input_data.bedrooms / input_data.area_sqft * 1000
        price_per_sqft = 0  # placeholder

        # encode
        city_enc = encoders["city"].transform([input_data.city])[0]
        type_enc = encoders["property_type"].transform([input_data.property_type])[0]
        furn_enc = encoders["furnishing"].transform([input_data.furnishing])[0]

        # build feature array
        num_features = np.array([[
            input_data.area_sqft, input_data.bedrooms, input_data.bathrooms,
            input_data.age_years, input_data.floor, input_data.parking,
            price_per_sqft, room_density
        ]])
        scaled = scaler.transform(num_features)

        # combine all features into dict
        features = {}
        for i, col in enumerate(config.NUMERICAL_COLS + config.DERIVED_COLS):
            features[col] = scaled[0][i]
        features["city_encoded"] = city_enc
        features["property_type_encoded"] = type_enc
        features["furnishing_encoded"] = furn_enc

        # align columns with training data
        feature_cols = [c for c in pd.read_csv(config.PROCESSED_DATA_PATH, nrows=0).columns
                        if c != config.TARGET_COL]
        input_df = pd.DataFrame([features]).reindex(columns=feature_cols, fill_value=0)

        prediction = float(model.predict(input_df)[0])

        return PredictionResponse(
            predicted_price_lakhs=round(prediction, 2),
            lower_bound=round(prediction * 0.88, 2),
            upper_bound=round(prediction * 1.12, 2),
            input_summary=f"{input_data.bedrooms}BHK {input_data.property_type} in {input_data.city}, {input_data.area_sqft} sqft"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
