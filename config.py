"""
Configuration file for Housing Price Predictor project.
Contains all paths, constants, and hyperparameter defaults.
"""

import os

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RAW_DATA_PATH = os.path.join(DATA_DIR, "housing_raw.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "housing_processed.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.joblib")

# ---- Dataset Constants ----
TARGET_COL = "price_lakhs"
RANDOM_STATE = 42
TEST_SIZE = 0.2

CATEGORICAL_COLS = ["city", "property_type", "furnishing"]
NUMERICAL_COLS = ["area_sqft", "bedrooms", "bathrooms", "age_years", "floor", "parking"]
DERIVED_COLS = ["price_per_sqft", "room_density"]

CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata"]
PROPERTY_TYPES = ["Apartment", "Villa", "Independent House", "Builder Floor"]
FURNISHING_OPTIONS = ["Furnished", "Semi-Furnished", "Unfurnished"]

# ---- Model Hyperparameters (defaults) ----
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}

GB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
}

# GridSearchCV params for XGBoost tuning
XGB_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.15],
    "subsample": [0.7, 0.8, 0.9],
}

# ---- Synthetic Data Generation ----
SYNTHETIC_SAMPLES = 5000
