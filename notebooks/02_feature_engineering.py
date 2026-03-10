"""
02 - Feature Engineering
=========================
Handles missing values, encodes categoricals, creates derived features,
and scales numerical columns. Saves processed data + fitted transformers.

Run: python notebooks/02_feature_engineering.py
"""

import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import load_data

print("=" * 50)
print("  FEATURE ENGINEERING")
print("=" * 50)

df = load_data()
print(f"Original shape: {df.shape}")

# --- 1. Handle Missing Values ---
print("\nHandling missing values...")
# fill numerical cols with median
for col in config.NUMERICAL_COLS:
    if col in df.columns and df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled {col} NaN with median={median_val}")

# sanity check
assert df[config.NUMERICAL_COLS].isnull().sum().sum() == 0, "Still have missing numericals!"

# --- 2. Create Derived Features ---
print("\nCreating derived features...")
df["price_per_sqft"] = (df["price_lakhs"] * 100000) / df["area_sqft"]  # in rupees
df["room_density"] = df["bedrooms"] / df["area_sqft"] * 1000  # rooms per 1000 sqft
# TODO: maybe add total_rooms = bedrooms + bathrooms?
print(f"  Added: price_per_sqft, room_density")

# --- 3. Encode Categorical Variables ---
print("\nEncoding categorical variables...")
label_encoders = {}
for col in config.CATEGORICAL_COLS:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: {list(le.classes_)}")

# save encoders
os.makedirs(config.MODEL_DIR, exist_ok=True)
joblib.dump(label_encoders, config.ENCODER_PATH)
print(f"  Saved label encoders to {config.ENCODER_PATH}")

# --- 4. Scale Numerical Features ---
print("\nScaling numerical features...")
# scale everything except target and original categoricals
feature_cols = config.NUMERICAL_COLS + config.DERIVED_COLS
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
joblib.dump(scaler, config.SCALER_PATH)
print(f"  Scaled {len(feature_cols)} columns")
print(f"  Saved scaler to {config.SCALER_PATH}")

# --- 5. Drop original categorical columns ---
# keep encoded versions only
df_processed = df.drop(columns=config.CATEGORICAL_COLS)

# --- 6. Final checks ---
print(f"\nProcessed shape: {df_processed.shape}")
print(f"Columns: {list(df_processed.columns)}")

# check for any remaining issues
print(f"\nNull values remaining: {df_processed.isnull().sum().sum()}")
print(f"Inf values: {np.isinf(df_processed.select_dtypes(include=[np.number])).sum().sum()}")

# --- 7. Save ---
os.makedirs(config.DATA_DIR, exist_ok=True)
df_processed.to_csv(config.PROCESSED_DATA_PATH, index=False)
print(f"\nSaved processed data to {config.PROCESSED_DATA_PATH}")

print("\n[DONE] Feature engineering complete.")
