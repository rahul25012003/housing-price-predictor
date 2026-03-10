"""
Data loading utilities for the Housing Price Predictor.
Includes synthetic data generation if the Kaggle dataset is not available.
"""

import os
import sys
import numpy as np
import pandas as pd

# add project root to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_data(filepath=None):
    """Load housing dataset from CSV. Falls back to synthetic if file missing."""
    if filepath is None:
        filepath = config.RAW_DATA_PATH

    if os.path.exists(filepath):
        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    else:
        print(f"[WARN] {filepath} not found. Generating synthetic data...")
        return generate_synthetic_data()


def generate_synthetic_data(n_samples=None, save=True):
    """
    Generate synthetic Indian housing data that roughly mimics real distributions.
    Prices vary by city — Mumbai being the most expensive, Kolkata the cheapest.
    """
    if n_samples is None:
        n_samples = config.SYNTHETIC_SAMPLES

    np.random.seed(config.RANDOM_STATE)

    # city base prices in lakhs per sqft (approx)
    city_base_price = {
        "Mumbai": 0.018, "Delhi": 0.014, "Bangalore": 0.012,
        "Hyderabad": 0.009, "Chennai": 0.010, "Pune": 0.009, "Kolkata": 0.007
    }

    cities = np.random.choice(config.CITIES, n_samples, p=[0.22, 0.18, 0.17, 0.13, 0.12, 0.10, 0.08])
    property_types = np.random.choice(config.PROPERTY_TYPES, n_samples, p=[0.45, 0.15, 0.25, 0.15])
    furnishing = np.random.choice(config.FURNISHING_OPTIONS, n_samples, p=[0.25, 0.45, 0.30])

    area_sqft = np.random.normal(1200, 450, n_samples).clip(400, 5000).astype(int)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.10, 0.30, 0.35, 0.18, 0.07])
    bathrooms = np.clip(bedrooms - np.random.choice([0, 1], n_samples, p=[0.6, 0.4]), 1, 5)
    age_years = np.random.exponential(8, n_samples).clip(0, 40).astype(int)
    floor_num = np.random.randint(0, 25, n_samples)
    parking = np.random.choice([0, 1, 2], n_samples, p=[0.25, 0.50, 0.25])

    # calculate price based on features + noise
    prices = []
    for i in range(n_samples):
        base = city_base_price[cities[i]] * area_sqft[i]
        # property type multiplier
        type_mult = {"Apartment": 1.0, "Villa": 1.4, "Independent House": 1.15, "Builder Floor": 0.95}
        base *= type_mult[property_types[i]]
        # furnishing bonus
        furn_add = {"Furnished": 8, "Semi-Furnished": 3, "Unfurnished": 0}
        base += furn_add[furnishing[i]]
        # age depreciation
        base *= max(0.5, 1 - age_years[i] * 0.012)
        # bedroom adjustment
        base += bedrooms[i] * 2.5
        # noise
        noise = np.random.normal(0, base * 0.12)
        price = max(5, base + noise)
        prices.append(round(price, 2))

    df = pd.DataFrame({
        "city": cities,
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "property_type": property_types,
        "furnishing": furnishing,
        "age_years": age_years,
        "floor": floor_num,
        "parking": parking,
        "price_lakhs": prices,
    })

    # introduce ~2% missing values randomly for realism
    mask = np.random.random(df.shape) < 0.02
    # only apply to some columns, not target
    for col in ["area_sqft", "bathrooms", "age_years", "floor", "parking"]:
        col_mask = mask[:, df.columns.get_loc(col)]
        df.loc[col_mask, col] = np.nan

    if save:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_csv(config.RAW_DATA_PATH, index=False)
        print(f"Saved synthetic data to {config.RAW_DATA_PATH} ({len(df)} rows)")

    return df


if __name__ == "__main__":
    # running directly generates the dataset
    df = generate_synthetic_data()
    print("\nSample:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
