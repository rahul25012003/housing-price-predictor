# Indian Housing Price Predictor

An end-to-end Machine Learning pipeline for predicting house prices across major Indian metropolitan cities. Built as a capstone project for B.Tech CSE (Data Science).

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This project implements a complete ML workflow — from raw data exploration to a deployed prediction API — targeting the Indian real estate market. The pipeline covers data cleaning, feature engineering, model comparison, and deployment via both a Streamlit dashboard and a FastAPI REST endpoint.

## Features

- **Exploratory Data Analysis** — Distribution plots, correlation heatmaps, city-wise price breakdowns, outlier detection
- **Feature Engineering** — Derived features like `price_per_sqft` and `room_density`, categorical encoding, scaling
- **Model Comparison** — Linear Regression, Random Forest, XGBoost, Gradient Boosting with cross-validation
- **Hyperparameter Tuning** — GridSearchCV on the best-performing model
- **Interactive Dashboard** — Streamlit app with prediction form and EDA visualizations
- **REST API** — FastAPI endpoint for programmatic predictions

## Results

| Model | R² Score | RMSE (lakhs) |
|---|---|---|
| Linear Regression | 0.72 | 14.31 |
| Random Forest | 0.85 | 10.22 |
| Gradient Boosting | 0.87 | 9.68 |
| **XGBoost** | **0.89** | **8.94** |

Best model: **XGBoost Regressor** with R² = 0.89 after hyperparameter tuning.

## Tech Stack

- **Language:** Python 3.9+
- **ML/Data:** Scikit-learn, XGBoost, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit, FastAPI, Uvicorn
- **Serialization:** Joblib

## Dataset

Uses the [Indian Housing Price Dataset](https://www.kaggle.com/) from Kaggle. If the dataset is unavailable, a synthetic data generation script is included in `src/data_loader.py`.

**Features include:** City, Area (sqft), Bedrooms, Bathrooms, Property Type, Furnishing Status, Age of Property, Floor, Parking, Price (in lakhs)

## Project Structure

```
housing-price-predictor/
├── app/
│   ├── streamlit_app.py        # Streamlit dashboard
│   └── api.py                  # FastAPI prediction endpoint
├── data/                       # Raw & processed datasets (not tracked)
├── models/                     # Saved model artifacts (not tracked)
├── notebooks/
│   ├── 01_eda.py               # Exploratory data analysis
│   ├── 02_feature_engineering.py
│   └── 03_model_training.py    # Model comparison & tuning
├── outputs/                    # Generated plots and reports
├── src/
│   ├── data_loader.py          # Data loading & synthetic generation
│   └── model_utils.py          # Model helper functions
├── config.py                   # Paths and constants
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

### 1. Clone and install dependencies

```bash
git clone https://github.com/rahul25012003/housing-price-predictor.git
cd housing-price-predictor
pip install -r requirements.txt
```

### 2. Generate synthetic data (if needed)

```bash
python src/data_loader.py
```

### 3. Run the pipeline

```bash
# Step 1: EDA
python notebooks/01_eda.py

# Step 2: Feature Engineering
python notebooks/02_feature_engineering.py

# Step 3: Model Training
python notebooks/03_model_training.py
```

### 4. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

### 5. Start the API server

```bash
uvicorn app.api:app --reload --port 8000
```

Then send a POST request to `http://localhost:8000/predict`:

```json
{
  "city": "Mumbai",
  "area_sqft": 1200,
  "bedrooms": 3,
  "bathrooms": 2,
  "property_type": "Apartment",
  "furnishing": "Semi-Furnished",
  "age_years": 5,
  "floor": 4,
  "parking": 1
}
```

## Future Improvements

- [ ] Add SHAP explanations for individual predictions
- [ ] Deploy on AWS/GCP with Docker
- [ ] Add time-series component for price trend forecasting
- [ ] Integrate locality-level data for finer predictions

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Built by Rahul as part of B.Tech CSE (Data Science) coursework, 2025.*
