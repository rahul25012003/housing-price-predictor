"""
Streamlit Dashboard for Indian Housing Price Predictor
========================================================
Interactive UI with prediction form, feature importance, and EDA tabs.

Run: streamlit run app/streamlit_app.py
"""

import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

st.set_page_config(page_title="Housing Price Predictor", page_icon="🏠", layout="wide")

# --- Load model and transformers ---
@st.cache_resource
def load_artifacts():
    model = joblib.load(config.BEST_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    encoders = joblib.load(config.ENCODER_PATH)
    return model, scaler, encoders

try:
    model, scaler, encoders = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.warning("Model not found. Please run the training pipeline first.")

st.title("Indian Housing Price Predictor")
st.markdown("Predict house prices across major Indian metro cities using ML")

tab1, tab2, tab3 = st.tabs(["Predict", "Feature Importance", "EDA"])

# ======= TAB 1: Prediction =======
with tab1:
    st.header("Price Prediction")

    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", config.CITIES)
        area = st.number_input("Area (sqft)", min_value=200, max_value=10000, value=1200, step=50)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1, 5, 2)

    with col2:
        prop_type = st.selectbox("Property Type", config.PROPERTY_TYPES)
        furnishing = st.selectbox("Furnishing", config.FURNISHING_OPTIONS)
        age = st.number_input("Age of Property (years)", min_value=0, max_value=50, value=5)
        floor_num = st.number_input("Floor Number", min_value=0, max_value=30, value=3)
        parking = st.selectbox("Parking Spaces", [0, 1, 2])

    if st.button("Predict Price", type="primary") and model_loaded:
        # build input dataframe
        price_per_sqft = 0  # placeholder, will be recalculated
        room_density = bedrooms / area * 1000

        input_data = {
            "area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "age_years": age, "floor": floor_num, "parking": parking,
            "price_per_sqft": price_per_sqft, "room_density": room_density,
        }

        # encode categoricals
        input_data["city_encoded"] = encoders["city"].transform([city])[0]
        input_data["property_type_encoded"] = encoders["property_type"].transform([prop_type])[0]
        input_data["furnishing_encoded"] = encoders["furnishing"].transform([furnishing])[0]

        # scale numerical
        num_vals = np.array([[area, bedrooms, bathrooms, age, floor_num, parking,
                              price_per_sqft, room_density]])
        scaled = scaler.transform(num_vals)
        for i, col in enumerate(config.NUMERICAL_COLS + config.DERIVED_COLS):
            input_data[col] = scaled[0][i]

        # predict
        input_df = pd.DataFrame([input_data])
        # make sure columns match training order
        feature_cols = [c for c in pd.read_csv(config.PROCESSED_DATA_PATH, nrows=0).columns
                        if c != config.TARGET_COL]
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        prediction = model.predict(input_df)[0]
        # rough confidence interval (not proper, just for display)
        lower = prediction * 0.88
        upper = prediction * 1.12

        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted Price", f"₹ {prediction:.2f} Lakhs")
        col_b.metric("Lower Estimate", f"₹ {lower:.2f} Lakhs")
        col_c.metric("Upper Estimate", f"₹ {upper:.2f} Lakhs")

        st.info(f"Prediction for a {bedrooms}BHK {prop_type} in {city}, {area} sqft, {furnishing}")

# ======= TAB 2: Feature Importance =======
with tab2:
    st.header("Feature Importance")

    if model_loaded and hasattr(model, "feature_importances_"):
        try:
            feature_cols = [c for c in pd.read_csv(config.PROCESSED_DATA_PATH, nrows=0).columns
                            if c != config.TARGET_COL]
            importances = model.feature_importances_
            imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
            imp_df = imp_df.sort_values("Importance", ascending=True)

            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title="Feature Importance (XGBoost)", color="Importance",
                         color_continuous_scale="viridis")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading feature importance: {e}")
    else:
        st.info("Train a tree-based model first to see feature importances.")

# ======= TAB 3: EDA =======
with tab3:
    st.header("Exploratory Data Analysis")

    # show saved plots if they exist
    plot_files = {
        "Distributions": "distributions.png",
        "Correlation Heatmap": "correlation_heatmap.png",
        "City Analysis": "city_analysis.png",
        "Area vs Price": "area_vs_price.png",
        "Categorical Analysis": "categorical_analysis.png",
    }

    for title, fname in plot_files.items():
        fpath = os.path.join(config.OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            st.subheader(title)
            st.image(fpath)
        else:
            st.caption(f"[{title}] Run notebooks/01_eda.py to generate this plot.")

st.markdown("---")
st.caption("Built by Rahul | B.Tech CSE (Data Science) | 2025")
