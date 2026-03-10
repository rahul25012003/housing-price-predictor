"""
03 - Model Training & Comparison
==================================
Trains multiple regression models, compares performance via cross-validation,
tunes the best model with GridSearchCV, and saves the final model.

Run: python notebooks/03_model_training.py
"""

import os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.model_utils import evaluate_model, save_model, get_feature_importance, print_comparison_table

# try importing xgboost
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    print("[WARN] XGBoost not installed, skipping XGB model")
    HAS_XGB = False

print("=" * 50)
print("  MODEL TRAINING & COMPARISON")
print("=" * 50)

# --- 1. Load Processed Data ---
df = pd.read_csv(config.PROCESSED_DATA_PATH)
print(f"Loaded processed data: {df.shape}")

# prepare features and target
feature_cols = [c for c in df.columns if c != config.TARGET_COL]
X = df[feature_cols]
y = df[config.TARGET_COL]

print(f"Features: {len(feature_cols)}")
print(f"Target: {config.TARGET_COL}")
print(f"  Mean: {y.mean():.2f}, Std: {y.std():.2f}")

# --- 2. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
)
print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# --- 3. Define Models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(**config.RF_PARAMS),
    "Gradient Boosting": GradientBoostingRegressor(**config.GB_PARAMS),
}
if HAS_XGB:
    models["XGBoost"] = XGBRegressor(**config.XGB_PARAMS)

# --- 4. Train & Evaluate ---
results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.time()

    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, model_name=name)
    metrics["train_time"] = round(train_time, 2)

    # cross validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    metrics["cv_mean"] = round(cv_scores.mean(), 4)
    metrics["cv_std"] = round(cv_scores.std(), 4)
    print(f"  CV R2: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    print(f"  Train time: {train_time:.2f}s")

    results.append(metrics)
    trained_models[name] = model

# --- 5. Comparison Table ---
print_comparison_table(results)

# --- 6. Hyperparameter Tuning (best model) ---
# find the best model by R2
best_result = max(results, key=lambda x: x["r2"])
best_name = best_result["model"]
print(f"\nBest model before tuning: {best_name} (R2={best_result['r2']})")

# only tune XGBoost if it's the best (most likely)
if best_name == "XGBoost" and HAS_XGB:
    print("\nRunning GridSearchCV on XGBoost...")
    print("(this might take a few minutes)")

    xgb_grid = GridSearchCV(
        XGBRegressor(random_state=config.RANDOM_STATE),
        param_grid=config.XGB_GRID,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=0
    )
    xgb_grid.fit(X_train, y_train)

    print(f"Best params: {xgb_grid.best_params_}")
    print(f"Best CV R2: {xgb_grid.best_score_:.4f}")

    # evaluate tuned model
    y_pred_tuned = xgb_grid.best_estimator_.predict(X_test)
    tuned_metrics = evaluate_model(y_test, y_pred_tuned, model_name="XGBoost (Tuned)")

    best_model = xgb_grid.best_estimator_
else:
    print(f"\nUsing {best_name} without further tuning.")
    best_model = trained_models[best_name]

# --- 7. Feature Importance ---
print("\n--- Feature Importance ---")
importances = get_feature_importance(best_model, feature_cols)
if importances:
    for feat, imp in importances[:10]:
        print(f"  {feat}: {imp}")

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(15, len(importances))
    feats = [x[0] for x in importances[:top_n]][::-1]
    vals = [x[1] for x in importances[:top_n]][::-1]
    ax.barh(feats, vals, color="steelblue")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("Saved: feature_importance.png")

# --- 8. Save Best Model ---
save_model(best_model, config.BEST_MODEL_PATH)

# TODO: add residual plot analysis
# TODO: try stacking/blending models

print("\n[DONE] Model training complete.")
print(f"Best model saved to {config.BEST_MODEL_PATH}")
