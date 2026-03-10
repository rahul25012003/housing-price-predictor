"""
Model helper functions — loading, saving, evaluation metrics.
"""

import os
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def save_model(model, filepath):
    """Save a trained model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load a saved model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model found at {filepath}")
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculate and return regression metrics.
    Returns dict with R2, RMSE, MAE.
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    metrics = {"model": model_name, "r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4)}

    print(f"\n--- {model_name} ---")
    print(f"  R2 Score : {metrics['r2']}")
    print(f"  RMSE     : {metrics['rmse']}")
    print(f"  MAE      : {metrics['mae']}")

    return metrics


def get_feature_importance(model, feature_names):
    """Extract feature importances from tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        print("[WARN] Model doesn't expose feature importances")
        return None

    # sort by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [(feature_names[i], round(importances[i], 4)) for i in indices]

    return sorted_features


def print_comparison_table(results_list):
    """Pretty-print model comparison results."""
    print("\n" + "=" * 55)
    print(f"{'Model':<25} {'R2':>8} {'RMSE':>10} {'MAE':>10}")
    print("-" * 55)
    for r in sorted(results_list, key=lambda x: x["r2"], reverse=True):
        print(f"{r['model']:<25} {r['r2']:>8.4f} {r['rmse']:>10.4f} {r['mae']:>10.4f}")
    print("=" * 55)
