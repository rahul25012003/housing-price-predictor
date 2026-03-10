"""
01 - Exploratory Data Analysis
================================
Generates visualizations and statistical summaries for the housing dataset.
Saves all plots to outputs/ directory.

Run: python notebooks/01_eda.py
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import load_data

# setup
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")

print("=" * 50)
print("  EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# load
df = load_data()
print(f"\nDataset shape: {df.shape}")

# --- 1. Basic Stats ---
print("\n--- Statistical Summary ---")
print(df.describe().round(2))

print("\n--- Data Types ---")
print(df.dtypes)

# --- 2. Missing Value Analysis ---
print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "percent": missing_pct})
missing_df = missing_df[missing_df["count"] > 0].sort_values("percent", ascending=False)
print(missing_df)

fig, ax = plt.subplots(figsize=(8, 4))
if len(missing_df) > 0:
    missing_df["percent"].plot(kind="barh", ax=ax, color="salmon")
    ax.set_title("Missing Values (%)")
    ax.set_xlabel("Percentage")
else:
    ax.text(0.5, 0.5, "No missing values!", ha="center", va="center", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "missing_values.png"), dpi=150)
plt.close()
print("Saved: missing_values.png")

# --- 3. Price Distribution ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(df["price_lakhs"].dropna(), bins=50, color="steelblue", edgecolor="black", alpha=0.7)
axes[0].set_title("Price Distribution (Lakhs)")
axes[0].set_xlabel("Price")

axes[1].hist(df["area_sqft"].dropna(), bins=50, color="teal", edgecolor="black", alpha=0.7)
axes[1].set_title("Area Distribution (sqft)")
axes[1].set_xlabel("Area")

bedrooms_counts = df["bedrooms"].value_counts().sort_index()
axes[2].bar(bedrooms_counts.index, bedrooms_counts.values, color="coral", edgecolor="black")
axes[2].set_title("Bedroom Distribution")
axes[2].set_xlabel("Bedrooms")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "distributions.png"), dpi=150)
plt.close()
print("Saved: distributions.png")

# --- 4. Correlation Heatmap ---
numerical = df.select_dtypes(include=[np.number])
corr_matrix = numerical.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")

# --- 5. City-wise Price Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

city_avg = df.groupby("city")["price_lakhs"].mean().sort_values(ascending=True)
city_avg.plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("Average Price by City (Lakhs)")
axes[0].set_xlabel("Avg Price (Lakhs)")

sns.boxplot(data=df, x="city", y="price_lakhs", ax=axes[1], palette="Set2")
axes[1].set_title("Price Distribution by City")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "city_analysis.png"), dpi=150)
plt.close()
print("Saved: city_analysis.png")

# --- 6. Outlier Detection (IQR method) ---
print("\n--- Outlier Detection (IQR) ---")
outlier_cols = ["price_lakhs", "area_sqft", "bedrooms"]
for col in outlier_cols:
    data = df[col].dropna()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((data < lower) | (data > upper)).sum()
    print(f"  {col}: {n_outliers} outliers ({n_outliers/len(data)*100:.1f}%)")

# --- 7. Scatter: Area vs Price ---
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(df["area_sqft"], df["price_lakhs"], c=df["bedrooms"],
                     cmap="viridis", alpha=0.4, s=15, edgecolors="none")
plt.colorbar(scatter, label="Bedrooms")
ax.set_title("Area vs Price (colored by Bedrooms)")
ax.set_xlabel("Area (sqft)")
ax.set_ylabel("Price (Lakhs)")
plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "area_vs_price.png"), dpi=150)
plt.close()
print("Saved: area_vs_price.png")

# --- 8. Property Type & Furnishing ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df.groupby("property_type")["price_lakhs"].mean().sort_values().plot(
    kind="barh", ax=axes[0], color="mediumpurple")
axes[0].set_title("Avg Price by Property Type")

df.groupby("furnishing")["price_lakhs"].mean().sort_values().plot(
    kind="barh", ax=axes[1], color="mediumseagreen")
axes[1].set_title("Avg Price by Furnishing")

plt.tight_layout()
plt.savefig(os.path.join(config.OUTPUT_DIR, "categorical_analysis.png"), dpi=150)
plt.close()
print("Saved: categorical_analysis.png")

print("\n[DONE] EDA complete. Check outputs/ for all plots.")
