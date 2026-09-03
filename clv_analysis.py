import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. SETUP
# ============================================================

os.makedirs("outputs", exist_ok=True)

DATA_PATH = "data/Online Retail.xlsx"

print("=" * 60)
print("CUSTOMER LIFETIME VALUE PREDICTION & SEGMENTATION")
print("=" * 60)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_excel(DATA_PATH)

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# 3. DATA CLEANING
# ============================================================

data = df.copy()

# Remove missing Customer IDs
data = data.dropna(subset=["CustomerID"])

# Convert CustomerID to integer
data["CustomerID"] = data["CustomerID"].astype(int)

# Remove cancelled transactions
data = data[
    ~data["InvoiceNo"].astype(str).str.startswith("C")
]

# Keep only valid purchases
data = data[
    (data["Quantity"] > 0) &
    (data["UnitPrice"] > 0)
]

# Create transaction value
data["TotalPrice"] = (
    data["Quantity"] * data["UnitPrice"]
)

print("Cleaned dataset shape:", data.shape)


# ============================================================
# 4. HISTORICAL / FUTURE SPLIT
# ============================================================

cutoff_date = pd.Timestamp("2011-09-01")

historical = data[
    data["InvoiceDate"] < cutoff_date
].copy()

future = data[
    data["InvoiceDate"] >= cutoff_date
].copy()

print("\nHistorical transactions:", len(historical))
print("Future transactions:", len(future))


# ============================================================
# 5. CUSTOMER FEATURE ENGINEERING
# ============================================================

customer_features = historical.groupby("CustomerID").agg(

    Recency=(
        "InvoiceDate",
        lambda x: (
            cutoff_date - x.max()
        ).days
    ),

    Frequency=(
        "InvoiceNo",
        "nunique"
    ),

    Monetary=(
        "TotalPrice",
        "sum"
    ),

    TotalQuantity=(
        "Quantity",
        "sum"
    ),

    UniqueProducts=(
        "StockCode",
        "nunique"
    )

).reset_index()

customer_features["AverageOrderValue"] = (
    customer_features["Monetary"] /
    customer_features["Frequency"]
)

print("\nCustomers analysed:", len(customer_features))


# ============================================================
# 6. RFM PREPARATION
# ============================================================

rfm = customer_features[
    [
        "Recency",
        "Frequency",
        "Monetary"
    ]
].copy()

# Reduce skew
rfm_log = np.log1p(rfm)

# Standardize
scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(
    rfm_log
)


# ============================================================
# 7. SILHOUETTE SCORE TEST
# ============================================================

print("\nSilhouette Scores")

silhouette_scores = {}

for k in range(2, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(rfm_scaled)

    score = silhouette_score(
        rfm_scaled,
        labels
    )

    silhouette_scores[k] = score

    print(
        f"K = {k} | "
        f"Silhouette Score = {score:.4f}"
    )


# ============================================================
# 8. CUSTOMER SEGMENTATION
# ============================================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

customer_features["Cluster"] = (
    kmeans.fit_predict(rfm_scaled)
)

segment_map = {
    0: "Regular / Mid-Value",
    1: "At-Risk / Low-Value",
    2: "Recent / Potential",
    3: "High-Value / Loyal"
}

customer_features["Segment"] = (
    customer_features["Cluster"]
    .map(segment_map)
)


# ============================================================
# 9. CLUSTER SUMMARY
# ============================================================

cluster_summary = (
    customer_features
    .groupby("Segment")
    .agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean")
    )
    .round(2)
)

print("\nCustomer Segment Summary")
print(cluster_summary)


# ============================================================
# 10. SEGMENT DISTRIBUTION PLOT
# ============================================================

segment_counts = (
    customer_features["Segment"]
    .value_counts()
)

plt.figure(figsize=(9, 5))

segment_counts.plot(
    kind="bar"
)

plt.title(
    "Customer Segment Distribution"
)

plt.xlabel(
    "Customer Segment"
)

plt.ylabel(
    "Number of Customers"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "outputs/customer_segments.png"
)

plt.show()


# ============================================================
# 11. CREATE FUTURE SPENDING TARGET
# ============================================================

future_spending = (
    future
    .groupby("CustomerID")["TotalPrice"]
    .sum()
    .reset_index()
)

future_spending.columns = [
    "CustomerID",
    "FutureSpend"
]

model_data = customer_features.merge(
    future_spending,
    on="CustomerID",
    how="left"
)

# No future purchase means future spend = 0
model_data["FutureSpend"] = (
    model_data["FutureSpend"]
    .fillna(0)
)


# ============================================================
# 12. MACHINE LEARNING FEATURES
# ============================================================

features = [
    "Recency",
    "Frequency",
    "Monetary",
    "AverageOrderValue",
    "TotalQuantity",
    "UniqueProducts"
]

X = model_data[features]

y = model_data["FutureSpend"]


# ============================================================
# 13. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
)

print(
    "\nTraining customers:",
    len(X_train)
)

print(
    "Testing customers:",
    len(X_test)
)


# ============================================================
# 14. LINEAR REGRESSION
# ============================================================

lr = LinearRegression()

lr.fit(
    X_train,
    y_train
)

lr_predictions = lr.predict(
    X_test
)

# Spending cannot be negative
lr_predictions = np.maximum(
    lr_predictions,
    0
)

lr_mae = mean_absolute_error(
    y_test,
    lr_predictions
)

lr_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        lr_predictions
    )
)

lr_r2 = r2_score(
    y_test,
    lr_predictions
)


# ============================================================
# 15. RANDOM FOREST REGRESSION
# ============================================================

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

rf_predictions = rf.predict(
    X_test
)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_predictions
    )
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


# ============================================================
# 16. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "MAE": [
        lr_mae,
        rf_mae
    ],

    "RMSE": [
        lr_rmse,
        rf_rmse
    ],

    "R²": [
        lr_r2,
        rf_r2
    ]

})

print("\nModel Comparison")
print(
    comparison.round(3)
)


# ============================================================
# 17. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    lr_predictions,
    alpha=0.5
)

maximum = max(
    y_test.max(),
    lr_predictions.max()
)

plt.plot(
    [0, maximum],
    [0, maximum],
    linestyle="--"
)

plt.xlabel(
    "Actual Future Spending"
)

plt.ylabel(
    "Predicted Future Spending"
)

plt.title(
    "Actual vs Predicted Customer Spending"
)

plt.tight_layout()

plt.savefig(
    "outputs/actual_vs_predicted.png"
)

plt.show()


# ============================================================
# 18. STANDARDIZED LINEAR REGRESSION COEFFICIENTS
# ============================================================

feature_scaler = StandardScaler()

X_train_scaled = (
    feature_scaler.fit_transform(
        X_train
    )
)

X_test_scaled = (
    feature_scaler.transform(
        X_test
    )
)

lr_scaled = LinearRegression()

lr_scaled.fit(
    X_train_scaled,
    y_train
)

standardized_coefficients = pd.DataFrame({

    "Feature": features,

    "StandardizedCoefficient":
        lr_scaled.coef_

})

standardized_coefficients[
    "AbsoluteCoefficient"
] = (
    standardized_coefficients[
        "StandardizedCoefficient"
    ].abs()
)

standardized_coefficients = (
    standardized_coefficients
    .sort_values(
        "AbsoluteCoefficient",
        ascending=False
    )
)

print(
    "\nStandardized Feature Coefficients"
)

print(
    standardized_coefficients.round(3)
)


# ============================================================
# 19. BUSINESS SUMMARY
# ============================================================

business_summary = (
    model_data
    .groupby("Segment")
    .agg(

        Customers=(
            "CustomerID",
            "count"
        ),

        AvgHistoricalSpend=(
            "Monetary",
            "mean"
        ),

        AvgFutureSpend=(
            "FutureSpend",
            "mean"
        ),

        AvgFrequency=(
            "Frequency",
            "mean"
        ),

        AvgRecency=(
            "Recency",
            "mean"
        )

    )
    .round(2)
)

print(
    "\nBusiness Segment Summary"
)

print(
    business_summary
)


# ============================================================
# 20. SAVE OUTPUT FILES
# ============================================================

customer_features.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

predictions = X_test.copy()

predictions[
    "ActualFutureSpend"
] = y_test.values

predictions[
    "PredictedFutureSpend"
] = lr_predictions

predictions.to_csv(
    "outputs/predictions.csv",
    index=False
)


comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

business_summary.to_csv(
    "outputs/business_summary.csv"
)


# ============================================================
# 21. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 60)

print(
    "PROJECT COMPLETED SUCCESSFULLY"
)

print("=" * 60)

print(
    f"Customers analysed: "
    f"{len(model_data)}"
)

print(
    "Customer segments: 4"
)

print("\nBest Model: Linear Regression")

print(
    f"MAE  : {lr_mae:.2f}"
)

print(
    f"RMSE : {lr_rmse:.2f}"
)

print(
    f"R²   : {lr_r2:.4f}"
)

print(
    "\nRandom Forest R²:",
    f"{rf_r2:.4f}"
)

print(
    "\nOutput files saved in "
    "the 'outputs' folder."
)
