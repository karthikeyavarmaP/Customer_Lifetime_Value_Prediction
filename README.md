# Customer Lifetime Value Prediction & Segmentation

A machine learning project that analyzes customer purchasing behavior, segments customers using RFM analysis and K-Means clustering, and predicts future customer spending using regression models.

## Project Overview

The project uses historical retail transaction data to:

- Clean and preprocess customer transaction data
- Engineer RFM and behavioral features
- Segment customers using K-Means clustering
- Predict future customer spending
- Compare Linear Regression and Random Forest models
- Generate business insights for different customer groups

## Dataset

The project uses the **UCI Online Retail Dataset** containing transactions from a UK-based online retailer.

- Raw Transactions: **541,909**
- Cleaned Transactions: **397,884**
- Customers Analyzed: **3,317**

The dataset is available inside:

`data/Online Retail.xlsx`

## Data Preprocessing

The following preprocessing steps were performed:

- Removed transactions with missing Customer IDs
- Removed cancelled orders
- Removed transactions with non-positive quantity or unit price
- Created transaction value using:

`TotalPrice = Quantity × UnitPrice`

The data was divided chronologically into:

- **Historical Period:** Before September 1, 2011
- **Future Period:** September 1, 2011 onwards

Historical customer behavior was used to predict future spending.

## Feature Engineering

Customer-level features included:

- **Recency** – Days since the customer's most recent purchase
- **Frequency** – Number of unique purchases
- **Monetary** – Total historical spending
- **Average Order Value**
- **Total Quantity**
- **Unique Products Purchased**

## Customer Segmentation

RFM features were log-transformed and standardized before applying **K-Means clustering**.

Four customer segments were created:

| Segment | Customers | Avg Historical Spend | Avg Frequency | Avg Recency |
|---|---:|---:|---:|---:|
| High-Value / Loyal | 495 | 6492.83 | 11.43 | 18.81 |
| Regular / Mid-Value | 968 | 1368.66 | 3.18 | 83.49 |
| Recent / Potential | 468 | 686.37 | 2.25 | 15.03 |
| At-Risk / Low-Value | 1386 | 273.59 | 1.17 | 150.41 |

## Machine Learning Models

Two regression models were compared for future spending prediction:

### Linear Regression

- **MAE:** 813.51
- **RMSE:** 3866.81
- **R²:** 0.675

### Random Forest Regression

- **MAE:** 949.24
- **RMSE:** 4934.32
- **R²:** 0.470

Linear Regression achieved the better test-set performance.

## Business Insights

- **High-Value / Loyal:** Focus on retention and loyalty programs
- **Regular / Mid-Value:** Target with cross-selling and upselling strategies
- **Recent / Potential:** Encourage repeat purchases and increase purchase frequency
- **At-Risk / Low-Value:** Use re-engagement and targeted promotional campaigns

Historical monetary value showed the strongest relationship with predicted future spending within the Linear Regression model.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- K-Means Clustering
- Linear Regression
- Random Forest Regression

## Repository Structure

```text
Customer_Lifetime_Value_Prediction/
│
├── clv_analysis.py
├── README.md
│
├── data/
│   └── Online Retail.xlsx
│
└── outputs/
    ├── customer_segments.csv
    ├── predictions.csv
    ├── model_comparison.csv
    ├── business_summary.csv
    ├── customer_segments.png
    └── actual_vs_predicted.png
