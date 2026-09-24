# Customer Segmentation & Retention Analysis

A churn prediction and RFM-based customer segmentation model built on the [Online Retail dataset](https://www.kaggle.com/datasets/vijayuv/onlineretail), combining churn risk and customer lifetime value (CLV) into a retention decision matrix.

## What it does

- Segments customers using RFM (Recency, Frequency, Monetary) scoring
- Predicts customer churn using a tuned XGBoost model
- Combines churn risk with CLV to recommend a retention action per customer (Reward, Retain, Nurture, or Let Go)

## Results

| Metric | Value |
|---|---|
| Model | XGBoost (tuned via RandomizedSearchCV) |
| Churn window | 60 days |
| CV ROC-AUC | 0.756 ± 0.008 |
| Test ROC-AUC | 0.762 |
| Test Accuracy | 0.71 (tuned threshold) |
| F1-score (churn class) | 0.75 |

Top churn predictors: purchase frequency and consistency of purchase timing (`Std_Gap_Days`) outweigh recency or monetary value — how often and how predictably a customer buys matters more than how much they've spent.

## Tech stack

pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn, plotly

## Notebook flow

1. Data cleaning
2. Time-based RFM & behavioral feature engineering
3. RFM segmentation (Champions, Loyal, At Risk, etc.)
4. CLV calculation
5. Churn model training, tuning, and threshold selection
6. Retention decision matrix
7. Visualizations

