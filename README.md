# House Price Prediction ML

A foundational machine learning project that predicts house prices from property features (area, bedrooms, age) using **Linear Regression**, implemented end-to-end with `pandas` and `scikit-learn`.

## Overview

This project demonstrates the core supervised learning workflow for a regression problem:

1. Structuring raw data into a `pandas` DataFrame
2. Splitting features and target variable
3. Creating a train/test split
4. Training a Linear Regression model
5. Evaluating performance with Mean Absolute Error (MAE)
6. Running inference on unseen data

## Dataset

The dataset is a small, illustrative housing dataset (5 records) defined directly in code, with the following features:

| Feature    | Description                          |
|------------|---------------------------------------|
| `Area`     | Property size (sq. ft.)               |
| `Bedrooms` | Number of bedrooms                    |
| `Age`      | Age of the property (years)           |
| `Price`    | Sale price (target variable, in $)    |

| Area | Bedrooms | Age | Price   |
|------|----------|-----|---------|
| 1000 | 2        | 10  | 200,000 |
| 1500 | 3        | 5   | 300,000 |
| 1800 | 3        | 8   | 350,000 |
| 2400 | 4        | 2   | 500,000 |
| 3000 | 5        | 1   | 650,000 |

> **Note:** This dataset is intentionally small and synthetic, built to clearly demonstrate the ML pipeline mechanics. See [Limitations](#limitations--future-work) below for plans to scale this up with a real-world dataset.

## Tech Stack

- **Python 3**
- **pandas** — data structuring and manipulation
- **scikit-learn** — `train_test_split`, `LinearRegression`, `mean_absolute_error`

## Methodology

- Features (`X`): `Area`, `Bedrooms`, `Age`
- Target (`y`): `Price`
- Split: 80% train / 20% test (`train_test_split`, `random_state=42`)
- Model: `sklearn.linear_model.LinearRegression`
- Evaluation metric: Mean Absolute Error (MAE)

## Results

Running the pipeline on the held-out test split produced:

| Metric                        | Value          |
|--------------------------------|---------------|
| Mean Absolute Error (MAE)      | **$50,000**   |
| Test set size                  | 1 sample (20% of 5 records) |

**Inference example** — predicting the price of a new, unseen property:

| Area (sq. ft.) | Bedrooms | Age (years) | Predicted Price |
|-----------------|----------|-------------|------------------|
| 2000            | 3        | 4           | **$350,000**     |

The fitted model learned `Bedrooms` as the dominant predictor of price in this dataset, with `Area` and `Age` contributing negligibly — a direct consequence of the small sample size, where all three features happen to be strongly correlated with each other. This is discussed further below.

## How to Run

```bash
pip install pandas scikit-learn
python "house prediction.py"
```

## Limitations & Future Work

This project is a deliberately minimal, from-scratch implementation to demonstrate the ML workflow clearly. Honest next steps to make it production-grade:

- **Replace the synthetic dataset** with a real-world housing dataset (e.g., Kaggle's Ames Housing or California Housing dataset) with hundreds/thousands of records
- **Address multicollinearity** between `Area`, `Bedrooms`, and `Age` using correlation analysis, VIF checks, or regularization (Ridge/Lasso)
- **Use k-fold cross-validation** instead of a single train/test split for a more reliable performance estimate
- **Compare additional models** (Random Forest, Gradient Boosting/XGBoost) against the Linear Regression baseline
- **Add feature scaling and engineering** (e.g., price per sq. ft., location-based features)
- **Track experiments** with metrics like R², RMSE, and MAPE alongside MAE

## Author

**Aditya Namdev**
B.Tech (AI & ML) | [GitHub](https://github.com/adityanamdev77) | [LeetCode](https://leetcode.com/u/adityanamdev24/) | [Portfolio](https://aditya549.lovable.app)
