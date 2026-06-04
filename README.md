# Credit Card Fraud Detection

## Project Overview

This project uses Machine Learning to detect fraudulent credit card transactions using the Credit Card Fraud Detection Dataset.

## Technologies Used

- Python
- Pandas
- Scikit-Learn
- Matplotlib
- Joblib

## Dataset

The dataset is not included in this repository due to GitHub file size limitations.

Dataset source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Exploratory Data Analysis

- Checked missing values
- Analyzed class distribution
- Generated descriptive statistics
- Created feature and target variables

## Model

### Logistic Regression

Features were scaled using StandardScaler before training.

## Results

### Classification Report

| Metric | Fraud Class |
|----------|----------|
| Precision | 0.88 |
| Recall | 0.68 |
| F1-Score | 0.77 |

### Confusion Matrix

```text
[[56854     9]
 [   32    67]]