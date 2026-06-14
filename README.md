# Credit Card Fraud Detection

## Project Overview

This project uses Machine Learning to detect fraudulent credit card transactions using the Credit Card Fraud Detection Dataset.

The objective of the project is to classify transactions as either legitimate or fraudulent and compare the performance of multiple machine learning models.

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* XGBoost
* Matplotlib
* Joblib

## Dataset

The dataset used in this project is the Credit Card Fraud Detection Dataset.

Dataset source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Due to GitHub file size limitations, the dataset is not included in this repository.

To run the project, download the dataset and place it inside:

```text
data/creditcard.csv
```

## Exploratory Data Analysis

The following EDA steps were performed:

* Loaded the dataset using Pandas
* Inspected dataset structure and data types
* Generated descriptive statistics
* Checked missing values
* Analyzed class distribution
* Created feature and target variables

### Dataset Information

* Total Transactions: 284,807
* Legitimate Transactions: 284,315
* Fraudulent Transactions: 492

The dataset is highly imbalanced, making fraud detection a challenging classification problem.

## Data Preprocessing

### Feature and Target Creation

Features:

```text
Time
V1 - V28
Amount
```

Target:

```text
Class
```

Where:

```text
0 = Legitimate Transaction
1 = Fraudulent Transaction
```

### Train-Test Split

The dataset was split into:

```text
80% Training Data
20% Testing Data
```

using:

```python
train_test_split(
    test_size=0.2,
    random_state=1918
)
```

### Feature Scaling

StandardScaler was applied before training the Logistic Regression model.

## Models Used

### Logistic Regression

Parameters:

```python
LogisticRegression(
    max_iter=1000
)
```

### Random Forest Classifier

Parameters:

```python
RandomForestClassifier(
    n_estimators=10,
    random_state=1918
)
```

### XGBoost Classifier

Parameters:

```python
XGBClassifier(
    n_estimators=10,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=1918
)
```

## Model Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

# Logistic Regression Results

### Fraud Class Metrics

| Metric    | Score |
| --------- | ----- |
| Precision | 0.88  |
| Recall    | 0.68  |
| F1-Score  | 0.77  |

### Confusion Matrix

```text
[[56854     9]
 [   32    67]]
```

### Interpretation

* 56,854 legitimate transactions were correctly classified.
* 67 fraudulent transactions were correctly detected.
* 9 legitimate transactions were incorrectly classified as fraud.
* 32 fraudulent transactions were missed.

---

# Random Forest Results

### Accuracy

**99.97%**

### Fraud Class Metrics

| Metric    | Score |
| --------- | ----- |
| Precision | 0.97  |
| Recall    | 0.84  |
| F1-Score  | 0.90  |

### Confusion Matrix

```text
[[56860     3]
 [   16    83]]
```

### Interpretation

* 56,860 legitimate transactions were correctly classified.
* 83 fraudulent transactions were correctly detected.
* Only 3 legitimate transactions were incorrectly flagged as fraud.
* 16 fraudulent transactions were missed.

### Conclusion

The Random Forest model significantly improved fraud detection performance compared to Logistic Regression.

---

# XGBoost Results

### Accuracy

**99.95%**

### Fraud Class Metrics

| Metric    | Score |
| --------- | ----- |
| Precision | 0.91  |
| Recall    | 0.79  |
| F1-Score  | 0.84  |

### Confusion Matrix

```text
[[56855     8]
 [   21    78]]
```

### Interpretation

* 56,855 legitimate transactions were correctly classified.
* 78 fraudulent transactions were correctly detected.
* 8 legitimate transactions were incorrectly flagged as fraud.
* 21 fraudulent transactions were missed.

### Conclusion

XGBoost achieved strong performance and successfully detected the majority of fraudulent transactions while maintaining a low false positive rate.

---

# Model Comparison

| Model               | Precision | Recall | F1-Score |
| ------------------- | --------- | ------ | -------- |
| Logistic Regression | 0.88      | 0.68   | 0.77     |
| Random Forest       | 0.97      | 0.84   | 0.90     |
| XGBoost             | 0.91      | 0.79   | 0.84     |

### Best Performing Model

Based on Precision, Recall, F1-Score, and the number of missed fraudulent transactions, **Random Forest** achieved the best overall performance on this dataset.

---

# Feature Importance Analysis

Feature importance analysis was performed using the Random Forest model.

The top features contributing to fraud detection were visualized and saved as:

```text
outputs/feature_importance.png
```

This analysis helps explain which variables had the greatest impact on model predictions.

---

# Model Persistence

All trained models were saved using Joblib.

Saved files:

```text
models/lr_model.pkl
models/rf_model.pkl
models/xgb_model.pkl
models/scaler.pkl
```

This allows the models to be loaded later without retraining.

---

# Project Structure

```text
FraudDetection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── lr_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   └── scaler.pkl
│
├── outputs/
│   └── feature_importance.png
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Future Improvements

* Hyperparameter Tuning
* Cross Validation
* SMOTE for Handling Class Imbalance
* Advanced XGBoost Optimization
* Real-Time Fraud Detection
* Model Deployment using Flask or FastAPI

---

# Final Conclusion

This project demonstrates a complete machine learning classification workflow for credit card fraud detection.

The project includes:

* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Feature Scaling
* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier
* Feature Importance Analysis
* Model Evaluation
* Model Persistence with Joblib

Among all tested models, **Random Forest achieved the best fraud detection performance**, with the highest Precision, Recall, and F1-Score while minimizing missed fraudulent transactions.
