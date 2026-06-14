# Credit Card Fraud Detection

## Project Overview

This project uses Machine Learning to detect fraudulent credit card transactions using the Credit Card Fraud Detection Dataset.

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Matplotlib
* Joblib

## Dataset

The dataset is not included in this repository due to GitHub file size limitations.

Dataset source:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Exploratory Data Analysis

* Checked missing values
* Analyzed class distribution
* Generated descriptive statistics
* Created feature and target variables

## Logistic Regression

Features were scaled using StandardScaler before training.

### Results

| Metric    | Fraud Class |
| --------- | ----------- |
| Precision | 0.88        |
| Recall    | 0.68        |
| F1-Score  | 0.77        |

### Confusion Matrix

```text
[[56854     9]
 [   32    67]]
```

### Interpretation

* 56,854 legitimate transactions were correctly classified.
* 67 fraudulent transactions were correctly detected.
* 9 legitimate transactions were incorrectly flagged as fraud.
* 32 fraudulent transactions were missed by the model.

---

## Random Forest Classifier

A Random Forest Classifier was trained and evaluated on the Credit Card Fraud Detection dataset.

### Accuracy

**99.97%**

### Classification Report (Fraud Class)

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
* 16 fraudulent transactions were missed by the model.

### Conclusion

The Random Forest Classifier achieved excellent performance in detecting fraudulent transactions. The model successfully identified the majority of fraud cases while maintaining a very low false positive rate.

The Random Forest model outperformed the Logistic Regression model by achieving higher Precision, Recall, and F1-Score while reducing the number of missed fraudulent transactions.