import matplotlib
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/creditcard.csv")

print("First 5 rows of credit card data:")
print(df.head())

print("\nData info:")
print(df.info())

print("\nStatistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

X = df.drop("Class", axis=1)
y = df["Class"]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1918)

print(X_train.shape)
print(X_test.shape)