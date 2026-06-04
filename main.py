import matplotlib
import pandas as pd
import joblib

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