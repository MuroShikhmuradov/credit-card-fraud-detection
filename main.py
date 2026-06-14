import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)

model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(
    n_estimators=10,
    random_state=1918
)

rf_model.fit(X_train, y_train)
rf_predict = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predict)

print("Accuracy: ", rf_accuracy)
print(classification_report(y_test, rf_predict))
print(confusion_matrix(y_test, rf_predict))

model.fit(X_train_scaled, y_train)
y_prediction = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_prediction)

print("Accuracy: ", accuracy)
print(classification_report(y_test, y_prediction))
print(confusion_matrix(y_test, y_prediction))

importance = pd.DataFrame({
    "Feature" : X.columns,
    "Importance" : rf_model.feature_importances_,
})

importance = importance.sort_values("Importance", ascending=False)

top10 = importance.head(10)

plt.figure(figsize=(10, 5))
plt.bar(top10["Feature"], top10["Importance"])

plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")

plt.savefig("outputs/feature_importance.png")