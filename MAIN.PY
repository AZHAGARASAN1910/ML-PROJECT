# ============================================
# HEART FAILURE PREDICTION
# Logistic Regression | Scikit-learn
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())

# EDA
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Target distribution
sns.countplot(x="DEATH_EVENT", data=df)
plt.title("Death Event Distribution")
plt.show()

# Select important features
features = [
    "time",
    "ejection_fraction",
    "serum_creatinine"
]

X = df[features]
y = df["DEATH_EVENT"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,cd D:
    fmt="d",
    cmap="Blues",
    xticklabels=["Survived", "Deceased"],
    yticklabels=["Survived", "Deceased"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Heart Failure Confusion Matrix")
plt.show()