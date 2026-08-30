import pandas as pd

# reading the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# checking first few rows
print(df.head())

# checking size of dataset
print(df.shape)

# checking column names
print(df.columns)

# checking info - data types and missing values
print(df.info())

# checking for missing values column by column
print(df.isnull().sum())

# checking how many churned vs not churned
print(df['Churn'].value_counts())

# checking data type of TotalCharges
print(df['TotalCharges'].dtype)

# checking Churn counts
print(df['Churn'].value_counts())

# converting TotalCharges to numbers, invalid ones become NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# now checking missing values again
print(df.isnull().sum())

# dropping rows where TotalCharges is missing
df = df.dropna(subset=['TotalCharges'])

# confirming it's fixed
print(df.isnull().sum())
print(df.shape)

# dropping customerID column, not useful for prediction
df = df.drop('customerID', axis=1)

import matplotlib.pyplot as plt

# plotting churn count
df['Churn'].value_counts().plot(kind='bar')
plt.title('Customers Churned vs Not Churned')
plt.xlabel('Churn')
plt.ylabel('Number of customers')
plt.show()

# churn by contract type
pd.crosstab(df['Contract'], df['Churn']).plot(kind='bar')
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of customers')
plt.show()

# churn by tenure (how long they've been a customer)
df[df['Churn']=='Yes']['tenure'].plot(kind='hist', bins=30)
plt.title('Tenure Distribution of Churned Customers')
plt.xlabel('Tenure (months)')
plt.show()

# converting Churn column to numbers (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# converting all other text columns to numbers using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

print(df.head())
print(df.shape)

from sklearn.model_selection import train_test_split

# separating input features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(y_train.value_counts())
print(y_train_smote.value_counts())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

#1st Model Logistic Regression

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_smote_scaled, y_train_smote)

log_pred = log_model.predict(X_test_scaled)

print(log_pred[:20])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("Accuracy:", accuracy_score(y_test, log_pred))
print("Precision:", precision_score(y_test, log_pred))
print("Recall:", recall_score(y_test, log_pred))
print("F1 Score:", f1_score(y_test, log_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_pred))
print("\nClassification Report:")
print(classification_report(y_test, log_pred))

#2nd Model Randomforest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

rf_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

#3rd Model XGBoost

from xgboost import XGBClassifier

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_smote, y_train_smote)

xgb_pred = xgb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Precision:", precision_score(y_test, xgb_pred))
print("Recall:", recall_score(y_test, xgb_pred))
print("F1 Score:", f1_score(y_test, xgb_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))

import joblib

# saving all 3 models
joblib.dump(log_model, 'log_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')

# saving the scaler too (needed for logistic regression predictions later)
joblib.dump(scaler, 'scaler.pkl')

print("all models saved")