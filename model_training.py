import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# loading dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# fixing TotalCharges column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# dropping customerID, not useful for prediction
df = df.drop('customerID', axis=1)

# converting Churn to numbers
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# encoding all other text columns
df = pd.get_dummies(df, drop_first=True)

# separating features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# handling class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# scaling data (needed for logistic regression)
scaler = StandardScaler()
X_train_smote_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# training logistic regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_smote_scaled, y_train_smote)
log_pred = log_model.predict(X_test_scaled)

print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("Precision:", precision_score(y_test, log_pred))
print("Recall:", recall_score(y_test, log_pred))
print("F1 Score:", f1_score(y_test, log_pred))
print(confusion_matrix(y_test, log_pred))

# training random forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_smote, y_train_smote)
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))

# doing cross validation properly - smote should happen inside each fold, not before
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import Pipeline

print("\nCross Validation for Random Forest")

cv_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])

cv_scores = cross_val_score(cv_pipeline, X_train, y_train, cv=5)
print("CV Accuracy scores:", cv_scores)
print("Average CV Accuracy:", cv_scores.mean())

# doing hyperparameter tuning for random forest

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1')
grid_search.fit(X_train_smote, y_train_smote)

print("\nBest parameters found:", grid_search.best_params_)
print("Best F1 score during search:", grid_search.best_score_)

# using the tuned model for final predictions
best_rf_model = grid_search.best_estimator_
best_rf_pred = best_rf_model.predict(X_test)

print("\nTuned Random Forest Results")
print("Accuracy:", accuracy_score(y_test, best_rf_pred))
print("Precision:", precision_score(y_test, best_rf_pred))
print("Recall:", recall_score(y_test, best_rf_pred))
print("F1 Score:", f1_score(y_test, best_rf_pred))
print(confusion_matrix(y_test, best_rf_pred))



# training xgboost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_smote, y_train_smote)
xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Precision:", precision_score(y_test, xgb_pred))
print("Recall:", recall_score(y_test, xgb_pred))
print("F1 Score:", f1_score(y_test, xgb_pred))
print(confusion_matrix(y_test, xgb_pred))

# saving models
joblib.dump(log_model, 'log_model.pkl')
joblib.dump(best_rf_model, 'rf_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# saving column names too, needed later for the app
joblib.dump(X.columns.tolist(), 'model_columns.pkl')

print("\nall models saved")