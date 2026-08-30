import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("Data Analysis Dashboard")

# loading dataset again for charts
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

# basic dataset summary
total_customers = df.shape[0]
churn_rate = round((df['Churn'].value_counts()['Yes'] / total_customers) * 100, 2)

summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    st.metric(label="Total Customers", value=total_customers)
with summary_col2:
    st.metric(label="Churn Rate", value=str(churn_rate) + "%")
    
# sidebar to choose which section to view
st.sidebar.header("Dashboard Options")
section = st.sidebar.radio("Select View", ["EDA Charts", "Model Comparison"])

if section == "EDA Charts":

    st.subheader("Customers Churned vs Not Churned")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        df['Churn'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_xlabel("Churn")
        ax1.set_ylabel("Number of customers")
        st.pyplot(fig1)

    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        pd.crosstab(df['Contract'], df['Churn']).plot(kind='bar', ax=ax2)
        ax2.set_xlabel("Contract Type")
        ax2.set_ylabel("Number of customers")
        ax2.set_title("Churn by Contract Type")
        st.pyplot(fig2)

    st.subheader("Tenure Distribution of Churned Customers")
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    df[df['Churn']=='Yes']['tenure'].plot(kind='hist', bins=30, ax=ax3)
    ax3.set_xlabel("Tenure (months)")
    st.pyplot(fig3)

else:

    st.subheader("Model Performance Comparison")

    # these accuracy numbers are from model_training.py output
    models = ["Logistic Regression", "Random Forest", "XGBoost"]

    accuracy_data = pd.DataFrame({"Accuracy": [0.7704, 0.7754, 0.7619]}, index=models)
    precision_data = pd.DataFrame({"Precision": [0.5603, 0.5763, 0.5506]}, index=models)
    recall_data = pd.DataFrame({"Recall": [0.6337, 0.5856, 0.5668]}, index=models)
    f1_data = pd.DataFrame({"F1 Score": [0.5947, 0.5809, 0.5586]}, index=models)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("Accuracy")
        st.bar_chart(accuracy_data)

        st.write("Recall")
        st.bar_chart(recall_data)

    with chart_col2:
        st.write("Precision")
        st.bar_chart(precision_data)

        st.write("F1 Score")
        st.bar_chart(f1_data)

    st.write("---")
    st.subheader("Confusion Matrix (Random Forest)")

    cm = np.array([[872, 161], [155, 219]])
    cm_df = pd.DataFrame(cm, index=["Actual: No", "Actual: Yes"], columns=["Predicted: No", "Predicted: Yes"])
    st.table(cm_df)

    st.write("---")
    st.subheader("Feature Importance (Random Forest)")

    rf_model = joblib.load('rf_model.pkl')
    model_columns = joblib.load('model_columns.pkl')

    importance_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": rf_model.feature_importances_
    })
    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)
    importance_df = importance_df.set_index("Feature")

    st.bar_chart(importance_df)

    st.write("---")
st.subheader("Model Validation Notes")

with st.container(border=True):
    val_col1, val_col2 = st.columns(2)

    with val_col1:
        st.write("**Cross Validation**")
        st.write("Method: 5-Fold Cross Validation")
        st.write("Average CV Accuracy: 78.58%")

    with val_col2:
        st.write("**Hyperparameter Tuning**")
        st.write("Method: GridSearchCV")
        st.write("Best Parameters: n_estimators=200, max_depth=None, min_samples_split=2")