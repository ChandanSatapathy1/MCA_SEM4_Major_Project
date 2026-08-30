import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("Customer Churn Prediction System")

st.write("""
This is a major project built to predict customer churn using machine learning.
Customer churn refers to customers who stop using a company's service. Predicting 
churn in advance helps businesses take action to retain customers before they leave.
""")

st.write("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("About This Project")
    st.write("""
    This app uses a real telecom customer dataset to predict whether a customer 
    is likely to leave (churn) or stay, based on their account details and 
    services used.

    Three machine learning models were trained and compared:
    - Logistic Regression
    - Random Forest
    - XGBoost

    The Random Forest model was further improved using hyperparameter tuning 
    and validated using 5-fold cross validation.
    """)

with col2:
    st.subheader("Dataset Overview")
    st.metric(label="Total Customers", value="7,032")
    st.metric(label="Overall Churn Rate", value="26.58%")
    st.metric(label="Features Used", value="30 (after encoding)")

st.write("---")
st.subheader("How to Use This App")
st.write("""
Use the sidebar to navigate between pages:
- **Data Understanding**: learn about each feature in the dataset
- **Predict**: enter customer details and get a churn prediction
- **Dashboard**: view data analysis charts and model performance comparison
""")

st.write("---")
st.subheader("Tech Stack I Used")
st.write("Python, Pandas, Scikit-learn, XGBoost, Imbalanced-learn, Streamlit, Matplotlib")