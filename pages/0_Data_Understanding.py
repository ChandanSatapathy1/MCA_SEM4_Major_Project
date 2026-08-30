import streamlit as st

st.set_page_config(page_title="Data Understanding", layout="wide")

st.title("Data Understanding")
st.write("This page explains each feature present in the dataset used for this project")

st.write("---")
st.subheader("Dataset Source")
st.write("Telco Customer Churn dataset from Kaggle, originally published by IBM Sample Data Sets")
st.write("Total records: 7043 customers (7032 after cleaning)")
st.write("Total original features: 19 (excluding customer ID and target column)")

st.write("---")
st.subheader("Feature Description")

col1, col2 = st.columns(2)

with col1:
    st.write("**Personal Information**")
    st.write("gender: Customer's gender (Male/Female)")
    st.write("SeniorCitizen: Whether customer is a senior citizen (1) or not (0)")
    st.write("Partner: Whether customer has a partner")
    st.write("Dependents: Whether customer has dependents")
    st.write("tenure: Number of months customer has stayed with the company")

    st.write("")
    st.write("**Services Subscribed**")
    st.write("PhoneService: Whether customer has phone service")
    st.write("MultipleLines: Whether customer has multiple phone lines")
    st.write("InternetService: Type of internet service (DSL, Fiber optic, No)")
    st.write("OnlineSecurity: Whether customer has online security add-on")
    st.write("OnlineBackup: Whether customer has online backup add-on")

with col2:
    st.write("**Additional Services**")
    st.write("DeviceProtection: Whether customer has device protection add-on")
    st.write("TechSupport: Whether customer has tech support add-on")
    st.write("StreamingTV: Whether customer has streaming TV service")
    st.write("StreamingMovies: Whether customer has streaming movies service")

    st.write("")
    st.write("**Billing Information**")
    st.write("Contract: Type of contract (Month-to-month, One year, Two year)")
    st.write("PaperlessBilling: Whether customer uses paperless billing")
    st.write("PaymentMethod: Method of payment used by customer")
    st.write("MonthlyCharges: Amount charged to customer monthly")
    st.write("TotalCharges: Total amount charged to customer")

st.write("---")
st.subheader("Target Variable")
st.write("Churn: Whether the customer left the company within the last month (Yes/No). This is what the model is trained to predict.")