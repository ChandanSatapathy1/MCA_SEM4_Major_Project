import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predict", layout="wide")

# loading saved models and scaler
log_model = joblib.load('log_model.pkl')
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Predict Customer Churn")
st.write("Enter customer details below to predict churn")

# model selector in sidebar
st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Random Forest", "XGBoost"])

# form for customer details, wrapped in a bordered container for separation
with st.container(border=True):

    with st.form("churn_form"):

        st.subheader("Customer Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Personal Info**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

        with col2:
            st.write("**Services**")
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            st.write("**Billing**")
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
            TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

        submitted = st.form_submit_button("Predict Churn")

if submitted:

    # putting all inputs into one row same as training columns
    input_dict = {}
    input_dict['gender'] = gender
    input_dict['SeniorCitizen'] = SeniorCitizen
    input_dict['Partner'] = Partner
    input_dict['Dependents'] = Dependents
    input_dict['tenure'] = tenure
    input_dict['PhoneService'] = PhoneService
    input_dict['MultipleLines'] = MultipleLines
    input_dict['InternetService'] = InternetService
    input_dict['OnlineSecurity'] = OnlineSecurity
    input_dict['OnlineBackup'] = OnlineBackup
    input_dict['DeviceProtection'] = DeviceProtection
    input_dict['TechSupport'] = TechSupport
    input_dict['StreamingTV'] = StreamingTV
    input_dict['StreamingMovies'] = StreamingMovies
    input_dict['Contract'] = Contract
    input_dict['PaperlessBilling'] = PaperlessBilling
    input_dict['PaymentMethod'] = PaymentMethod
    input_dict['MonthlyCharges'] = MonthlyCharges
    input_dict['TotalCharges'] = TotalCharges

    input_df = pd.DataFrame([input_dict])

    # encoding same way as training data
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # picking model based on sidebar choice
    if model_choice == "Logistic Regression":
        input_scaled = scaler.transform(input_encoded)
        prediction = log_model.predict(input_scaled)[0]
        probability = log_model.predict_proba(input_scaled)[0][1]
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(input_encoded)[0]
        probability = rf_model.predict_proba(input_encoded)[0][1]
    else:
        prediction = xgb_model.predict(input_encoded)[0]
        probability = xgb_model.predict_proba(input_encoded)[0][1]

    # deciding risk level
    if probability >= 0.7:
        risk = "High Risk"
    elif probability >= 0.4:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    st.write("")

    with st.container(border=True):
        st.subheader("Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.write("Model used")
            st.write(model_choice)

        with result_col2:
            st.write("Prediction")
            if prediction == 1:
                st.write("Customer will CHURN")
            else:
                st.write("Customer will STAY")

        with result_col3:
            st.write("Risk Level")
            st.write(risk)

        st.write("")
        st.metric(label="Churn Probability", value=str(round(probability * 100, 2)) + "%")