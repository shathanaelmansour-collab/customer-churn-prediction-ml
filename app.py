import streamlit as st
import pandas as pd
import joblib

# Load trained model and fitted scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("Customer Churn Prediction")

st.write(
    "Enter customer information below to estimate "
    "the probability of customer churn."
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

tenure = st.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

gender = st.selectbox("Gender", ["Female", "Male"])

partner = st.selectbox("Partner", ["No", "Yes"])

dependents = st.selectbox("Dependents", ["No", "Yes"])

phone_service = st.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["No", "Yes", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["No", "Yes", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["No", "Yes", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["No", "Yes", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["No", "Yes", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["No", "Yes", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["No", "Yes", "No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

customer = pd.DataFrame({
    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],

    'gender_Male': [gender == "Male"],
    'Partner_Yes': [partner == "Yes"],
    'Dependents_Yes': [dependents == "Yes"],
    'PhoneService_Yes': [phone_service == "Yes"],

    'MultipleLines_No phone service': [
        multiple_lines == "No phone service"
    ],
    'MultipleLines_Yes': [multiple_lines == "Yes"],

    'InternetService_Fiber optic': [
        internet_service == "Fiber optic"
    ],
    'InternetService_No': [internet_service == "No"],

    'OnlineSecurity_No internet service': [
        online_security == "No internet service"
    ],
    'OnlineSecurity_Yes': [online_security == "Yes"],

    'OnlineBackup_No internet service': [
        online_backup == "No internet service"
    ],
    'OnlineBackup_Yes': [online_backup == "Yes"],

    'DeviceProtection_No internet service': [
        device_protection == "No internet service"
    ],
    'DeviceProtection_Yes': [device_protection == "Yes"],

    'TechSupport_No internet service': [
        tech_support == "No internet service"
    ],
    'TechSupport_Yes': [tech_support == "Yes"],

    'StreamingTV_No internet service': [
        streaming_tv == "No internet service"
    ],
    'StreamingTV_Yes': [streaming_tv == "Yes"],

    'StreamingMovies_No internet service': [
        streaming_movies == "No internet service"
    ],
    'StreamingMovies_Yes': [streaming_movies == "Yes"],

    'Contract_One year': [contract == "One year"],
    'Contract_Two year': [contract == "Two year"],

    'PaperlessBilling_Yes': [
        paperless_billing == "Yes"
    ],

    'PaymentMethod_Credit card (automatic)': [
        payment_method == "Credit card (automatic)"
    ],
    'PaymentMethod_Electronic check': [
        payment_method == "Electronic check"
    ],
    'PaymentMethod_Mailed check': [
        payment_method == "Mailed check"
    ]
})

# Ensure the input features exactly match the columns and order used during training
expected_columns = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

customer = customer[expected_columns]

numeric_columns = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

customer[numeric_columns] = scaler.transform(
    customer[numeric_columns]
)

if st.button("Predict Churn"):

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]

    if prediction:
        st.error("Customer is at risk of churn.")
    else:
        st.success("Customer is predicted to stay.")

    st.metric(
        "Churn Probability",
        f"{probability:.1%}"
    )