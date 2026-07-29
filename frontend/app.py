import streamlit as st

from api import predict

from utils import risk_color

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Customer Churn Prediction")

st.markdown(
"""
Predict whether a telecom customer is likely to churn
using a Gradient Boosting machine learning model.
"""
)

st.sidebar.header("About")

st.sidebar.write(
"""
Algorithm:
Gradient Boosting

Dataset:
IBM Telco Customer Churn (Enhanced)

Target:
Customer Churn
"""
)

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0,1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes","No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes","No"]
    )

    tenure = st.slider(
        "Tenure Months",
        0,
        72,
        24
    )

with col2:

    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1500.0
    )

    cltv = st.number_input(
        "CLTV",
        min_value=0,
        value=4000
    )

if st.button("Predict Churn"):
    customer = {

    "Gender": gender,
    "Senior_Citizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "Tenure_Months": tenure,
    "Internet_Service": internet,
    "Contract": contract,
    "Monthly_Charges": monthly,
    "Total_Charges": total,
    "Paperless_Billing": "Yes",
    "Payment_Method": "Electronic check",
    "CLTV": cltv,
}
    result = predict(customer)

    st.metric(
    "Churn Probability",
    f"{result['churn_probability']:.1%}"
)
    if result["prediction"]:

        st.error("Customer is likely to churn")

    else:

        st.success("Customer is likely to stay")

    color = risk_color(
    result["risk_level"]
    )

    st.markdown(
    f"""
    ### Risk Level

    <span style='color:{color};
    font-size:24px;
    font-weight:bold;'>

    {result["risk_level"]}

    </span>
    """,
    unsafe_allow_html=True,
    )

    st.progress(
        result["churn_probability"]
    )

    if result["risk_level"] == "High":

        st.warning(
    """
    Recommended Action

    • Contact customer immediately.

    • Offer contract discount.

    • Assign retention specialist.

    • Review service issues.
    """
    )

    elif result["risk_level"] == "Medium":

        st.info(
    """
    Recommended Action

    Monitor customer closely.

    Offer loyalty rewards.
    """
    )

    else:

        st.success(
    """
    Customer appears loyal.

    Continue regular engagement.
    """
    )

st.divider()

st.caption(
"""
Customer Churn Prediction

Developed using

FastAPI + Streamlit + Scikit-Learn
"""
)