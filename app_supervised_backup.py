import streamlit as st
import requests


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏦 Loan Approval Predictor")

st.write(
    "Enter the applicant details below to predict "
    "the loan approval status."
)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("Applicant Details")


income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)


credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650,
    step=1
)


loan_amount = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=25000.0,
    step=1000.0
)


employment_years = st.number_input(
    "Employment Years",
    min_value=0,
    value=5,
    step=1
)


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

if st.button("🔍 Predict Loan Approval"):

    try:

        # Send applicant details to FastAPI

        response = requests.post(
            "http://127.0.0.1:8000/predict",

            params={
                "income": income,
                "credit_score": credit_score,
                "loan_amount": loan_amount,
                "employment_years": employment_years
            }
        )


        # --------------------------------------------------
        # CHECK RESPONSE
        # --------------------------------------------------

        if response.status_code == 200:

            result = response.json()

            status = result["loan_status"]

            probability = result["approval_probability"]


            # --------------------------------------------------
            # DISPLAY RESULT
            # --------------------------------------------------

            st.subheader("Prediction Result")


            if status == "Approved":

                st.success("✅ Loan Approved")

            else:

                st.error("❌ Loan Rejected")


            st.metric(
                "Approval Probability",
                f"{probability}%"
            )


        else:

            st.error(
                f"Backend returned an error: {response.status_code}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the FastAPI backend. "
            "Please make sure the backend is running."
        )