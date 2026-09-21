from fastapi import FastAPI
import joblib
import pandas as pd


# --------------------------------------------------
# 1. CREATE FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(title="Loan Approval Prediction API")


# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------

model = joblib.load("loan_model.pkl")

scaler = joblib.load("preprocessor.pkl")


# --------------------------------------------------
# 3. HOME PAGE
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Loan Approval Prediction API is running"
    }


# --------------------------------------------------
# 4. LOAN PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict_loan(
    income: float,
    credit_score: float,
    loan_amount: float,
    employment_years: float
):

    # Create DataFrame from user input

    input_data = pd.DataFrame(
        [[
            income,
            credit_score,
            loan_amount,
            employment_years
        ]],
        columns=[
            "income",
            "credit_score",
            "loan_amount",
            "employment_years"
        ]
    )


    # Apply the same preprocessing
    # used during model training

    input_scaled = scaler.transform(input_data)


    # Make prediction

    prediction = model.predict(input_scaled)[0]


    # Get prediction probabilities

    probability = model.predict_proba(input_scaled)[0]


    # Convert 0/1 into readable result

    if prediction == 1:

        status = "Approved"

    else:

        status = "Rejected"


    # Return result

    return {
        "loan_status": status,
        "prediction": int(prediction),
        "approval_probability": round(
            float(probability[1]) * 100,
            2
        )
    }