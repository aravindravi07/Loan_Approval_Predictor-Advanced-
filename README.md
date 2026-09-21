# 🏦 Advanced Loan Approval Prediction System

## 📌 Project Overview

The **Advanced Loan Approval Prediction System** is a supervised machine learning project developed to predict whether a loan application is likely to be **Approved** or **Rejected** based on applicant-related financial and employment information.

The project is designed as a complete end-to-end machine learning application.

It starts from a structured loan dataset, performs data preparation and preprocessing, trains a machine learning model, evaluates the model, saves the trained model, exposes the prediction functionality through a FastAPI backend, and finally provides an interactive Streamlit web interface for users.

### Main Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression
- StandardScaler
- Joblib
- FastAPI
- Uvicorn
- Streamlit
- Matplotlib
- Seaborn

---

# 🎯 1. Project Objective

The main objective of this project is to build a machine learning-based system that can analyze loan application information and predict the corresponding loan status.

The system takes the following information as input:

- Applicant Income
- Credit Score
- Loan Amount
- Employment Years

Based on these inputs, the trained machine learning model predicts:

- **Approved**
- **Rejected**

The project also demonstrates how a machine learning model can be integrated into a complete software application using a backend API and an interactive frontend.

---

# 💡 2. Problem Statement

Loan approval is a classification problem where an application needs to be categorized into one of two possible outcomes.

Instead of manually analyzing the input values, a supervised machine learning model can learn patterns from previously available loan application data.

In this project:

```text
Applicant Information
        ↓
Machine Learning Model
        ↓
Loan Status Prediction
        ↓
Approved / Rejected
loan-approval-predictor/
│
├── app_supervised.py
│       └── Streamlit frontend
│
├── main_supervised.py
│       └── FastAPI backend
│
├── train_supervised.py
│       └── Model training and evaluation
│
├── loans.csv
│       └── Dataset
│
├── loan_model.pkl
│       └── Trained Logistic Regression model
│
├── preprocessor.pkl
│       └── Trained StandardScaler
│
├── requirements.txt
│       └── Required Python packages
│
├── README.md
│       └── Project documentation
│
├── .streamlit/
│   └── config.toml
│       └── Streamlit configuration
│
└── venv/
        └── Local Python virtual environment

        We built an end-to-end supervised machine learning application for loan approval prediction. We first prepared the loan dataset, selected four input features, split the data into training and testing sets, standardized the numerical features using StandardScaler, and trained a Logistic Regression classifier. We evaluated the model on unseen test data and saved both the trained model and preprocessing scaler using Joblib. We then created a FastAPI backend for prediction and a Streamlit frontend that allows users to enter loan details and view the prediction interactively.

loans.csv
→ Dataset used for training and testing.

train_supervised.py
→ Builds, trains, evaluates, and saves the ML model.

loan_model.pkl
→ Saved trained Logistic Regression model.

preprocessor.pkl
→ Saved StandardScaler used during preprocessing.

main_supervised.py
→ FastAPI backend that provides prediction functionality.

app_supervised.py
→ Streamlit frontend used by the user.

requirements.txt
→ Python dependencies required by the project.

.streamlit/config.toml
→ Streamlit application configuration.

README.md
→ Complete project documentation.