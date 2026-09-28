import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

st.set_page_config(
    page_title="Advanced Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)
st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(99, 102, 241, 0.14),
                transparent 28%
            ),
            radial-gradient(
                circle at 10% 45%,
                rgba(14, 165, 233, 0.10),
                transparent 25%
            ),
            linear-gradient(
                135deg,
                #F4F7FC 0%,
                #EEF3FA 50%,
                #F8FAFD 100%
            );
    }


        max-width: 1450px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }



    h1 {
        font-size: 2.65rem !important;

        font-weight: 800 !important;

        letter-spacing: -1.2px;

        background:
            linear-gradient(
                90deg,
                #172554,
                #2563EB,
                #7C3AED
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color: transparent;
    }


    h2 {
        color: #172554 !important;

        font-weight: 750 !important;

        letter-spacing: -0.5px;
    }


    h3 {
        color: #1E293B !important;

        font-weight: 700 !important;
    }


    p {
        color: #475569;
    }



    .stCaption {
        color: #64748B !important;

        font-size: 0.95rem;
    }

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #0F172A 0%,
                #172554 48%,
                #1E1B4B 100%
            );

        border-right: 1px solid
        rgba(255,255,255,0.08);

        box-shadow:
            8px 0 30px
            rgba(15,23,42,0.15);
    }


    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }


    section[data-testid="stSidebar"] h1 {

        background: none;

        -webkit-text-fill-color: #FFFFFF;

        font-size: 1.35rem !important;

        letter-spacing: -0.4px;
    }


    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {

        color: #CBD5E1 !important;

        font-size: 0.78rem !important;

        text-transform: uppercase;

        letter-spacing: 1.5px;
    }


    section[data-testid="stSidebar"] hr {

        border-color:
        rgba(255,255,255,0.12);
    }



    div[data-testid="stMetric"] {

        position: relative;

        background:
            linear-gradient(
                145deg,
                rgba(255,255,255,0.98),
                rgba(248,250,252,0.94)
            ) !important;

        border: 1px solid
        rgba(148,163,184,0.25);

        border-radius: 20px;

        padding: 22px 22px 20px;

        min-height: 125px;

        box-shadow:
            0 10px 30px
            rgba(15,23,42,0.07);

        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease;
    }


    div[data-testid="stMetric"]:hover {

        transform:
            translateY(-6px);

        box-shadow:
            0 18px 38px
            rgba(37,99,235,0.15);
    }


    div[data-testid="stMetric"] label {

        color: #64748B !important;

        font-size: 0.82rem !important;

        font-weight: 650 !important;

        text-transform: uppercase;

        letter-spacing: 0.5px;
    }


    div[data-testid="stMetricValue"] {

        color: #0F172A !important;

        font-size: 2rem !important;

        font-weight: 800 !important;
    }

    div[data-testid="stHorizontalBlock"]
    div[data-testid="stMetric"]:nth-child(1) {

        border-top: 4px solid #2563EB;
    }


    div[data-testid="stHorizontalBlock"]
    div[data-testid="stMetric"]:nth-child(2) {

        border-top: 4px solid #10B981;
    }


    div[data-testid="stHorizontalBlock"]
    div[data-testid="stMetric"]:nth-child(3) {

        border-top: 4px solid #F43F5E;
    }


    div[data-testid="stHorizontalBlock"]
    div[data-testid="stMetric"]:nth-child(4) {

        border-top: 4px solid #8B5CF6;
    }

    div[role="radiogroup"] {

        display: flex;

        gap: 8px;

        background:
            rgba(255,255,255,0.82);

        backdrop-filter:
            blur(12px);

        border: 1px solid
        rgba(148,163,184,0.28);

        border-radius: 18px;

        padding: 8px;

        box-shadow:
            0 8px 25px
            rgba(15,23,42,0.06);
    }


    div[role="radiogroup"] label {

        color: #475569 !important;

        background: transparent;

        border-radius: 12px;

        padding: 9px 17px;

        font-weight: 650 !important;

        transition:
            all 0.2s ease;
    }


    div[role="radiogroup"] label:hover {

        background:
            rgba(37,99,235,0.08);

        color: #2563EB !important;
    }


    div[data-baseweb="input"] {

        background: #FFFFFF !important;

        border: 1px solid #CBD5E1;

        border-radius: 12px;

        transition:
            all 0.2s ease;
    }


    div[data-baseweb="input"]:focus-within {

        border-color: #2563EB;

        box-shadow:
            0 0 0 3px
            rgba(37,99,235,0.12);
    }


    div[data-testid="stNumberInput"] label {

        color: #334155 !important;

        font-weight: 650 !important;
    }


    div.stButton > button {

        background:
            linear-gradient(
                135deg,
                #2563EB,
                #4F46E5,
                #7C3AED
            ) !important;

        color: #FFFFFF !important;

        border: none !important;

        border-radius: 13px;

        min-height: 52px;

        font-size: 16px;

        font-weight: 750;

        letter-spacing: 0.2px;

        box-shadow:
            0 8px 20px
            rgba(79,70,229,0.25);

        transition:
            all 0.25s ease;
    }


    div.stButton > button:hover {

        transform:
            translateY(-3px);

        box-shadow:
            0 14px 28px
            rgba(79,70,229,0.32);

        filter:
            brightness(1.08);
    }


    div.stButton > button:active {

        transform:
            translateY(0);
    }


    div[data-testid="stAlert"] {

        border-radius: 16px !important;

        border: none !important;

        box-shadow:
            0 8px 22px
            rgba(15,23,42,0.06);

        animation:
            slideUp 0.45s ease;
    }


    div[data-testid="stDataFrame"] {

        background: #FFFFFF;

        border-radius: 16px;

        border: 1px solid
        #E2E8F0;

        overflow: hidden;

        box-shadow:
            0 8px 24px
            rgba(15,23,42,0.06);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }


    div[data-testid="stDataFrame"]:hover {

        box-shadow:
            0 12px 30px
            rgba(15,23,42,0.09);
    }



    div[data-testid="stImage"],
    div[data-testid="stPyplot"] {

        background: #FFFFFF;

        border-radius: 18px;

        padding: 8px;

        border: 1px solid
        #E2E8F0;

        box-shadow:
            0 8px 24px
            rgba(15,23,42,0.06);
    }



    div[data-testid="stAlert"] p {

        color: inherit !important;
    }


    hr {

        border: none;

        height: 1px;

        background:
            linear-gradient(
                90deg,
                transparent,
                #CBD5E1,
                transparent
            );

        margin-top: 1.5rem;

        margin-bottom: 1.5rem;
    }


        from {

            opacity: 0;

            transform:
                translateY(12px);
        }

        to {

            opacity: 1;

            transform:
                translateY(0);
        }
    }


    @keyframes glow {

        0% {

            box-shadow:
                0 8px 20px
                rgba(37,99,235,0.18);
        }

        50% {

            box-shadow:
                0 12px 30px
                rgba(124,58,237,0.28);
        }

        100% {

            box-shadow:
                0 8px 20px
                rgba(37,99,235,0.18);
        }
    }

    .main .block-container {

        animation:
            slideUp 0.55s ease;
    }


    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:

    st.title("🏦 Advanced Loan Approval Prediction System")

    st.caption(
        "Machine Learning Based Loan Approval Analysis and Prediction"
    )

    st.divider()

    st.subheader("System")

    st.write(
        "Supervised Machine Learning"
    )

    st.write(
        "Logistic Regression"
    )

    st.write(
        "StandardScaler"
    )

    st.divider()

    st.subheader("Dataset")

    st.write(
        "Loans Dataset"
    )

    st.write(
        "1,000 applications"
    )

    


df = pd.read_csv("loans.csv")

model = joblib.load("loan_model.pkl")

scaler = joblib.load("preprocessor.pkl")


features = [
    "income",
    "credit_score",
    "loan_amount",
    "employment_years"
]

X = df[features]

y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# MODEL PREDICTIONS
# =========================================================

X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)


# =========================================================
# PERFORMANCE METRICS
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)


total_applications = len(df)

approved = int(
    (df["loan_status"] == 1).sum()
)

rejected = int(
    (df["loan_status"] == 0).sum()
)


approval_rate = (
    approved / total_applications
) * 100


rejection_rate = (
    rejected / total_applications
) * 100



approved_df = df[
    df["loan_status"] == 1
]

rejected_df = df[
    df["loan_status"] == 0
]


approved_income = approved_df["income"].mean()
rejected_income = rejected_df["income"].mean()

approved_credit = approved_df["credit_score"].mean()
rejected_credit = rejected_df["credit_score"].mean()

approved_loan = approved_df["loan_amount"].mean()
rejected_loan = rejected_df["loan_amount"].mean()

approved_employment = approved_df[
    "employment_years"
].mean()

rejected_employment = rejected_df[
    "employment_years"
].mean()


st.title(
    "🏦 Advanced Loan Approval Prediction System"
)

st.caption(
    "Machine Learning Based Loan Approval Analysis and Prediction"
)


st.divider()

st.header("Dataset Overview")

st.write(
    "Summary of the dataset used for model development "
    "and evaluation."
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Applications",
        total_applications
    )


with col2:

    st.metric(
        "Approved",
        approved
    )


with col3:

    st.metric(
        "Rejected",
        rejected
    )


with col4:

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )


st.write("")

st.divider()


st.header("Project Modules")

page = st.radio(
    "Select a module",
    [
        "Overview",
        "Prediction",
        "Data Analysis",
        "Model Performance"
    ],
    horizontal=True
)


st.divider()


if page == "Overview":

    st.header("Project Overview")

    st.write(
        "This system uses supervised machine learning to "
        "classify loan applications as approved or rejected "
        "based on applicant attributes."
    )


    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


    st.subheader("Dataset Structure")

    structure = pd.DataFrame(
        {
            "Feature": [
                "income",
                "credit_score",
                "loan_amount",
                "employment_years",
                "loan_status"
            ],

            "Description": [
                "Applicant income",
                "Applicant credit score",
                "Requested loan amount",
                "Years of employment",
                "Loan approval outcome"
            ],

            "Role": [
                "Input",
                "Input",
                "Input",
                "Input",
                "Target"
            ]
        }
    )


    st.dataframe(
        structure,
        use_container_width=True,
        hide_index=True
    )


    st.subheader("Machine Learning Configuration")


    configuration = pd.DataFrame(
        {
            "Component": [
                "Learning Type",
                "Algorithm",
                "Preprocessing",
                "Training Data",
                "Testing Data",
                "Target Type"
            ],

            "Value": [
                "Supervised Learning",
                "Logistic Regression",
                "StandardScaler",
                "80%",
                "20%",
                "Binary Classification"
            ]
        }
    )


    st.dataframe(
        configuration,
        use_container_width=True,
        hide_index=True
    )


elif page == "Prediction":

    st.header("Loan Approval Prediction")

    st.write(
        "Enter applicant information to generate a prediction "
        "from the trained Logistic Regression model."
    )


    col1, col2 = st.columns(2)


    with col1:

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


    with col2:

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


    st.write("")


    predict_button = st.button(
        "Predict Loan Approval",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        try:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                params={
                    "income": income,
                    "credit_score": credit_score,
                    "loan_amount": loan_amount,
                    "employment_years": employment_years
                }
            )


            if response.status_code == 200:

                result = response.json()

                status = result["loan_status"]

                probability = result[
                    "approval_probability"
                ]


                st.divider()

                st.subheader(
                    "Prediction Result"
                )


                if status == "Approved":

                    st.success(
                        f"Loan Approved\n\n"
                        f"Model-estimated approval probability: "
                        f"{probability}%"
                    )

                else:

                    st.error(
                        f"Loan Rejected\n\n"
                        f"Model-estimated approval probability: "
                        f"{probability}%"
                    )


                result_col1, result_col2 = st.columns(2)


                with result_col1:

                    st.metric(
                        "Predicted Status",
                        status
                    )


                with result_col2:

                    st.metric(
                        "Approval Probability",
                        f"{probability}%"
                    )


            else:

                st.error(
                    f"Backend returned error: "
                    f"{response.status_code}"
                )


        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI. "
                "Please make sure the backend is running."
            )


elif page == "Data Analysis":

    st.header("Data Analysis")

    st.write(
        "This analysis examines the distribution of loan "
        "outcomes and compares applicant characteristics "
        "between approved and rejected applications."
    )

    st.subheader(
        "1. Loan Approval Distribution"
    )


    status_data = pd.DataFrame(
        {
            "Status": [
                "Approved",
                "Rejected"
            ],

            "Applications": [
                approved,
                rejected
            ]
        }
    )


    col1, col2 = st.columns([1.5, 1])


    with col1:

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")


        sns.barplot(
            data=status_data,
            x="Status",
            y="Applications",
            ax=ax
        )


        ax.set_title(
            "Loan Application Outcomes"
        )

        ax.set_xlabel("")

        ax.set_ylabel(
            "Number of Applications"
        )


        for container in ax.containers:

            ax.bar_label(
                container
            )


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    with col2:

        st.metric(
            "Approval Rate",
            f"{approval_rate:.2f}%"
        )

        st.metric(
            "Rejection Rate",
            f"{rejection_rate:.2f}%"
        )


        st.info(
            "This chart shows the distribution of the "
            "observed loan outcomes in the dataset."
        )


    st.divider()



    st.subheader(
        "2. Credit Score Analysis"
    )


    col1, col2 = st.columns(2)


    with col1:

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )


        sns.boxplot(
            data=df.assign(
                Loan_Status=df["loan_status"].map(
                    {
                        0: "Rejected",
                        1: "Approved"
                    }
                )
            ),
            x="Loan_Status",
            y="credit_score",
            ax=ax
        )


        ax.set_title(
            "Credit Score by Loan Status"
        )

        ax.set_xlabel(
            "Loan Status"
        )

        ax.set_ylabel(
            "Credit Score"
        )


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    with col2:

        st.metric(
            "Approved Avg. Credit Score",
            f"{approved_credit:.1f}"
        )

        st.metric(
            "Rejected Avg. Credit Score",
            f"{rejected_credit:.1f}"
        )


        difference = (
            approved_credit -
            rejected_credit
        )


        st.info(
            f"The average credit score difference "
            f"between the two groups is approximately "
            f"{abs(difference):.1f} points."
        )


    st.divider()


    # =====================================================
    # 3. INCOME ANALYSIS
    # =====================================================

    st.subheader(
        "3. Income Analysis"
    )


    col1, col2 = st.columns(2)


    with col1:

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )


        sns.boxplot(
            data=df.assign(
                Loan_Status=df["loan_status"].map(
                    {
                        0: "Rejected",
                        1: "Approved"
                    }
                )
            ),
            x="Loan_Status",
            y="income",
            ax=ax
        )


        ax.set_title(
            "Income by Loan Status"
        )

        ax.set_xlabel(
            "Loan Status"
        )

        ax.set_ylabel(
            "Annual Income"
        )


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    with col2:

        st.metric(
            "Approved Avg. Income",
            f"{approved_income:,.0f}"
        )

        st.metric(
            "Rejected Avg. Income",
            f"{rejected_income:,.0f}"
        )


        difference = (
            approved_income -
            rejected_income
        )


        st.info(
            f"The difference in average income "
            f"between the two groups is approximately "
            f"{abs(difference):,.0f}."
        )


    st.divider()


    st.subheader(
        "4. Loan Amount Analysis"
    )


    col1, col2 = st.columns(2)


    with col1:

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )


        sns.boxplot(
            data=df.assign(
                Loan_Status=df["loan_status"].map(
                    {
                        0: "Rejected",
                        1: "Approved"
                    }
                )
            ),
            x="Loan_Status",
            y="loan_amount",
            ax=ax
        )


        ax.set_title(
            "Loan Amount by Loan Status"
        )

        ax.set_xlabel(
            "Loan Status"
        )

        ax.set_ylabel(
            "Loan Amount"
        )


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    with col2:

        st.metric(
            "Approved Avg. Loan",
            f"{approved_loan:,.0f}"
        )

        st.metric(
            "Rejected Avg. Loan",
            f"{rejected_loan:,.0f}"
        )


        difference = (
            approved_loan -
            rejected_loan
        )


        st.info(
            f"The difference in average requested "
            f"loan amount is approximately "
            f"{abs(difference):,.0f}."
        )


    st.divider()

    st.subheader(
        "5. Employment Experience Analysis"
    )


    col1, col2 = st.columns(2)


    with col1:

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )


        sns.boxplot(
            data=df.assign(
                Loan_Status=df["loan_status"].map(
                    {
                        0: "Rejected",
                        1: "Approved"
                    }
                )
            ),
            x="Loan_Status",
            y="employment_years",
            ax=ax
        )


        ax.set_title(
            "Employment Years by Loan Status"
        )

        ax.set_xlabel(
            "Loan Status"
        )

        ax.set_ylabel(
            "Employment Years"
        )


        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    with col2:

        st.metric(
            "Approved Avg. Employment",
            f"{approved_employment:.1f} years"
        )

        st.metric(
            "Rejected Avg. Employment",
            f"{rejected_employment:.1f} years"
        )


        difference = (
            approved_employment -
            rejected_employment
        )


        st.info(
            f"The difference in average employment "
            f"experience is approximately "
            f"{abs(difference):.1f} years."
        )


    st.divider()


    st.subheader(
        "6. Comparative Statistical Summary"
    )


    comparison = pd.DataFrame(
        {
            "Feature": [
                "Income",
                "Credit Score",
                "Loan Amount",
                "Employment Years"
            ],

            "Approved Average": [
                approved_income,
                approved_credit,
                approved_loan,
                approved_employment
            ],

            "Rejected Average": [
                rejected_income,
                rejected_credit,
                rejected_loan,
                rejected_employment
            ]
        }
    )


    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


    st.info(
        "The comparisons above describe patterns observed "
        "in this dataset. They do not establish that any "
        "individual feature causes loan approval or rejection."
    )

elif page == "Model Performance":

    st.header("Model Performance")

    st.write(
        "Evaluation of the Logistic Regression model "
        "using the held-out test dataset."
    )

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )


    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )


    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )


    st.divider()

    st.subheader(
        "Confusion Matrix"
    )


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Rejected",
            "Approved"
        ],
        yticklabels=[
            "Rejected",
            "Approved"
        ],
        ax=ax
    )


    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        "Loan Approval Confusion Matrix"
    )


    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    st.divider()



    st.subheader(
        "Model Configuration"
    )


    configuration = pd.DataFrame(
        {
            "Parameter": [
                "Algorithm",
                "Preprocessing",
                "Training Data",
                "Testing Data",
                "Input Features",
                "Target"
            ],

            "Value": [
                "Logistic Regression",
                "StandardScaler",
                "80%",
                "20%",
                "Income, Credit Score, Loan Amount, Employment Years",
                "Loan Status"
            ]
        }
    )


    st.dataframe(
        configuration,
        use_container_width=True,
        hide_index=True
    )


    st.divider()

    st.subheader(
        "Performance Interpretation"
    )


    st.write(
        f"""
        The model achieved an accuracy of
        **{accuracy * 100:.2f}%** on the test dataset.

        Precision of **{precision * 100:.2f}%** indicates the
        proportion of predicted approved cases that were
        correctly classified.

        Recall of **{recall * 100:.2f}%** indicates how many
        of the actual approved cases were correctly identified.

        The F1 score of **{f1 * 100:.2f}%** combines precision
        and recall into a single performance measure.
        """
    )

st.divider()

st.caption(
    "Advanced Loan Approval Prediction System"
)