import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# 1. LOAD DATASET
df = pd.read_csv("loans.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# 2. SEPARATE FEATURES AND TARGET

X = df[
    [
        "income",
        "credit_score",
        "loan_amount",
        "employment_years"
    ]
]

y = df["loan_status"]


# 3. SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 4. SCALE FEATURES

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. CREATE AND TRAIN MODEL

model = LogisticRegression()

model.fit(X_train_scaled, y_train)


# 6. MAKE PREDICTIONS

y_pred = model.predict(X_test_scaled)


# 7. EVALUATE MODEL

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 8. SAVE MODEL

joblib.dump(model, "loan_model.pkl")

joblib.dump(scaler, "preprocessor.pkl")


print("\nModel saved as loan_model.pkl")
print("Preprocessor saved as preprocessor.pkl")