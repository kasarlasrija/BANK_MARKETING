import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config("Bank Marketing Prediction", layout="wide")

st.title("🏦 Bank Marketing – Term Deposit Prediction App")

# ------------------------------
# ------------------------------
# LOAD DATA (same folder as app.py)
# ------------------------------
st.sidebar.header("📂 Dataset Loaded Automatically")

@st.cache_data
def load_data():
    df = pd.read_csv("bank.csv")   # same folder
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()

st.sidebar.success("bank.csv loaded successfully")


# ------------------------------
# SIDEBAR MENU
# ------------------------------
menu = st.sidebar.radio(
    "Navigation Menu",
    ["Dataset Overview", 
     "Train Model", 
     "Model Evaluation", 
     "Predict Customer Outcome", 
     "Decision Tree Visualization"]
)

# ------------------------------
# COMMON PREPROCESSING
# ------------------------------
df["deposit"] = df["deposit"].astype(str).str.lower().str.strip()
df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})
df = df.dropna(subset=["deposit"])

# Fill missing values
df["age"] = df["age"].fillna(df["age"].median())
df["balance"] = df["balance"].fillna(df["balance"].median())

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode().iloc[0])

# Label encode
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# features and target
X = df.drop("deposit", axis=1)
y = df["deposit"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
model.fit(X_train, y_train)

# ------------------------------
# PAGE 1 — DATASET OVERVIEW
# ------------------------------
if menu == "Dataset Overview":
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Information")
    st.write(df.describe())

    st.subheader("📌 Attributes of Interest")
    st.write("""
    - Age  
    - Job  
    - Account Balance  
    - Loan Status  
    - Contact History  
    """)

# ------------------------------
# PAGE 2 — TRAIN MODEL
# ------------------------------
elif menu == "Train Model":
    st.subheader("🤖 Model Trained Successfully")
    st.write("Decision Tree Classifier trained on dataset.")

# ------------------------------
# PAGE 3 — MODEL EVALUATION
# ------------------------------
elif menu == "Model Evaluation":
    st.subheader("📈 Model Performance")

    y_pred = model.predict(X_test)

    st.write("### ✅ Accuracy")
    st.write(round(accuracy_score(y_test, y_pred), 4))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    st.write("### 🔲 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # classification report table
    st.write("### 📋 Classification Report (Table)")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

# ------------------------------
# PAGE 4 — PREDICT CUSTOMER OUTCOME
# ------------------------------
elif menu == "Predict Customer Outcome":
    st.subheader("🔮 Predict Term Deposit Subscription")

    st.write("Enter customer details below")

    user_inputs = {}

    for col in X.columns:
        if df[col].dtype in [np.int64, np.float64]:
            user_inputs[col] = st.number_input(
                f"{col}",
                value=float(df[col].median())
            )
        else:
            user_inputs[col] = st.selectbox(
                f"{col}",
                sorted(df[col].unique())
            )

    input_df = pd.DataFrame([user_inputs])
    input_df = input_df[X.columns]

    pred = model.predict(input_df)

    st.write("### Result")
    if pred[0] == 1:
        st.success("💰 Customer is LIKELY to subscribe to a term deposit")
    else:
        st.error("❌ Customer is NOT likely to subscribe")

# ------------------------------
# PAGE 5 — DECISION TREE VISUALIZATION
# ------------------------------
elif menu == "Decision Tree Visualization":
    st.subheader("🌳 Decision Tree")

    fig = plt.figure(figsize=(16, 10))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["No Deposit", "Deposit"],
        filled=True,
        rounded=True
    )
    st.pyplot(fig)
