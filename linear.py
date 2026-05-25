# app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Linear Regression",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 Linear Regression")
st.write("Upload a CSV dataset and train a Linear Regression model.")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# -----------------------------
# MAIN APP
# -----------------------------
if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # -----------------------------
    # DATA PREVIEW
    # -----------------------------
    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # DATASET INFO
    # -----------------------------
    st.subheader("📌 Dataset Shape")
    st.write(df.shape)

    st.subheader("📌 Columns")
    st.write(list(df.columns))

    # -----------------------------
    # TARGET COLUMN
    # -----------------------------
    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    # -----------------------------
    # FEATURES & TARGET
    # -----------------------------
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # -----------------------------
    # TARGET VALIDATION
    # -----------------------------
    try:
        y = pd.to_numeric(y)
    except:
        st.error(
            "❌ Selected target column is not numeric. "
            "Please select a numeric column."
        )
        st.stop()

    # -----------------------------
    # COLUMN TYPES
    # -----------------------------
    numerical_cols = X.select_dtypes(
        include=['int64', 'float64']
    ).columns

    categorical_cols = X.select_dtypes(
        include=['object']
    ).columns

    # -----------------------------
    # DISPLAY COLUMN TYPES
    # -----------------------------
    st.subheader("📌 Numerical Columns")
    st.write(list(numerical_cols))

    st.subheader("📌 Categorical Columns")
    st.write(list(categorical_cols))

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].mean())

    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # -----------------------------
    # ENCODE CATEGORICAL COLUMNS
    # -----------------------------
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(
            X[col].astype(str)
        )
        label_encoders[col] = le

    # -----------------------------
    # TRAIN TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------
    # FEATURE SCALING
    # -----------------------------
    scaler = StandardScaler()

    X_train[numerical_cols] = scaler.fit_transform(
        X_train[numerical_cols]
    )

    X_test[numerical_cols] = scaler.transform(
        X_test[numerical_cols]
    )

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    model = LinearRegression()

    model.fit(X_train, y_train)

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    y_pred = model.predict(X_test)

    # -----------------------------
    # EVALUATION METRICS
    # -----------------------------
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    # -----------------------------
    # PERFORMANCE DISPLAY
    # -----------------------------
    st.subheader("📈 Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("MAE", round(mae, 2))
        st.metric("MSE", round(mse, 2))

    with col2:
        st.metric("RMSE", round(rmse, 2))
        st.metric("R² Score", round(r2, 2))

    # -----------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------
    st.subheader("📋 Actual vs Predicted")

    results_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred
    })

    st.dataframe(results_df.head(20))

    # -----------------------------
    # MODEL COEFFICIENTS
    # -----------------------------
    st.subheader("📌 Model Coefficients")

    coeff_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    st.dataframe(coeff_df)

else:
    st.info("Please upload a CSV file to continue.")