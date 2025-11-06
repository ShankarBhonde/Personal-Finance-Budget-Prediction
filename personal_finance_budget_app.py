# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 16:04:52 2025

@author: 91957
"""

# app.py
# Personal Finance Budget Planner with ML + Streamlit UI

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load and Train Model
# -----------------------------

def train_model():
    # Load dataset (replace with your file name)
    df = pd.read_csv("budget_dataset.csv")

    # Select input features and target
    X = df[[
        "Monthly Income",
        "Monthly Fixed Expenses",
        "Monthly Variable Expenses",
        "Savings Goal",
        "Impulse Purchases",
        "Uses Budgeting App"
    ]]
    y = df["Likelihood of Sticking to Budget"]

    # Encode categorical columns
    le_app = LabelEncoder()
    X["Uses Budgeting App"] = le_app.fit_transform(X["Uses Budgeting App"])  # Yes=1, No=0

    le_target = LabelEncoder()
    y = le_target.fit_transform(y)  # High/Medium/Low

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    acc = accuracy_score(y_test, model.predict(X_test)) * 100

    return model, scaler, le_app, le_target, acc


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Personal Finance Budget Planner", layout="centered")

st.title("💰 Personal Finance Budget Planner (ML)")
st.markdown("---")
st.write("Predict your likelihood of sticking to your monthly budget based on your income, expenses, and habits.")

# Train model
model, scaler, le_app, le_target, accuracy = train_model()
st.sidebar.success(f"Model trained with {accuracy:.2f}% accuracy")

# User input form
st.header("Enter Your Financial Details")

income = st.number_input("Monthly Income ($)", min_value=0, max_value=100000, value=3000, step=100)
fixed = st.number_input("Monthly Fixed Expenses ($)", min_value=0, max_value=5000, value=1200, step=100)
variable = st.number_input("Monthly Variable Expenses ($)", min_value=0, max_value=3000, value=800, step=100)
savings = st.number_input("Savings Goal ($)", min_value=0, max_value=2000, value=500, step=50)
impulse = st.slider("Number of Impulse Purchases (Last Month)", 0, 10, 2)
app_use = st.selectbox("Do you use a budgeting app?", ["Yes", "No"])

if st.button("🔍 Predict Budget Adherence"):
    # Prepare input
    input_df = pd.DataFrame([{
        "Monthly Income": income,
        "Monthly Fixed Expenses": fixed,
        "Monthly Variable Expenses": variable,
        "Savings Goal": savings,
        "Impulse Purchases": impulse,
        "Uses Budgeting App": 1 if app_use == "Yes" else 0
    }])

    # Scale data
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]
    predicted_label = le_target.inverse_transform([prediction])[0]

    # Convert probability to %
    prob_percent = round(max(probability) * 100, 2)

    # Recommendation logic
    if predicted_label == "High":
        recommendation = "✅ Maintain habits and continue tracking expenses."
    elif predicted_label == "Medium":
        recommendation = "⚠️ Control impulse purchases and review spending trends."
    else:
        recommendation = "❌ Re-evaluate expenses, reduce variable costs, and consider using a budgeting app."

    # Display result
    st.subheader("📊 Prediction Result")
    st.write(f"**Likelihood of Sticking to Budget:** {predicted_label}")
    st.write(f"**Confidence Level:** {prob_percent}%")
    st.write(f"**Recommendation:** {recommendation}")

st.markdown("---")
st.subheader("Raw Data")
data = pd.read_csv("budget_dataset.csv")

st.write(data.head())
# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & scikit-learn")
