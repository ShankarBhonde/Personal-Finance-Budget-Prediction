# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 13:29:18 2025

@author: 91957
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Training Data (Synthetic Example)
# -----------------------------
# Normally you would train on a real dataset. For now, we'll simulate.
np.random.seed(42)

data = pd.DataFrame({
    "income": np.random.randint(1000, 10000, 200),
    "fixed_expenses": np.random.randint(200, 5000, 200),
    "variable_expenses": np.random.randint(100, 3000, 200),
    "savings_goal": np.random.randint(0, 2000, 200),
    "impulse_purchases": np.random.randint(0, 10, 200),
    "uses_app": np.random.randint(0, 2, 200)
})

# Simple rule for labeling
data["label"] = ((data["income"] - (data["fixed_expenses"] + data["variable_expenses"] + data["savings_goal"])) > 0).astype(int)

# Features and target
X = data.drop("label", axis=1)
y = data["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💰 Personal Finance Budget Planner")
st.write("Predict your likelihood of sticking to your monthly budget and get personalized recommendations.")

# Input Form
income = st.slider("Monthly Income ($)", 0, 10000, 3000, 100)
fixed_expenses = st.slider("Monthly Fixed Expenses ($)", 0, 5000, 1200, 50)
variable_expenses = st.slider("Monthly Variable Expenses ($)", 0, 3000, 800, 50)
savings_goal = st.slider("Savings Goal ($)", 0, 2000, 500, 50)
impulse_purchases = st.slider("Impulse Purchases Last Month", 0, 10, 2, 1)
uses_app = st.radio("Do you use a budgeting app?", ["Yes", "No"])

# Convert Yes/No to numeric
uses_app_num = 1 if uses_app == "Yes" else 0

# Prediction
input_data = pd.DataFrame([{
    "income": income,
    "fixed_expenses": fixed_expenses,
    "variable_expenses": variable_expenses,
    "savings_goal": savings_goal,
    "impulse_purchases": impulse_purchases,
    "uses_app": uses_app_num
}])

prob = model.predict_proba(input_data)[0][1]  # likelihood of sticking to budget
prediction = "Yes" if prob >= 0.5 else "No"

st.subheader(f"Budget Adherence: {prediction} ({prob*100:.1f}% likelihood)")

# -----------------------------
# Recommendations
# -----------------------------
st.subheader("💡 Recommendations")

recommendations = []

# Expense Analysis
if fixed_expenses + variable_expenses > 0.8 * income:
    recommendations.append("Your expenses exceed 80% of your income. Try cutting down variable expenses.")

if variable_expenses > 0.4 * income:
    recommendations.append("Reduce variable expenses (entertainment, dining, shopping) by at least 10%.")

# Savings Analysis
if savings_goal > (income - (fixed_expenses + variable_expenses)):
    recommendations.append("Your savings goal is too high for your current budget. Consider lowering it or reducing expenses.")

# Impulse Purchases
if impulse_purchases > 5:
    recommendations.append("Impulse purchases are high. Set a monthly limit to control spending.")

# Budget App Usage
if uses_app_num == 0:
    recommendations.append("Try using a budgeting app to track your spending and build discipline.")

if not recommendations:
    recommendations.append("Your budget looks healthy. Keep it up!")

for rec in recommendations:
    st.write(f"- {rec}")
