import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load("heart_model_5.pkl")

st.set_page_config(page_title="Heart Risk App", layout="wide")

st.title("❤️ Heart Attack Risk Prediction Dashboard")

# ================= SIDEBAR INPUT =================
st.sidebar.header("Patient Input")

age = st.sidebar.number_input("Age")
cholesterol = st.sidebar.number_input("Cholesterol")
heart_rate = st.sidebar.number_input("Heart Rate")
family_history = st.sidebar.number_input("Family History (0/1)")
exercise = st.sidebar.number_input("Exercise Hours Per Week")
prev_heart = st.sidebar.number_input("Previous Heart Problems (0/1)")
med_use = st.sidebar.number_input("Medication Use (0/1)")
stress = st.sidebar.number_input("Stress Level")
sedentary = st.sidebar.number_input("Sedentary Hours Per Day")
income = st.sidebar.number_input("Income")
bmi = st.sidebar.number_input("BMI")
triglycerides = st.sidebar.number_input("Triglycerides")
physical_days = st.sidebar.number_input("Physical Activity Days/Week")
sleep = st.sidebar.number_input("Sleep Hours/Day")

systolic = st.sidebar.number_input("Systolic")
diastolic = st.sidebar.number_input("Diastolic")

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
diabetes = st.sidebar.selectbox("Diabetes", ["Yes", "No"])
smoking = st.sidebar.selectbox("Smoking", ["Yes", "No"])
obesity = st.sidebar.selectbox("Obesity", ["Yes", "No"])
alcohol = st.sidebar.selectbox("Alcohol", ["Yes", "No"])
diet = st.sidebar.selectbox("Diet", ["Healthy", "Average", "Unhealthy"])

# ================= INPUT DF =================
input_df = pd.DataFrame([{
    "Age": age,
    "Cholesterol": cholesterol,
    "Heart Rate": heart_rate,
    "Family History": family_history,
    "Exercise Hours Per Week": exercise,
    "Previous Heart Problems": prev_heart,
    "Medication Use": med_use,
    "Stress Level": stress,
    "Sedentary Hours Per Day": sedentary,
    "Income": income,
    "BMI": bmi,
    "Triglycerides": triglycerides,
    "Physical Activity Days Per Week": physical_days,
    "Sleep Hours Per Day": sleep,
    "Systolic": systolic,
    "Diastolic": diastolic,

    "Sex": sex,
    "Diabetes": diabetes,
    "Smoking": smoking,
    "Obesity": obesity,
    "Alcohol Consumption": alcohol,
    "Diet": diet,
}])

# ================= PREDIKSI =================
if st.button("Predict Risk"):

    proba = model.predict_proba(input_df)[0]
    pred = model.predict(input_df)[0]

    col1, col2 = st.columns(2)

    with col1:
        if pred == 1:
            st.error("🔴 High Risk of Heart Attack")
        else:
            st.success("🟢 Low Risk of Heart Attack")

    with col2:
        st.metric("Risk Probability", f"{max(proba)*100:.2f}%")

    # ================= FEATURE IMPORTANCE =================
    st.subheader("📊 Feature Importance")

    # ambil model random forest dari pipeline
    rf_model = model.named_steps['model']
    preprocessor = model.named_steps['preprocessor']

    # ambil nama fitur setelah preprocessing
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]

    cat_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    all_features = list(num_cols) + cat_features

    importance = rf_model.feature_importances_

    feat_imp = pd.DataFrame({
        "Feature": all_features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(15)

    fig, ax = plt.subplots()
    ax.barh(feat_imp["Feature"][::-1], feat_imp["Importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importance")

    st.pyplot(fig)