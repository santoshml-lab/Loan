import streamlit as st
import pandas as pd
import joblib

# ================= LOAD MODEL =================
model = joblib.load("model.pkl")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Loan Approval System", layout="centered")

st.title("🏦 Loan Approval Prediction System")
st.markdown("AI-based loan risk analysis dashboard 🚀")

# ================= INPUT FORM =================
st.sidebar.header("Enter Customer Details")

no_of_dependents = st.sidebar.number_input("No of Dependents", 0, 10, 2)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

income_annum = st.sidebar.number_input("Annual Income", 0, 10000000, 5000000)
loan_amount = st.sidebar.number_input("Loan Amount", 0, 50000000, 10000000)
loan_term = st.sidebar.number_input("Loan Term (years)", 1, 30, 10)
cibil_score = st.sidebar.number_input("CIBIL Score", 300, 900, 700)

residential_assets_value = st.sidebar.number_input("Residential Assets", 0, 50000000, 2000000)
commercial_assets_value = st.sidebar.number_input("Commercial Assets", 0, 50000000, 1000000)
luxury_assets_value = st.sidebar.number_input("Luxury Assets", 0, 50000000, 3000000)
bank_asset_value = st.sidebar.number_input("Bank Assets", 0, 50000000, 1000000)

# ================= PREDICTION =================
if st.button("🚀 Predict Loan Status"):

    input_df = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Result")

    if pred == "Approved":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.metric("Approval Probability", f"{round(proba*100,2)} %")

    st.subheader("🧠 Insights")

    if cibil_score < 600:
        st.warning("⚠ Low CIBIL Score Risk")

    if income_annum < loan_amount:
        st.warning("⚠ High Loan-to-Income Ratio")

    if loan_term > 20:
        st.info("ℹ Long-term loan increases risk")

    if cibil_score > 750:
        st.success("✔ Strong credit profile")
