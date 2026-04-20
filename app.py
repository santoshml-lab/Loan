
import streamlit as st
import joblib
import pandas as pd

# ---------- CONFIG ----------
st.set_page_config(page_title="Loan AI System", page_icon="🏦", layout="centered")

st.title("🏦 Loan Approval AI System")
st.write("Smart ML-based loan prediction with confidence score")

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")

# ---------- FORM ----------
with st.form("loan_form"):

    st.subheader("Customer Information")

    no_of_dependents = st.number_input("No of Dependents", min_value=0, step=1)

    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    income_annum = st.number_input("Annual Income")
    loan_amount = st.number_input("Loan Amount")
    loan_term = st.number_input("Loan Term (years)")

    cibil_score = st.number_input("CIBIL Score")

    residential_assets_value = st.number_input("Residential Assets Value")
    commercial_assets_value = st.number_input("Commercial Assets Value")
    luxury_assets_value = st.number_input("Luxury Assets Value")
    bank_asset_value = st.number_input("Bank Asset Value")

    submit = st.form_submit_button("🚀 Predict Loan Status")

# ---------- PREDICTION ----------
if submit:

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

    # probability (safe handling)
    try:
        prob = model.predict_proba(input_df)[0][1]
    except:
        prob = 0.5

    st.markdown("---")

    # ---------- RESULT DASHBOARD ----------
    if pred == 1 or pred == "Approved":
        st.success("✅ LOAN APPROVED")
        st.progress(int(prob * 100))
        st.metric("Confidence Score", f"{prob*100:.2f}%")

        st.info("📊 Reason: Strong financial profile, high repayment capability")

    else:
        st.error("❌ LOAN REJECTED")
        st.progress(int(prob * 100))
        st.metric("Confidence Score", f"{prob*100:.2f}%")

        st.warning("📊 Reason: High risk profile, low repayment capability")
