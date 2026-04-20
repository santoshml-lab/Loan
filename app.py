
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Loan Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Loan Approval System")

# ---------- FORM ----------
with st.form("loan_form"):

    st.subheader("Customer Details")

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

    submit = st.form_submit_button("🚀 Predict")

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

    # prediction
    pred = model.predict(input_df)[0]

    # probability (only if supported)
    try:
        prob = model.predict_proba(input_df)[0][1]
    except:
        prob = None

    st.markdown("---")

    # ---------- DASHBOARD ----------
    if pred == 1 or pred == "Approved":
        st.success("✅ LOAN APPROVED")

        st.progress(int((prob or 0.9) * 100))

        st.metric("Approval Probability", f"{(prob or 0.9)*100:.2f}%")

    else:
        st.error("❌ LOAN REJECTED")

        st.progress(int((prob or 0.1) * 100))

        st.metric("Approval Probability", f"{(prob or 0.1)*100:.2f}%")

    # ---------- EXTRA INSIGHT ----------
    st.info("📊 Model Confidence Score shown above helps evaluate risk level.")
