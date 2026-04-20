
import streamlit as st
import joblib
import pandas as pd

import shap
import numpy as np
# ---------- CONFIG ----------
st.set_page_config(
    page_title="Loan AI Pro",
    page_icon="🏦",
    layout="centered"
)

# ---------- UI STYLE ----------
st.markdown("""
    <style>
        .main {
            background-color: #0f1117;
        }
        h1, h2, h3 {
            color: #00ffcc;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 Loan Approval AI - PRO SYSTEM")
st.write("Smart AI-powered loan prediction with risk analysis")

# ---------- LOAD MODEL ----------
model = joblib.load("model.pkl")

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

    submit = st.form_submit_button("🚀 Predict Loan Status")

# ---------- RESET ----------
if st.button("🔄 Reset Form"):
    st.experimental_rerun()

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

    try:
        prob = model.predict_proba(input_df)[0][1]
    except:
        prob = 0.5

    risk_score = prob * 100

    st.markdown("---")
    st.markdown("## 📊 AI Decision Report")

    st.markdown("### 📈 Risk Probability")
    st.progress(int(risk_score))
    st.metric("Confidence Score", f"{risk_score:.2f}%")

    if risk_score >= 70:
        st.success("🟢 LOW RISK CUSTOMER")
        st.write("✔ Strong financial profile")
        st.write("✔ High repayment capability")

    elif risk_score >= 40:
        st.warning("🟡 MEDIUM RISK CUSTOMER")
        st.write("⚠ Mixed financial signals")
        st.write("⚠ Manual review recommended")

    else:
        st.error("🔴 HIGH RISK CUSTOMER")
        st.write("❌ High default probability")
        st.write("❌ Weak financial profile")
        st.markdown("## 🧠 Why this prediction? (SHAP)")

try:
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    st.write("Feature impact visualization:")

    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot()

except:
    st.warning("SHAP explanation not available for this model type")
        
        
    

        


    
    

    



