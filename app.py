"""
Smart Loan/Credit Risk Approval System with AI-Generated Explanations
Streamlit demo app

Run with: streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

st.set_page_config(
    page_title="Smart Loan Risk Assessment",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    rf = joblib.load("models/credit_risk_rf.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    X_test_sample = pd.read_pickle("models/X_test_sample.pkl")
    shap_values_class1 = np.load("models/shap_values_class1.npy")
    return rf, feature_columns, X_test_sample, shap_values_class1


@st.cache_resource
def load_gemini_client():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


rf, feature_columns, X_test_sample, shap_values_class1 = load_artifacts()
client = load_gemini_client()


def get_applicant_shap_explanation(idx, X_sample, shap_values, top_n=5):
    applicant_shap = shap_values[idx]
    shap_df = pd.DataFrame({
        "feature": X_sample.columns,
        "shap_value": applicant_shap,
        "feature_value": X_sample.iloc[idx].values
    })
    shap_df["abs_impact"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_impact", ascending=False).head(top_n)

    factors = []
    for _, row in shap_df.iterrows():
        direction = "increases risk" if row["shap_value"] > 0 else "decreases risk"
        factors.append({
            "feature": row["feature"],
            "value": row["feature_value"],
            "effect": direction,
            "impact_score": round(row["abs_impact"], 4)
        })
    return factors, shap_df


def generate_ai_explanation(prediction, probability, factors, audience="loan officer"):
    if client is None:
        return ("Gemini API key not found. Add GOOGLE_API_KEY to your .env file "
                "to enable AI-generated explanations.")

    factors_text = "\n".join([
        f"- {f['feature']} (value: {f['value']}): {f['effect']} (impact score: {f['impact_score']})"
        for f in factors
    ])

    prompt = f"""
You are explaining a credit risk model's decision to a {audience}.

Prediction: {"High Risk of Default" if prediction == 1 else "Low Risk (Likely to Repay)"}
Predicted probability of default: {probability:.2%}

Top contributing factors (from SHAP analysis):
{factors_text}

Write a clear, professional explanation (3-5 sentences) of why this applicant
received this risk classification. Avoid technical jargon like "SHAP" or "impact score"
in the explanation itself - translate them into plain business language.
If audience is "applicant", be empathetic and constructive.
If audience is "loan officer", be precise and analytical.

IMPORTANT FAIRNESS CONSTRAINT: Do not cite age, gender, region/geographic location,
or education level as a reason in your explanation, even if they appear in the
contributing factors above. These attributes may raise fair-lending or discrimination
concerns and should not be presented to the applicant or loan officer as justification
for a credit decision. Focus your explanation only on financial and credit-history
related factors (e.g., credit scores, income, loan amount, employment history,
repayment history, existing debt). If the top factors are dominated by excluded
attributes, focus on whichever remaining factors are most relevant, and note briefly
that the assessment also considered demographic factors that are not detailed here
for fairness reasons.

Do not include any preamble, meta-commentary, or phrases like "Here is an explanation" -
respond with only the explanation itself, starting directly with the first sentence.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
            return ("⚠️ The Gemini API free-tier quota has been reached for now. "
                    "Please wait a minute and try again, or check your quota at "
                    "https://ai.google.dev/gemini-api/docs/rate-limits")
        return f"⚠️ An error occurred while generating the explanation: {e}"


st.sidebar.title("📋 Select Applicant")

probs = rf.predict_proba(X_test_sample)[:, 1]
risk_labels = [f"Applicant #{i} — {p:.1%} default risk" for i, p in enumerate(probs)]
selected = st.sidebar.selectbox("Choose a sample applicant", options=range(len(X_test_sample)),
                                 format_func=lambda i: risk_labels[i])

audience = st.sidebar.radio("Explanation audience", ["Loan Officer", "Applicant"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "This is a portfolio/educational project using the Home Credit Default Risk "
    "dataset. It is not a real financial decision system. Credit scoring models "
    "can carry bias and fairness risks that require careful review in production."
)

st.title("🏦 Smart Loan Risk Assessment")
st.caption("Random Forest prediction + SHAP explainability + Gemini natural-language explanation")

idx = selected
prediction = rf.predict(X_test_sample.iloc[[idx]])[0]
probability = rf.predict_proba(X_test_sample.iloc[[idx]])[0][1]
factors, shap_df = get_applicant_shap_explanation(idx, X_test_sample, shap_values_class1)

col1, col2 = st.columns([1, 2])

with col1:
    st.metric(
        "Risk Classification",
        "High Risk" if prediction == 1 else "Low Risk",
        delta=f"{probability:.1%} probability of default",
        delta_color="inverse"
    )

with col2:
    st.subheader("Top contributing factors")
    display_df = shap_df[["feature", "feature_value", "shap_value"]].rename(
        columns={"feature": "Feature", "feature_value": "Value", "shap_value": "SHAP impact"}
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader(f"💬 AI-Generated Explanation ({audience} view)")

with st.spinner("Generating explanation with Gemini..."):
    explanation = generate_ai_explanation(
        prediction, probability, factors,
        audience="loan officer" if audience == "Loan Officer" else "applicant"
    )

st.info(explanation)

st.markdown("---")
st.caption(
    "Built with scikit-learn (Random Forest), SHAP (explainability), and the "
    "Gemini API (natural-language generation). Dataset: Home Credit Default Risk (Kaggle)."
)