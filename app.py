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
import shap
from dotenv import load_dotenv
from google import genai

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Risk Assessment",
    page_icon="\U0001F4CA",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Load artifacts (cached so they only load once per session)
# ---------------------------------------------------------------------------
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


@st.cache_resource
def get_explainer(_rf):
    return shap.TreeExplainer(_rf)


explainer = get_explainer(rf)

# ---------------------------------------------------------------------------
# Helper functions (same logic as the notebook)
# ---------------------------------------------------------------------------
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
        if "UNAVAILABLE" in str(e) or "503" in str(e):
            return ("⚠️ Gemini's servers are experiencing high demand right now. "
                    "This is temporary — please try again in a moment.")
        return f"⚠️ An error occurred while generating the explanation: {e}"


def build_new_applicant_row(template_row, feature_columns, inputs):
    """
    Start from a real applicant's full feature vector (so every one-hot group
    stays structurally valid), then overwrite only the fields the user
    actually provided in the form.
    """
    row = template_row.copy()

    # --- direct numeric overwrites ---
    numeric_map = {
        "AMT_INCOME_TOTAL": inputs["income"],
        "AMT_CREDIT": inputs["credit_amount"],
        "AMT_ANNUITY": inputs["annuity"],
        "AMT_GOODS_PRICE": inputs["goods_price"],
        "CNT_CHILDREN": inputs["num_children"],
        "DAYS_BIRTH": -inputs["age"] * 365,
        "AGE_YEARS": inputs["age"],
        "DAYS_EMPLOYED": -inputs["years_employed"] * 365,
        "EXT_SOURCE_1": inputs["ext_source_1"],
        "EXT_SOURCE_2": inputs["ext_source_2"],
        "EXT_SOURCE_3": inputs["ext_source_3"],
    }
    for col, val in numeric_map.items():
        if col in row.index:
            row[col] = val

    # --- one-hot categorical overwrites ---
    def set_one_hot(prefix, selected_value):
        group_cols = [c for c in feature_columns if c.startswith(prefix)]
        for c in group_cols:
            row[c] = 0
        target = f"{prefix}{selected_value}"
        if target in row.index:
            row[target] = 1

    set_one_hot("NAME_EDUCATION_TYPE_", inputs["education"])
    set_one_hot("FLAG_OWN_CAR_", inputs["own_car"])
    set_one_hot("FLAG_OWN_REALTY_", inputs["own_realty"])

    return row


def get_shap_explanation_for_row(row, explainer, feature_columns, top_n=5):
    row_df = pd.DataFrame([row[feature_columns].values], columns=feature_columns)
    shap_vals = explainer.shap_values(row_df)

    # Handle both possible SHAP output shapes across versions
    if isinstance(shap_vals, list):
        applicant_shap = shap_vals[1][0]
    elif shap_vals.ndim == 3:
        applicant_shap = shap_vals[0, :, 1]
    else:
        applicant_shap = shap_vals[0]

    shap_df = pd.DataFrame({
        "feature": feature_columns,
        "shap_value": applicant_shap,
        "feature_value": row[feature_columns].values
    })
    shap_df["abs_impact"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("abs_impact", ascending=False).head(top_n)

    factors = []
    for _, r in shap_df.iterrows():
        direction = "increases risk" if r["shap_value"] > 0 else "decreases risk"
        factors.append({
            "feature": r["feature"],
            "value": r["feature_value"],
            "effect": direction,
            "impact_score": round(r["abs_impact"], 4)
        })
    return factors, shap_df, row_df


# ---------------------------------------------------------------------------
# Sidebar - mode selection
# ---------------------------------------------------------------------------
st.sidebar.title("\U0001F4CB Settings")
audience = st.sidebar.radio("Explanation audience", ["Loan Officer", "Applicant"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "This is a portfolio/educational project using the Home Credit Default Risk "
    "dataset. It is not a real financial decision system. Credit scoring models "
    "can carry bias and fairness risks that require careful review in production."
)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("\U0001F3E6 Smart Loan Risk Assessment")
st.caption("Random Forest prediction + SHAP explainability + Gemini natural-language explanation")

tab_sample, tab_new = st.tabs(["\U0001F4C1 Browse Sample Applicants", "\U0001F195 Assess a New Applicant"])

audience_key = "loan officer" if audience == "Loan Officer" else "applicant"


def render_result(prediction, probability, factors, shap_df):
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
    st.subheader(f"\U0001F4AC AI-Generated Explanation ({audience} view)")

    with st.spinner("Generating explanation with Gemini..."):
        explanation = generate_ai_explanation(prediction, probability, factors, audience=audience_key)

    st.info(explanation)


# ---------------------------------------------------------------------------
# Tab 1: browse existing sample applicants (original behaviour)
# ---------------------------------------------------------------------------
with tab_sample:
    probs = rf.predict_proba(X_test_sample)[:, 1]
    risk_labels = [f"Applicant #{i} — {p:.1%} default risk" for i, p in enumerate(probs)]
    selected = st.selectbox("Choose a sample applicant", options=range(len(X_test_sample)),
                             format_func=lambda i: risk_labels[i])

    idx = selected
    prediction = rf.predict(X_test_sample.iloc[[idx]])[0]
    probability = rf.predict_proba(X_test_sample.iloc[[idx]])[0][1]
    factors, shap_df = get_applicant_shap_explanation(idx, X_test_sample, shap_values_class1)

    render_result(prediction, probability, factors, shap_df)

# ---------------------------------------------------------------------------
# Tab 2: enter a brand-new applicant's details and get a live prediction
# ---------------------------------------------------------------------------
with tab_new:
    st.markdown(
        "Enter an applicant's details below. Fields not listed here are filled "
        "using realistic defaults so the model still receives a complete, valid "
        "input — this keeps the demo simple while remaining representative of "
        "the full model."
    )

    with st.form("new_applicant_form"):
        st.markdown("**Financial details**")
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input("Annual income", min_value=0, value=180000, step=10000)
            credit_amount = st.number_input("Requested loan amount", min_value=0, value=500000, step=10000)
        with c2:
            annuity = st.number_input("Annual loan annuity (repayment amount)", min_value=0, value=25000, step=1000)
            goods_price = st.number_input("Price of goods being financed", min_value=0, value=450000, step=10000)

        st.markdown("**Applicant background**")
        c3, c4, c5 = st.columns(3)
        with c3:
            age = st.number_input("Age (years)", min_value=18, max_value=75, value=35)
        with c4:
            years_employed = st.number_input("Years at current job", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
        with c5:
            num_children = st.number_input("Number of children", min_value=0, max_value=10, value=0)

        c6, c7, c8 = st.columns(3)
        with c6:
            education = st.selectbox("Education level", [
                "Higher education", "Secondary / secondary special",
                "Incomplete higher", "Lower secondary", "Academic degree"
            ])
        with c7:
            own_car = st.selectbox("Owns a car?", ["Y", "N"], format_func=lambda v: "Yes" if v == "Y" else "No")
        with c8:
            own_realty = st.selectbox("Owns property?", ["Y", "N"], format_func=lambda v: "Yes" if v == "Y" else "No")

        st.markdown("**External credit bureau scores** (0 = weakest, 1 = strongest)")
        st.caption(
            "In a real system these would come from a credit bureau API, not "
            "from the applicant. They are included here because they are the "
            "model's strongest predictors — adjust the sliders to see their effect."
        )
        e1, e2, e3 = st.columns(3)
        with e1:
            ext_source_1 = st.slider("External score 1", 0.0, 1.0, 0.5, 0.01)
        with e2:
            ext_source_2 = st.slider("External score 2", 0.0, 1.0, 0.5, 0.01)
        with e3:
            ext_source_3 = st.slider("External score 3", 0.0, 1.0, 0.5, 0.01)

        submitted = st.form_submit_button("Assess Risk")

    if submitted:
        inputs = {
            "income": income, "credit_amount": credit_amount, "annuity": annuity,
            "goods_price": goods_price, "age": age, "years_employed": years_employed,
            "num_children": num_children, "education": education,
            "own_car": own_car, "own_realty": own_realty,
            "ext_source_1": ext_source_1, "ext_source_2": ext_source_2, "ext_source_3": ext_source_3,
        }

        template_row = X_test_sample.iloc[0]
        new_row = build_new_applicant_row(template_row, feature_columns, inputs)

        prediction_arr = rf.predict(pd.DataFrame([new_row[feature_columns].values], columns=feature_columns))
        probability_arr = rf.predict_proba(pd.DataFrame([new_row[feature_columns].values], columns=feature_columns))
        prediction = prediction_arr[0]
        probability = probability_arr[0][1]

        with st.spinner("Computing SHAP explanation..."):
            factors, shap_df, _ = get_shap_explanation_for_row(new_row, explainer, feature_columns)

        render_result(prediction, probability, factors, shap_df)

st.markdown("---")
st.caption(
    "Built with scikit-learn (Random Forest), SHAP (explainability), and the "
    "Gemini API (natural-language generation). Dataset: Home Credit Default Risk (Kaggle)."
)