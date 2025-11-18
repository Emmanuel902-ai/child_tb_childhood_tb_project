import os
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Paths & loaders
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "figures")

@st.cache_resource
def load_artifacts():
    # Load calibrated model
    model_path = os.path.join(MODELS_DIR, "calibrated_xgb.pkl")
    with open(model_path, "rb") as f:
        calibrated_model = pickle.load(f)

    # Load feature order
    feat_path = os.path.join(MODELS_DIR, "feature_order.json")
    with open(feat_path, "r") as f:
        feature_order = json.load(f)

    # Load threshold
    thr_path = os.path.join(MODELS_DIR, "threshold.json")
    with open(thr_path, "r") as f:
        thr_data = json.load(f)
    threshold = float(thr_data["threshold"])

    # Load example patient
    example_path = os.path.join(MODELS_DIR, "example_patient.json")
    with open(example_path, "r") as f:
        example_patient = json.load(f)

    # Load summary text if available
    summary_path = os.path.join(BASE_DIR, "results_summary.txt")
    summary_text = None
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_text = f.read()

    return calibrated_model, feature_order, threshold, example_patient, summary_text


# ---------------------------------------------------------
# Helper: build input row and predict
# ---------------------------------------------------------

def build_input_df(user_inputs: dict, feature_order):
    """
    Convert dict of inputs to a 1-row DataFrame
    with columns in the correct order.
    """
    data = {col: [user_inputs.get(col)] for col in feature_order}
    return pd.DataFrame(data)


def predict_risk(model, X_row):
    """
    Return predicted probability (float between 0 and 1).
    """
    proba = model.predict_proba(X_row)[:, 1][0]
    return float(proba)


def interpret_risk(prob):
    """
    Give a simple textual interpretation of risk.
    """
    if prob < 0.2:
        return "Low risk"
    elif prob < 0.5:
        return "Moderate risk"
    else:
        return "High risk"


# ---------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------

def page_overview(threshold):
    st.title("🧒 AI-Enabled Clinical Prediction for Childhood Tuberculosis")

    st.markdown(
        """
This prototype explores how **machine learning** can support clinicians in
identifying **pulmonary TB in children ≤ 15 years** using **routine clinical data**.

**Why this problem matters:**
- Childhood TB is often **underdiagnosed**: non-specific symptoms, difficulty obtaining sputum, and atypical X-ray findings.
- WHO **Treatment Decision Algorithms (TDAs)** use fixed rules that may **underperform across settings**.
- AI can provide **probabilistic, calibrated risk scores** instead of yes/no rules.
        """
    )

    st.subheader("What this prototype does")
    st.markdown(
        f"""
- Uses a **synthetic cohort** of 1 500 children with WHO-inspired feature distributions  
- Trains several models and selects a **calibrated XGBoost** classifier  
- Tunes an operating **threshold ≈ {threshold:.3f}** to keep high sensitivity  
- Evaluates **fairness** across age, HIV status, malnutrition, and sex  
- Exposes an interactive **risk calculator** for a single child
        """
    )

    st.info(
        "This is a **research prototype** built on synthetic data. "
        "It is **not a clinical tool** and should not be used for real patient decisions."
    )


# ---------------------------------------------------------
# Page 2: Single-patient risk calculator
# ---------------------------------------------------------

def page_single_patient(model, feature_order, threshold, example_patient):
    st.title("🧒 Single-Patient TB Risk Calculator")

    st.markdown(
        """
Use this page to simulate a child with presumptive TB and estimate
the model's **predicted risk** and decision (TB+ / TB−) at the current threshold.
        """
    )

    # Use example_patient as defaults
    ep = example_patient

    with st.form("tb_risk_form"):
        st.subheader("Clinical and epidemiological inputs")

        col1, col2 = st.columns(2)

        with col1:
            age_years = st.slider(
                "Age (years)", 0, 15, int(ep.get("age_years", 7))
            )
            sex_label = st.selectbox(
                "Sex",
                options=["Female (0)", "Male (1)"],
                index=int(ep.get("sex", 0))
            )
            sex = 1 if "Male" in sex_label else 0

            cough_weeks = st.slider(
                "Cough duration (weeks)", 0, 8, int(ep.get("cough_weeks", 3))
            )

            weight_zscore = st.slider(
                "Weight-for-age z-score",
                -4.0, 3.0,
                float(ep.get("weight_zscore", -2.0)),
                step=0.1,
            )

            malnutrition = st.selectbox(
                "Malnutrition flag",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("malnutrition", 1)),
            )
            malnutrition = 1 if "Yes" in malnutrition else 0

        with col2:
            fever = st.selectbox(
                "Fever",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("fever", 1)),
            )
            fever = 1 if "Yes" in fever else 0

            night_sweats = st.selectbox(
                "Night sweats",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("night_sweats", 1)),
            )
            night_sweats = 1 if "Yes" in night_sweats else 0

            contact_TB = st.selectbox(
                "Known contact with TB case",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("contact_TB", 1)),
            )
            contact_TB = 1 if "Yes" in contact_TB else 0

            HIV_positive = st.selectbox(
                "HIV positive",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("HIV_positive", 0)),
            )
            HIV_positive = 1 if "Yes" in HIV_positive else 0

            BCG_scar = st.selectbox(
                "BCG scar present",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("BCG_scar", 1)),
            )
            BCG_scar = 1 if "Yes" in BCG_scar else 0

            CXR_abnormal = st.selectbox(
                "Chest X-ray abnormal",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("CXR_abnormal", 1)),
            )
            CXR_abnormal = 1 if "Yes" in CXR_abnormal else 0

            GeneXpert_available = st.selectbox(
                "GeneXpert available",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("GeneXpert_available", 0)),
            )
            GeneXpert_available = 1 if "Yes" in GeneXpert_available else 0

        submitted = st.form_submit_button("Estimate TB Risk")

    if not submitted:
        st.caption("Fill the fields above and click **Estimate TB Risk**.")
        return

    # Build input dict
    user_inputs = {
        "age_years": age_years,
        "sex": sex,
        "cough_weeks": cough_weeks,
        "fever": fever,
        "night_sweats": night_sweats,
        "contact_TB": contact_TB,
        "HIV_positive": HIV_positive,
        "weight_zscore": weight_zscore,
        "malnutrition": malnutrition,
        "BCG_scar": BCG_scar,
        "CXR_abnormal": CXR_abnormal,
        "GeneXpert_available": GeneXpert_available,
    }

    # Convert to DataFrame with correct column order
    X_row = build_input_df(user_inputs, feature_order)

    # Predict
    prob = predict_risk(model, X_row)
    label = int(prob >= threshold)
    interpretation = interpret_risk(prob)

    st.subheader("Model output")

    colA, colB = st.columns(2)
    with colA:
        st.metric(
            "Predicted TB risk (probability)",
            f"{prob*100:.1f}%",
        )
        st.write(f"**Risk level:** {interpretation}")

    with colB:
        decision_text = "TB-positive (above threshold)" if label == 1 else "TB-negative (below threshold)"
        st.metric(
            "Model decision at operating threshold",
            decision_text,
            help=f"Threshold used: {threshold:.3f}",
        )

    st.info(
        "This is a **research prototype** using synthetic data. "
        "In reality, such a tool would need extensive clinical validation "
        "and must be used alongside, not instead of, clinician judgment."
    )


# ---------------------------------------------------------
# Page 3: Model performance & figures
# ---------------------------------------------------------

def page_performance(summary_text):
    st.title("📈 Model Performance & Fairness")

    st.markdown(
        """
This page summarises the **final calibrated model** and visualizes
its performance on the synthetic test set.
        """
    )

    if summary_text:
        st.subheader("Text summary")
        st.code(summary_text, language="text")
    else:
        st.warning("No `results_summary.txt` found in project root.")

    st.subheader("Key diagnostic plots")

    roc_path = os.path.join(FIG_DIR, "roc_curve_test.png")
    calib_path = os.path.join(FIG_DIR, "calibration_curve_test.png")
    subgroup_path = os.path.join(FIG_DIR, "subgroup_auc_bar.png")
    uncert_path = os.path.join(FIG_DIR, "uncertainty_hist.png")

    if os.path.exists(roc_path):
        st.markdown("**ROC curve (Test set)**")
        st.image(roc_path, width='stretch')
    else:
        st.caption("ROC curve image not found.")

    if os.path.exists(calib_path):
        st.markdown("**Calibration curve (Test set)**")
        st.image(calib_path, width='stretch')
    else:
        st.caption("Calibration curve image not found.")

    if os.path.exists(subgroup_path):
        st.markdown("**Subgroup AUROC by age / HIV / malnutrition / sex**")
        st.image(subgroup_path, width='stretch')
    else:
        st.caption("Subgroup AUROC image not found.")

    if os.path.exists(uncert_path):
        st.markdown("**Uncertainty distribution (prediction confidence)**")
        st.image(uncert_path, width='stretch')
    else:
        st.caption("Uncertainty histogram not found.")


# ---------------------------------------------------------
# Page 4: Methods & limitations
# ---------------------------------------------------------

def page_methods():
    st.title("ℹ️ Methods, Data & Limitations")

    st.subheader("Synthetic cohort design")
    st.markdown(
        """
- **N = 1 500** children aged 0–15 years  
- Features sampled from distributions inspired by **WHO childhood TB guidelines**  
- TB labels generated using a **latent logistic risk model** so that high-risk profiles
  (e.g. TB contact, HIV+, abnormal CXR, malnutrition) are more likely to be TB-positive  
- Overall TB prevalence ≈ **23%**, consistent with high-risk presumptive TB cohorts
        """
    )

    st.subheader("Models and training")
    st.markdown(
        """
- Compared **Logistic Regression, Random Forest, XGBoost, LightGBM**  
- Selected **XGBoost** as the champion model  
- Applied **isotonic regression** for probability calibration  
- Chose an operating threshold using **Youden’s J**, with
  **sensitivity ≥ 0.85** as a clinical constraint  
- Checked performance across subgroups: **age bands, HIV status, malnutrition, sex**
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
- Dataset is **fully synthetic** → real-world performance may differ  
- No real imaging data (e.g. detailed CXR) or lab values were used  
- Model does **not** incorporate treatment history, drug resistance, or co-infections  
- This is a **proof-of-concept** only, not a deployable medical device  
        """
    )

    st.subheader("Future directions")
    st.markdown(
        """
- Validate on **real multicountry paediatric TB datasets**  
- Involve clinicians to refine thresholds and interpretability  
- Integrate into a **clinical decision support tool** with strict governance, audit logs, and training  
        """
    )

    st.info(
        "This project is designed to demonstrate how a junior data scientist can "
        "build an **ethically-aware, uncertainty-conscious AI prototype** for a "
        "high-impact clinical problem using only synthetic data."
    )


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Childhood TB Risk Prototype",
        page_icon="🧒",
        layout="centered",
    )

    calibrated_model, feature_order, threshold, example_patient, summary_text = load_artifacts()

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "🏠 Overview",
            "🧒 Risk calculator",
            "📈 Performance",
            "ℹ️ Methods & limitations",
        ],
    )

    if page == "🏠 Overview":
        page_overview(threshold)
    elif page == "🧒 Risk calculator":
        page_single_patient(calibrated_model, feature_order, threshold, example_patient)
    elif page == "📈 Performance":
        page_performance(summary_text)
    elif page == "ℹ️ Methods & limitations":
        page_methods()


if __name__ == "__main__":
    main()
