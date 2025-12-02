import os
import json
import pickle
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# -------------------------------------------------------------------
# Paths, logging, and artifact loading
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "figures")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Basic logging configuration
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_artifacts():
    """
    Load model, feature order, classification threshold,
    example patient profile, and results summary.
    """

    # Load calibrated model
    model_path = os.path.join(MODELS_DIR, "calibrated_xgb.pkl")
    if not os.path.exists(model_path):
        st.error(f"Missing model file: {model_path}")
        st.stop()

    with open(model_path, "rb") as f:
        calibrated_model = pickle.load(f)
    logger.info("Loaded calibrated model from %s", model_path)

    # Feature order
    feat_path = os.path.join(MODELS_DIR, "feature_order.json")
    if not os.path.exists(feat_path):
        st.error(f"Missing feature order file: {feat_path}")
        st.stop()

    with open(feat_path, "r") as f:
        feature_order = json.load(f)
    logger.info("Loaded feature order with %d features.", len(feature_order))

    # Threshold
    thr_path = os.path.join(MODELS_DIR, "threshold.json")
    if not os.path.exists(thr_path):
        st.error(f"Missing threshold file: {thr_path}")
        st.stop()

    with open(thr_path, "r") as f:
        thr_data = json.load(f)
    threshold = float(thr_data["threshold"])
    logger.info("Loaded operating threshold: %.4f", threshold)

    # Example patient
    example_path = os.path.join(MODELS_DIR, "example_patient.json")
    if not os.path.exists(example_path):
        st.error(f"Missing example patient file: {example_path}")
        st.stop()

    with open(example_path, "r") as f:
        example_patient = json.load(f)
    logger.info("Loaded example patient defaults.")

    # Optional text summary
    summary_path = os.path.join(BASE_DIR, "results_summary.txt")
    summary_text = None
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_text = f.read()
        logger.info("Loaded results_summary.txt.")
    else:
        logger.warning("results_summary.txt not found.")

    return calibrated_model, feature_order, threshold, example_patient, summary_text


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def build_input_df(user_inputs, feature_order):
    """Create a DataFrame in correct feature order."""
    return pd.DataFrame({f: [user_inputs.get(f)] for f in feature_order})


def predict_risk(model, X_row):
    """Return predicted probability."""
    proba = model.predict_proba(X_row)[:, 1][0]
    return float(proba)


def interpret_risk(prob):
    """Human-readable risk interpretation."""
    if prob < 0.2:
        return "Low risk profile"
    elif prob < 0.5:
        return "Moderate risk profile"
    return "High risk profile"


def risk_band(prob):
    """Return (band_label, colour) for use in UI."""
    if prob < 0.2:
        return "Low", "#198754"  # green
    elif prob < 0.5:
        return "Moderate", "#ffc107"  # amber
    return "High", "#dc3545"  # red


def log_prediction(user_inputs, prob, label, threshold, patient_name):
    """
    Append a simple audit record to logs/predictions.csv
    (timestamp, inputs, prob, label).
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "patient_name": patient_name,
        "probability": prob,
        "label": label,
        "threshold": threshold,
    }
    # Flatten inputs
    for k, v in user_inputs.items():
        record[k] = v

    csv_path = os.path.join(LOG_DIR, "predictions.csv")
    df_row = pd.DataFrame([record])
    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)
    logger.info(
        "Logged prediction: label=%s, prob=%.4f (threshold=%.4f)",
        label,
        prob,
        threshold,
    )


def get_underlying_xgb(model):
    """
    Try to recover the underlying XGBClassifier from a calibrated model.
    If this fails, return None and skip explanations.
    """
    try:
        # Typical structure: CalibratedClassifierCV -> calibrated_classifiers_[0].base_estimator
        if hasattr(model, "calibrated_classifiers_"):
            return model.calibrated_classifiers_[0].base_estimator
        # Fallback: direct estimator
        if hasattr(model, "base_estimator"):
            return model.base_estimator
    except Exception as e:
        logger.warning("Could not extract underlying XGB model: %s", e)
    return None


def shap_like_contributions(xgb_model, X_row, feature_order):
    """
    Use XGBoost's pred_contribs=True to obtain SHAP-like contributions
    without requiring the external shap package.

    Returns a sorted DataFrame with columns: feature, contribution.
    """
    try:
        dmatrix = xgb.DMatrix(X_row[feature_order])
        contribs = xgb_model.get_booster().predict(dmatrix, pred_contribs=True)
        contribs = contribs[0]  # single row
        feature_contribs = contribs[:-1]  # last term is bias
        df_contrib = pd.DataFrame(
            {"feature": feature_order, "contribution": feature_contribs}
        )
        df_contrib["abs_contribution"] = df_contrib["contribution"].abs()
        df_contrib = df_contrib.sort_values("abs_contribution", ascending=False)
        return df_contrib
    except Exception as e:
        logger.warning("Failed to compute SHAP-like contributions: %s", e)
        return None


def build_html_report(patient_name, user_inputs, prob, label, threshold, interpretation):
    """
    Build a simple HTML report for one patient.
    The user can download this and export to PDF via browser.
    """
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in user_inputs.items()
    )
    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Childhood TB Risk Report – {patient_name}</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 24px;
}}
h1 {{
    color: #005eb8;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin-top: 16px;
}}
th, td {{
    border: 1px solid #ccc;
    padding: 8px;
    font-size: 12px;
}}
th {{
    background-color: #f5f5f5;
}}
.summary {{
    border: 1px solid #005eb8;
    padding: 12px;
    margin-top: 16px;
}}
</style>
</head>
<body>
<h1>Childhood Tuberculosis Risk Report</h1>
<p><strong>Patient name:</strong> {patient_name or "Not provided"}</p>
<div class="summary">
  <p><strong>Predicted probability of TB:</strong> {prob*100:.1f}%</p>
  <p><strong>Model decision (threshold {threshold:.3f}):</strong> {label}</p>
  <p><strong>Risk interpretation:</strong> {interpretation}</p>
</div>

<h2>Clinical input summary</h2>
<table>
  <tr><th>Variable</th><th>Value</th></tr>
  {rows}
</table>

<p style="margin-top: 24px; font-size: 11px; color: #666;">
This report is generated from a research prototype trained on synthetic data.
It is not intended for clinical use or individual patient management.
</p>
</body>
</html>
"""
    return html


# -------------------------------------------------------------------
# Overview page
# -------------------------------------------------------------------

def page_overview(threshold):
    st.title("Childhood Tuberculosis Risk Prediction Prototype")

    st.markdown("""
This application presents a research prototype demonstrating calibrated
machine learning–based risk prediction for **childhood pulmonary tuberculosis**
using routine clinical and epidemiological data (ages 0–15).
""")

    st.subheader("Scope of the prototype")
    st.markdown(f"""
- Synthetic cohort (~1 500 children) using WHO-inspired distributions  
- Multiple models compared; **calibrated XGBoost** selected  
- Operating threshold ≈ **{threshold:.3f}**, prioritising sensitivity  
- Subgroup fairness evaluated (age, HIV, malnutrition, sex)  
- Includes an interactive single-patient calculator  
""")

    st.info(
        "This is a research and teaching prototype based on synthetic data. "
        "It is not clinically validated and must not be used for patient care."
    )


# -------------------------------------------------------------------
# Risk calculator page
# -------------------------------------------------------------------

def page_single_patient(model, feature_order, threshold, example_patient):
    st.title("Single-Patient Risk Calculator")

    st.markdown("""
Specify the clinical profile of a child with presumptive TB to obtain
a risk estimate and model decision at the current operating threshold.
""")

    ep = example_patient

    with st.form("tb_form"):
        st.subheader("Clinical parameters")

        # NEW: Patient/child name
        patient_name = st.text_input(
            "Patient/Child Name",
            value="",
            placeholder="Enter patient name (optional but recommended)",
        )

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age (years)", 0, 15, ep.get("age_years", 7))
            sex_opt = st.selectbox(
                "Sex", ["Female (0)", "Male (1)"], index=ep.get("sex", 0)
            )
            sex = 1 if "Male" in sex_opt else 0

            cough_weeks = st.slider(
                "Cough duration (weeks)", 0, 8, ep.get("cough_weeks", 3)
            )
            weight_z = st.slider(
                "Weight-for-age z-score",
                -4.0,
                3.0,
                ep.get("weight_zscore", -2.0),
                step=0.1,
            )

            maln_opt = st.selectbox(
                "Malnutrition (moderate/severe)",
                ["No (0)", "Yes (1)"],
                index=ep.get("malnutrition", 1),
            )
            maln = 1 if "Yes" in maln_opt else 0

        with col2:
            fever_opt = st.selectbox(
                "Fever", ["No (0)", "Yes (1)"], index=ep.get("fever", 1)
            )
            fever = 1 if "Yes" in fever_opt else 0

            night_opt = st.selectbox(
                "Night sweats", ["No (0)", "Yes (1)"], index=ep.get("night_sweats", 1)
            )
            night = 1 if "Yes" in night_opt else 0

            contact_opt = st.selectbox(
                "TB contact (household/close)",
                ["No (0)", "Yes (1)"],
                index=ep.get("contact_TB", 1),
            )
            contact = 1 if "Yes" in contact_opt else 0

            hiv_opt = st.selectbox(
                "HIV-positive",
                ["No (0)", "Yes (1)"],
                index=ep.get("HIV_positive", 0),
            )
            hiv = 1 if "Yes" in hiv_opt else 0

            cxr_opt = st.selectbox(
                "Chest X-ray abnormal",
                ["No (0)", "Yes (1)"],
                index=ep.get("CXR_abnormal", 1),
            )
            cxr = 1 if "Yes" in cxr_opt else 0

            gx_opt = st.selectbox(
                "GeneXpert available",
                ["No (0)", "Yes (1)"],
                index=ep.get("GeneXpert_available", 0),
            )
            gx = 1 if "Yes" in gx_opt else 0

            # NEW: interactive BCG scar input
            bcg_opt = st.selectbox(
                "BCG scar present",
                ["No (0)", "Yes (1)"],
                index=ep.get("BCG_scar", 1),
            )
            bcg_scar = 1 if "Yes" in bcg_opt else 0

        submitted = st.form_submit_button("Estimate tuberculosis risk")

    if not submitted:
        st.caption("Fill the fields above and click "
                   "\"Estimate tuberculosis risk\" to obtain a prediction.")
        return

    inputs = {
        "age_years": age,
        "sex": sex,
        "cough_weeks": cough_weeks,
        "fever": fever,
        "night_sweats": night,
        "contact_TB": contact,
        "HIV_positive": hiv,
        "weight_zscore": weight_z,
        "malnutrition": maln,
        "BCG_scar": bcg_scar,
        "CXR_abnormal": cxr,
        "GeneXpert_available": gx,
    }

    X = build_input_df(inputs, feature_order)
    prob = predict_risk(model, X)
    label = "TB-positive" if prob >= threshold else "TB-negative"
    interpretation = interpret_risk(prob)
    band_label, band_color = risk_band(prob)

    log_prediction(inputs, prob, label, threshold, patient_name)

    st.subheader("Model Output")

    st.write(f"**Patient name:** {patient_name or 'Not provided'}")

    # Risk band panel
    st.markdown(
        f"""
<div style="
    border-left: 6px solid {band_color};
    border-radius: 4px;
    padding: 8px 12px;
    background-color: #f8f9fa;
">
<strong>Risk band: {band_label}</strong><br>
Predicted probability of tuberculosis: <strong>{prob*100:.1f}%</strong><br>
Interpretation: <strong>{interpretation}</strong>
</div>
        """,
        unsafe_allow_html=True,
    )

    colA, colB = st.columns(2)

    with colA:
        st.metric("Predicted probability", f"{prob*100:.1f}%")
        st.write(f"Risk category: **{interpretation}**")

    with colB:
        st.metric("Model decision", label, help=f"Threshold = {threshold:.3f}")

    # Explanation section (SHAP-like contributions)
    st.subheader("Model explanation (feature contributions)")

    xgb_model = get_underlying_xgb(model)
    contrib_df = None
    if xgb_model is not None:
        contrib_df = shap_like_contributions(xgb_model, X, feature_order)

    if contrib_df is not None:
        top_k = contrib_df.head(8).copy()
        st.markdown("Top contributing features for this prediction:")
        st.dataframe(
            top_k[["feature", "contribution"]],
            use_container_width=True,
        )
        st.bar_chart(
            top_k.set_index("feature")["contribution"],
            use_container_width=True,
        )
    else:
        st.caption(
            "Feature-level contributions are not available for this model configuration."
        )

    # Downloadable report
    st.subheader("Downloadable report")
    report_html = build_html_report(
        patient_name, inputs, prob, label, threshold, interpretation
    )
    st.download_button(
        label="Download patient report (HTML, PDF-ready)",
        data=report_html,
        file_name="tb_risk_report.html",
        mime="text/html",
    )

    st.info(
        "Predictions and explanations are based entirely on synthetic training data. "
        "In real-world settings, such tools require extensive clinical validation "
        "and regulatory review."
    )


# -------------------------------------------------------------------
# Performance page
# -------------------------------------------------------------------

def page_performance(summary_text):
    st.title("Model Performance and Fairness")

    if summary_text:
        st.subheader("Summary")
        st.code(summary_text)
    else:
        st.warning("results_summary.txt not found.")

    st.subheader("Diagnostic Plots")

    figs = {
        "ROC curve (test set)": "roc_curve_test.png",
        "Calibration curve (test set)": "calibration_curve_test.png",
        "Subgroup AUROC": "subgroup_auc_bar.png",
        "Uncertainty distribution": "uncertainty_hist.png",
    }

    for title, fname in figs.items():
        path = os.path.join(FIG_DIR, fname)
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            # use_container_width=True avoids the deprecation warning
            st.image(path, use_container_width=True)
        else:
            st.caption(f"{fname} not found in 'figures/'.")


# -------------------------------------------------------------------
# Methods page
# -------------------------------------------------------------------

def page_methods():
    st.title("Methods and Limitations")

    st.subheader("Synthetic Cohort")
    st.markdown("""
- ~1500 children aged 0–15  
- WHO-inspired feature distributions  
- TB labels via latent logistic model  
- Prevalence ≈ 20–25%  
""")

    st.subheader("Model Approach")
    st.markdown("""
- Logistic regression, Random Forest, XGBoost, LightGBM evaluated  
- **Calibrated XGBoost** selected  
- Probability calibration via isotonic regression  
- Threshold chosen using Youden’s J  
- Subgroup fairness checked (age, HIV, malnutrition, sex)  
""")

    st.subheader("Limitations")
    st.markdown("""
- Fully synthetic data  
- No imaging or microbiology data  
- Not clinically validated  
- Not a medical device  
""")

    st.info(
        "This prototype demonstrates transparent and calibrated risk modelling "
        "for an important paediatric health problem using only synthetic data."
    )


# -------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Childhood TB Risk Prototype",
        layout="centered",
    )

    # Simple WHO-like blue/grey styling
    st.markdown(
        """
<style>
    .stApp {
        background-color: #f5f7fa;
    }
    h1, h2, h3 {
        color: #005eb8;
    }
</style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Childhood TB Risk Prototype")
    st.sidebar.markdown("""
**Version:** 1.0  
**Author:** Nahimana Emmanuel  

Research and educational prototype for calibrated TB risk prediction
using synthetic data.
""")

    model, feat_order, thr, example, summary = load_artifacts()

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Risk calculator", "Performance", "Methods and limitations"],
    )

    if page == "Overview":
        page_overview(thr)
    elif page == "Risk calculator":
        page_single_patient(model, feat_order, thr, example)
    elif page == "Performance":
        page_performance(summary)
    else:
        page_methods()

    st.markdown(
        """
<hr>
<small>
This prototype is built on synthetic data and is not intended for clinical use.
It is presented for research and teaching purposes only.
</small>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
