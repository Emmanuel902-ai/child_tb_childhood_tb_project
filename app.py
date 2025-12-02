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
    """Load artifacts: model, feature order, threshold, example patient, summary text."""
    model_path = os.path.join(MODELS_DIR, "calibrated_xgb.pkl")
    if not os.path.exists(model_path):
        st.error(f"Missing model file: {model_path}")
        st.stop()
    with open(model_path, "rb") as f:
        calibrated_model = pickle.load(f)

    feat_path = os.path.join(MODELS_DIR, "feature_order.json")
    if not os.path.exists(feat_path):
        st.error(f"Missing feature order file: {feat_path}")
        st.stop()
    with open(feat_path, "r") as f:
        feature_order = json.load(f)

    thr_path = os.path.join(MODELS_DIR, "threshold.json")
    if not os.path.exists(thr_path):
        st.error(f"Missing threshold file: {thr_path}")
        st.stop()
    with open(thr_path, "r") as f:
        threshold = float(json.load(f)["threshold"])

    example_path = os.path.join(MODELS_DIR, "example_patient.json")
    if not os.path.exists(example_path):
        st.error(f"Missing example patient file: {example_path}")
        st.stop()
    with open(example_path, "r") as f:
        example_patient = json.load(f)

    summary_path = os.path.join(BASE_DIR, "results_summary.txt")
    summary_text = None
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_text = f.read()

    return calibrated_model, feature_order, threshold, example_patient, summary_text


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def build_input_df(user_inputs, feature_order):
    return pd.DataFrame({col: [user_inputs.get(col)] for col in feature_order})


def predict_risk(model, X_row):
    return float(model.predict_proba(X_row)[:, 1][0])


def interpret_risk(prob):
    if prob < 0.2:
        return "Low risk profile"
    if prob < 0.5:
        return "Moderate risk profile"
    return "High risk profile"


def risk_band(prob):
    if prob < 0.2:
        return "Low", "#198754"      # green
    elif prob < 0.5:
        return "Moderate", "#ffc107" # amber
    return "High", "#dc3545"         # red


def log_prediction(user_inputs, prob, label, threshold, patient_name):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "patient_name": patient_name,
        "probability": prob,
        "label": label,
        "threshold": threshold,
    }
    for k, v in user_inputs.items():
        record[k] = v

    csv_path = os.path.join(LOG_DIR, "predictions.csv")
    df_row = pd.DataFrame([record])

    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)


def build_html_report(patient_name, user_inputs, prob, label, threshold, interpretation):
    rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in user_inputs.items()
    )

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TB Risk Report – {patient_name}</title>
<style>
body {{ font-family: Arial; margin: 24px; }}
h1 {{ color: #005eb8; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 8px; font-size: 12px; }}
.summary {{ border: 1px solid #005eb8; padding: 12px; margin-top: 16px; }}
</style>
</head>
<body>

<h1>Childhood TB Risk Report</h1>
<p><strong>Patient name:</strong> {patient_name}</p>

<div class="summary">
  <p><strong>Predicted probability:</strong> {prob*100:.1f}%</p>
  <p><strong>Decision threshold:</strong> {threshold:.3f}</p>
  <p><strong>Model decision:</strong> {label}</p>
  <p><strong>Risk interpretation:</strong> {interpretation}</p>
</div>

<h2>Clinical Inputs</h2>
<table>
<tr><th>Variable</th><th>Value</th></tr>
{rows}
</table>

</body>
</html>
"""


# -------------------------------------------------------------------
# Risk Calculator Page
# -------------------------------------------------------------------

def page_single_patient(model, feature_order, threshold, example_patient):

    st.title("Single-Patient Risk Calculator")

    with st.form("tb_form"):
        st.subheader("Patient Information")

        patient_name = st.text_input(
            "Patient/Child Name",
            value="",
            placeholder="Enter patient name (optional but recommended)"
        )

        col1, col2 = st.columns(2)

        # First column
        with col1:
            age = st.slider("Age (years)", 0, 15, example_patient.get("age_years", 7))
            sex_opt = st.selectbox("Sex", ["Female (0)", "Male (1)"], index=example_patient.get("sex", 0))
            sex = 1 if "Male" in sex_opt else 0
            cough_weeks = st.slider("Cough duration (weeks)", 0, 8, example_patient.get("cough_weeks", 3))
            weight_z = st.slider(
                "Weight-for-age z-score", -4.0, 3.0, example_patient.get("weight_zscore", -2.0), step=0.1
            )
            maln_opt = st.selectbox(
                "Malnutrition", ["No (0)", "Yes (1)"], index=example_patient.get("malnutrition", 1)
            )
            maln = 1 if "Yes" in maln_opt else 0

        # Second column
        with col2:
            fever = 1 if "Yes" in st.selectbox(
                "Fever", ["No (0)", "Yes (1)"], index=example_patient.get("fever", 1)
            ) else 0
            night = 1 if "Yes" in st.selectbox(
                "Night sweats", ["No (0)", "Yes (1)"], index=example_patient.get("night_sweats", 1)
            ) else 0
            contact = 1 if "Yes" in st.selectbox(
                "TB contact", ["No (0)", "Yes (1)"], index=example_patient.get("contact_TB", 1)
            ) else 0
            hiv = 1 if "Yes" in st.selectbox(
                "HIV-positive", ["No (0)", "Yes (1)"], index=example_patient.get("HIV_positive", 0)
            ) else 0
            cxr = 1 if "Yes" in st.selectbox(
                "Chest X-ray abnormal", ["No (0)", "Yes (1)"], index=example_patient.get("CXR_abnormal", 1)
            ) else 0
            gx = 1 if "Yes" in st.selectbox(
                "GeneXpert available", ["No (0)", "Yes (1)"], index=example_patient.get("GeneXpert_available", 0)
            ) else 0

        submitted = st.form_submit_button("Estimate tuberculosis risk")

    if not submitted:
        return

    # Build input record
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
        "BCG_scar": example_patient.get("BCG_scar"),
        "CXR_abnormal": cxr,
        "GeneXpert_available": gx,
    }

    X = build_input_df(inputs, feature_order)
    prob = predict_risk(model, X)
    label = "TB-positive" if prob >= threshold else "TB-negative"
    interpretation = interpret_risk(prob)
    band_label, band_color = risk_band(prob)

    log_prediction(inputs, prob, label, threshold, patient_name)

    st.subheader("Prediction Summary")
    st.write(f"**Patient name:** {patient_name or 'Not provided'}")

    st.markdown(
        f"""
<div style="
    border-left: 6px solid {band_color};
    padding: 12px;
    border-radius: 4px;
    background-color: #f8f9fa;
">
<strong>Risk band: {band_label}</strong><br>
Predicted probability: <strong>{prob*100:.1f}%</strong><br>
Interpretation: <strong>{interpretation}</strong>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Download report
    report_html = build_html_report(
        patient_name, inputs, prob, label, threshold, interpretation
    )

    st.download_button(
        label="Download patient report (HTML, PDF-ready)",
        data=report_html,
        file_name=f"tb_risk_report_{patient_name.replace(' ', '_') or 'patient'}.html",
        mime="text/html",
    )


# -------------------------------------------------------------------
# Simple pages (overview, performance, methods)
# -------------------------------------------------------------------

def page_overview(threshold):
    st.title("Childhood TB Risk Prediction Prototype")
    st.markdown("""
A calibrated machine-learning prototype for childhood pulmonary tuberculosis,
built using synthetic data for demonstration and educational purposes.
""")
    st.write(f"Operating threshold: **{threshold:.3f}**")


def page_performance(summary_text):
    st.title("Model Performance and Fairness")
    if summary_text:
        st.code(summary_text)

    figs = {
        "ROC curve": "roc_curve_test.png",
        "Calibration curve": "calibration_curve_test.png",
        "Subgroup AUROC": "subgroup_auc_bar.png",
        "Uncertainty distribution": "uncertainty_hist.png",
    }

    for title, fname in figs.items():
        path = os.path.join(FIG_DIR, fname)
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, use_container_width=True)
        else:
            st.caption(f"{fname} not found.")


def page_methods():
    st.title("Methods and Limitations")
    st.write("""
- Synthetic cohort (~1500 children)
- WHO-inspired feature distributions
- Calibrated XGBoost classifier
- Subgroup fairness evaluation
- No clinical validation (research prototype)
""")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Childhood TB Risk Prototype", layout="centered")

    st.sidebar.title("Navigation")
    st.sidebar.markdown("**Version:** 1.0")

    model, feature_order, threshold, example_patient, summary_text = load_artifacts()

    page = st.sidebar.radio(
        "Menu",
        ["Overview", "Risk calculator", "Performance", "Methods and limitations"],
    )

    if page == "Overview":
        page_overview(threshold)
    elif page == "Risk calculator":
        page_single_patient(model, feature_order, threshold, example_patient)
    elif page == "Performance":
        page_performance(summary_text)
    else:
        page_methods()


if __name__ == "__main__":
    main()
