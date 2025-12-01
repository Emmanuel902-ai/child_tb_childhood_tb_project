import os
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# Paths and artifact loading
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "figures")


@st.cache_resource
def load_artifacts():
    """
    Load the calibrated model, feature order, classification threshold,
    example patient profile, and textual results summary.

    Raises a clear Streamlit error if required artifacts are missing.
    """
    # Load calibrated model
    model_path = os.path.join(MODELS_DIR, "calibrated_xgb.pkl")
    if not os.path.exists(model_path):
        st.error(
            f"Missing model file: {model_path}. "
            "Please ensure the trained calibrated_xgb.pkl is available "
            "in the 'models' directory."
        )
        st.stop()

    with open(model_path, "rb") as f:
        calibrated_model = pickle.load(f)

    # Load feature order
    feat_path = os.path.join(MODELS_DIR, "feature_order.json")
    if not os.path.exists(feat_path):
        st.error(
            f"Missing feature order file: {feat_path}. "
            "Please ensure feature_order.json is available in the 'models' directory."
        )
        st.stop()

    with open(feat_path, "r") as f:
        feature_order = json.load(f)

    # Load decision threshold
    thr_path = os.path.join(MODELS_DIR, "threshold.json")
    if not os.path.exists(thr_path):
        st.error(
            f"Missing threshold file: {thr_path}. "
            "Please ensure threshold.json is available in the 'models' directory."
        )
        st.stop()

    with open(thr_path, "r") as f:
        thr_data = json.load(f)
    threshold = float(thr_data["threshold"])

    # Load example patient (for defaults in the UI)
    example_path = os.path.join(MODELS_DIR, "example_patient.json")
    if not os.path.exists(example_path):
        st.error(
            f"Missing example patient file: {example_path}. "
            "Please ensure example_patient.json is available in the 'models' directory."
        )
        st.stop()

    with open(example_path, "r") as f:
        example_patient = json.load(f)

    # Load text summary (optional)
    summary_path = os.path.join(BASE_DIR, "results_summary.txt")
    summary_text = None
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary_text = f.read()

    return calibrated_model, feature_order, threshold, example_patient, summary_text


# -------------------------------------------------------------------
# Helper functions: input construction and prediction
# -------------------------------------------------------------------

def build_input_df(user_inputs: dict, feature_order):
    """
    Convert a dictionary of user inputs to a single-row DataFrame
    with columns in the correct model feature order.
    """
    data = {col: [user_inputs.get(col)] for col in feature_order}
    return pd.DataFrame(data)


def predict_risk(model, X_row: pd.DataFrame) -> float:
    """
    Return the predicted probability of tuberculosis (float in [0, 1]).
    """
    proba = model.predict_proba(X_row)[:, 1][0]
    return float(proba)


def interpret_risk(prob: float) -> str:
    """
    Provide a simple textual interpretation of the predicted risk.
    """
    if prob < 0.2:
        return "Low risk profile"
    elif prob < 0.5:
        return "Moderate risk profile"
    else:
        return "High risk profile"


# -------------------------------------------------------------------
# Page 1: Overview
# -------------------------------------------------------------------

def page_overview(threshold: float):
    st.title("Childhood Tuberculosis Risk Prediction Prototype")

    st.markdown(
        """
This web application presents a research prototype for **clinical risk prediction**
in **childhood pulmonary tuberculosis (TB)** using routine clinical and epidemiological
data for children aged 0–15 years.

The system operates on a **synthetic dataset** designed to reflect the distributions
of key risk factors described in international childhood TB guidance. It is intended
to demonstrate how calibrated machine learning models could support structured
clinical decision-making in high-burden settings.
        """
    )

    st.subheader("Scope and intent of this prototype")
    st.markdown(
        f"""
- Developed using a **synthetic cohort** of approximately 1 500 children with
  WHO-inspired feature distributions  
- Trains and evaluates several models, selecting a **calibrated XGBoost classifier**  
- Chooses an operating **decision threshold of approximately {threshold:.3f}**
  to prioritise high sensitivity  
- Assesses performance and fairness across subgroups (age, HIV status, nutritional status, sex)  
- Provides an interactive **single-patient risk calculator** for demonstration
  and educational purposes  
        """
    )

    st.info(
        "Important: This is a research and teaching prototype built on synthetic data. "
        "It is not a medical device, has not been clinically validated, and must not be "
        "used to guide real patient care."
    )


# -------------------------------------------------------------------
# Page 2: Single-patient risk calculator
# -------------------------------------------------------------------

def page_single_patient(model, feature_order, threshold: float, example_patient: dict):
    st.title("Single-Patient Risk Calculator")

    st.markdown(
        """
This page allows users to specify the clinical and epidemiological profile of a child
with presumptive TB and obtain the model’s **predicted risk** of tuberculosis,
together with the associated decision at the current operating threshold.
        """
    )

    ep = example_patient

    with st.form("tb_risk_form"):
        st.subheader("Clinical and epidemiological inputs")

        col1, col2 = st.columns(2)

        with col1:
            age_years = st.slider(
                "Age (years)",
                min_value=0,
                max_value=15,
                value=int(ep.get("age_years", 7)),
            )
            sex_label = st.selectbox(
                "Sex",
                options=["Female (0)", "Male (1)"],
                index=int(ep.get("sex", 0)),
            )
            sex = 1 if "Male" in sex_label else 0

            cough_weeks = st.slider(
                "Cough duration (weeks)",
                min_value=0,
                max_value=8,
                value=int(ep.get("cough_weeks", 3)),
            )

            weight_zscore = st.slider(
                "Weight-for-age z-score",
                min_value=-4.0,
                max_value=3.0,
                value=float(ep.get("weight_zscore", -2.0)),
                step=0.1,
            )

            malnutrition_label = st.selectbox(
                "Moderate or severe malnutrition",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("malnutrition", 1)),
            )
            malnutrition = 1 if "Yes" in malnutrition_label else 0

        with col2:
            fever_label = st.selectbox(
                "Fever",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("fever", 1)),
            )
            fever = 1 if "Yes" in fever_label else 0

            night_sweats_label = st.selectbox(
                "Night sweats",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("night_sweats", 1)),
            )
            night_sweats = 1 if "Yes" in night_sweats_label else 0

            contact_label = st.selectbox(
                "Known household or close contact with TB",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("contact_TB", 1)),
            )
            contact_TB = 1 if "Yes" in contact_label else 0

            hiv_label = st.selectbox(
                "HIV-positive",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("HIV_positive", 0)),
            )
            HIV_positive = 1 if "Yes" in hiv_label else 0

            bcg_label = st.selectbox(
                "BCG scar present",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("BCG_scar", 1)),
            )
            BCG_scar = 1 if "Yes" in bcg_label else 0

            cxr_label = st.selectbox(
                "Chest X-ray abnormal",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("CXR_abnormal", 1)),
            )
            CXR_abnormal = 1 if "Yes" in cxr_label else 0

            gx_label = st.selectbox(
                "GeneXpert available",
                options=["No (0)", "Yes (1)"],
                index=int(ep.get("GeneXpert_available", 0)),
            )
            GeneXpert_available = 1 if "Yes" in gx_label else 0

        submitted = st.form_submit_button("Estimate tuberculosis risk")

    if not submitted:
        st.caption("Specify the input parameters above and click "
                   "\"Estimate tuberculosis risk\" to obtain a prediction.")
        return

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

    X_row = build_input_df(user_inputs, feature_order)

    prob = predict_risk(model, X_row)
    label = int(prob >= threshold)
    interpretation = interpret_risk(prob)

    st.subheader("Model output")

    colA, colB = st.columns(2)

    with colA:
        st.metric(
            "Predicted probability of tuberculosis",
            f"{prob*100:.1f} %",
        )
        st.write(f"Risk interpretation: **{interpretation}**")

    with colB:
        decision_text = (
            "Classified as TB-positive (probability above threshold)"
            if label == 1
            else "Classified as TB-negative (probability below threshold)"
        )
        st.metric(
            "Model decision at current operating threshold",
            decision_text,
            help=f"Current threshold: {threshold:.3f}",
        )

    st.info(
        "This output is generated from a model trained entirely on synthetic data. "
        "In real settings, any similar tool would require rigorous clinical validation, "
        "regulatory review, and integration with local clinical guidelines."
    )


# -------------------------------------------------------------------
# Page 3: Model performance and figures
# -------------------------------------------------------------------

def page_performance(summary_text: str | None):
    st.title("Model Performance and Fairness Assessment")

    st.markdown(
        """
This page summarises the performance of the final calibrated model on the
synthetic test set and highlights diagnostic plots and subgroup analyses.
        """
    )

    if summary_text:
        st.subheader("Textual summary of results")
        st.code(summary_text, language="text")
    else:
        st.warning(
            "No results summary file (results_summary.txt) was found in the project root."
        )

    st.subheader("Diagnostic plots")

    roc_path = os.path.join(FIG_DIR, "roc_curve_test.png")
    calib_path = os.path.join(FIG_DIR, "calibration_curve_test.png")
    subgroup_path = os.path.join(FIG_DIR, "subgroup_auc_bar.png")
    uncert_path = os.path.join(FIG_DIR, "uncertainty_hist.png")

    if os.path.exists(roc_path):
        st.markdown("**Receiver operating characteristic (ROC) curve – test set**")
        st.image(roc_path, use_column_width=True)
    else:
        st.caption("ROC curve image not found in 'figures/roc_curve_test.png'.")

    if os.path.exists(calib_path):
        st.markdown("**Calibration curve – test set**")
        st.image(calib_path, use_column_width=True)
    else:
        st.caption("Calibration curve image not found in 'figures/calibration_curve_test.png'.")

    if os.path.exists(subgroup_path):
        st.markdown("**AUROC by subgroup (age, HIV status, nutritional status, sex)**")
        st.image(subgroup_path, use_column_width=True)
    else:
        st.caption("Subgroup AUROC figure not found in 'figures/subgroup_auc_bar.png'.")

    if os.path.exists(uncert_path):
        st.markdown("**Distribution of prediction uncertainty**")
        st.image(uncert_path, use_column_width=True)
    else:
        st.caption("Uncertainty histogram not found in 'figures/uncertainty_hist.png'.")


# -------------------------------------------------------------------
# Page 4: Methods and limitations
# -------------------------------------------------------------------

def page_methods():
    st.title("Methods, Data Generation and Limitations")

    st.subheader("Synthetic cohort design")
    st.markdown(
        """
- Cohort size of approximately **1 500 children** aged 0–15 years  
- Features sampled from distributions motivated by **WHO childhood TB guidance**  
- Tuberculosis labels generated via a **latent logistic risk model** such that
  high-risk profiles (e.g. TB contact, HIV-positivity, abnormal chest X-ray,
  malnutrition) are more likely to be TB-positive  
- Overall TB prevalence in the synthetic dataset around **20–25 %**, reflecting
  a high-risk presumptive TB population  
        """
    )

    st.subheader("Models and training approach")
    st.markdown(
        """
- Compared several models, including **logistic regression, random forests, XGBoost and LightGBM**  
- Selected **XGBoost** as the final model based on discrimination, calibration
  and stability on the synthetic test set  
- Applied **isotonic regression** for probability calibration  
- Defined an operating threshold using **Youden’s J statistic**, constrained to
  maintain **high sensitivity** in line with clinical priorities  
- Evaluated performance across subgroups (age bands, HIV status, malnutrition, sex)
  to assess potential fairness concerns  
        """
    )

    st.subheader("Key limitations")
    st.markdown(
        """
- The dataset is **fully synthetic**; real-world performance may differ substantially  
- No detailed imaging, microbiology or longitudinal treatment data are included  
- The prototype does **not** incorporate drug resistance, previous TB history or
  other co-morbidities  
- The application has **not** undergone clinical validation or regulatory review  
        """
    )

    st.subheader("Future directions")
    st.markdown(
        """
- External validation on **real, multicountry paediatric TB datasets**  
- Co-design with clinicians to refine decision thresholds, explanations and workflow integration  
- Incorporation into a broader **clinical decision support framework** with clear
  governance, audit logging and training  
        """
    )

    st.info(
        "This project illustrates how a junior data scientist can design a transparent, "
        "calibrated and fairness-aware risk prediction prototype for a high-impact "
        "clinical question, using only synthetic data and open-source tools."
    )


# -------------------------------------------------------------------
# Application entry point
# -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Childhood Tuberculosis Risk Prototype",
        page_icon=None,
        layout="centered",
    )

    # Sidebar: project information and navigation
    st.sidebar.title("Childhood TB Risk Prototype")
    st.sidebar.markdown(
        """
**Version:** 1.0  
**Author:** Nahimana Emmanuel  

This prototype demonstrates calibrated risk prediction for childhood TB
using synthetic data. It is intended for research and educational use only.
        """
    )

    calibrated_model, feature_order, threshold, example_patient, summary_text = load_artifacts()

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "Overview",
            "Risk calculator",
            "Performance",
            "Methods and limitations",
        ],
    )

    if page == "Overview":
        page_overview(threshold)
    elif page == "Risk calculator":
        page_single_patient(calibrated_model, feature_order, threshold, example_patient)
    elif page == "Performance":
        page_performance(summary_text)
    elif page == "Methods and limitations":
        page_methods()

    # Footer
    st.markdown(
        """
<hr>
<small>
This application is a research and teaching prototype built on synthetic data.  
It does not constitute medical advice and must not be used to guide individual patient care.
</small>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
