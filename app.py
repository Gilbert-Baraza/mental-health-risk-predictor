from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import altair as alt
import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from utils.preprocessing import (
    COUNTRY_OPTIONS,
    EXERCISE_FREQUENCY_OPTIONS,
    GENDER_OPTIONS,
    OCCUPATION_OPTIONS,
    SLEEP_QUALITY_OPTIONS,
    SOCIAL_SUPPORT_OPTIONS,
    get_feature_columns,
    get_model,
    preprocess_input,
)


APP_TITLE = "Mental Health Risk Prediction System"
MODEL_PATHS = [Path("mental_health_model.pkl"), Path("mental_health_app_model.pkl")]
RISK_LABELS = ["Low Risk", "Moderate Risk", "High Risk"]


class ModelLoadError(RuntimeError):
    """Raised when no usable model artifact can be loaded."""


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    """Load the trained artifact from disk.

    The app first checks the filename requested by the project brief and then
    falls back to the filename found in this workspace.
    """
    existing_paths = [path for path in MODEL_PATHS if path.exists()]
    if not existing_paths:
        searched = ", ".join(str(path) for path in MODEL_PATHS)
        raise ModelLoadError(f"No model file found. Expected one of: {searched}")

    last_error: Exception | None = None
    for path in existing_paths:
        try:
            artifact = joblib.load(path)
            if not hasattr(get_model(artifact), "predict"):
                raise ModelLoadError(f"{path} does not contain a predict-capable estimator.")
            return artifact
        except Exception as exc:
            last_error = exc

    raise ModelLoadError(f"Model loading failed: {last_error}")


def _risk_label(value: Any, class_count: int | None = None) -> str:
    text = str(value).strip()
    normalized = text.lower().replace("_", " ")

    if "low" in normalized:
        return "Low Risk"
    if "moderate" in normalized or "medium" in normalized:
        return "Moderate Risk"
    if "high" in normalized or "severe" in normalized:
        return "High Risk"

    try:
        numeric = int(float(text))
    except ValueError:
        return text if text.endswith("Risk") else f"{text} Risk"

    if class_count == 2:
        return ["Low Risk", "High Risk"][max(0, min(numeric, 1))]
    if 0 <= numeric < len(RISK_LABELS):
        return RISK_LABELS[numeric]
    return f"Class {numeric}"


def predict_risk(artifact: Any, raw_input: dict[str, Any]) -> dict[str, Any]:
    """Preprocess form input and return prediction, probabilities, and confidence."""
    model = get_model(artifact)
    features = preprocess_input(raw_input, artifact)

    prediction = model.predict(features)[0]
    classes = list(getattr(model, "classes_", []))
    class_count = len(classes) or None
    predicted_label = _risk_label(prediction, class_count)

    probabilities: pd.DataFrame
    if hasattr(model, "predict_proba"):
        proba_values = model.predict_proba(features)[0]
        labels = [_risk_label(class_value, class_count) for class_value in classes]
        probabilities = pd.DataFrame(
            {"Risk Level": labels, "Probability": [float(value) for value in proba_values]}
        )
        confidence = float(max(proba_values))
    else:
        probabilities = pd.DataFrame(
            {"Risk Level": [predicted_label], "Probability": [1.0]}
        )
        confidence = 1.0

    return {
        "prediction": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "features": features,
    }


def generate_recommendations(risk_level: str) -> list[str]:
    """Return practical, non-diagnostic recommendations for the predicted risk."""
    normalized = risk_level.lower()
    if "high" in normalized:
        return [
            "Consider speaking with a licensed mental health professional soon.",
            "Share how you are feeling with someone you trust today.",
            "Reduce avoidable stressors and protect sleep, meals, and hydration.",
            "If you feel unsafe or at risk of harming yourself, contact local emergency services or a crisis hotline immediately.",
        ]
    if "moderate" in normalized:
        return [
            "Schedule regular breaks and use a simple stress-management routine.",
            "Aim for consistent sleep and light physical activity most days.",
            "Talk with a trusted person, mentor, counselor, or clinician if symptoms persist.",
            "Track stress, anxiety, and sleep for one to two weeks to spot patterns.",
        ]
    return [
        "Maintain healthy sleep, exercise, and social connection habits.",
        "Keep monitoring changes in stress, mood, appetite, and focus.",
        "Use preventive practices such as journaling, mindfulness, or planned downtime.",
        "Seek support early if symptoms become frequent or disruptive.",
    ]


def build_prediction_report(
    raw_input: dict[str, Any],
    prediction: dict[str, Any],
    recommendations: list[str],
) -> str:
    """Create a downloadable text report for the current assessment."""
    probabilities = prediction["probabilities"].copy()
    probabilities["Probability"] = probabilities["Probability"].map(lambda value: f"{value:.1%}")

    return "\n".join(
        [
            APP_TITLE,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Prediction",
            f"Risk level: {prediction['prediction']}",
            f"Model confidence: {prediction['confidence']:.1%}",
            "",
            "Class probabilities",
            probabilities.to_string(index=False),
            "",
            "Submitted inputs",
            json.dumps(raw_input, indent=2),
            "",
            "Recommendations",
            *[f"- {item}" for item in recommendations],
            "",
            "Important note: This tool is for educational decision support only and is not a medical diagnosis.",
        ]
    )


def render_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"], .stApp {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        :root {
            --primary: #0f766e;
            --primary-light: #14b8a6;
            --bg-main: #f8fafc;
            --card-bg: #ffffff;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            
            --low-risk-bg: #f0fdf4;
            --low-risk-border: #bbf7d0;
            --low-risk-text: #166534;
            
            --mod-risk-bg: #fffbeb;
            --mod-risk-border: #fef3c7;
            --mod-risk-text: #92400e;
            
            --high-risk-bg: #fef2f2;
            --high-risk-border: #fee2e2;
            --high-risk-text: #991b1b;
        }

        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            max-width: 1250px !important;
        }

        /* Premium Hero Section */
        .hero {
            background: linear-gradient(135deg, #0f766e 0%, #115e59 40%, #042f2e 100%);
            color: #ffffff;
            padding: 2.2rem 2.2rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(15, 118, 110, 0.15), 0 8px 10px -6px rgba(15, 118, 110, 0.15);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }

        .hero::after {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 300px;
            height: 300px;
            background: radial-gradient(circle, rgba(20, 184, 166, 0.2) 0%, transparent 70%);
            border-radius: 50%;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 700;
            color: #ffffff !important;
            letter-spacing: -0.025em;
            line-height: 1.2;
        }

        .hero p {
            color: #ccfbf1;
            font-size: 1.02rem;
            font-weight: 300;
            line-height: 1.5;
            margin-top: 0.6rem;
            margin-bottom: 0;
            max-width: 850px;
        }

        /* Sidebar Custom Styling */
        [data-testid="stSidebar"] {
            background-color: #f1f5f9 !important;
            border-right: 1px solid #e2e8f0;
        }

        .sidebar-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--primary);
            margin-top: 0.5rem;
            margin-bottom: 1.5rem;
        }

        .sidebar-section {
            background-color: #ffffff;
            padding: 1.1rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }

        .sidebar-section h4 {
            margin-top: 0;
            margin-bottom: 0.6rem;
            font-size: 0.92rem;
            font-weight: 600;
            color: var(--text-dark);
        }

        .sidebar-section p, .sidebar-section li {
            font-size: 0.82rem;
            color: var(--text-muted);
            line-height: 1.45;
        }

        .sidebar-section ul {
            margin: 0;
            padding-left: 1.1rem;
            margin-top: 0.25rem;
        }

        /* Form and UI inputs */
        .stForm {
            background-color: #ffffff !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 16px !important;
            padding: 1.8rem !important;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -2px rgba(0,0,0,0.02) !important;
            margin-bottom: 0px;
        }

        /* Form subheaders */
        .form-subheader {
            font-size: 1.35rem;
            font-weight: 600;
            color: var(--text-dark);
            margin-bottom: 1.5rem;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 0.5rem;
        }

        /* Custom Risk Assessment Card */
        .result-card {
            border-radius: 16px;
            padding: 1.6rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.02);
            animation: fadeIn 0.4s ease-out;
            border-left: 6px solid transparent;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-card.risk-low {
            background-color: var(--low-risk-bg);
            border: 1px solid var(--low-risk-border);
            border-left: 6px solid #22c55e;
            color: var(--low-risk-text);
        }

        .result-card.risk-moderate {
            background-color: var(--mod-risk-bg);
            border: 1px solid var(--mod-risk-border);
            border-left: 6px solid #eab308;
            color: var(--mod-risk-text);
        }

        .result-card.risk-high {
            background-color: var(--high-risk-bg);
            border: 1px solid var(--high-risk-border);
            border-left: 6px solid #ef4444;
            color: var(--high-risk-text);
        }

        .result-card h2 {
            font-size: 1.7rem;
            font-weight: 700;
            margin: 0 0 0.4rem 0;
            color: inherit !important;
        }

        .result-card p {
            font-size: 1rem;
            margin: 0;
            opacity: 0.9;
        }

        /* Recommendation items */
        .recommendation-box {
            background-color: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.01);
        }

        .recommendation-item {
            display: flex;
            align-items: flex-start;
            gap: 0.6rem;
            margin-bottom: 0.6rem;
            font-size: 0.92rem;
            color: var(--text-dark);
            line-height: 1.45;
        }

        .recommendation-item:last-child {
            margin-bottom: 0;
        }

        .recommendation-icon {
            color: var(--primary);
            font-weight: bold;
            flex-shrink: 0;
        }

        /* Styled Submit Button */
        button[kind="primaryFormSubmit"] {
            background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%) !important;
            color: #ffffff !important;
            border: none !important;
            padding: 0.7rem 2rem !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 10px rgba(15, 118, 110, 0.15) !important;
            transition: all 0.2s ease !important;
            height: auto !important;
        }

        button[kind="primaryFormSubmit"]:hover {
            background: linear-gradient(135deg, #115e59 0%, #0f766e 100%) !important;
            box-shadow: 0 6px 14px rgba(15, 118, 110, 0.25) !important;
            transform: translateY(-1px);
        }

        /* Info notices */
        .notice-box {
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            color: #1e3a8a;
            font-size: 0.9rem;
            margin-top: 1.5rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <p>
                An interactive decision-support tool designed to estimate mental health risk levels based on lifestyle, sleep quality, workload, stress, and supportive indicators. This screening aid is built to support proactive mental wellness and self-reflection.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def collect_user_input() -> dict[str, Any]:
    with st.form("risk_assessment_form", border=False):
        st.markdown('<div class="form-subheader">Lifestyle & Wellness Questionnaire</div>', unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown('<p style="font-weight: 600; color: #0f766e; margin-bottom: 0.5rem; font-size: 0.95rem;">👥 Demographics</p>', unsafe_allow_html=True)
            age = st.number_input("Age", min_value=13, max_value=100, value=28, step=1)
            gender = st.selectbox("Gender", GENDER_OPTIONS)
            country = st.selectbox("Country", COUNTRY_OPTIONS, index=9)
            occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS, index=6)
            work_hours = st.slider("Work hours per week", 0, 100, 40)

        with col_b:
            st.markdown('<p style="font-weight: 600; color: #0f766e; margin-bottom: 0.5rem; font-size: 0.95rem;">💤 Lifestyle & Habits</p>', unsafe_allow_html=True)
            sleep_hours = st.slider("Sleep hours", 0.0, 14.0, 7.0, 0.5)
            sleep_quality = st.selectbox("Sleep quality", SLEEP_QUALITY_OPTIONS, index=2)
            exercise = st.selectbox("Exercise frequency", EXERCISE_FREQUENCY_OPTIONS, index=2)
            social_support = st.selectbox("Social support", SOCIAL_SUPPORT_OPTIONS)
            screen_time = st.slider("Screen time hours", 0.0, 18.0, 6.0, 0.5)

        with col_c:
            st.markdown('<p style="font-weight: 600; color: #0f766e; margin-bottom: 0.5rem; font-size: 0.95rem;">🧠 Wellness History & Scores</p>', unsafe_allow_html=True)
            therapy_history = st.radio("Therapy history", ["No", "Yes"], horizontal=True)
            family_history = st.radio(
                "Family history of mental illness", ["No", "Yes"], horizontal=True
            )
            stress_score = st.slider("Stress score", 0, 10, 5)
            anxiety_score = st.slider("Anxiety score", 0, 10, 4)
            depression_score = st.slider("Depression score", 0, 10, 4)

        st.markdown('<div style="margin-top: 1rem; margin-bottom: 0.5rem;"></div>', unsafe_allow_html=True)
        col_d, col_e = st.columns(2)
        with col_d:
            st.markdown('<p style="font-weight: 600; color: #0f766e; margin-bottom: 0.25rem; font-size: 0.95rem;">💼 Pressures</p>', unsafe_allow_html=True)
            pressure = st.slider("Academic/job pressure", 0, 10, 5)
        with col_e:
            st.markdown('<p style="font-weight: 600; color: #0f766e; margin-bottom: 0.25rem; font-size: 0.95rem;">💰 Finance</p>', unsafe_allow_html=True)
            financial_stress = st.slider("Financial stress score", 0, 10, 4)

        st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    values = {
        "age": age,
        "gender": gender,
        "country": country,
        "occupation": occupation,
        "work_hours_per_week": work_hours,
        "screen_time_hours": screen_time,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "exercise_frequency": exercise,
        "social_support": social_support,
        "therapy_history": therapy_history,
        "family_history_mental_illness": family_history,
        "stress_score": stress_score,
        "anxiety_score": anxiety_score,
        "depression_score": depression_score,
        "academic_or_job_pressure": pressure,
        "financial_stress_score": financial_stress,
    }
    values["_submitted"] = submitted
    return values


def render_probability_chart(probabilities: pd.DataFrame, predicted_label: str) -> None:
    chart_data = probabilities.copy()
    chart_data["ProbabilityLabel"] = chart_data["Probability"].map(lambda value: f"{value:.1%}")
    chart_data["Predicted"] = chart_data["Risk Level"].eq(predicted_label)

    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Risk Level:N", sort=RISK_LABELS, title=None, axis=alt.Axis(labelAngle=0, labelFont="Outfit", labelFontSize=12, labelColor="#475569")),
            y=alt.Y("Probability:Q", title="Probability Score", axis=alt.Axis(format="%", labelFont="Outfit", labelFontSize=11, labelColor="#475569", titleFont="Outfit", titleFontSize=12, titleColor="#475569")),
            color=alt.Color(
                "Predicted:N",
                scale=alt.Scale(domain=[True, False], range=["#0f766e", "#cbd5e1"]),
                legend=None,
            ),
            tooltip=["Risk Level:N", alt.Tooltip("Probability:Q", format=".1%")],
        )
        .properties(height=240)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )
    st.altair_chart(chart, use_container_width=True)


def render_prediction_result(raw_input: dict[str, Any], prediction: dict[str, Any]) -> None:
    risk = prediction["prediction"]
    risk_class = "risk-high" if "High" in risk else "risk-moderate" if "Moderate" in risk else "risk-low"
    recommendations = generate_recommendations(risk)

    st.markdown(
        f"""
        <div class="result-card {risk_class}">
            <h2>{risk}</h2>
            <p>Model confidence: <strong>{prediction['confidence']:.1%}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="form-subheader" style="margin-top: 1.5rem; margin-bottom: 1rem;">Probability Analysis</div>', unsafe_allow_html=True)
    render_probability_chart(prediction["probabilities"], risk)

    st.markdown('<div class="form-subheader" style="margin-top: 1.5rem; margin-bottom: 1rem;">Wellness Action Plan</div>', unsafe_allow_html=True)
    
    rec_html = '<div class="recommendation-box">'
    for rec in recommendations:
        rec_html += f'<div class="recommendation-item"><span class="recommendation-icon">✦</span><span>{rec}</span></div>'
    rec_html += '</div>'
    st.markdown(rec_html, unsafe_allow_html=True)

    report = build_prediction_report(raw_input, prediction, recommendations)
    st.download_button(
        "Download Full Prediction Report",
        data=report,
        file_name=f"mental_health_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="assets/logo.png", layout="wide")
    render_css()

    with st.sidebar:
        logo_path = Path("assets/logo.png")
        if logo_path.exists():
            st.image(str(logo_path), width=90)
            
        st.markdown('<div class="sidebar-title">Mental Health Risk Portal</div>', unsafe_allow_html=True)
        
        st.markdown(
            """
            <div class="sidebar-section">
                <h4>About this Tool</h4>
                <p>
                    This app evaluates demographic, lifestyle, sleep, and self-reported wellness factors using a machine learning model to estimate mental health risk levels.
                </p>
            </div>
            <div class="sidebar-section">
                <h4>Support Helpline Resources</h4>
                <p><strong>In Immediate Danger?</strong> Call your local emergency services (e.g., 911, 999, 112).</p>
                <p><strong>Crisis Support lines:</strong></p>
                <ul>
                    <li><strong>US/Canada:</strong> Call or text 988</li>
                    <li><strong>UK:</strong> Call 111 / 116 123</li>
                </ul>
                <p style="margin-top:0.5rem;"><a href="https://findahelpline.com/" target="_blank" style="color:#0f766e; font-weight:600; text-decoration:none;">Find a Global Helpline ↗</a></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("---")
        st.caption("Educational decision support only. This is not a medical diagnosis.")

    render_header()

    try:
        artifact = load_model()
    except ModelLoadError as exc:
        st.error(str(exc))
        st.info("Place the trained model file in the project root and restart the app.")
        return
    except Exception as exc:
        st.error("The model file was found, but loading failed.")
        st.exception(exc)
        return

    # Two columns for form and results
    col_form, col_results = st.columns([1.1, 0.9], gap="large")

    with col_form:
        user_input = collect_user_input()
        submitted = bool(user_input.pop("_submitted"))

    with col_results:
        if submitted:
            try:
                prediction = predict_risk(artifact, user_input)
                render_prediction_result(user_input, prediction)
            except ValueError as exc:
                st.warning(f"Invalid input: {exc}")
            except Exception as exc:
                st.error("Prediction failed. Please verify that the model and preprocessing artifacts match.")
                st.exception(exc)
        else:
            # Let's show a beautiful initial prompt card
            st.markdown(
                """
                <div style="border: 2px dashed #cbd5e1; border-radius: 16px; padding: 3rem 2rem; text-align: center; color: #64748b; margin-top: 1rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📋</div>
                    <h3 style="margin: 0 0 0.5rem 0; color: #334155; font-size: 1.25rem;">Awaiting Assessment</h3>
                    <p style="margin: 0; font-size: 0.95rem; line-height: 1.5;">
                        Please complete the personal and lifestyle questionnaire on the left and click <strong>Predict Risk</strong> to view the assessment results.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            st.markdown(
                """
                <div class="notice-box">
                    <strong>Disclaimer:</strong> This application is meant to offer educational decision support only. 
                    It is not a clinical diagnosis or medical evaluation.
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

