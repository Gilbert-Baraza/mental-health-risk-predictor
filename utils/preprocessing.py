"""Preprocessing utilities for the mental health risk model.

The uploaded artifact is a joblib dictionary, not an sklearn Pipeline. It
stores the fitted estimator plus separate preprocessing artifacts. This module
rebuilds the exact final 35-column feature matrix expected by the trained
DecisionTreeClassifier.
"""

from __future__ import annotations

from typing import Any, Mapping
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


FEATURE_COLUMNS: list[str] = [
    "age",
    "work_hours_per_week",
    "screen_time_hours",
    "sleep_hours",
    "sleep_quality",
    "exercise_frequency",
    "stress_score",
    "anxiety_score",
    "depression_score",
    "social_support",
    "therapy_history",
    "family_history_mental_illness",
    "academic_or_job_pressure",
    "financial_stress_score",
    "gender_Male",
    "gender_Non-binary",
    "gender_Prefer not to say",
    "country_Bangladesh",
    "country_Canada",
    "country_Germany",
    "country_India",
    "country_Pakistan",
    "country_Saudi Arabia",
    "country_UAE",
    "country_UK",
    "country_USA",
    "occupation_Designer",
    "occupation_Doctor",
    "occupation_Freelancer",
    "occupation_Manager",
    "occupation_Researcher",
    "occupation_Software Engineer",
    "occupation_Student",
    "occupation_Teacher",
    "occupation_Unemployed",
]

NUMERIC_COLUMNS: list[str] = [
    "age",
    "work_hours_per_week",
    "screen_time_hours",
    "sleep_hours",
    "stress_score",
    "anxiety_score",
    "depression_score",
    "therapy_history",
    "family_history_mental_illness",
    "academic_or_job_pressure",
    "financial_stress_score",
]

SLEEP_QUALITY_OPTIONS = ["Poor", "Average", "Good", "Excellent"]
EXERCISE_FREQUENCY_OPTIONS = ["Never", "Rarely", "Sometimes", "Regularly"]
SOCIAL_SUPPORT_OPTIONS = ["Moderate", "Strong", "Weak"]

GENDER_OPTIONS = ["Female / baseline", "Male", "Non-binary", "Prefer not to say"]
COUNTRY_OPTIONS = [
    "Other / baseline",
    "Bangladesh",
    "Canada",
    "Germany",
    "India",
    "Pakistan",
    "Saudi Arabia",
    "UAE",
    "UK",
    "USA",
]
OCCUPATION_OPTIONS = [
    "Other / baseline",
    "Designer",
    "Doctor",
    "Freelancer",
    "Manager",
    "Researcher",
    "Software Engineer",
    "Student",
    "Teacher",
    "Unemployed",
]


def yes_no_to_int(value: str | bool | int) -> int:
    """Convert UI yes/no values to the numeric representation used in training."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value > 0)
    return 1 if str(value).strip().lower() in {"yes", "y", "true", "1"} else 0


def _artifact_part(artifact: Any, key: str) -> Any | None:
    if isinstance(artifact, Mapping):
        return artifact.get(key)
    return None


def get_model(artifact: Any) -> Any:
    """Return the fitted estimator from either a dict artifact or direct model."""
    return _artifact_part(artifact, "model") or artifact


def get_feature_columns(artifact: Any | None = None) -> list[str]:
    """Return the expected final feature order.

    The model artifact stores this as ``feature_columns``. The constant above is
    the same order extracted from the uploaded pickle and is used as a fallback.
    """
    artifact_columns = _artifact_part(artifact, "feature_columns")
    if artifact_columns is not None:
        return [str(column) for column in artifact_columns]

    model = get_model(artifact) if artifact is not None else None
    model_columns = getattr(model, "feature_names_in_", None)
    if model_columns is not None:
        return [str(column) for column in model_columns]

    return FEATURE_COLUMNS.copy()


def _encode_with_artifact(encoder: Any | None, value: str, fallback_values: list[str]) -> float:
    if encoder is not None and hasattr(encoder, "transform"):
        try:
            return float(encoder.transform([[value]])[0][0])
        except Exception:
            return float(encoder.transform([value])[0])

    fallback = {name: index for index, name in enumerate(fallback_values)}
    return float(fallback[value])


def _assign_one_hot(row: dict[str, float], prefix: str, value: str) -> None:
    column = f"{prefix}_{value}"
    if column in row:
        row[column] = 1.0


def preprocess_input(raw_input: Mapping[str, Any], artifact: Any) -> pd.DataFrame:
    """Build a scaled single-row feature frame for model prediction.

    Parameters
    ----------
    raw_input:
        Values collected from the Streamlit form.
    artifact:
        The loaded joblib artifact. For this model it is expected to contain
        ``model``, ``scaler``, ``oe_sleep``, ``oe_exercise``, ``le_support``,
        and ``feature_columns``.
    """
    feature_columns = get_feature_columns(artifact)
    row = {column: 0.0 for column in feature_columns}

    for column in NUMERIC_COLUMNS:
        if column in row:
            value = raw_input.get(column)
            if value is None:
                raise ValueError(f"Missing required input: {column}")
            row[column] = float(yes_no_to_int(value) if "history" in column else value)

    row["sleep_quality"] = _encode_with_artifact(
        _artifact_part(artifact, "oe_sleep"),
        str(raw_input["sleep_quality"]),
        SLEEP_QUALITY_OPTIONS,
    )
    row["exercise_frequency"] = _encode_with_artifact(
        _artifact_part(artifact, "oe_exercise"),
        str(raw_input["exercise_frequency"]),
        EXERCISE_FREQUENCY_OPTIONS,
    )
    row["social_support"] = _encode_with_artifact(
        _artifact_part(artifact, "le_support"),
        str(raw_input["social_support"]),
        SOCIAL_SUPPORT_OPTIONS,
    )

    _assign_one_hot(row, "gender", str(raw_input["gender"]).replace(" / baseline", ""))
    _assign_one_hot(row, "country", str(raw_input["country"]).replace(" / baseline", ""))
    _assign_one_hot(row, "occupation", str(raw_input["occupation"]).replace(" / baseline", ""))

    engineered = pd.DataFrame([row], columns=feature_columns)
    scaler = _artifact_part(artifact, "scaler")
    model = get_model(artifact)
    model_name = model.__class__.__name__.lower() if model is not None else ""
    needs_scaling = any(name in model_name for name in ["logistic", "neighbor", "svm", "svc"])

    if scaler is None or not needs_scaling:
        # Tree-based models (Decision Tree, Random Forest, Gradient Boosting) were fit on unscaled features
        return engineered

    scaled_values = scaler.transform(engineered)
    return pd.DataFrame(scaled_values, columns=feature_columns)
