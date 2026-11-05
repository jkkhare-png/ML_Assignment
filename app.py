"""Interactive Streamlit evaluation app for the five saved Spambase models."""

from __future__ import annotations

import io

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from project_config import (
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_FILES,
    TARGET_COLUMN,
    TEST_DATA_PATH,
)


st.set_page_config(
    page_title="Spambase Model Evaluation",
    page_icon="📧",
    layout="wide",
)


@st.cache_resource
def load_model(model_name: str):
    model_path = MODEL_DIR / MODEL_FILES[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path.name}")
    return joblib.load(model_path)


@st.cache_data
def load_bundled_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_DATA_PATH)


def validate_test_data(data: pd.DataFrame) -> pd.DataFrame:
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    extra = [column for column in data.columns if column not in required]

    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if extra:
        st.warning("Extra columns will be ignored: " + ", ".join(extra))

    validated = data[required].copy()
    for column in required:
        validated[column] = pd.to_numeric(validated[column], errors="coerce")

    if validated.isna().any().any():
        bad_columns = validated.columns[validated.isna().any()].tolist()
        raise ValueError(
            "Missing or non-numeric values found in: " + ", ".join(bad_columns)
        )
    if not set(validated[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("The spam target must contain only 0 (non-spam) and 1 (spam).")
    if validated[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Both target classes are required to calculate ROC AUC.")

    return validated


def get_scores(model, features: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def evaluate_model(model_name: str, features: pd.DataFrame, target: pd.Series):
    model = load_model(model_name)
    predictions = model.predict(features)
    scores = get_scores(model, features)
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, scores),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }
    return predictions, metrics


def display_metric_cards(metrics: dict[str, float]) -> None:
    columns = st.columns(6)
    for column, (name, value) in zip(columns, metrics.items()):
        column.metric(name, f"{value:.4f}")


def display_diagnostics(target: pd.Series, predictions) -> None:
    left, right = st.columns([1, 1.35])
    with left:
        st.subheader("Confusion matrix")
        matrix = confusion_matrix(target, predictions, labels=[0, 1])
        figure, axis = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Non-spam", "Spam"],
            yticklabels=["Non-spam", "Spam"],
            ax=axis,
        )
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Actual class")
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    with right:
        st.subheader("Classification report")
        report = classification_report(
            target,
            predictions,
            labels=[0, 1],
            target_names=["Non-spam", "Spam"],
            output_dict=True,
            zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")


st.title("📧 Spambase Classification Model Evaluation")
st.write(
    "Evaluate five pre-trained classifiers on the bundled held-out test set or "
    "upload a compatible CSV. The app only performs inference; it never retrains a model."
)

with st.sidebar:
    st.header("Evaluation controls")
    uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])
    model_choice = st.selectbox(
        "Select model",
        ["Compare All Models", *MODEL_FILES.keys()],
    )
    st.caption("The CSV must contain all 57 input features and the `spam` target column.")

try:
    if uploaded_file is None:
        raw_data = load_bundled_test_data()
        source_name = "Bundled test_data.csv"
    else:
        raw_data = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        source_name = uploaded_file.name

    data = validate_test_data(raw_data)
except Exception as error:
    st.error(f"Unable to use the test data: {error}")
    st.stop()

st.success(f"Using {source_name}: {len(data):,} rows and 57 input features.")
with st.expander("Preview test data"):
    st.dataframe(data.head(20), width="stretch")

x_test = data[FEATURE_COLUMNS]
y_test = data[TARGET_COLUMN].astype(int)

if model_choice == "Compare All Models":
    st.header("Model comparison")
    comparison_rows = []
    evaluation_cache = {}
    for name in MODEL_FILES:
        predictions, metrics = evaluate_model(name, x_test, y_test)
        comparison_rows.append({"ML Model Name": name, **metrics})
        evaluation_cache[name] = (predictions, metrics)

    comparison = pd.DataFrame(comparison_rows).set_index("ML Model Name")
    st.dataframe(
        comparison.style.format("{:.4f}").highlight_max(axis=0, color="#b7f7c2"),
        width="stretch",
    )

    detail_model = st.selectbox("Show detailed diagnostics for", list(MODEL_FILES))
    detail_predictions, detail_metrics = evaluation_cache[detail_model]
    st.subheader(detail_model)
    display_metric_cards(detail_metrics)
    display_diagnostics(y_test, detail_predictions)
else:
    st.header(model_choice)
    predictions, metrics = evaluate_model(model_choice, x_test, y_test)
    display_metric_cards(metrics)
    display_diagnostics(y_test, predictions)

st.caption(
    "Target definition: spam = 1 and non-spam = 0. AUC uses predicted spam probabilities."
)

