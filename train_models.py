"""Train, evaluate, and save the five classifiers required by the assignment."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from project_config import (
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_FILES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
)


def load_dataset() -> pd.DataFrame:
    """Load the official UCI data file and assign readable column names."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RAW_DATA_PATH}. Download spambase.data from the UCI "
            "Spambase repository and place it in dataset-source/."
        )

    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    data = pd.read_csv(RAW_DATA_PATH, header=None, names=columns)

    if data.shape != (4601, 58):
        raise ValueError(f"Expected dataset shape (4601, 58), found {data.shape}.")
    if data.isna().any().any():
        raise ValueError("Unexpected missing values found in Spambase data.")
    if set(data[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("The target column must contain only 0 and 1.")

    return data


def build_models() -> dict[str, BaseEstimator]:
    """Return reproducible model pipelines with preprocessing included."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB()),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def positive_class_scores(model: BaseEstimator, features: pd.DataFrame):
    """Return continuous scores for binary ROC AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    raise TypeError("Model does not expose probability or decision scores.")


def calculate_metrics(
    model: BaseEstimator, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    predictions = model.predict(features)
    scores = positive_class_scores(model, features)
    return {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": roc_auc_score(target, scores),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "MCC": matthews_corrcoef(target, predictions),
    }


def main() -> None:
    data = load_dataset()
    train_data, test_data = train_test_split(
        data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COLUMN],
    )

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]
    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    test_data.to_csv(TEST_DATA_PATH, index=False)

    results: list[dict[str, float | str]] = []
    for model_name, model in build_models().items():
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[model_name])
        results.append({"ML Model Name": model_name, **calculate_metrics(model, x_test, y_test)})

    metrics = pd.DataFrame(results)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False, float_format="%.6f")

    metadata = {
        "dataset": "UCI Spambase",
        "dataset_url": "https://archive.ics.uci.edu/dataset/94/spambase",
        "random_state": RANDOM_STATE,
        "split": "80% stratified training / 20% stratified testing",
        "training_rows": len(train_data),
        "test_rows": len(test_data),
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": __import__("sklearn").__version__,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nHeld-out test results:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved {len(test_data)} test rows to {TEST_DATA_PATH.name}.")


if __name__ == "__main__":
    main()

