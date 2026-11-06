# Spambase Classification and Streamlit Evaluation

This repository implements and compares five machine-learning classification models on the UCI Spambase dataset. It also contains an interactive Streamlit application that evaluates the saved models on an uploaded test CSV without retraining them.

## a. Problem statement

Unsolicited email creates security, productivity, and storage problems. The objective of this project is to classify an email as spam or non-spam from word-frequency, character-frequency, and capital-letter sequence measurements. Five required classifiers are trained on the same training partition and compared on one unseen test partition.

## b. Dataset description

- **Dataset:** Spambase
- **Source:** UCI Machine Learning Repository
- **Dataset page:** https://archive.ics.uci.edu/dataset/94/spambase
- **Instances:** 4,601 email records
- **Input features:** 57 numeric features
- **Target:** `spam` (`1` = spam, `0` = non-spam)
- **Class distribution:** 1,813 spam and 2,788 non-spam records
- **Missing values:** None
- **Split used:** 80% training and 20% testing, stratified by target
- **Random state:** 42

The first 48 features measure word frequencies, the next 6 measure character frequencies, and the final 3 describe uninterrupted sequences of capital letters. The included `test_data.csv` is the held-out test partition and contains all 57 features plus the target column.

## c. GitHub Repository Link

**GitHub Repository:** https://github.com/jkkhare-png/ML_Assignment

**Live Streamlit App:** https://mlassignment-lgvdzcn82vgksryq9o2s2i.streamlit.app/

## d. Models used

All models were trained on the same stratified training partition. Standardization is included inside the saved pipelines for Logistic Regression, KNN, Gaussian Naive Bayes, and SVM. The Decision Tree operates on the original numeric feature values.

### Comparison table

The following values were measured on the 921-row held-out test partition. AUC is calculated from predicted spam probabilities rather than hard class predictions.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9294 | 0.9702 | 0.9209 | 0.8981 | 0.9093 | 0.8518 |
| Decision Tree | 0.9186 | 0.9211 | 0.9211 | 0.8678 | 0.8936 | 0.8287 |
| K-Nearest Neighbors | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| Gaussian Naive Bayes | 0.8328 | 0.9376 | 0.7146 | 0.9587 | 0.8188 | 0.6946 |
| Support Vector Machine | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

### Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Produced the highest Accuracy, AUC, F1, and MCC. Its recall of 0.8981 also shows that it detected most spam while retaining strong precision. |
| Decision Tree | Achieved strong and interpretable results, including 0.9211 precision, but its lower AUC indicates weaker probability ranking than the scaled linear and kernel models. |
| K-Nearest Neighbors | Performed reasonably after standardization, but had lower Accuracy, F1, and MCC than Logistic Regression, Decision Tree, and SVM. Its prediction cost also grows with the stored training data. |
| Gaussian Naive Bayes | Obtained the highest Recall (0.9587), so it missed the fewest spam messages. Its much lower Precision (0.7146) indicates more legitimate messages were incorrectly marked as spam. |
| Support Vector Machine | Was a close second overall and achieved the highest Precision (0.9277), making it attractive when false spam alarms are especially costly. Its Accuracy and MCC were slightly below Logistic Regression. |
| Overall winner | **Logistic Regression** is the overall winner because it led four major metrics—Accuracy, AUC, F1, and MCC—while maintaining a strong balance between Precision and Recall. |

## Repository structure

```text
project-folder/
|-- app.py
|-- train_models.py
|-- project_config.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- dataset-source/
|   |-- spambase.data
|   |-- spambase.names
|   `-- spambase.DOCUMENTATION
`-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- gaussian_naive_bayes.joblib
    |-- support_vector_machine.joblib
    |-- metrics.csv
    `-- metadata.json
```

## Reproducing the experiment

1. Create and activate a Python virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run `python train_models.py` to recreate the split, models, metrics, and test CSV.
4. Run `streamlit run app.py` to start the application locally.

The training script validates the official data shape and target before fitting any model. Re-running it with the included source data and fixed random state reproduces the saved experiment.

## Streamlit application features

- Upload a test CSV containing every input feature and the target column.
- Use the bundled `test_data.csv` when no file is uploaded.
- Select one model or compare all five models.
- Display Accuracy, AUC, Precision, Recall, F1, and MCC.
- Display a confusion matrix and classification report.
- Load saved model pipelines and perform inference without retraining.

## Deployment

After pushing this folder to GitHub:

1. Sign in to Streamlit Community Cloud with GitHub.
2. Create a new app and select the repository and branch.
3. Set the main file path to `app.py`.
4. Deploy the app.
5. Add the final GitHub and Streamlit URLs to this README and the submission PDF.

## Academic integrity and reproducibility

The repository retains the official UCI dataset documentation, explicit preprocessing choices, a fixed split, and saved metadata. The README observations are tied to the generated metric values rather than copied benchmark results.

