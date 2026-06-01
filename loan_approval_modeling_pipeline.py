Original file is located at
    https://colab.research.google.com/drive/1ZWPtFvNEQaVKXecOikJhwNJmUAwkn3XM

# Loan Approval Prediction - Supervised Learning

Goal: Predict loan approval (`Loan_Status`) using borrower features.

This notebook covers:
- Missing value handling
- Categorical encoding
- Feature scaling
- Class imbalance handling using SMOTE
- Model comparison: Logistic Regression, Decision Tree, Random Forest
- Evaluation: Precision, Recall, F1-score, ROC-AUC
- Business interpretation and threshold suggestion
"""

# Install only if needed in Colab
# !pip install imbalanced-learn

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("loan_prediction.csv")

print("Shape:", df.shape)
display(df.head())
display(df.info())

# Check missing values and target balance
print("Missing values:")
display(df.isnull().sum())

print("\nTarget distribution:")
display(df["Loan_Status"].value_counts())
display(df["Loan_Status"].value_counts(normalize=True) * 100)

"""## Preprocessing plan

- Drop `Loan_ID` because it is only an ID, not a useful predictive feature.
- Convert target: `Y = 1`, `N = 0`.
- Numeric missing values: fill with median.
- Categorical missing values: fill with most frequent value.
- Categorical columns: One-Hot Encoding.
- Numeric columns: Standard Scaling.
- Use SMOTE only on training data to handle class imbalance.

"""

# Target conversion
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Features and target
X = df.drop(columns=["Loan_Status", "Loan_ID"])
y = df["Loan_Status"]

# Column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    sparse_threshold=0
)

# Train-test split with stratify to keep same class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Models to compare
models = {
    "Logistic Regression + SMOTE": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree + SMOTE": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
    "Random Forest + SMOTE": RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )
}

results = []
trained_pipelines = {}

for model_name, model in models.items():
    pipeline = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    results.append({
        "Model": model_name,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Confusion_Matrix": confusion_matrix(y_test, y_pred)
    })

    trained_pipelines[model_name] = pipeline

results_df = pd.DataFrame(results)
display(results_df[["Model", "Precision", "Recall", "F1", "ROC_AUC"]])

# Detailed report for the best model based on ROC-AUC
best_model_name = results_df.sort_values("ROC_AUC", ascending=False).iloc[0]["Model"]
best_pipeline = trained_pipelines[best_model_name]

print("Best model:", best_model_name)

y_prob = best_pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Threshold comparison for deployment decision
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(y_test, y_pred_threshold),
        "Recall": recall_score(y_test, y_pred_threshold),
        "F1": f1_score(y_test, y_pred_threshold)
    })

threshold_df = pd.DataFrame(threshold_results)
display(threshold_df)

"""## Business interpretation

- Logistic Regression is easier to explain to business teams and regulators.
- Random Forest can capture non-linear patterns, but it is less explainable.
- Precision matters when wrongly approving risky borrowers is costly.
- Recall matters when rejecting good borrowers causes lost business.
- A threshold of **0.50** is a balanced starting point.
- If the company wants more growth and accepts higher risk, use **0.40**.
- If the company wants safer approvals, use **0.60**.

Suggested deployment threshold: **0.50** initially, then tune using real loan default/cost data.

"""
