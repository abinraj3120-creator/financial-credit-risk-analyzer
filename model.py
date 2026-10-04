"""Data cleaning + ML model for the Loan Approval project.

Run `python model.py` to train from the command line and save `loan_model.joblib`.
The Streamlit app (app.py) imports the same functions, so both stay in sync.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Loan_Status"
MODEL_PATH = Path(__file__).parent / "loan_model.joblib"

# Personal attributes that should not drive a lending decision.
# Remove a name from this list if you want the model to use it.
EXCLUDE = ["Gender", "Married"]

NUMERIC = [
    "Dependents",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "TotalIncome",
]
CATEGORICAL = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]

NUMERIC_FEATURES = [c for c in NUMERIC if c not in EXCLUDE]
CATEGORICAL_FEATURES = [c for c in CATEGORICAL if c not in EXCLUDE]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------------- #
# Cleaning (same steps as the notebook, so raw or cleaned CSVs both work)
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy().drop_duplicates()
    clean = clean.drop(columns=["Loan_ID"], errors="ignore")

    clean["Dependents"] = pd.to_numeric(
        clean["Dependents"].replace("3+", "3"), errors="coerce"
    )

    num_cols = clean.select_dtypes(include="number").columns
    cat_cols = clean.select_dtypes(exclude="number").columns
    for col in num_cols:
        clean[col] = clean[col].fillna(clean[col].median())
    for col in cat_cols:
        clean[col] = clean[col].fillna(clean[col].mode()[0])
    clean["Dependents"] = clean["Dependents"].astype(int)

    clean["TotalIncome"] = clean["ApplicantIncome"] + clean["CoapplicantIncome"]
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "TotalIncome"]:
        q1, q3 = clean[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        clean[col] = clean[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return clean.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _pipeline(estimator) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("prep", prep), ("model", estimator)])


def train(df: pd.DataFrame) -> dict:
    """Clean the data, compare two models with cross-validation, keep the best."""
    data = clean_data(df)
    X = data[FEATURES]
    y = (data[TARGET] == "Y").astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, est in candidates.items():
        pipe = _pipeline(est)
        rows.append(
            {
                "Model": name,
                "CV ROC-AUC": cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean(),
                "CV F1": cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1").mean(),
                "CV Accuracy": cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy").mean(),
            }
        )
    comparison = pd.DataFrame(rows).round(3)
    best_name = comparison.sort_values("CV ROC-AUC", ascending=False).iloc[0]["Model"]

    pipe = _pipeline(candidates[best_name]).fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, proba)

    imp = permutation_importance(
        pipe, X_te, y_te, scoring="roc_auc", n_repeats=20, random_state=42
    )
    importance = (
        pd.DataFrame({"Feature": FEATURES, "Importance": imp.importances_mean})
        .sort_values("Importance")
        .reset_index(drop=True)
    )

    fill = {c: data[c].median() for c in NUMERIC_FEATURES if c != "TotalIncome"}
    fill.update({c: data[c].mode()[0] for c in CATEGORICAL_FEATURES})
    caps = {c: (data[c].min(), data[c].max())
            for c in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "TotalIncome"]}

    return {
        "fill": fill,
        "caps": caps,
        "pipeline": pipe,
        "model_name": best_name,
        "comparison": comparison,
        "metrics": {
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, zero_division=0),
            "Recall": recall_score(y_te, pred, zero_division=0),
            "F1": f1_score(y_te, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_te, proba),
        },
        "confusion": confusion_matrix(y_te, pred),
        "roc": {"fpr": fpr, "tpr": tpr},
        "importance": importance,
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }


def predict_proba(bundle: dict, applicant: dict) -> float:
    """Probability (0-1) that the loan is approved for one applicant."""
    row = dict(applicant)
    row["TotalIncome"] = row["ApplicantIncome"] + row["CoapplicantIncome"]
    X = pd.DataFrame([row])[FEATURES]
    return float(bundle["pipeline"].predict_proba(X)[0, 1])


def predict_batch(bundle: dict, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Score many applications (e.g. the test file). Missing values are filled
    with the values learned from the training data."""
    ids = (df["Loan_ID"].reset_index(drop=True) if "Loan_ID" in df.columns
           else pd.Series(range(1, len(df) + 1), name="Row"))
    d = df.drop(columns=["Loan_ID", TARGET], errors="ignore").reset_index(drop=True)

    d["Dependents"] = pd.to_numeric(d["Dependents"].replace("3+", "3"), errors="coerce")
    for col, value in bundle["fill"].items():
        d[col] = d[col].fillna(value)
    d["TotalIncome"] = d["ApplicantIncome"] + d["CoapplicantIncome"]
    for col, (lo, hi) in bundle["caps"].items():
        d[col] = d[col].clip(lo, hi)
    d["Dependents"] = d["Dependents"].astype(int)

    prob = bundle["pipeline"].predict_proba(d[FEATURES])[:, 1]
    out = pd.DataFrame({ids.name: ids, "Approval_Probability": prob.round(3)})
    out["Prediction"] = pd.Series(prob >= threshold).map({True: "Approved", False: "Rejected"})
    return out


def save(bundle: dict, path: Path = MODEL_PATH) -> None:
    joblib.dump(bundle, path)


def load(path: Path = MODEL_PATH) -> dict | None:
    return joblib.load(path) if Path(path).exists() else None


if __name__ == "__main__":
    import sys

    csv = sys.argv[1] if len(sys.argv) > 1 else "loan_cleaned.csv"
    bundle = train(pd.read_csv(csv))
    save(bundle)
    print(f"Best model: {bundle['model_name']}")
    print(bundle["comparison"].to_string(index=False))
    for k, v in bundle["metrics"].items():
        print(f"{k:>10}: {v:.3f}")
    print(f"Saved to {MODEL_PATH}")
