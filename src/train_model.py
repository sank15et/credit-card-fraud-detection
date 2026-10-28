
from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------
# 1. Project paths and settings
# --------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "creditcard.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    start_time = time.time()

    # --------------------------------
    # 2. Load and validate dataset
    # --------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain the Class target column.")

    if df["Class"].isna().any():
        raise ValueError("The Class column contains missing values.")

    if df.empty or df["Class"].nunique() != 2:
        raise ValueError("Expected a non-empty binary classification dataset.")

    if not set(df["Class"].unique()).issubset({0, 1}):
        raise ValueError("Class must contain only 0 and 1.")

    if df.drop(columns=["Class"]).select_dtypes(
        exclude=np.number
    ).shape[1] > 0:
        raise ValueError("All predictor features must be numeric.")

    print("\n========== DATA PREPARATION ==========")
    print("Original dataset shape:", df.shape)

    # Exact duplicate rows can cause leakage if copies are
    # distributed across train and test. Remove them before splitting.
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    print("Exact duplicate rows removed:", duplicate_count)
    print("Clean dataset shape:", df.shape)

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    feature_names = X.columns.tolist()

    print("\nClass distribution after deduplication:")
    print(y.value_counts().sort_index().rename(
        index={0: "Legitimate", 1: "Fraud"}
    ))

    # --------------------------------
    # 3. Split before fitting transforms
    # --------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTraining transactions:", len(X_train))
    print("Test transactions:", len(X_test))
    print("Training fraud cases:", int(y_train.sum()))
    print("Test fraud cases:", int(y_test.sum()))

    # --------------------------------
    # 4. Define candidate models
    # --------------------------------
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }

    # --------------------------------
    # 5. Train and evaluate models
    # --------------------------------
    results = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"\n========== {model_name} ==========")

        model_start = time.time()
        model.fit(X_train, y_train)

        # Default classification threshold: 0.5
        predictions = model.predict(X_test)

        # Positive-class probabilities for ranking metrics
        probabilities = model.predict_proba(X_test)[:, 1]

        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )
        average_precision = average_precision_score(
            y_test, probabilities
        )
        roc_auc = roc_auc_score(y_test, probabilities)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        metrics = {
            "model": model_name,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "average_precision": float(average_precision),
            "roc_auc": float(roc_auc),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "threshold": 0.5,
            "training_seconds": round(time.time() - model_start, 2),
        }

        results.append(metrics)
        fitted_models[model_name] = model

        print(f"Precision:       {precision:.4f}")
        print(f"Recall:          {recall:.4f}")
        print(f"F1-score:        {f1:.4f}")
        print(f"Average precision: {average_precision:.4f}")
        print(f"ROC-AUC:         {roc_auc:.4f}")

        print("\nConfusion matrix:")
        print(np.array([[tn, fp], [fn, tp]]))

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        ))

        # Save per-model predictions for further inspection.
        prediction_report = pd.DataFrame({
            "actual_class": y_test.to_numpy(),
            "predicted_class": predictions,
            "fraud_probability": probabilities,
        })

        prediction_report.to_csv(
            REPORT_DIR
            / f"{model_name.lower().replace(' ', '_')}_predictions.csv",
            index=False,
        )

    # --------------------------------
    # 6. Compare models
    # --------------------------------
    results_df = pd.DataFrame(results).sort_values(
        by="average_precision",
        ascending=False,
    )

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n========== MODEL COMPARISON ==========")
    print(results_df[
        [
            "model",
            "precision",
            "recall",
            "f1_score",
            "average_precision",
            "roc_auc",
            "false_positives",
            "false_negatives",
        ]
    ].to_string(index=False))

    best_model_name = str(results_df.iloc[0]["model"])
    best_model = fitted_models[best_model_name]

    # Save a bundle containing the fitted pipeline and feature schema.
    # The Streamlit application will use this bundle.
    bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "threshold": 0.5,
        "selection_metric": "average_precision",
        "random_state": RANDOM_STATE,
    }

    joblib.dump(bundle, MODEL_DIR / "fraud_detection_model.joblib")

    metadata = {
        "selected_model": best_model_name,
        "selection_metric": "average_precision",
        "classification_threshold": 0.5,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "original_rows": int(pd.read_csv(DATA_PATH, usecols=["Class"]).shape[0]),
        "duplicates_removed": duplicate_count,
        "clean_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_names": feature_names,
    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print(f"\nSelected model: {best_model_name}")
    print("Saved model: models/fraud_detection_model.joblib")
    print("Saved metadata: models/model_metadata.json")
    print("Saved comparison: reports/model_comparison.csv")
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
    print("\nModel training and evaluation completed!")


if __name__ == "__main__":
    main()
