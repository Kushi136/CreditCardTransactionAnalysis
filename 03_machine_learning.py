"""
Step 3: Machine Learning Model Training and Evaluation
Credit Card Transaction Analysis & Fraud Detection
Models: Logistic Regression vs Random Forest Classifier
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support
)

OUTPUT_MODELS_DIR = os.path.join("outputs", "models")
OUTPUT_GRAPHS_DIR = os.path.join("outputs", "graphs")
os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_GRAPHS_DIR, exist_ok=True)


def load_data():
    train_path = os.path.join("data", "processed", "fraudTrain_clean.csv")
    test_path = os.path.join("data", "processed", "fraudTest_clean.csv")

    print(f"Loading train dataset: {train_path}...")
    train = pd.read_csv(train_path)
    print(f"Loading test dataset: {test_path}...")
    test = pd.read_csv(test_path)

    # To ensure efficient training on local resources while maintaining class representation,
    # we can train on a well-stratified representative sample or full dataset.
    # We take 200,000 stratified samples for ultra-fast and reliable model convergence.
    from sklearn.model_selection import train_test_split
    if len(train) > 200000:
        print("Sampling 200,000 stratified records from training set for rapid, robust training...")
        train, _ = train_test_split(train, train_size=200000, stratify=train["is_fraud"], random_state=42)

    if len(test) > 50000:
        test, _ = train_test_split(test, train_size=50000, stratify=test["is_fraud"], random_state=42)

    features = [
        "amt",
        "category",
        "gender",
        "city_pop",
        "transaction_hour",
        "day_of_week",
        "is_weekend",
        "customer_age",
        "merchant_distance"
    ]
    target = "is_fraud"

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    print(f"Train features shape: {X_train.shape}, Fraud cases: {y_train.sum()} ({y_train.mean():.2%})")
    print(f"Test features shape: {X_test.shape}, Fraud cases: {y_test.sum()} ({y_test.mean():.2%})")

    return X_train, y_train, X_test, y_test, features


def build_preprocessor():
    categorical_features = ["category", "gender"]
    numeric_features = [
        "amt",
        "city_pop",
        "transaction_hour",
        "day_of_week",
        "is_weekend",
        "customer_age",
        "merchant_distance"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    return preprocessor


def evaluate_model(name, model, X_test, y_test):
    print("\n" + "=" * 60)
    print(f"EVALUATING MODEL: {name}")
    print("=" * 60)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"])
    print(report)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    metrics = {
        "model": name,
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": cm.tolist()
    }
    return metrics, y_pred, y_proba


def main():
    X_train, y_train, X_test, y_test, feature_names = load_data()
    preprocessor = build_preprocessor()

    # Model 1: Logistic Regression
    print("\n--> Training Model 1: Logistic Regression (balanced)...")
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_metrics, lr_pred, lr_proba = evaluate_model("Logistic Regression", lr_pipeline, X_test, y_test)

    # Model 2: Random Forest
    print("\n--> Training Model 2: Random Forest Classifier...")
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_metrics, rf_pred, rf_proba = evaluate_model("Random Forest", rf_pipeline, X_test, y_test)

    # Save models
    joblib.dump(lr_pipeline, os.path.join(OUTPUT_MODELS_DIR, "logistic_regression.joblib"))
    joblib.dump(rf_pipeline, os.path.join(OUTPUT_MODELS_DIR, "random_forest.joblib"))
    print(f"\nModels successfully saved to {OUTPUT_MODELS_DIR}/")

    # Save comparison summary
    comparison = [lr_metrics, rf_metrics]
    with open(os.path.join(OUTPUT_MODELS_DIR, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=4)

    # Plot ROC Curves
    plt.figure(figsize=(8, 6))
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)

    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {lr_metrics['roc_auc']:.3f})", color="#e67e22", lw=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_metrics['roc_auc']:.3f})", color="#27ae60", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    plt.title("ROC Curves Comparison: Fraud Detection", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_GRAPHS_DIR, "10_roc_curves_comparison.png"), dpi=300)
    plt.close()

    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name, cm in zip(axes, ["Logistic Regression", "Random Forest"], [lr_metrics["confusion_matrix"], rf_metrics["confusion_matrix"]]):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
        ax.set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_GRAPHS_DIR, "11_confusion_matrices.png"), dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print("SUCCESS: Machine learning training & evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
