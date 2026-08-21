"""
MLflow Demo 1: Core Experiment Tracking on Enterprise Churn Dataset
Demonstrates tracking hyperparameters, multi-metric business evaluation (Accuracy, F1, ROC-AUC),
and packaging trained models as versioned artifacts.
"""
import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add parent directory to path if run directly
sys.path.insert(0, os.path.dirname(__file__))
from mlflow_utils import setup_tracking

# Path to enterprise dataset
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "customer_churn_5k.csv"))

print("=" * 70)
print("  MLflow Demo 1: Core Experiment Tracking (Customer Churn 5K)")
print("=" * 70)

status = setup_tracking("lecture-churn-tracking", "http://127.0.0.1:5000")
print(f"\n[Tracking Status] {status}")

# 1. Load Data
print("\n1. Ingesting Enterprise Customer Churn Dataset (5,000 Records)...")
df = pd.read_csv(DATA_PATH)
X = df.drop("churn_target", axis=1)
y = df["churn_target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"   • Total Samples:    {len(df):,} customers across 18 behavioral features")
print(f"   • Training Split:   {len(X_train):,} samples (75%)")
print(f"   • Test Split:       {len(X_test):,} samples (25%)")
print(f"   • Class Imbalance:  70% Retained (0) vs. 30% Churned (1)")

# 2. Train & Track with MLflow
print("\n2. Training Baseline Logistic Regression & Logging to MLflow...")
with mlflow.start_run(run_name="Logistic Regression - Baseline") as run:
    # Hyperparameters
    params = {
        "model_type": "LogisticRegression",
        "scaler": "StandardScaler",
        "solver": "lbfgs",
        "max_iter": 500,
        "C": 1.0,
        "random_state": 42
    }
    mlflow.log_params(params)
    mlflow.set_tag("use_case", "customer_churn")
    mlflow.set_tag("dataset_version", "v1.0_5k")
    mlflow.set_tag("stage", "baseline_prototype")
    
    # Train Pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver=params["solver"],
            max_iter=params["max_iter"],
            C=params["C"],
            random_state=params["random_state"]
        )
    )
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Log Multi-Metric Performance
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    
    # Log Model Artifact
    mlflow.sklearn.log_model(pipeline, "model")
    
    run_id = run.info.run_id

    print("   ✓ Run logged successfully!")
    print(f"   • Run ID:    {run_id}")
    print(f"   • Accuracy:  {acc:.4f} (Raw overall correct predictions)")
    print(f"   • Precision: {prec:.4f} (True churners among flagged)")
    print(f"   • Recall:    {rec:.4f} (Churners successfully caught)")
    print(f"   • F1-Score:  {f1:.4f} (Harmonic balance)")
    print(f"   • ROC-AUC:   {auc:.4f} (Ranking discrimination ability)")

# 3. Presenter Guidance
print("\n" + "-" * 70)
print("3. Instructor Teaching Note on Metrics:")
print("   • Why Accuracy Alone Lies: Predicting all 0s gives 70% accuracy but 0.0 Recall!")
print("   • Multi-Metric Observability: Tracking F1 and ROC-AUC is mandatory for business impact.")
print("   • Web UI: Open http://localhost:5000 -> Review 'lecture-churn-tracking'.")
print("=" * 70)
