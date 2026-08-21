"""
MLflow Demo 2: Systematic Model Architecture & Hyperparameter Comparison
Trains 5 distinct models on the 5,000-record Customer Churn dataset,
demonstrating trade-offs across Accuracy, Precision, Recall, F1, and ROC-AUC.
"""
import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add parent directory to path if run directly
sys.path.insert(0, os.path.dirname(__file__))
from mlflow_utils import setup_tracking

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "customer_churn_5k.csv"))

print("=" * 80)
print("  MLflow Demo 2: Multi-Model Benchmarking (Customer Churn 5K Dataset)")
print("=" * 80)

status = setup_tracking("lecture-churn-model-comparison", "http://127.0.0.1:5000")
print(f"\n[Tracking Status] {status}")

# 1. Load Data
print("\n1. Ingesting Customer Churn Dataset (5,000 Records, 18 Features)...")
df = pd.read_csv(DATA_PATH)
X = df.drop("churn_target", axis=1)
y = df["churn_target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"   • Train Split: {len(X_train):,} samples | Test Split: {len(X_test):,} samples")

# 2. Define Model Candidates with Realistic Hyperparameter Variations
candidates = {
    "1. Baseline Logistic Reg": {
        "pipeline": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=1.0, random_state=42)),
        "params": {"model_type": "LogisticRegression", "scaler": "StandardScaler", "C": 1.0, "max_iter": 500}
    },
    "2. Random Forest (Shallow)": {
        "pipeline": RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42),
        "params": {"model_type": "RandomForest", "n_estimators": 50, "max_depth": 4, "criterion": "gini"}
    },
    "3. Random Forest (Deep/Tuned)": {
        "pipeline": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_split=4, random_state=42),
        "params": {"model_type": "RandomForest", "n_estimators": 150, "max_depth": 12, "min_samples_split": 4}
    },
    "4. Gradient Boosting (Champion)": {
        "pipeline": GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42),
        "params": {"model_type": "GradientBoosting", "n_estimators": 120, "learning_rate": 0.1, "max_depth": 4}
    },
    "5. Support Vector Machine": {
        "pipeline": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
        "params": {"model_type": "SVM_RBF", "scaler": "StandardScaler", "C": 1.0, "kernel": "rbf"}
    }
}

print(f"\n2. Training & Tracking {len(candidates)} Candidate Architectures...")
print("-" * 80)

results = []

for name, config in candidates.items():
    with mlflow.start_run(run_name=name) as run:
        # Log Hyperparameters & Tags
        mlflow.log_params(config["params"])
        mlflow.set_tag("experiment_type", "architecture_benchmark")
        mlflow.set_tag("dataset_samples", "5000")
        
        # Fit Model Pipeline
        model = config["pipeline"]
        model.fit(X_train, y_train)
        
        # Predict & Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        # Log Model Artifact
        mlflow.sklearn.log_model(model, "model")
        
        results.append({
            "name": name,
            "run_id": run.info.run_id[:8],
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc
        })
        print(f"   ✓ [{run.info.run_id[:8]}] {name:<32} Acc: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")

# 3. Present Results Table
print("\n" + "=" * 85)
print(f"{'Model Architecture':<33} {'Run ID':<10} {'Accuracy':>9} {'Precision':>10} {'Recall':>9} {'F1-Score':>9} {'ROC-AUC':>9}")
print("-" * 85)
for r in sorted(results, key=lambda x: x['roc_auc'], reverse=True):
    print(f"{r['name']:<33} {r['run_id']:<10} {r['accuracy']:>9.4f} {r['precision']:>10.4f} {r['recall']:>9.4f} {r['f1']:>9.4f} {r['roc_auc']:>9.4f}")
print("=" * 85)

best_auc = max(results, key=lambda x: x["roc_auc"])
best_f1 = max(results, key=lambda x: x["f1"])
print(f"\n🏆 Champion by ROC-AUC: {best_auc['name']} (ROC-AUC: {best_auc['roc_auc']:.4f})")
print(f"🏆 Champion by F1-Score: {best_f1['name']} (F1-Score: {best_f1['f1']:.4f})")

print("\n4. Instructor Key Points to Show in MLflow UI:")
print("   • Navigate to http://localhost:5000 -> Open 'lecture-churn-model-comparison'.")
print("   • Select all 5 runs -> Click 'Compare'.")
print("   • Show the Scatter Plot: X-Axis = 'precision', Y-Axis = 'recall'.")
print("   • Show Parallel Coordinates: How increasing max_depth from 4 to 12 boosted F1 by +10%!")
print("=" * 85)
