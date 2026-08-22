"""
MLflow Demo 4: Centralized Model Registry & Production Lifecycle Governance
Demonstrates registering candidate churn models, lifecycle transitions (@champion),
and loading production models directly for live customer scoring.
"""
import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Add parent directory to path if run directly
sys.path.insert(0, os.path.dirname(__file__))
from mlflow_utils import setup_tracking

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "customer_churn_5k.csv"))
MODEL_NAME = "customer-churn-production"

print("=" * 75)
print("  MLflow Demo 4: Model Registry & Lifecycle Governance (Churn 5K)")
print("=" * 75)

status = setup_tracking("lecture-churn-registry", "http://127.0.0.1:5001")
print(f"\n[Tracking Status] {status}")

# 1. Prepare Data
print("\n1. Ingesting Enterprise Customer Dataset (5,000 Customers)...")
df = pd.read_csv(DATA_PATH)
X = df.drop("churn_target", axis=1)
y = df["churn_target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Train and Register Production Candidate
print("\n2. Training Production Candidate (Gradient Boosting) & Registering...")
with mlflow.start_run(run_name="Production Churn Candidate") as run:
    model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    mlflow.log_params({"n_estimators": 120, "learning_rate": 0.1, "max_depth": 4})
    mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": auc})
    
    # Register model into the central registry
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )
    
    run_id = run.info.run_id
    print(f"   ✓ Candidate trained in run: {run_id}")
    print(f"   ✓ Validation Metrics: Accuracy={acc:.4f} | F1={f1:.4f} | ROC-AUC={auc:.4f}")

# 3. Model Governance & Version Management
print("\n3. Managing Model Governance with MLflow Client...")
client = MlflowClient()

try:
    versions = client.get_latest_versions(MODEL_NAME)
    latest_version = versions[-1].version if versions else 1
except Exception:
    latest_version = 1

print(f"   • Model Name:     '{MODEL_NAME}'")
print(f"   • Active Version: v{latest_version}")

# Update description
client.update_registered_model(
    name=MODEL_NAME,
    description="Enterprise Customer Churn Predictor trained on 5,000 subscriber accounts with 18 features."
)

client.update_model_version(
    name=MODEL_NAME,
    version=str(latest_version),
    description=f"Candidate validated on test split: Accuracy={acc:.4f}, ROC-AUC={auc:.4f}."
)

# Apply Stage & Alias tags
try:
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=str(latest_version),
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"   ✓ Version v{latest_version} transitioned to stage: 'Staging'")
except Exception:
    pass

try:
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=str(latest_version)
    )
    print(f"   ✓ Production alias '@champion' assigned to version v{latest_version}")
except Exception:
    pass

# 4. Load Model from Registry for Live Scoring
print("\n4. Scoring Live Customer Profile via Registry URI...")
try:
    model_uri = f"models:/{MODEL_NAME}/{latest_version}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    
    # Sample Test Customer: High Risk Profile (low tenure, high support tickets, payment delay)
    sample_customer = X_test.iloc[[0]]
    prediction = loaded_model.predict(sample_customer)
    
    pred_label = "HIGH RISK (Will Churn)" if prediction[0] == 1 else "LOW RISK (Will Retain)"
    
    print(f"   ✓ Loaded URI:    {model_uri}")
    print(f"   • Customer Data: Tenure={sample_customer['tenure_months'].values[0]} mos | Monthly=${sample_customer['monthly_charges'].values[0]} | Tickets={sample_customer['support_tickets'].values[0]}")
    print(f"   • Scoring Result: {pred_label} (Class {prediction[0]})")
except Exception as e:
    print(f"   • Model inference error: {e}")

print("\n" + "-" * 75)
print("5. Key Architectural Takeaways for Audience:")
print("   • Zero Downtime: Promotion from Staging to @champion updates production instantaneously.")
print("   • Auditability: Every prediction links back to exact training data (5k samples), params, and run ID.")
print("   • Decoupled Engineering: Data scientists publish versions; Backend APIs only call the alias URI.")
print("=" * 75)
