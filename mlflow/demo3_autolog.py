"""
MLflow Demo 3: Zero-Boilerplate Autologging on 5,000-Record Dataset
Demonstrates how mlflow.sklearn.autolog() automatically captures
hyperparameters, training/validation metrics, confusion matrices, and model artifacts.
"""
import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path if run directly
sys.path.insert(0, os.path.dirname(__file__))
from mlflow_utils import setup_tracking

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "customer_churn_5k.csv"))

print("=" * 70)
print("  MLflow Demo 3: Zero-Code Autologging (Customer Churn 5K)")
print("=" * 70)

# 1. Enable Autologging FIRST
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)

status = setup_tracking("lecture-churn-autolog-demo", "http://127.0.0.1:5000")
print(f"\n[Tracking Status] {status}")

# 2. Data
print("\n1. Ingesting Dataset (5,000 Customers)...")
df = pd.read_csv(DATA_PATH)
X = df.drop("churn_target", axis=1)
y = df["churn_target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Fit Model inside MLflow Run
print("\n2. Training RandomForest with Autologging Activated...")
with mlflow.start_run(run_name="RandomForest - Autologged (5k)") as run:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    # Autolog automatically intercepts fit() and score()
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print("   ✓ Training complete!")
    print(f"   • Run ID:     {run.info.run_id}")
    print(f"   • Test Score: {test_score:.4f}")

# 4. Presenter Summary
print("\n" + "-" * 70)
print("3. What MLflow Captured Automatically with Zero Custom Code:")
print("   ✓ 18 Feature Schema & Signatures (tenure, monthly_charges, etc.)")
print("   ✓ Model Hyperparameters (n_estimators=100, max_depth=10, min_samples_split=5)")
print("   ✓ Performance Metrics (training score, test score, log_loss)")
print("   ✓ Feature Importances across all 18 variables")
print("   ✓ Packaged MLmodel artifact with dependencies for deployment")
print("=" * 70)
