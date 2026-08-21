"""
H2O AutoML Master Demo (Enterprise Customer Churn 5K Dataset)
Demonstrates enterprise AutoML using H2O:
1. Initialize local embedded H2O distributed cluster
2. Automated Feature Engineering, Model Training & Ensembling (GBM, DeepLearning, GLM, DRF, StackedEnsemble)
3. Leaderboard extraction & Model Performance Evaluation
4. Champion model inference and live customer scoring
"""
import os
import sys
import time
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "customer_churn_5k.csv"))

print("=" * 75)
print("  AutoML Demo: Enterprise AutoML & Leaderboards (Churn 5K Dataset)")
print("=" * 75)

try:
    import h2o
    from h2o.automl import H2OAutoML
except ImportError:
    print("❌ H2O is not installed. Please run: pip install h2o")
    sys.exit(1)

# 1. Initialize H2O Cluster
print("\n1. Starting Local Embedded H2O Engine...")
try:
    h2o.init(max_mem_size="2G", nthreads=-1, verbose=False)
except Exception as e:
    print(f"   Failed to start H2O engine: {e}")
    sys.exit(1)

# 2. Ingest Data
print(f"\n2. Ingesting Customer Churn Dataset ({DATA_PATH})...")
if not os.path.exists(DATA_PATH):
    from data.generate_dataset import get_churn_dataset
    df = get_churn_dataset(5000)
    df.to_csv(DATA_PATH, index=False)

hf = h2o.import_file(DATA_PATH, header=1)
target_col = "churn_target"
hf[target_col] = hf[target_col].asfactor()

train_hf, test_hf = hf.split_frame(ratios=[0.75], seed=42)
feature_cols = [c for c in hf.columns if c != target_col]

print(f"   • Total Records:  {hf.nrow:,} rows across {len(feature_cols)} behavioral features")
print(f"   • Train Split:    {train_hf.nrow:,} rows (75%)")
print(f"   • Test Split:     {test_hf.nrow:,} rows (25%)")
print(f"   • Target Classes: 0 (Retained) vs. 1 (Churned)")

# 3. Launch AutoML Search
print("\n3. Launching H2O AutoML Engine (30s Budget: Evaluating StackedEnsembles, GBMs, DeepLearning, GLMs, DRFs)...")
aml = H2OAutoML(
    max_runtime_secs=30,
    max_models=10,
    seed=42,
    verbosity="warn",
    sort_metric="AUC"
)

t0 = time.time()
aml.train(x=feature_cols, y=target_col, training_frame=train_hf, leaderboard_frame=test_hf)
elapsed = time.time() - t0

# 4. Display Leaderboard
print("\n" + "=" * 80)
print(f"4. AutoML Completed in {elapsed:.2f}s! Top Leaderboard Models (Sorted by AUC):")
print("=" * 80)
lb = aml.leaderboard.as_data_frame()
print(lb[["model_id", "auc", "logloss", "rmse"]].head(6).to_string(index=False))

# 5. Best Leader Model Inference
print("\n" + "-" * 80)
print("5. Champion Model Evaluation on Test Split:")
leader_model = aml.leader
perf = leader_model.model_performance(test_hf)
print(f"   • Champion Model ID: {leader_model.model_id}")
print(f"   • Champion Test AUC: {perf.auc():.4f}")
print(f"   • Champion LogLoss:  {perf.logloss():.4f}")

# Sample prediction
preds = leader_model.predict(test_hf.head(3))
print("\n   Sample Predictions for First 3 Accounts (Predicted Class & Probabilities):")
print(preds.as_data_frame().to_string(index=False))

print("\n" + "=" * 80)
print("6. Key Takeaways for H2O AutoML:")
print("   • Realistic Metric Comparison: Clear separation between linear baselines and ensemble models.")
print("   • Ensembling Superiority: Stacked Ensembles combine top models to maximize AUC.")
print("   • MLOps Production Bridge: Champion model can be exported as high-speed MOJO artifact.")
print("=" * 80)

# Clean shutdown
try:
    h2o.cluster().shutdown(prompt=False)
except Exception:
    pass
