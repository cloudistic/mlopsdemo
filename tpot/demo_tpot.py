"""
TPOT Live AutoML Demonstration
Demonstrates automated machine learning pipeline optimization using genetic programming.
TPOT automatically searches hundreds of scikit-learn combinations (preprocessors,
feature selectors, model architectures, and hyperparameters) and exports clean Python code.
"""
import os
import sys
import time
from tpot import TPOTClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=" * 70)
print("  AutoML Demo: Genetic Pipeline Optimization with TPOT")
print("=" * 70)

# 1. Load Data
print("\n1. Loading Multiclass Wine Classification Dataset...")
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"   • Features: {X.shape[1]} | Classes: {len(set(y))} | Samples: {len(X)}")

# 2. Configure TPOT for Fast Live Demo
print("\n2. Initializing Genetic Pipeline Optimizer (TPOT)...")
print("   • Generations: 3")
print("   • Population Size: 8")
print("   • Search Space: All Scikit-Learn Classifiers + Feature Transformers")

tpot = TPOTClassifier(
    generations=3,
    population_size=8,
    scoring='accuracy',
    cv=3,
    random_state=42,
    verbosity=2,
    max_time_mins=1
)

# 3. Fit (Genetic Algorithm Evolution)
print("\n3. Evolving Pipelines (Genetic Search in Progress)...")
t0 = time.time()
tpot.fit(X_train, y_train)
search_time = time.time() - t0

# 4. Evaluate Discovered Best Pipeline
test_score = tpot.score(X_test, y_test)
print("\n" + "-" * 70)
print(f"4. Search Completed in {search_time:.2f}s!")
print(f"   • Discovered Pipeline Test Accuracy: {test_score * 100:.2f}%")

# 5. Export Python Code
export_path = os.path.join(os.path.dirname(__file__), "discovered_pipeline.py")
tpot.export(export_path)
print(f"   ✓ Optimal pipeline exported to: {export_path}")

print("\n5. Key Instructor Takeaways for AutoML:")
print("   • Efficiency: Replaces days of manual hyperparameter tuning with automated search.")
print("   • No Black Box: TPOT exports standard, inspectable Scikit-Learn Python code.")
print("   • MLOps Integration: Discovered pipeline can be tracked in MLflow or scheduled in Airflow.")
print("=" * 70)
