"""
TPOT Generated Pipeline: Tabular Classification Pipeline
Demonstrates feature selection combined with Random Forest classifier discovered by TPOT.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

print("=" * 65)
print("  TPOT Discovered Pipeline: Feature Selection + Random Forest")
print("=" * 65)

# 1. Load Data
print("\n1. Loading Tabular Classification Dataset...")
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"   • Training samples: {len(X_train)} | Original Features: {X_train.shape[1]}")
print(f"   • Test samples:     {len(X_test)}")

# 2. Pipeline architecture discovered by TPOT
print("\n2. Executing TPOT Discovered Pipeline:")
print("   Stage 1: Feature Selection (SelectFwe with f_classif)")
print("   Stage 2: RandomForestClassifier(n_estimators=100, min_samples_leaf=9)")

pipeline = make_pipeline(
    SelectFwe(score_func=f_classif, alpha=0.01),
    RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=9,
        min_samples_split=17,
        max_features=0.85,
        random_state=42
    )
)

pipeline.fit(X_train, y_train)

# 3. Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n3. Evaluation Results:")
print(f"   • Test Accuracy: {acc * 100:.2f}%")
print(f"   • F1-Score:      {f1:.4f}")
print("=" * 65)
