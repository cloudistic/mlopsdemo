"""
TPOT Generated Pipeline: Digits Classification
Demonstrates an optimized machine learning pipeline discovered via Genetic Algorithm search.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 65)
print("  TPOT Discovered Pipeline: Optical Recognition of Handwritten Digits")
print("=" * 65)

# 1. Load Data
print("\n1. Loading Digits Dataset (8x8 pixel images, 10 classes)...")
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"   • Training samples: {len(X_train)} | Features: {X_train.shape[1]}")
print(f"   • Test samples:     {len(X_test)}")

# 2. Optimized Pipeline discovered by TPOT genetic search
print("\n2. Executing Best Pipeline Discovered by TPOT Genetic Search:")
print("   Pipeline Architecture: KNeighborsClassifier(n_neighbors=8, p=1, weights='uniform')")

pipeline = KNeighborsClassifier(n_neighbors=8, p=1, weights="uniform")
pipeline.fit(X_train, y_train)

# 3. Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n3. Evaluation Results:")
print(f"   • Test Accuracy: {acc * 100:.2f}%")
print(f"   • Error Rate:    {(1 - acc) * 100:.2f}%")
print("\n" + "=" * 65)
