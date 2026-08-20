# Module 2: Experimentation & Model Lifecycle

**Session Length:** 30 minutes  
**Audience:** Professionals learning production-grade ML workflows

---

## 1) Why Experimentation Discipline Matters

As ML teams scale, ad hoc training runs become impossible to manage.

Without tracking and governance:
- you cannot reproduce results,
- you cannot compare models objectively,
- and you cannot safely promote models to production.

---

## 2) The Reproducibility Triad

A training result is reproducible only when three dimensions are coupled:

1. **Code Commit (Git)**
   - exact source revision used for training
2. **Dataset Snapshot**
   - immutable or versioned reference to the training data state
3. **Environment Configuration**
   - package versions, runtime, and system context

If any one is missing, exact recreation is unreliable.

### Practical implementation pattern

For each run, log:
- `git_commit` (tag/param)
- `dataset_version` (tag/param)
- environment details (`python_version`, dependency file hash)

---

## 3) Experiment Tracking Architecture

### What should be centralized

- Hyperparameters
- Evaluation metrics
- Dataset/version references
- Model artifacts
- Runtime/system metadata (CPU, memory, duration)
- User/owner and timestamps

### Why this matters

- Enables side-by-side model comparison
- Creates evidence for release decisions
- Supports collaboration across data scientists and MLOps engineers

### Common architecture

- Training job emits logs to tracking backend (e.g., MLflow Tracking Server)
- Artifact store keeps model binaries, plots, reports
- UI/API allows filtering, search, and lineage inspection

---

## 4) Model Registry & Governance

### Lifecycle stages

- **Development**: newly registered candidate model
- **Staging**: validated candidate for integration tests/canary
- **Production**: approved model serving live traffic

### Governance controls

- Stage transition criteria (metric thresholds, checks)
- Approval workflows (human-in-the-loop)
- Audit logs (who promoted/changed what and when)
- Lineage traceability (run → model artifact → registry version → deployment)

---

## 5) Practical Demo: MLflow Tracking + Registry + Comparison

### Objective

Train estimators and demonstrate:
1. **Basic manual logging** (params/metrics/artifacts)
2. **Autologging** with `mlflow.sklearn.autolog()`
3. **Model comparison** across runs in one experiment
4. Registering the best model to MLflow Registry
5. Inspecting run artifact trees and lineage references

### Tooling

- Python
- scikit-learn
- MLflow
- pandas

---

## 6) Demo Walkthrough (10 minutes)

1. Start MLflow server (or point to existing tracking URI)
2. Run manual logging + comparison script
3. Open MLflow UI and compare candidate runs
4. Show generated comparison artifact (`model_comparison.csv/.md`)
5. Run autolog script and compare captured metadata richness
6. Verify model registration/version in Registry tab

---

## 7) Key Takeaways

- Reproducibility requires coupling code, data, and environment.
- Tracking is the system of record for model experimentation.
- Comparison-driven selection improves model promotion decisions.
- Registry + governance enables safe, auditable model promotion.

---

## 8) Transition to Module 3

Next module can cover deployment automation patterns, CI/CD/CT integration, and production monitoring feedback loops.
