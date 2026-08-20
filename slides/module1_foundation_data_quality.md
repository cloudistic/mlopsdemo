# Module 1: The Foundation & Data Quality

**Session Length:** 30 minutes  
**Audience:** Professionals new to practical MLOps

---

## 1) Why MLOps Exists

### DevOps vs. MLOps

Traditional DevOps assumes software behavior is primarily determined by **code**.

In ML systems, behavior is determined by:
- code,
- data,
- model artifacts,
- and serving context.

So while application code may be deterministic, **model quality is dynamic** and can degrade as data changes.

### Why classic software workflows fail for ML

- Unit tests can pass while model quality fails.
- A successful build does not guarantee production model performance.
- Data and feature pipelines create additional operational risk.

---

## 2) Hidden Technical Debt in ML Systems

### The “Scaffolding” problem

In many ML systems, the model is a small part of the total system. Most complexity lives in scaffolding:

- data ingestion and cleaning,
- feature transformation,
- training orchestration,
- model packaging and serving,
- monitoring and alerting,
- rollback and governance.

### Typical debt patterns

- **Code vs. configuration sprawl:** behavior spread across notebooks, YAML files, scripts.
- **Feature entanglement:** one feature used by many models with undocumented dependencies.
- **Pipeline glue code:** brittle scripts connecting tools without contracts.

---

## 3) ML Maturity Framework (L0 → L2)

### Level 0 — Manual

- Notebook-driven experimentation
- Manual data pulls and model retraining
- Ad hoc deployment

### Level 1 — Automated Pipelines

- Repeatable training pipeline
- Data validation and feature checks
- Versioned model artifacts
- Basic orchestration (scheduled or trigger-based)

### Level 2 — CI/CD/CT for ML

- Continuous Integration for code + pipeline logic
- Continuous Delivery for model serving assets
- Continuous Training with monitored retraining triggers
- Governance, reproducibility, and safe rollout strategies

---

## 4) Data Integrity & Feature Management

### Key risk: Train/Serve Skew

A model is trained on one feature definition/distribution, but served on another.

Common causes:
- schema drift,
- transformation mismatch,
- stale feature definitions,
- online/offline pipeline inconsistency.

### Controls

- enforce schema contracts,
- centralize feature definitions,
- validate statistical bounds,
- gate pipelines when quality checks fail.

---

## 5) Practical Demo: Data Quality Gating

### Objective

Implement runtime validation that **blocks pipeline execution** if incoming payloads violate:
- schema types,
- null constraints,
- value ranges,
- simple statistical drift thresholds.

### Tooling

- Python
- Pandera
- Pandas

### What the demo shows

1. A **good payload** passes checks and pipeline continues.
2. A **bad payload** fails checks and pipeline halts with actionable errors.

---

## 6) Demo Walkthrough (10 minutes)

1. Install dependencies.
2. Run validation with good input.
3. Run validation with bad input.
4. Show failure output and discuss quality gates in CI/CD/CT.

---

## 7) Takeaways

- MLOps is not optional for production ML.
- Data quality is a first-class production concern.
- Pipeline gating is a foundational control before scaling to full automation.

---

## 8) Transition to Module 2

Next module will build on this foundation by introducing experiment tracking, reproducibility, and model registry workflows.
