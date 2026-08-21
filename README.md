# Modern MLOps Masterclass: From Prototype to Production

Welcome to the **Modern MLOps Masterclass** codebase. This repository contains complete, runnable, production-grade implementations of the four core pillars of enterprise MLOps: **DVC**, **MLflow**, **Apache Airflow**, and **H2O AutoML**, culminating in a unified **End-to-End Automated Retraining DAG**.

---

## 📋 Table of Contents
1. [Quick Start (Run in 2 Minutes)](#-quick-start)
2. [Case Study Dataset: Customer Churn 5K](#-case-study-dataset)
3. [Repository Structure & Demos](#-repository-structure--demos)
4. [Detailed Pillar Walkthroughs](#-detailed-pillar-walkthroughs)
   - [Pillar 1: Data Version Control (DVC)](#pillar-1-data-version-control-dvc)
   - [Pillar 2: Experiment Tracking & Model Registry (MLflow)](#pillar-2-experiment-tracking--model-registry-mlflow)
   - [Pillar 3: Workflow Orchestration (Apache Airflow)](#pillar-3-workflow-orchestration-apache-airflow)
   - [Pillar 4: Accelerated Modeling (H2O AutoML)](#pillar-4-accelerated-modeling-h2o-automl)
   - [🚀 Capstone: Unified End-to-End Retraining DAG](#-capstone-unified-end-to-end-retraining-dag)
5. [Automated Test Suite](#-automated-test-suite)
6. [Presentation & Instructor Materials](#-presentation--instructor-materials)

---

## ⚡ Quick Start

### 1. Clone & Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/cloudistic/mlopsdemo.git
cd mlopsdemo

# Create and activate Python virtual environment (Python 3.10 - 3.13)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Master Demo Suite
```bash
python run_all_demos.py
```
This opens an interactive CLI menu allowing you to run any of the 10 demos individually or run the automated pre-flight test suite.

---

## Case Study Dataset: Customer Churn 5K

To reflect real-world enterprise scenarios, all demos utilize our **5,000-subscriber Enterprise Customer Churn dataset** ([`data/customer_churn_5k.csv`](data/customer_churn_5k.csv)):
- **Volume:** 5,000 subscriber accounts
- **Features:** 18 behavioral & contract attributes (tenure, monthly charges, support tickets, payment delays, etc.)
- **Class Imbalance:** 70% Retained (Class 0) vs. 30% Churned (Class 1)
- **Multi-Metric Evaluation:** Accuracy alone is deceptive on imbalanced data (predicting all 0s gives 70% accuracy but 0.0 Recall). All scripts evaluate **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.

---

## Repository Structure & Demos

```
mlopsdemo/
├── data/
│   ├── customer_churn_5k.csv           # Primary enterprise case study dataset (5k rows, 18 features)
│   └── generate_dataset.py             # Reusable synthetic generator script
├── dvc/
│   ├── run_dvc_demo.py                 # Interactive DVC runner (pointers, ASCII DAG, dvc repro)
│   └── dvctutorial/                    # 4-stage DVC pipeline (prepare -> featurize -> train -> evaluate)
│       ├── dvc.yaml                    # DVC pipeline specification
│       ├── params.yaml                 # Tunable hyperparameters
│       └── src/                        # Modular stage scripts
├── mlflow/
│   ├── mlflow_utils.py                 # Non-blocking tracking server setup & fallback
│   ├── demo1_simple.py                 # Core tracking: params, multi-metric logs, artifacts
│   ├── demo2_compare.py                # 5-Model architecture benchmark & comparison table
│   ├── demo3_autolog.py                # 1-Line zero-code autologging
│   └── demo4_registry.py               # Model Registry: versioning, '@champion' alias & live inference
├── airflow/
│   └── dags/
│       ├── hello_world_dag.py          # Introduction to DAGs & PythonOperator
│       ├── ml_training_pipeline.py     # 8-stage production pipeline with automated Quality Gates
│       ├── ml_pipeline_with_branching.py # Dynamic conditional routing based on runtime data quality
│       └── end_to_end_mlops_pipeline.py  # 🚀 Capstone: Unified 9-stage retraining DAG
├── h20ai/
│   └── demo_h2o.py                     # Enterprise AutoML, Leaderboard & Stacked Ensembles
├── run_all_demos.py                    # Interactive master menu & automated test runner
├── generate_presentation.py            # Generates the 23-slide PowerPoint deck
├── MLOps_2Hour_Masterclass.pptx        # Complete 16:9 widescreen presentation deck
├── INSTRUCTOR_LECTURE_NOTES.md         # Printable slide-by-slide master speaking guide
└── INSTRUCTOR_QUICK_REFERENCE.md       # Presenter cheat sheet
```

---

## Detailed Pillar Walkthroughs

### Pillar 1: Data Version Control (DVC)
Tracks large datasets and machine learning pipeline dependencies using lightweight text pointers (`.dvc` files) committed to Git.
```bash
python dvc/run_dvc_demo.py
```
* **Key Concept:** Content-addressable storage. `dvc repro` detects changes across code, data, and parameters, skipping unchanged stages in 0.0s via smart caching.

---

### Pillar 2: Experiment Tracking & Model Registry (MLflow)
Logs parameters, metrics, tags, and model artifacts with decentralized governance.

```bash
# Optional: Launch MLflow Tracking UI in another terminal
mlflow server --host 0.0.0.0 --port 5000

# Run the 4 MLflow demos
python mlflow/demo1_simple.py
python mlflow/demo2_compare.py
python mlflow/demo3_autolog.py
python mlflow/demo4_registry.py
```
* **Key Concept:** Decoupling training from serving. Applications load `models:/customer-churn-production@champion`. Updating the `@champion` alias promotes new models with zero downtime and zero code deployments.

---

### Pillar 3: Workflow Orchestration (Apache Airflow)
Automates retraining pipelines as Directed Acyclic Graphs (DAGs) with automated Quality Gates and conditional branching.

```bash
# Run standalone (no background cluster needed for local testing)
python airflow/dags/hello_world_dag.py
python airflow/dags/ml_training_pipeline.py
python airflow/dags/ml_pipeline_with_branching.py
```
* **Key Concept:** Quality Gates. Gate 1 verifies data volume and missing values; Gate 2 enforces `ROC-AUC >= 0.88` and `F1 >= 0.75` before promoting candidate models.

---

### Pillar 4: Accelerated Modeling (H2O AutoML)
Explores dozens of machine learning algorithms in parallel and generates Stacked Ensembles.

```bash
python h20ai/demo_h2o.py
```
* **Key Concept:** Automated Leaderboards. H2O evaluates Gradient Boosting, Deep Learning, GLMs, and builds a **Stacked Ensemble** reaching **~0.953 AUC** on our churn dataset in seconds.

---

### Capstone: Unified End-to-End Retraining DAG
Unites all 4 pillars into a single automated 9-stage pipeline:

```bash
python airflow/dags/end_to_end_mlops_pipeline.py
```
```
1. Ingest Data      ──> Ingests 5,000 enterprise customer churn records
2. Data Gate        ──> Validates schema, 0 missing values, and >=4,000 records
3. DVC Versioning   ──> Computes MD5 content hash and locks data lineage in .dvc pointer
4. Preprocessing    ──> Stratified 75/25 Train/Test split across 18 features
5. MLflow Tracking  ──> Trains candidate model & logs params, metrics (Accuracy, F1, ROC-AUC)
6. Performance Gate ──> Enforces ROC-AUC >= 0.88 & F1 >= 0.75 before promotion
7. Model Registry   ──> Registers model version and assigns '@champion' alias
8. Deploy & Probe   ──> Promotes model artifact and runs live test inference (<1ms latency)
9. Broadcast Alert  ──> Emits completion audit broadcast with full lineage record
```

---

## Automated Test Suite

Run the full pre-flight test suite to verify that 100% of all 10 demo modules pass cleanly:
```bash
python run_all_demos.py --test-all
```

---

---

## 📄 License
MIT License. Created for the Executive MLOps Masterclass Series.
