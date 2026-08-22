# Modern MLOps Masterclass: Student Lecture Notes & Hands-On Guide

**Course Title:** *From Prototype to Production: The Modern MLOps Architecture*  
**Repository:** `https://github.com/cloudistic/mlopsdemo`  
**Dataset:** Enterprise Customer Churn Dataset (`data/customer_churn_5k.csv` – 5,000 Records, 18 Features)  
**Tools Covered:** Data Version Control (DVC), MLflow, Apache Airflow, and H2O AutoML  

---

## Course Overview & Learning Objectives

In enterprise AI, discovering a high-performing model in a Jupyter Notebook is only the first 10% of the journey. The remaining 90% is the engineering, data versioning, experiment tracking, workflow orchestration, and automated governance required to make machine learning reliable and reproducible in production.

By the end of this course, you will be able to:
1. **Version Datasets & Pipelines (DVC):** Decouple large data storage from Git using content-addressable hash pointers and reproducible DAGs (`dvc.yaml`).
2. **Track Experiments & Govern Models (MLflow):** Log hyperparameters, evaluate models across multiple business metrics (Accuracy, F1, ROC-AUC), and manage deployments using Model Registry aliases (`@champion`).
3. **Orchestrate Retraining Workflows (Apache Airflow):** Build automated retraining DAGs with strict Data and Performance Quality Gates and conditional dynamic routing.
4. **Accelerate Baseline Discovery (H2O AutoML):** Rapidly explore diverse algorithms and train high-performance Stacked Ensembles in minutes.
5. **Run Unified Production Pipelines:** Execute a single capstone DAG that combines data ingestion, validation, versioning, training, gating, and live scoring.

---

## Environment Setup & Quick Start

### 1. Clone & Activate Virtual Environment
```bash
# Clone the repository
git clone https://github.com/cloudistic/mlopsdemo.git
cd mlopsdemo

# Create and activate Python virtual environment (Python 3.10+)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Interactive Master Demo Suite
To launch an interactive CLI menu that allows you to execute any demonstration:
```bash
python run_all_demos.py
```
To run the automated pre-flight test suite verifying all 10 modules:
```bash
python run_all_demos.py --test-all
```

---

## How to Start Web UIs & Visual Dashboards

| Tool | Purpose | How to Start | Web URL / Command |
| :--- | :--- | :--- | :--- |
| **MLflow UI** | Experiment comparison, scatter plots, model registry | `mlflow server --host 0.0.0.0 --port 5000` | `http://localhost:5000` |
| **Airflow UI** | DAG visualization, task status, execution logs | `airflow standalone` | `http://localhost:8080` (credentials in terminal) |
| **DVC Plots** | Interactive HTML charts for ROC, PR, and Confusion Matrices | `cd dvc/dvctutorial && dvc plots show` | Opens interactive browser dashboard |

---

# Complete Module-by-Module Lecture Content & Code

---

## ACT I: The MLOps Imperative & Foundations

### 1. The Ad-Hoc Trap: Why 85% of ML Models Never Reach Production
* **The Reproducibility Void:** Models evaluated on toy datasets getting 100% accuracy fail on real enterprise data. Weights saved as unversioned local files (`model_v2_final.pkl`) cannot be linked back to the exact code or data version.
* **The Operational Wall:** Data scientists work in exploratory Python notebooks, while DevOps teams operate in CI/CD and Kubernetes. Handing off raw notebooks leads to error-prone rewrites.
* **Business & Regulatory Impact:** Unversioned models create severe compliance risks under GDPR, HIPAA, and the EU AI Act.

### 2. DevOps vs. MLOps: The Trinity of Code + Data + Models
* **DevOps (Deterministic):** Core asset is **Source Code**. If code does not change, application behavior remains identical. Testing focuses on unit and integration tests.
* **MLOps (Probabilistic):** Behavior is generated dynamically by **Code + Data + Weights + Hyperparameters**. 
* **The Key Insight:** You can freeze source code completely with zero Git commits, yet if customer behavior shifts (**Data Drift** / **Concept Drift**), your model's real-world accuracy will silently degrade.

### 3. The MLOps Maturity Levels
* **Level 0 (Manual):** Notebooks, manual data extracts, local pickle files, zero automated monitoring.
* **Level 1 (DevOps Only):** Git for code, CI unit tests, but manual model training and zero data lineage.
* **Level 2 (Automated Training):** Data versioning (DVC), experiment tracking and model registry (MLflow).
* **Level 3 & 4 (CI/CD for ML & Autonomous Retraining):** Automated Airflow DAGs, quality gates, canary deployments, and continuous drift monitoring.

---

## ACT II: Pillar 1 – Data & Asset Versioning with DVC

### Core Concept: Decoupled Pointer Architecture
Git is engineered for text diffs, not 50GB binary datasets. DVC solves this by decoupling metadata from storage:
- **Git** tracks tiny metadata pointer files (`.dvc` files, ~90 bytes) containing the MD5 hash and file size.
- **DVC** pushes/pulls the actual dataset payload to remote object storage (AWS S3, Google Cloud Storage, Azure Blob).

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│ Git Repository                  │       │ Remote Storage (S3/GCS/Blob)    │
│  ├── train.py                   │       │                                 │
│  └── data/data.xml.dvc (90 B)   │ ────> │  └── [MD5: 22a1a29...] (15 MB)  │
└─────────────────────────────────┘       └─────────────────────────────────┘
```

### Hands-On DVC Execution

#### 1. Inspecting the DVC Pointer File
```bash
cat dvc/data/data.xml.dvc
```
*Output:*
```yaml
outs:
- md5: 22a1a2931c8370d3aeedd7183606fd7f
  size: 14445097
  path: data.xml
```

#### 2. Visualizing the Pipeline DAG
```bash
cd dvc/dvctutorial
dvc dag
```
*Output:*
```
       +---------+       
       | prepare |       
       +---------+       
            |            
            v            
      +-----------+      
      | featurize |      
      +-----------+      
            |            
            v            
        +-------+        
        | train |        
        +-------+        
            |            
            v            
       +----------+      
       | evaluate |      
       +----------+      
```

#### 3. Reproducing the Pipeline (`dvc repro`)
```bash
# Run all stages and generate metrics:
dvc repro

# Force recalculation from scratch without cache:
dvc repro -f
```

#### 4. Viewing Interactive DVC Browser Charts
```bash
dvc plots show
```
*This generates and automatically opens an interactive HTML dashboard in your default browser showing the ROC Curve, Precision-Recall Curve, and Confusion Matrix.*

---

## ACT III: Pillar 2 – Experiment Tracking & Model Registry with MLflow

### The Enterprise Case Study: Customer Churn (5,000 Records)
In our 5,000-subscriber dataset (`data/customer_churn_5k.csv`), 70% of accounts are retained (Class 0) and 30% churn (Class 1).
- **The Accuracy Paradox:** A dummy model that blindly predicts 0 for everyone gets **70% accuracy**, but catches **0% of churners** (Recall = 0.0, F1 = 0.0).
- **Multi-Metric Observability:** We evaluate **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.

### Hands-On MLflow Scripts

#### Demo 1: Core Experiment Tracking
Tracks hyperparameters, multi-metric business evaluation, and logs model artifacts.
```bash
python mlflow/demo1_simple.py
```

#### Demo 2: Multi-Model Architecture Comparison
Trains 5 candidate models and outputs a comparative benchmark table:
```bash
python mlflow/demo2_compare.py
```
*Benchmark Results Table:*
```
Model Architecture                Run ID       Accuracy   Precision    Recall  F1-Score   ROC-AUC
-------------------------------------------------------------------------------------------------
5. Support Vector Machine         e192c73d       0.9256      0.8950    0.8500    0.8724    0.9600
4. Gradient Boosting (Champion)   4b21d89a       0.9064      0.8710    0.8110    0.8404    0.9520
3. Random Forest (Tuned, d=12)    8d529d32       0.9040      0.8650    0.8000    0.8310    0.9533
2. Random Forest (Shallow, d=4)   b45d230d       0.8624      0.8120    0.6830    0.7420    0.9204
1. Baseline Logistic Regression   941e796e       0.8248      0.7410    0.6550    0.6950    0.8505
```
*Teaching Takeaway:* Tuning `max_depth` from 4 to 12 in Random Forest produced an immediate **+10% gain in F1 score**!

#### Demo 3: Zero-Code Autologging
Activates `mlflow.sklearn.autolog()` to automatically capture parameters, training curves, confusion matrices, and model signatures with a single line of code.
```bash
python mlflow/demo3_autolog.py
```

#### Demo 4: Central Model Registry & Production Governance
Registers candidate models into the central store, transitions stages (`Staging` -> `Production`), assigns the `@champion` alias, and loads the active model for live customer scoring.
```bash
python mlflow/demo4_registry.py
```
*Key Takeaway:* Downstream applications load models using `models:/customer-churn-production@champion`. Updating the `@champion` alias promotes new models with **zero downtime and zero code changes**.

---

## ACT IV: Pillar 3 – Workflow Orchestration with Apache Airflow

### Why Cron Breaks in Machine Learning
Cron runs blindly based on time without dependency awareness. If data ingestion fails or hangs at 2:00 AM, cron triggers model training at 2:30 AM on corrupt or empty data. Airflow solves this using Directed Acyclic Graphs (DAGs), automatic retries, and explicit Quality Gates.

### Hands-On Airflow Workflows

#### 1. Basic Operator Syntax
```bash
python airflow/dags/hello_world_dag.py
```

#### 2. Production Retraining Pipeline with Quality Gates
An 8-stage automated workflow enforcing data integrity and model accuracy thresholds:
```bash
python airflow/dags/ml_training_pipeline.py
```
*Pipeline Stages:*
1. `create_directories` -> Setup staging area
2. `extract_data` -> Ingest 5k customer batch
3. `validate_data` [Gate 1] -> Verifies 0 missing values, >=4,000 records, 19 columns
4. `preprocess_data` -> Stratified 75/25 split
5. `train_model` -> Train Gradient Boosting classifier
6. `evaluate_model` [Gate 2] -> **Strictly enforces ROC-AUC >= 0.88 & F1 >= 0.75**
7. `deploy_model` -> Promote artifact to production endpoint
8. `send_notification` -> Broadcast team alert

#### 3. Dynamic Conditional Branching
Calculates a runtime data quality score and routes execution dynamically:
```bash
python airflow/dags/ml_pipeline_with_branching.py
```
- **Quality Score >= 0.80:** Routes to Deep Feature Engineering Pipeline.
- **Quality Score >= 0.60:** Routes to Robust Imputation & Linear Model.
- **Quality Score < 0.60:** Halts pipeline and triggers high-priority engineer alert.

---

## ACT V: Pillar 4 – Accelerated Modeling with H2O AutoML

AutoML acts as a productivity multiplier that explores dozens of algorithm configurations in parallel and builds **Stacked Ensembles** to establish competitive baselines.

```bash
python h20ai/demo_h2o.py
```
*What H2O Does in 15 Seconds:*
1. Starts an embedded, multi-threaded H2O computational engine.
2. Trains Gradient Boosting Machines (GBM), Deep Neural Networks, GLMs, and Random Forests.
3. Combines top models into a **Stacked Ensemble** reaching **~0.953 AUC** on our customer churn dataset.
4. Outputs live test customer predictions with exact class probabilities.

---

## 🎬 ACT VI: 🚀 Capstone – Unified End-to-End Retraining DAG

This master pipeline integrates all four pillars into a single automated, production-grade workflow:

```bash
python airflow/dags/end_to_end_mlops_pipeline.py
```

```
┌──────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│ 1. Ingest Data   │ ──> │ 2. Data Quality Gate │ ──> │ 3. DVC Data Hash Lock │
│ (5,000 Records)  │     │ (0 missing, >=4k)    │     │ (MD5 Lineage Pointer) │
└──────────────────┘     └──────────────────────┘     └───────────────────────┘
                                                                 │
                                                                 ▼
┌──────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│ 6. Performance   │ <── │ 5. MLflow Tracking   │ <── │ 4. Stratified Split   │
│    Quality Gate  │     │    (Params, Metrics) │     │    (75/25 Partition)  │
│ (AUC>=0.88,F1)   │     └──────────────────────┘     └───────────────────────┘
└──────────────────┘
        │
        ▼
┌──────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│ 7. MLflow Model  │ ──> │ 8. Production Deploy │ ──> │ 9. Broadcast Alert    │
│    Registry      │     │    & Live Test Probe │     │    & Lineage Audit    │
│ (@champion Tag)  │     │    (<1ms Latency)    │     │    (Terminal Banner)  │
└──────────────────┘     └──────────────────────┘     └───────────────────────┘
```

---

## Top Review Questions & Self-Study Exercises

1. **Exercise 1 (DVC):** Modify a hyperparameter in `dvc/dvctutorial/params.yaml` (e.g., change `n_est` from 50 to 100). Run `dvc repro`. Which stages re-ran, and which stages hit the cache?
2. **Exercise 2 (MLflow):** Open `http://localhost:5000`. In the `lecture-churn-model-comparison` experiment, create a Scatter Plot with X-axis as `precision` and Y-axis as `recall`. Which model offers the best balance?
3. **Exercise 3 (Airflow):** In `airflow/dags/ml_training_pipeline.py`, change `MIN_ROC_AUC_THRESHOLD` to `0.99`. Run the script. Observe how the Quality Gate exception halts deployment to protect production.
4. **Exercise 4 (AutoML):** In `h20ai/demo_h2o.py`, increase `max_runtime_secs` from 30 to 60. How does the Leaderboard AUC change?

