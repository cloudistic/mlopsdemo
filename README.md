# Modern MLOps Masterclass: From Prototype to Production

Welcome to the **Modern MLOps Masterclass** codebase. This repository contains complete, runnable, production-grade implementations of the four core pillars of enterprise MLOps: **DVC**, **MLflow**, **Apache Airflow**, and **H2O AutoML**, culminating in a unified **End-to-End Automated Retraining DAG**.

---

## Quick Start

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
To launch the interactive demonstration launcher:
```bash
python run_all_demos.py
```
To run the automated pre-flight test suite:
```bash
python run_all_demos.py --test-all
```

---

## Complete Student Study Guide

For complete, in-depth lecture notes, architectural explanations, code walkthroughs, and self-study exercises, please refer to:
 **[`STUDENT_LECTURE_NOTES.md`](STUDENT_LECTURE_NOTES.md)**

---

## Web Dashboards & Services

| Service | Terminal Command | Web URL |
| :--- | :--- | :--- |
| **MLflow Tracking Server** | `mlflow server --host 0.0.0.0 --port 5000` | `http://localhost:5000` |
| **Apache Airflow Webserver** | `airflow standalone` | `http://localhost:8080` |
| **DVC Plots Dashboard** | `cd dvc/dvctutorial && dvc plots show` | Opens interactive browser dashboard |

---
