# Module 2 Demo — Experiment Tracking & Model Registry (MLflow)

This module demonstrates end-to-end experiment tracking and model registration with MLflow, including:

- **Basic manual logging**
- **Autologging**
- **Multi-run model comparison**

## What this demo does

- Trains classifiers on the same dataset
- Logs params, metrics, tags, and artifacts manually
- Demonstrates `mlflow.sklearn.autolog()`
- Compares multiple runs and writes a summary artifact
- Registers the best model in MLflow Model Registry
- Prints an artifact tree from the winning run

## Project structure

- `demo_mlflow_tracking_registry.py` — main script (basic logging + comparison + registry)
- `demo_mlflow_autolog.py` — autologging-focused script
- `data/sample_training_data.csv` — sample dataset used for training
- `requirements.txt` — dependencies
- `run_local_mlflow.sh` — helper script to run local MLflow server

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r module2/requirements.txt
```

## Start local MLflow server

```bash
bash module2/run_local_mlflow.sh
```

This starts MLflow at: `http://127.0.0.1:5000`

## 1) Basic logging + model comparison + registry

```bash
python module2/demo_mlflow_tracking_registry.py \
  --tracking-uri http://127.0.0.1:5000 \
  --experiment-name module2-experiment-demo \
  --registered-model-name credit-risk-classifier
```

## 2) Autologging demonstration

```bash
python module2/demo_mlflow_autolog.py \
  --tracking-uri http://127.0.0.1:5000 \
  --experiment-name module2-autolog-demo
```

## Inspect in UI

1. Open `http://127.0.0.1:5000`
2. Open experiments:
   - `module2-experiment-demo`
   - `module2-autolog-demo`
3. Inspect:
   - run params/metrics/tags
   - artifacts (`model/`, `reports/`)
4. Open **Models** tab and inspect the registered model/version

## Suggested 10-minute live flow

1. Explain reproducibility triad tags (`git_commit`, `dataset_version`, `python_version`)
2. Run **basic logging** script and show manual logs
3. Show **comparison report artifact** and best-run selection
4. Run **autolog** script and compare amount of captured metadata
5. Show registered model and discuss Dev → Staging → Prod governance
