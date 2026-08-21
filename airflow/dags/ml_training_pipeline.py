"""
Complete Production ML Training Pipeline DAG (Customer Churn 5K Dataset)
Demonstrates an end-to-end enterprise ML retraining pipeline with Airflow:
Data Ingestion -> Quality Validation Gate -> Preprocessing -> Model Training ->
Performance Gate (ROC-AUC >= 0.88) -> Deployment -> Alerting.

Can be scheduled in Apache Airflow or executed directly via CLI for live lecture demos.
"""
import os
import json
import pickle
import shutil
import sys
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

from airflow import DAG
try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'ml-engineering',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='ml_training_pipeline',
    default_args=default_args,
    description='Enterprise customer churn retraining pipeline with automated quality gates',
    schedule='0 2 * * *',  # Runs daily at 2:00 AM UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'production', 'churn'],
)

DATA_DIR = '/tmp/ml_churn_pipeline'
MIN_ROC_AUC_THRESHOLD = 0.88
MIN_F1_THRESHOLD = 0.75


# Task 1: Environment Setup
def create_directories():
    """Task 1: Prepare staging directories"""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"   [1/8 create_directories] Staging directory verified: {DATA_DIR}")


# Task 2: Data Extraction
def extract_data():
    """Task 2: Extract raw customer dataset (5,000 records) from data warehouse"""
    src_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "customer_churn_5k.csv"))
    if not os.path.exists(src_data):
        from data.generate_dataset import get_churn_dataset
        df = get_churn_dataset(5000)
    else:
        df = pd.read_csv(src_data)
        
    output_path = f"{DATA_DIR}/raw_churn_data.csv"
    df.to_csv(output_path, index=False)
    print(f"   [2/8 extract_data] Ingested {len(df):,} customer records across {df.shape[1]} features -> {output_path}")
    return output_path


# Task 3: Data Quality Validation Gate
def validate_data():
    """Task 3: Automated data quality checks before compute allocation"""
    raw_path = f"{DATA_DIR}/raw_churn_data.csv"
    df = pd.read_csv(raw_path)
    
    # Check 1: Missing values
    missing = int(df.isnull().sum().sum())
    if missing > 0:
        raise ValueError(f"Data Validation Failed: {missing} missing values detected!")
        
    # Check 2: Minimum volume threshold (must have at least 4,000 records)
    if len(df) < 4000:
        raise ValueError(f"Data Validation Failed: Insufficient volume ({len(df)} < 4,000)!")
        
    # Check 3: Schema verification (18 features + 1 target)
    expected_cols = 19
    if df.shape[1] < expected_cols:
        raise ValueError(f"Schema Violation: Expected {expected_cols} columns, got {df.shape[1]}!")
        
    print(f"   [3/8 validate_data] Quality Gate PASSED: 0 missing values, {len(df):,} samples, {df.shape[1]} columns verified.")


# Task 4: Preprocessing & Split
def preprocess_data():
    """Task 4: Feature engineering and stratified train/test split"""
    df = pd.read_csv(f"{DATA_DIR}/raw_churn_data.csv")
    X = df.drop("churn_target", axis=1)
    y = df["churn_target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    X_train.to_csv(f"{DATA_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{DATA_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{DATA_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{DATA_DIR}/y_test.csv", index=False)
    print(f"   [4/8 preprocess_data] Stratified split: Train={len(X_train):,} (75%) | Test={len(X_test):,} (25%)")


# Task 5: Model Training
def train_model():
    """Task 5: Train Gradient Boosting classifier on preprocessed dataset"""
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
    
    params = {
        'n_estimators': 120,
        'learning_rate': 0.1,
        'max_depth': 4,
        'random_state': 42
    }
    
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    
    model_path = f"{DATA_DIR}/model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    meta = {
        'model_type': 'GradientBoostingClassifier',
        'params': params,
        'trained_at': datetime.now().isoformat(),
        'train_samples': len(X_train)
    }
    with open(f"{DATA_DIR}/model_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)
        
    print(f"   [5/8 train_model] Model trained and serialized -> {model_path}")


# Task 6: Evaluation & Quality Gate
def evaluate_model():
    """Task 6: Evaluate candidate against multi-metric performance threshold before deployment"""
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()
    
    with open(f"{DATA_DIR}/model.pkl", 'rb') as f:
        model = pickle.load(f)
        
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': auc,
        'evaluated_at': datetime.now().isoformat()
    }
    with open(f"{DATA_DIR}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # QUALITY GATE CHECKS
    if auc < MIN_ROC_AUC_THRESHOLD:
        raise ValueError(
            f"Quality Gate Failed: Model ROC-AUC ({auc:.4f}) below threshold ({MIN_ROC_AUC_THRESHOLD})!"
        )
    if f1 < MIN_F1_THRESHOLD:
        raise ValueError(
            f"Quality Gate Failed: Model F1 ({f1:.4f}) below threshold ({MIN_F1_THRESHOLD})!"
        )
        
    print(f"   [6/8 evaluate_model] Quality Gate PASSED: ROC-AUC={auc:.4f} >= {MIN_ROC_AUC_THRESHOLD} | F1={f1:.4f} >= {MIN_F1_THRESHOLD}")


# Task 7: Production Deployment
def deploy_model():
    """Task 7: Promote validated model artifact to production endpoint"""
    src_model = f"{DATA_DIR}/model.pkl"
    src_meta = f"{DATA_DIR}/model_metadata.json"
    prod_model = f"{DATA_DIR}/production_model.pkl"
    prod_meta = f"{DATA_DIR}/production_metadata.json"
    
    shutil.copy(src_model, prod_model)
    shutil.copy(src_meta, prod_meta)
    
    deployment_record = {
        'status': 'ACTIVE_PRODUCTION',
        'deployed_at': datetime.now().isoformat(),
        'model_artifact': prod_model,
        'metadata_artifact': prod_meta
    }
    with open(f"{DATA_DIR}/deployment_info.json", 'w') as f:
        json.dump(deployment_record, f, indent=2)
        
    print(f"   [7/8 deploy_model] Model promoted to production -> {prod_model}")


# Task 8: Notification & Audit
def send_notification():
    """Task 8: Emit completion alert to engineering teams"""
    with open(f"{DATA_DIR}/metrics.json") as f:
        metrics = json.load(f)
    print(f"   [8/8 send_notification] 📧 Alert broadcast: Churn Retraining Succeeded (ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f})")


# Define Tasks for Airflow Scheduler
t_dirs = PythonOperator(task_id='create_directories', python_callable=create_directories, dag=dag)
t_extract = PythonOperator(task_id='extract_data', python_callable=extract_data, dag=dag)
t_validate = PythonOperator(task_id='validate_data', python_callable=validate_data, dag=dag)
t_prep = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data, dag=dag)
t_train = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)
t_eval = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model, dag=dag)
t_deploy = PythonOperator(task_id='deploy_model', python_callable=deploy_model, dag=dag)
t_notify = PythonOperator(task_id='send_notification', python_callable=send_notification, dag=dag)

# Dependency Sequence:
t_dirs >> t_extract >> t_validate >> t_prep >> t_train >> t_eval >> t_deploy >> t_notify


# Direct CLI Execution for Live Demonstration
if __name__ == "__main__":
    print("=" * 75)
    print("  Airflow Demo 2: Production Churn Retraining Pipeline (5K Records)")
    print("=" * 75)
    print("Pipeline DAG: create_dirs -> extract -> validate -> preprocess ->")
    print("              train -> evaluate (quality gate) -> deploy -> notify\n")
    print("Executing Pipeline Tasks in Sequence...")
    print("-" * 75)
    
    t0 = datetime.now()
    create_directories()
    extract_data()
    validate_data()
    preprocess_data()
    train_model()
    evaluate_model()
    deploy_model()
    send_notification()
    
    elapsed = (datetime.now() - t0).total_seconds()
    print("-" * 75)
    print(f"✓ Pipeline execution completed successfully in {elapsed:.2f}s!")
    print(f"  Artifacts stored in: {DATA_DIR}")
    print("=" * 75)
