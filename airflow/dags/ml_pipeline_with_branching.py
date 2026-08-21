"""
ML Pipeline with Conditional Branching
Demonstrates dynamic workflow routing in Airflow based on runtime data quality checks:
Ingest Data -> Quality Assessment -> [High Quality Path / Medium Quality Path / Alert Path] -> Final Summary.

Can be scheduled in Apache Airflow or executed directly via CLI for live lecture demos.
"""
import random
import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

from airflow import DAG
try:
    from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:
    from airflow.operators.python import PythonOperator, BranchPythonOperator
    from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'ml-engineering',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='ml_pipeline_with_branching',
    default_args=default_args,
    description='Dynamic ML pipeline with conditional routing based on data quality',
    schedule=None,  # Manual trigger / event-driven
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'advanced', 'branching'],
)


def check_data_quality(**context):
    """Assess data quality and return next downstream task ID"""
    score = random.uniform(0.4, 0.99)
    print(f"   [check_data_quality] Calculated Data Quality Score: {score:.2f}")
    
    # Push to XCom if inside Airflow context
    if 'task_instance' in context:
        context['task_instance'].xcom_push(key='quality_score', value=score)
        
    if score >= 0.80:
        print("   -> Routing to 'high_quality_pipeline' (Deep Feature Engineering)")
        return 'high_quality_pipeline'
    elif score >= 0.60:
        print("   -> Routing to 'medium_quality_pipeline' (Robust Imputation & Linear Model)")
        return 'medium_quality_pipeline'
    else:
        print("   -> Routing to 'data_quality_alert' (Threshold Breached < 0.60)")
        return 'data_quality_alert'


def high_quality_pipeline(**context):
    print("   [high_quality_pipeline] Executing complex polynomial & interaction feature engineering...")
    time.sleep(0.3)
    print("   [high_quality_pipeline] Training ensemble model with extensive hyperparameter tuning...")


def medium_quality_pipeline(**context):
    print("   [medium_quality_pipeline] Applying median imputation and outlier capping...")
    time.sleep(0.3)
    print("   [medium_quality_pipeline] Training regularized linear model...")


def send_data_quality_alert(**context):
    print("   [data_quality_alert] ⚠️ ALERT: Severe data drift / corruption detected!")
    print("   [data_quality_alert] Halting automated retraining. On-call engineer alerted.")


def final_summary(**context):
    print(f"   [final_summary] Pipeline execution recorded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")


# Task Definitions
task_ingest = BashOperator(
    task_id='ingest_data',
    bash_command='echo "   [ingest_data] Streaming batch partition from data lake..."',
    dag=dag,
)

task_check = BranchPythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag,
)

task_high = PythonOperator(
    task_id='high_quality_pipeline',
    python_callable=high_quality_pipeline,
    dag=dag,
)

task_med = PythonOperator(
    task_id='medium_quality_pipeline',
    python_callable=medium_quality_pipeline,
    dag=dag,
)

task_alert = PythonOperator(
    task_id='data_quality_alert',
    python_callable=send_data_quality_alert,
    dag=dag,
)

task_final = PythonOperator(
    task_id='final_summary',
    python_callable=final_summary,
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)

# Workflow Routing
task_ingest >> task_check
task_check >> [task_high, task_med, task_alert]
task_high >> task_final
task_med >> task_final
task_alert >> task_final


# Direct CLI Execution for Live Demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("  Airflow Demo 3: Conditional Branching & Dynamic Routing")
    print("=" * 70)
    print("Workflow Graph:")
    print("                    ┌─> high_quality_pipeline ─┐")
    print("  ingest -> check ──┼─> medium_quality_pipeline ┼─> final_summary")
    print("                    └─> data_quality_alert ────┘\n")
    print("Executing Branching Workflow...")
    print("-" * 70)
    
    import subprocess
    subprocess.run('echo "   [ingest_data] Streaming batch partition from data lake..."', shell=True)
    
    chosen_path = check_data_quality()
    if chosen_path == 'high_quality_pipeline':
        high_quality_pipeline()
    elif chosen_path == 'medium_quality_pipeline':
        medium_quality_pipeline()
    else:
        send_data_quality_alert()
        
    final_summary()
    print("-" * 70)
    print("✓ Dynamic conditional routing demonstrated successfully.")
    print("=" * 70)
