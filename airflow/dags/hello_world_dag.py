"""
Simple Hello World DAG for Airflow Tutorial
Demonstrates core Airflow concepts: DAG definition, operators, dependencies, and execution.
Can be executed within Apache Airflow or directly via Python CLI for live lecture demos.
"""
import sys
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

from airflow import DAG
try:
    from airflow.providers.standard.operators.python import PythonOperator
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator

# Default arguments applied to all tasks
default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG (using modern `schedule` parameter)
dag = DAG(
    dag_id='hello_world',
    default_args=default_args,
    description='A simple hello world DAG for learning Airflow basics',
    schedule=timedelta(days=1),  # Run daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['tutorial', 'beginner'],
)

# Python functions that will be executed as tasks
def print_hello():
    """Print a hello message"""
    print("   [Task 1: say_hello] Hello from Apache Airflow Python Operator!")
    return "Hello task completed"

def print_date():
    """Print current date and time"""
    current_time = datetime.now()
    print(f"   [Task 2: print_current_date] Current Execution Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    return "Date task completed"

def print_context(**context):
    """Print task context information"""
    exec_date = context.get('execution_date', datetime.now().isoformat())
    task_id = context.get('task', None)
    task_name = task_id.task_id if task_id else "print_context"
    print(f"   [Task 3: print_context] Task '{task_name}' executed. Execution date: {exec_date}")
    return "Context task completed"

# Task Definitions
task_hello = PythonOperator(
    task_id='say_hello',
    python_callable=print_hello,
    dag=dag,
)

task_date = PythonOperator(
    task_id='print_current_date',
    python_callable=print_date,
    dag=dag,
)

task_context = PythonOperator(
    task_id='print_context',
    python_callable=print_context,
    dag=dag,
)

task_bash = BashOperator(
    task_id='bash_hello',
    bash_command='echo "   [Task 4: bash_hello] BashOperator successfully executed from shell"',
    dag=dag,
)

task_final = BashOperator(
    task_id='final_task',
    bash_command='echo "   [Task 5: final_task] Pipeline workflow completed successfully! 🎉"',
    dag=dag,
)

# Define task dependencies (Execution sequence)
task_hello >> task_date >> task_context >> task_bash >> task_final


# Direct CLI Execution for Live Demos without Airflow Scheduler
if __name__ == "__main__":
    print("=" * 65)
    print("  Airflow Demo 1: Hello World Workflow Execution")
    print("=" * 65)
    print("Workflow Graph: say_hello -> print_date -> print_context -> bash_hello -> final_task\n")
    
    print("Executing DAG Tasks Sequentially...")
    print("-" * 65)
    print_hello()
    print_date()
    print_context(execution_date=datetime.now().isoformat())
    import subprocess
    subprocess.run('echo "   [Task 4: bash_hello] BashOperator successfully executed from shell"', shell=True)
    subprocess.run('echo "   [Task 5: final_task] Pipeline workflow completed successfully! 🎉"', shell=True)
    print("-" * 65)
    print("✓ All DAG tasks completed successfully in standalone test mode.")
    print("=" * 65)
