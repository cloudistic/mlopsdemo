# Apache Airflow Tutorial - ML Pipeline Orchestration

## 🎯 Learning Objectives (45 minutes)

In this module, you will learn:
1. What Apache Airflow is and why it's essential for ML workflows
2. How to create and schedule ML pipelines (DAGs)
3. How to handle dependencies and retries
4. How to monitor pipeline execution
5. How to integrate Airflow with ML workflows

## 📖 What is Apache Airflow?

Apache Airflow is a platform to programmatically author, schedule, and monitor workflows.

**Key Features:**
- **DAGs (Directed Acyclic Graphs):** Define workflows as code
- **Scheduling:** Run pipelines on a schedule or trigger-based
- **Monitoring:** Web UI for tracking pipeline execution
- **Scalability:** Handle complex, large-scale workflows
- **Extensibility:** Rich ecosystem of operators and integrations

**Why Airflow for ML?** Orchestrate end-to-end ML pipelines from data ingestion to model deployment.

## 🔧 Prerequisites

- Python 3.8+ with Airflow installed
- Basic understanding of Python
- Understanding of ML workflow stages

## 📚 Tutorial Steps

### Part 1: Airflow Setup (10 minutes)

#### Step 1: Installation

```bash
# Install Airflow (use constraints file for compatibility)
AIRFLOW_VERSION=2.7.0
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Verify installation
airflow version
```

#### Step 2: Initialize Airflow

```bash
# Set Airflow home (optional, defaults to ~/airflow)
export AIRFLOW_HOME=~/airflow

# Initialize the database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

#### Step 3: Start Airflow

You need two terminals:

**Terminal 1 - Start the web server:**
```bash
airflow webserver --port 8080
```

**Terminal 2 - Start the scheduler:**
```bash
airflow scheduler
```

**Access the UI:**
Open http://localhost:8080 in your browser
- Username: `admin`
- Password: `admin`

#### Step 4: Configure DAG Directory

Create a directory for your DAGs:

```bash
mkdir -p ~/airflow/dags
```

**Important:** Airflow looks for DAGs in the `~/airflow/dags` directory by default.

### Part 2: Creating Your First DAG (15 minutes)

#### Step 5: Simple Hello World DAG

Create `~/airflow/dags/hello_world_dag.py`:

```python
"""
Simple Hello World DAG
Demonstrates basic Airflow concepts
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Default arguments for the DAG
default_args = {
    'owner': 'data-science-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'hello_world',
    default_args=default_args,
    description='A simple hello world DAG',
    schedule_interval=timedelta(days=1),  # Run daily
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Don't run for past dates
    tags=['tutorial', 'beginner'],
)

# Define Python functions for tasks
def print_hello():
    print("Hello from Airflow!")
    return "Hello task completed"

def print_date():
    print(f"Current date: {datetime.now()}")
    return "Date task completed"

# Define tasks
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

task_bash = BashOperator(
    task_id='bash_task',
    bash_command='echo "Hello from Bash!"',
    dag=dag,
)

# Define task dependencies
task_hello >> task_date >> task_bash
```

**Key Concepts:**
- `DAG`: The workflow container
- `PythonOperator`: Executes Python functions
- `BashOperator`: Executes bash commands
- `>>`: Sets task dependencies (task_hello runs before task_date)

#### Step 6: Verify Your DAG

```bash
# List all DAGs
airflow dags list

# Check for errors in your DAG
airflow dags list-import-errors

# Test a specific task
airflow tasks test hello_world say_hello 2024-01-01
```

#### Step 7: Trigger Your DAG

**Option 1: Via Web UI**
1. Go to http://localhost:8080
2. Find "hello_world" DAG
3. Toggle the DAG to "On" (unpause)
4. Click "Trigger DAG" button

**Option 2: Via CLI**
```bash
airflow dags trigger hello_world
```

Watch it run in the UI! You can see:
- Task status (success, running, failed)
- Logs for each task
- Execution time

### Part 3: ML Pipeline DAG (20 minutes)

#### Step 8: Complete ML Training Pipeline

Create `~/airflow/dags/ml_training_pipeline.py`:

```python
"""
ML Training Pipeline DAG
Demonstrates a complete ML workflow with Airflow
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json
import os

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='End-to-end ML training pipeline',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'production'],
)

# Define data directory
DATA_DIR = '/tmp/ml_pipeline'

def create_directories():
    """Create necessary directories"""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Created directory: {DATA_DIR}")

def extract_data():
    """Extract data from source"""
    print("Extracting data from source...")
    
    # Load iris dataset
    iris = load_iris()
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['target'] = iris.target
    
    # Save raw data
    output_path = f"{DATA_DIR}/raw_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Data extracted: {output_path}")
    print(f"Shape: {df.shape}")
    
    return output_path

def validate_data():
    """Validate data quality"""
    print("Validating data quality...")
    
    df = pd.read_csv(f"{DATA_DIR}/raw_data.csv")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        raise ValueError(f"Data contains {missing} missing values!")
    
    # Check shape
    if df.shape[0] < 100:
        raise ValueError("Insufficient data samples!")
    
    print("* Data validation passed")
    print(f"Samples: {df.shape[0]}, Features: {df.shape[1]}")

def preprocess_data():
    """Preprocess and split data"""
    print("Preprocessing data...")
    
    df = pd.read_csv(f"{DATA_DIR}/raw_data.csv")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save processed data
    X_train.to_csv(f"{DATA_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{DATA_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{DATA_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{DATA_DIR}/y_test.csv", index=False)
    
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

def train_model():
    """Train ML model"""
    print("Training model...")
    
    # Load training data
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Save model
    model_path = f"{DATA_DIR}/model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"* Model trained and saved: {model_path}")

def evaluate_model():
    """Evaluate model performance"""
    print("Evaluating model...")
    
    # Load test data
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()
    
    # Load model
    with open(f"{DATA_DIR}/model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
        'classification_report': report
    }
    
    with open(f"{DATA_DIR}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"* Model Accuracy: {accuracy:.4f}")
    
    # Quality gate: accuracy must be > 0.85
    if accuracy < 0.85:
        raise ValueError(f"Model accuracy {accuracy:.4f} below threshold 0.85!")

def deploy_model():
    """Deploy model (simulation)"""
    print("Deploying model...")
    
    # In production, this would deploy to a serving platform
    # For demo, we'll just copy the model to a "production" location
    
    import shutil
    prod_path = f"{DATA_DIR}/production_model.pkl"
    shutil.copy(f"{DATA_DIR}/model.pkl", prod_path)
    
    print(f"* Model deployed to: {prod_path}")

def send_notification():
    """Send success notification"""
    print("📧 Sending notification...")
    print("* ML Pipeline completed successfully!")
    print(f"Timestamp: {datetime.now()}")
    
    # In production, send email/Slack notification
    with open(f"{DATA_DIR}/metrics.json", 'r') as f:
        metrics = json.load(f)
    
    print(f"Model Accuracy: {metrics['accuracy']:.4f}")

# Define tasks
task_create_dirs = PythonOperator(
    task_id='create_directories',
    python_callable=create_directories,
    dag=dag,
)

task_extract = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

task_validate = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

task_evaluate = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

task_deploy = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

task_notify = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag,
)

# Define workflow
task_create_dirs >> task_extract >> task_validate >> task_preprocess
task_preprocess >> task_train >> task_evaluate >> task_deploy >> task_notify
```

**This DAG demonstrates:**
- Complete ML pipeline (data → training → evaluation → deployment)
- Data quality checks
- Model validation gates
- Error handling with retries
- Sequential dependencies

#### Step 9: Test the ML Pipeline

```bash
# Test individual tasks
airflow tasks test ml_training_pipeline extract_data 2024-01-01
airflow tasks test ml_training_pipeline train_model 2024-01-01

# Trigger the entire DAG
airflow dags trigger ml_training_pipeline
```

Watch the pipeline execute in the UI:
1. Go to http://localhost:8080
2. Click on "ml_training_pipeline"
3. View the Graph view to see task dependencies
4. Click on tasks to see logs

#### Step 10: Advanced DAG with Branching

Create `~/airflow/dags/ml_pipeline_with_branching.py`:

```python
"""
ML Pipeline with Conditional Branching
Demonstrates branching logic based on data quality
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
import random

default_args = {
    'owner': 'ml-team',
    'retries': 1,
}

dag = DAG(
    'ml_pipeline_branching',
    default_args=default_args,
    description='ML pipeline with conditional logic',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'advanced'],
)

def check_data_quality(**context):
    """Check data quality and decide next step"""
    # Simulate data quality check
    quality_score = random.uniform(0, 1)
    print(f"Data quality score: {quality_score:.2f}")
    
    # Push score to XCom for other tasks
    context['task_instance'].xcom_push(key='quality_score', value=quality_score)
    
    if quality_score > 0.8:
        return 'high_quality_pipeline'
    elif quality_score > 0.5:
        return 'medium_quality_pipeline'
    else:
        return 'data_quality_alert'

def high_quality_process():
    print("Running full feature engineering pipeline")

def medium_quality_process():
    print("Running simplified feature engineering")

def send_alert():
    print("⚠️ ALERT: Data quality is too low!")
    print("Pipeline stopped - manual review required")

# Define tasks
task_check = BranchPythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag,
)

task_high_quality = PythonOperator(
    task_id='high_quality_pipeline',
    python_callable=high_quality_process,
    dag=dag,
)

task_medium_quality = PythonOperator(
    task_id='medium_quality_pipeline',
    python_callable=medium_quality_process,
    dag=dag,
)

task_alert = PythonOperator(
    task_id='data_quality_alert',
    python_callable=send_alert,
    dag=dag,
)

task_final = BashOperator(
    task_id='final_step',
    bash_command='echo "Pipeline completed"',
    trigger_rule='none_failed',  # Run if no upstream tasks failed
    dag=dag,
)

# Define branching workflow
task_check >> [task_high_quality, task_medium_quality, task_alert]
task_high_quality >> task_final
task_medium_quality >> task_final
```

**This demonstrates:**
- Conditional branching based on runtime decisions
- XCom for passing data between tasks
- Trigger rules for task execution

## 🎓 Key Concepts Summary

### 1. DAG (Directed Acyclic Graph)
- **Directed:** Tasks have a specific order
- **Acyclic:** No circular dependencies
- **Graph:** Visual representation of workflow

### 2. Operators
- **PythonOperator:** Execute Python functions
- **BashOperator:** Execute bash commands
- **BranchPythonOperator:** Conditional branching
- **Many more:** SQL, HTTP, Docker, Kubernetes, etc.

### 3. Task Dependencies
```python
# Sequential
task_a >> task_b >> task_c

# Parallel
task_a >> [task_b, task_c] >> task_d

# Complex
task_a >> task_b
task_a >> task_c
[task_b, task_c] >> task_d
```

### 4. Scheduling
```python
# Run daily at 2 AM
schedule_interval='0 2 * * *'

# Run every 6 hours
schedule_interval=timedelta(hours=6)

# Manual only
schedule_interval=None
```

### 5. XCom (Cross-Communication)
Share data between tasks:
```python
# Push data
ti.xcom_push(key='my_key', value='my_value')

# Pull data
value = ti.xcom_pull(key='my_key', task_ids='other_task')
```

## 💡 Best Practices

1. **Idempotency**
   - Tasks should produce same results on reruns
   - Handle partial failures gracefully

2. **Small Tasks**
   - Break complex workflows into small, testable tasks
   - Easier to debug and retry

3. **Error Handling**
   - Set appropriate retries and retry delays
   - Use trigger rules for complex dependencies

4. **Testing**
   - Test tasks individually before running full DAG
   - Use `airflow tasks test` command

5. **Monitoring**
   - Use tags to organize DAGs
   - Set up alerts for failures
   - Monitor execution times

## 🚀 Common Use Cases in ML

### 1. Daily Model Retraining
```python
schedule_interval='0 2 * * *'  # Every day at 2 AM
```

### 2. Data Pipeline → ML Pipeline
```python
data_ingestion >> data_validation >> feature_engineering
feature_engineering >> model_training >> model_evaluation
```

### 3. A/B Testing Pipeline
```python
prepare_data >> [train_model_a, train_model_b]
[train_model_a, train_model_b] >> compare_models >> deploy_winner
```

### 4. Batch Prediction Pipeline
```python
fetch_new_data >> preprocess >> load_model >> batch_predict >> store_results
```

## 🔍 Common Commands Reference

| Command | Description |
|---------|-------------|
| `airflow db init` | Initialize database |
| `airflow webserver` | Start web server |
| `airflow scheduler` | Start scheduler |
| `airflow dags list` | List all DAGs |
| `airflow tasks test <dag> <task> <date>` | Test a task |
| `airflow dags trigger <dag>` | Trigger a DAG |
| `airflow dags pause <dag>` | Pause a DAG |
| `airflow dags unpause <dag>` | Unpause a DAG |

## 🎯 Practice Exercise

Create your own ML pipeline:

1. Create a DAG that:
   - Fetches data from a CSV file
   - Validates data quality
   - Trains two different models (in parallel)
   - Compares their performance
   - Deploys the better model
2. Add retry logic
3. Include a notification task
4. Test it thoroughly

## 📚 Additional Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Airflow Providers](https://airflow.apache.org/docs/apache-airflow-providers/)
- [Awesome Apache Airflow](https://github.com/jghoman/awesome-apache-airflow)

## ❓ Common Questions

**Q: When should I use Airflow vs. other tools?**
A: Use Airflow for:
- Complex workflows with dependencies
- Scheduled/recurring jobs
- Need for monitoring and alerting
- Team collaboration on workflows

**Q: Can Airflow scale?**
A: Yes! Airflow can run on Kubernetes, supports multiple executors (Celery, Kubernetes), and handles thousands of tasks.

**Q: How do I handle secrets?**
A: Use Airflow's built-in Connections and Variables, or integrate with secret management systems (HashiCorp Vault, AWS Secrets Manager).

**Q: Can I use Airflow for real-time?**
A: No. Airflow is designed for batch workflows. For real-time, consider Apache Kafka or Apache Flink.

## 🔗 Integration with Other Tools

### Airflow + MLflow
```python
def log_to_mlflow():
    import mlflow
    mlflow.log_metric("accuracy", 0.95)
```

### Airflow + DVC
```python
def pull_data_with_dvc():
    import subprocess
    subprocess.run(["dvc", "pull"])
```

---

**Congratulations!** You now know how to orchestrate ML pipelines with Apache Airflow!

**Previous Module:** [MLflow - Experiment Tracking](../mlflow/README.md)
