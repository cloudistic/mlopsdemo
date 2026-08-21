# Quick Start Guide - 2-Hour MLOps Lecture

This guide provides a streamlined path through the demos for a 2-hour lecture session.

## 🎯 Session Overview

**Total Duration:** 2 hours  
**Target Audience:** Analytical industry professionals  
**Prerequisites:** Basic Python and ML knowledge

## 📅 Lecture Schedule

### ⏰ Session Breakdown

| Time | Duration | Module | Activity |
|------|----------|--------|----------|
| 0:00 - 0:10 | 10 min | Introduction | MLOps overview and setup verification |
| 0:10 - 0:40 | 30 min | DVC | Data versioning and pipeline demo |
| 0:40 - 1:25 | 45 min | MLflow | Experiment tracking and model registry |
| 1:25 - 1:55 | 30 min | Airflow | Workflow orchestration demo |
| 1:55 - 2:00 | 5 min | Wrap-up | Q&A and next steps |

---

## 🚀 Pre-Lecture Setup (Instructor)

### 1. Environment Preparation

```bash
# Clone the repository
git clone https://github.com/cloudistic/scalingsystems.git
cd scalingsystems

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies only (lighter installation)
pip install dvc mlflow scikit-learn pandas numpy matplotlib jupyter
```

### 2. Verify Installation

```bash
# Check installations
dvc version
mlflow --version
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"
```

---

## 📖 Module 1: Introduction (10 minutes)

### Talking Points

1. **What is MLOps?** (3 min)
   - Software engineering practices applied to ML
   - Versioning, automation, monitoring
   - Bridge between data science and production

2. **The MLOps Challenge** (3 min)
   - Jupyter notebooks everywhere
   - "Works on my machine"
   - Lost experiments
   - Manual deployments

3. **Today's Tools** (2 min)
   - DVC: Git for data
   - MLflow: Experiment tracking
   - Airflow: Workflow automation

4. **Setup Check** (2 min)
   - Ensure participants have Python 3.8+
   - Quick installation check

---

## 📖 Module 2: DVC Demo (30 minutes)

### 🎬 Demo Script

**Time: 0:10 - 0:40**

#### Demo 1: Basic Data Tracking (10 min)

```bash
# Navigate to DVC demo
cd dvc

# Create a simple project
mkdir dvc-demo && cd dvc-demo
git init
dvc init

# Show what DVC created
ls -la .dvc/
git status
```

**Key Points to Emphasize:**
- DVC works on top of Git
- `.dvc` directory for configuration
- Separates data from code

```bash
# Download sample data
dvc get https://github.com/iterative/dataset-registry \
    get-started/data.xml -o data.xml

# Track with DVC
dvc add data.xml

# Show what was created
ls -la
cat data.xml.dvc  # Show the metadata file
```

**Key Points:**
- `.dvc` file is small metadata
- Actual data in `.dvc/cache`
- Only metadata goes to Git

```bash
# Commit to Git
git add data.xml.dvc .gitignore
git commit -m "Track data with DVC"
```

#### Demo 2: Remote Storage (5 min)

```bash
# Configure remote storage
mkdir ~/dvc-storage
dvc remote add -d myremote ~/dvc-storage

# Push data
dvc push

# Simulate team member: delete and pull
rm -rf .dvc/cache data.xml
dvc pull
```

**Key Points:**
- Remote storage for collaboration
- In production: use S3, GCS, Azure
- Team members only need `.dvc` files

#### Demo 3: Pipeline Preview (15 min)

```bash
# Show a pre-built pipeline
cd ../dvctutorial

# Visualize the pipeline
dvc dag
```

**Show `dvc.yaml` structure:**
```yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - data/data.xml
    outs:
      - data/prepared
```

**Key Points:**
- Declarative pipeline definition
- Auto-detects what needs to re-run
- Reproducibility built-in

```bash
# Run the pipeline
dvc repro

# Show metrics
dvc metrics show
```

**Wrap-up:**
- DVC = Git for data + Make for ML
- Version data without Git bloat
- Reproducible pipelines

---

## 📖 Module 3: MLflow Demo (45 minutes)

### 🎬 Demo Script

**Time: 0:40 - 1:25**

#### Setup (5 min)

```bash
# Start MLflow server (in separate terminal)
cd ../../mlflow
mlflow server --host 0.0.0.0 --port 5000

# Open UI in browser
# http://localhost:5000
```

**Show participants the UI:**
- Clean slate
- Experiments tab
- Models tab

#### Demo 1: First Experiment (10 min)

**Create `demo1_simple.py`:**

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("lecture-demo")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and log
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 100)
    
    # Train
    model = LogisticRegression(max_iter=100)
    model.fit(X_train, y_train)
    
    # Evaluate and log metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Logged! Accuracy: {accuracy:.4f}")
```

**Run it:**
```bash
python demo1_simple.py
```

**In MLflow UI:**
- Show the run
- Parameters
- Metrics
- Model artifact

**Key Points:**
- Everything tracked automatically
- Timestamp, user, code version
- Never lose an experiment again

#### Demo 2: Model Comparison (15 min)

**Create `demo2_compare.py`:**

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model-comparison")

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try different models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"{name}: {accuracy:.4f}")
```

**Run it:**
```bash
python demo2_compare.py
```

**In MLflow UI:**
1. Navigate to "model-comparison" experiment
2. Select all 3 runs
3. Click "Compare"
4. Show parallel coordinates plot
5. Show scatter plot

**Key Points:**
- Easy model comparison
- Visual comparisons
- Data-driven decisions

#### Demo 3: Autologging (10 min)

**Create `demo3_autolog.py`:**

```python
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Enable autologging - that's it!
mlflow.sklearn.autolog()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("autolog-demo")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    # Just train - MLflow logs everything!
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Score: {score:.4f}")
```

**Run it:**
```bash
python demo3_autolog.py
```

**In MLflow UI - show what was logged automatically:**
- All model parameters
- Training metrics
- Model artifact
- Even feature importance!

**Key Points:**
- Autologging = minimal code
- Captures more than manual logging
- Production-ready

#### Demo 4: Model Registry (5 min)

**In MLflow UI:**
1. Go to a successful run
2. Click "Register Model"
3. Name it: "iris-classifier"
4. Show Models tab
5. Show version management
6. Show stage transitions (None → Staging → Production)

**Key Points:**
- Centralized model store
- Version management
- Deployment workflow

**Wrap-up:**
- MLflow = complete ML lifecycle
- Track everything
- Compare experiments
- Manage models

---

## 📖 Module 4: Airflow Demo (30 minutes)

### 🎬 Demo Script

**Time: 1:25 - 1:55**

#### Setup (5 min)

```bash
# Initialize Airflow (if not done)
export AIRFLOW_HOME=~/airflow
airflow db init

# Create user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start services (two terminals)
# Terminal 1:
airflow webserver --port 8080

# Terminal 2:
airflow scheduler
```

**Open UI:**
- http://localhost:8080
- Login: admin/admin

#### Demo 1: Hello World DAG (10 min)

**Show the `hello_world_dag.py` from our repo:**

```bash
cd ../../airflow/dags
cat hello_world_dag.py
```

**In Airflow UI:**
1. Show DAGs list
2. Find "hello_world"
3. Unpause it
4. Show Graph view
5. Trigger manually
6. Watch it run
7. Click on tasks to see logs

**Key Concepts to Explain:**
- DAG = Directed Acyclic Graph
- Tasks = individual units of work
- Dependencies = execution order
- Operators = task types

#### Demo 2: ML Pipeline DAG (15 min)

**Show the `ml_training_pipeline.py`:**

```bash
cat ml_training_pipeline.py
```

**Explain the structure:**
1. Data extraction
2. Validation
3. Preprocessing
4. Training
5. Evaluation
6. Deployment
7. Notification

**In Airflow UI:**
1. Find "ml_training_pipeline"
2. Show Graph view - visualize the workflow
3. Trigger the DAG
4. Watch execution in real-time
5. Show color coding (success = green, running = light green, failed = red)
6. Click on tasks to show logs

**Show logs for "evaluate_model":**
- Accuracy metrics
- Quality gate check

**Key Points:**
- End-to-end ML workflow
- Automated execution
- Built-in retry logic
- Quality gates

**Show schedule configuration:**
```python
schedule_interval='0 2 * * *'  # Daily at 2 AM
```

**Explain use cases:**
- Daily model retraining
- Data pipeline automation
- A/B testing workflows

**Wrap-up:**
- Airflow = workflow automation
- DAGs define dependencies
- Monitor execution
- Production-ready scheduling

---

## 📖 Module 5: Wrap-up (5 minutes)

### Key Takeaways

**DVC:**
- * Version data like code
- * Create reproducible pipelines
- * Collaborate on data

**MLflow:**
- * Track every experiment
- * Compare models easily
- * Manage model lifecycle

**Airflow:**
- * Automate ML workflows
- * Schedule and monitor
- * Handle complex dependencies

### The Complete MLOps Stack

```
Code (Git) + Data (DVC) + Experiments (MLflow) + Automation (Airflow)
= Production-Ready ML System
```

### Next Steps for Participants

1. **Try the demos** in this repository
2. **Apply to your projects** - start small
3. **Explore integrations** - these tools work together
4. **Join communities** - active open-source communities

### Resources Provided

- 📁 All demo code in this repository
- 📖 Detailed README for each tool
- 🎯 Practice exercises
- 🔗 Links to official documentation

### Q&A Time

Common questions to be ready for:
- Can these tools work together? (Yes!)
- Do I need all three? (Start with what you need)
- Are they free? (Yes, all open-source)
- Production-ready? (Yes, used by major companies)

---

## 🎬 Instructor Tips

### Before the Lecture

1. **Test all demos** - run through everything
2. **Prepare backups** - pre-run outputs in case of issues
3. **Check internet** - some demos download data
4. **Setup multiple terminals** - have them ready

### During the Lecture

1. **Go slow** - 2 hours goes fast
2. **Show, don't just tell** - live demos are key
3. **Handle errors gracefully** - they happen, explain them
4. **Encourage questions** - but manage time
5. **Share screen effectively** - zoom in on code

### Common Issues and Solutions

**MLflow UI not loading:**
- Check port 5000 is free
- Try different port: `mlflow server --port 5001`

**Airflow tasks failing:**
- Check logs in UI
- Verify Python packages installed
- Check file paths

**DVC remote not working:**
- Use local directory for demo
- Explain cloud storage is for production

### Time Management

- **If running late:** Skip autologging demo, shorten Q&A
- **If running early:** Add more model comparisons in MLflow
- **Keep breaks optional** - participants may prefer continuous

### Engagement Tips

1. **Ask questions:** "Who has lost track of experiments?"
2. **Show pain points first:** Then show solutions
3. **Use real examples:** "Imagine you trained 100 models..."
4. **Make it relatable:** "This happened to me..."

---

## 📋 Checklist for Lecture Day

- [ ] All software installed and tested
- [ ] Repository cloned
- [ ] MLflow server starts successfully
- [ ] Airflow webserver and scheduler start
- [ ] Demo scripts tested
- [ ] Presentation slides ready (if any)
- [ ] Backup plan for internet issues
- [ ] Participant materials ready
- [ ] Zoom/screen sharing tested
- [ ] Water bottle ready 💧

---

## 🎯 Success Metrics

**Participants should leave able to:**
1. Explain what DVC, MLflow, and Airflow do
2. Run basic examples of each tool
3. Identify use cases in their work
4. Know where to find documentation
5. Feel confident to start using these tools

---

**Good luck with your lecture! 🚀**
