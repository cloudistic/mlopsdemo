# MLflow Tutorial - Experiment Tracking and Model Management

## 🎯 Learning Objectives (45 minutes)

In this module, you will learn:
1. What MLflow is and its core components
2. How to track ML experiments systematically
3. How to log parameters, metrics, and artifacts
4. How to use the Model Registry
5. How to compare and visualize experiments

## 📖 What is MLflow?

MLflow is an open-source platform for managing the ML lifecycle, including:
- **Tracking:** Record and query experiments
- **Projects:** Package code for reproducibility
- **Models:** Manage and deploy models
- **Registry:** Centralized model store

**Why MLflow?** Solves the challenge of tracking hundreds of experiments and managing model versions.

## 🔧 Prerequisites

- Python 3.8+ with MLflow installed
- Basic knowledge of scikit-learn
- Jupyter Notebook (optional but recommended)

## 📚 Tutorial Steps

### Part 1: MLflow Tracking Basics (20 minutes)

#### Step 1: Installation and Setup

```bash
# Install MLflow
pip install mlflow

# Install additional dependencies
pip install scikit-learn pandas numpy psutil

# Verify installation
mlflow --version
```

#### Step 2: Start MLflow Tracking Server

```bash
# Start the MLflow UI server
mlflow server --host 0.0.0.0 --port 5000

# Access UI in browser
# http://localhost:5000
```

**Keep this terminal running!** Open a new terminal for the next steps.

#### Step 3: Your First MLflow Experiment

Create a file `simple_experiment.py`:

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set tracking URI (if using remote server)
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment name
mlflow.set_experiment("iris-classification")

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run():
    # Define parameters
    params = {
        "solver": "lbfgs",
        "max_iter": 100,
        "random_state": 42
    }
    
    # Log parameters
    mlflow.log_params(params)
    
    # Train model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Accuracy: {accuracy}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

Run the experiment:
```bash
python simple_experiment.py
```

#### Step 4: Explore the MLflow UI

1. Open http://localhost:5000 in your browser
2. You should see your experiment "iris-classification"
3. Click on it to see the run details
4. Explore:
   - **Parameters:** solver, max_iter, random_state
   - **Metrics:** accuracy
   - **Artifacts:** logged model

**Key Concept:** Every run is tracked automatically with timestamps, user, and source code version.

#### Step 5: Compare Multiple Runs

Create `compare_experiments.py`:

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("model-comparison")

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Define different models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

# Train and log each model
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Log model type as parameter
        mlflow.log_param("model_type", model_name)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Calculate and log metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")

print("\n* All models trained! Check MLflow UI to compare.")
```

Run it:
```bash
python compare_experiments.py
```

#### Step 6: Compare Results in UI

1. Go to MLflow UI (http://localhost:5000)
2. Navigate to "model-comparison" experiment
3. Select all runs (checkbox)
4. Click "Compare" button
5. View parallel coordinates plot and scatter plots
6. Sort by accuracy to find best model

### Part 2: Advanced Features (25 minutes)

#### Step 7: Logging Additional Artifacts

Create `log_artifacts.py`:

```python
import mlflow
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-with-artifacts")

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

with mlflow.start_run():
    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Create confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                   display_labels=iris.target_names)
    disp.plot()
    plt.title("Confusion Matrix")
    
    # Save and log the plot
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    
    # Log a text file with notes
    with open("notes.txt", "w") as f:
        f.write("Model: Logistic Regression\n")
        f.write("Dataset: Iris\n")
        f.write("Notes: Baseline model for comparison\n")
    mlflow.log_artifact("notes.txt")
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print("* Artifacts logged successfully!")
```

Run it:
```bash
python log_artifacts.py
```

Check the UI - you'll see the confusion matrix plot and notes in the Artifacts section!

#### Step 8: Using MLflow Model Registry

Create `model_registry.py`:

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("production-models")

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    # Train a good model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log everything
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log and register model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="iris-classifier"
    )
    
    print(f"Model registered! Accuracy: {accuracy:.4f}")
    print(f"Run ID: {run.info.run_id}")
```

Run it:
```bash
python model_registry.py
```

#### Step 9: Managing Model Versions

In the MLflow UI:
1. Click on "Models" tab
2. Find "iris-classifier"
3. You'll see version 1
4. You can:
   - Add descriptions
   - Tag versions (e.g., "production", "staging")
   - Compare versions
   - Transition to different stages

To transition a model to production:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition model to production
client.transition_model_version_stage(
    name="iris-classifier",
    version=1,
    stage="Production"
)
```

#### Step 10: Loading Models from Registry

```python
import mlflow.pyfunc

# Load production model
model_uri = "models:/iris-classifier/Production"
model = mlflow.pyfunc.load_model(model_uri)

# Use the model
import numpy as np
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example iris measurement
prediction = model.predict(sample_data)
print(f"Prediction: {prediction}")
```

#### Step 11: Autologging (Easy Mode!)

MLflow can automatically log parameters and metrics:

```python
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Enable autologging
mlflow.sklearn.autolog()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("autolog-demo")

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

with mlflow.start_run():
    # Just train - MLflow logs everything automatically!
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Even scores on test data
    score = model.score(X_test, y_test)
    
    print(f"* Everything logged automatically! Score: {score:.4f}")
```

**Autologging captures:**
- All model parameters
- Training metrics
- Model artifacts
- Feature importance plots (for supported models)

## 🎓 Key Concepts Summary

### 1. MLflow Tracking
- **Runs:** Individual executions of your code
- **Experiments:** Groups of related runs
- **Parameters:** Input hyperparameters (e.g., learning_rate=0.01)
- **Metrics:** Output measurements (e.g., accuracy=0.95)
- **Artifacts:** Output files (models, plots, data files)

### 2. MLflow Models
- Standardized format for packaging models
- Supports multiple frameworks (sklearn, TensorFlow, PyTorch, etc.)
- Easy deployment across different platforms

### 3. Model Registry
- Centralized hub for model versions
- Lifecycle stages: None → Staging → Production → Archived
- Version control for models
- Approval workflows

## 💡 Best Practices

1. **Use Descriptive Experiment Names**
   - ❌ "experiment1", "test"
   - * "customer-churn-model", "price-prediction-v2"

2. **Log Everything Relevant**
   - Parameters that affect the model
   - Multiple metrics (accuracy, precision, recall, F1)
   - Visualizations (confusion matrices, feature importance)
   - Model artifacts

3. **Use Autologging When Possible**
   - Reduces boilerplate code
   - Ensures consistency
   - Captures more information automatically

4. **Tag Your Runs**
   ```python
   mlflow.set_tag("team", "data-science")
   mlflow.set_tag("priority", "high")
   ```

5. **Document Model Versions**
   - Add descriptions in the Model Registry
   - Document what changed between versions
   - Include performance benchmarks

## 🚀 Common Use Cases

### Hyperparameter Tuning

```python
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("hyperparameter-tuning")

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15]
}

with mlflow.start_run():
    # Grid search
    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Log best parameters
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_score", grid_search.best_score_)
```

### A/B Testing Models

```python
# Compare production model vs new candidate
prod_model = mlflow.pyfunc.load_model("models:/my-model/Production")
candidate_model = mlflow.pyfunc.load_model("models:/my-model/Staging")

# Evaluate both on new data
prod_score = evaluate(prod_model, test_data)
candidate_score = evaluate(candidate_model, test_data)

if candidate_score > prod_score:
    # Promote candidate to production
    client.transition_model_version_stage(
        name="my-model",
        version=candidate_version,
        stage="Production"
    )
```

## 🔍 Common Commands Reference

| Command | Description |
|---------|-------------|
| `mlflow server` | Start tracking server |
| `mlflow.set_experiment()` | Set active experiment |
| `mlflow.start_run()` | Start a new run |
| `mlflow.log_param()` | Log a parameter |
| `mlflow.log_metric()` | Log a metric |
| `mlflow.log_artifact()` | Log a file |
| `mlflow.sklearn.log_model()` | Log a model |
| `mlflow.sklearn.autolog()` | Enable autologging |

## 🎯 Practice Exercise

Create your own experiment:

1. Choose a dataset (e.g., wine quality, boston housing)
2. Train 3 different models
3. Log parameters, metrics, and a visualization
4. Compare results in the UI
5. Register the best model
6. Transition it to "Staging"

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Examples](https://github.com/mlflow/mlflow/tree/master/examples)
- [MLflow Model Registry Guide](https://mlflow.org/docs/latest/model-registry.html)

## ❓ Common Questions

**Q: Where is data stored?**
A: By default in `./mlruns` directory. Can be configured to use databases.

**Q: Can I use MLflow with TensorFlow/PyTorch?**
A: Yes! MLflow supports all major ML frameworks.

**Q: How do I deploy MLflow models?**
A: MLflow provides CLI tools and APIs for deployment to various platforms (Docker, AWS SageMaker, Azure ML, etc.)

**Q: Is MLflow free?**
A: Yes, MLflow is open-source and free.

---

**Next Module:** [Apache Airflow - Workflow Orchestration](../airflow/README.md)
