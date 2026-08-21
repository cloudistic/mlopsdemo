# Setup Script for MLOps Lecture Demos

This script helps you set up the environment for the MLOps demos.

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/cloudistic/scalingsystems.git
cd scalingsystems

# Run the setup script
bash setup.sh
```

### Option 2: Manual Setup

#### 1. Prerequisites

Ensure you have:
- Python 3.8 or higher
- pip (Python package manager)
- Git

Check your versions:
```bash
python --version  # Should be 3.8+
pip --version
git --version
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

For all demos (full installation):
```bash
pip install -r requirements.txt
```

For quick start (minimal installation):
```bash
# Core ML libraries
pip install numpy pandas scikit-learn matplotlib

# MLOps tools (pick what you need)
pip install dvc mlflow

# For Airflow (optional, larger installation)
pip install apache-airflow==2.7.0
```

#### 4. Verify Installation

```bash
# Check DVC
dvc version

# Check MLflow
mlflow --version

# Check Airflow (if installed)
airflow version
```

## Module-Specific Setup

### DVC Setup

```bash
# Initialize DVC (in your project)
cd dvc
git init  # If not already a git repo
dvc init

# Verify
dvc version
```

### MLflow Setup

```bash
# Start MLflow tracking server
cd mlflow
mlflow server --host 0.0.0.0 --port 5000

# In a new terminal, run demo scripts
python demo1_simple.py
```

Access MLflow UI at: http://localhost:5000

### Airflow Setup

```bash
# Set Airflow home
export AIRFLOW_HOME=~/airflow

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Copy DAG files
cp airflow/dags/*.py ~/airflow/dags/

# Start webserver (terminal 1)
airflow webserver --port 8080

# Start scheduler (terminal 2)
airflow scheduler
```

Access Airflow UI at: http://localhost:8080 (admin/admin)

## Troubleshooting

### Common Issues

#### Issue: `command not found: python`
**Solution:** Try `python3` instead of `python`

#### Issue: `pip install` fails with permissions error
**Solution:** 
- Use virtual environment (recommended)
- Or use `pip install --user`

#### Issue: MLflow UI not loading
**Solution:**
- Check if port 5000 is already in use
- Try a different port: `mlflow server --port 5001`
- Check firewall settings

#### Issue: Airflow tasks failing
**Solution:**
- Check logs in Airflow UI
- Verify all Python packages are installed
- Ensure file paths are correct

#### Issue: DVC remote storage not working
**Solution:**
- For demos, use local storage: `dvc remote add -d myremote ~/dvc-storage`
- For production, configure cloud storage (S3, GCS, Azure)

### Getting Help

1. Check the README files in each module directory
2. Review the QUICK_START_LECTURE.md guide
3. Consult official documentation:
   - [DVC Docs](https://dvc.org/doc)
   - [MLflow Docs](https://mlflow.org/docs/latest/index.html)
   - [Airflow Docs](https://airflow.apache.org/docs/)

## Testing Your Setup

### Test DVC
```bash
cd dvc
dvc version
# Should display version number
```

### Test MLflow
```bash
cd mlflow
python demo1_simple.py
# Should run without errors and show run ID
```

### Test Airflow
```bash
airflow dags list
# Should display list of DAGs including hello_world and ml_training_pipeline
```

## Clean Up

To remove everything and start fresh:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv

# Remove Airflow home (optional)
rm -rf ~/airflow

# Remove DVC cache (optional)
rm -rf .dvc/cache
```

## Next Steps

1. Read the main [README.md](README.md)
2. Follow the [QUICK_START_LECTURE.md](QUICK_START_LECTURE.md) guide
3. Try the demos in each module:
   - [DVC Tutorial](dvc/README.md)
   - [MLflow Tutorial](mlflow/README.md)
   - [Airflow Tutorial](airflow/README.md)

Happy learning! 🚀
