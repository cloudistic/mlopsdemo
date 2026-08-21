# DVC (Data Version Control) Tutorial

## 🎯 Learning Objectives (30 minutes)

In this module, you will learn:
1. What DVC is and why it's essential for ML projects
2. How to version control large datasets
3. How to create reproducible ML pipelines
4. How to track experiments and metrics

## 📖 What is DVC?

DVC (Data Version Control) is a version control system for machine learning projects that:
- Tracks large data files and ML models (like Git, but for data)
- Creates reproducible ML pipelines
- Enables efficient data sharing across teams
- Integrates seamlessly with Git

**Think of it as:** Git for data + Make for ML pipelines

## 🔧 Prerequisites

- Git installed and configured
- Python 3.8+ with DVC installed
- Basic understanding of Git concepts

## 📚 Tutorial Steps

### Part 1: Basic DVC Setup (10 minutes)

#### Step 1: Installation

```bash
# Install DVC
pip install dvc

# Verify installation
dvc version
```

#### Step 2: Initialize DVC in a Git Repository

```bash
# Create a new project directory
mkdir my-ml-project
cd my-ml-project

# Initialize Git (DVC requires Git)
git init

# Initialize DVC
dvc init

# Check what DVC created
git status
```

**What happened?** DVC created `.dvc` directory and configuration files.

#### Step 3: Track Your First Dataset

```bash
# Create data directory
mkdir -p data

# Download sample data (or use your own dataset)
dvc get https://github.com/iterative/dataset-registry \
    get-started/data.xml -o data/data.xml

# Track the data file with DVC
dvc add data/data.xml

# Check what was created
ls -la data/
```

**Key concept:** `dvc add` creates a `.dvc` file that tracks the data file. The actual data is stored in `.dvc/cache`.

#### Step 4: Commit Changes to Git

```bash
# Add DVC files to Git (NOT the actual data!)
git add data/data.xml.dvc data/.gitignore

# Commit
git commit -m "Add dataset tracking with DVC"
```

**Important:** Git tracks the `.dvc` file (small), not the actual data file (large).

#### Step 5: Configure Remote Storage

```bash
# Create a local remote storage (for demo purposes)
mkdir -p ~/dvc-storage

# Configure DVC remote
dvc remote add -d myremote ~/dvc-storage

# Add remote config to Git
git add .dvc/config
git commit -m "Configure DVC remote storage"

# Push data to remote
dvc push
```

**In production:** Use cloud storage (S3, GCS, Azure Blob) instead of local directories.

#### Step 6: Simulate Collaboration - Pull Data

```bash
# Remove local cache and data file
rm -rf .dvc/cache
rm -f data/data.xml

# Pull data from remote
dvc pull

# Verify data is restored
ls -la data/
```

**Key takeaway:** Team members only need the `.dvc` files to retrieve the actual data!

### Part 2: Building ML Pipelines (20 minutes)

#### Step 7: Download Pipeline Code

```bash
# Download sample ML pipeline code
wget https://code.dvc.org/get-started/code.zip
unzip code.zip && rm -f code.zip

# Check the structure
ls -la src/
```

The code includes:
- `prepare.py` - Data preparation
- `featurization.py` - Feature engineering
- `train.py` - Model training
- `evaluate.py` - Model evaluation

#### Step 8: Create Pipeline Parameters

Create a `params.yaml` file to store hyperparameters:

```yaml
prepare:
  split: 0.20
  seed: 20170428

featurize:
  max_features: 500
  ngrams: 1

train:
  seed: 20170428
  n_est: 50
  min_split: 2
```

#### Step 9: Define Pipeline Stages

**Stage 1: Data Preparation**
```bash
dvc stage add -n prepare \
    -p prepare.seed,prepare.split \
    -d src/prepare.py -d data/data.xml \
    -o data/prepared \
    python src/prepare.py data/data.xml
```

**Stage 2: Feature Engineering**
```bash
dvc stage add -n featurize \
    -p featurize.max_features,featurize.ngrams \
    -d src/featurization.py -d data/prepared \
    -o data/features \
    python src/featurization.py data/prepared data/features
```

**Stage 3: Model Training**
```bash
dvc stage add -n train \
    -p train.seed,train.n_est,train.min_split \
    -d src/train.py -d data/features \
    -o model.pkl \
    python src/train.py data/features model.pkl
```

**Stage 4: Model Evaluation**
```bash
dvc stage add -n evaluate \
    -d src/evaluate.py -d model.pkl -d data/features \
    -M eval/metrics.json \
    python src/evaluate.py model.pkl data/features
```

#### Step 10: Visualize the Pipeline

```bash
# View pipeline as ASCII DAG
dvc dag

# Output shows:
#   +---------+
#   | prepare |
#   +---------+
#        *
#        *
#        *
# +-----------+
# | featurize |
# +-----------+
#        *
#        *
#        *
#   +-------+
#   | train |
#   +-------+
#        *
#        *
#        *
# +----------+
# | evaluate |
# +----------+
```

#### Step 11: Run the Pipeline

```bash
# Run the entire pipeline
dvc repro

# DVC will:
# 1. Check which stages need to run
# 2. Execute stages in correct order
# 3. Cache outputs for reproducibility
```

#### Step 12: Track Experiments

```bash
# View metrics
dvc metrics show

# Modify parameters in params.yaml (e.g., change n_est to 100)
# Run pipeline again
dvc repro

# Compare experiments
dvc metrics diff
```

#### Step 13: Commit Pipeline to Git

```bash
git add .gitignore data/.gitignore dvc.yaml dvc.lock params.yaml
git commit -m "Create ML pipeline with DVC"
```

## 🎓 Key Concepts Summary

1. **DVC Tracking (`.dvc` files)**
   - Small metadata files tracked by Git
   - Point to actual data in cache or remote storage
   - Enable data versioning without storing large files in Git

2. **DVC Pipelines (`dvc.yaml`)**
   - Define stages and dependencies
   - Automatically detect what needs to re-run
   - Ensure reproducibility

3. **DVC Lock File (`dvc.lock`)**
   - Records exact versions of dependencies
   - Ensures reproducible results
   - Like `package-lock.json` for ML

4. **Remote Storage**
   - Centralized data storage
   - Team collaboration
   - Backup and sharing

## 💡 Best Practices

1. **Always use Git with DVC** - They work together, not separately
2. **Keep data out of Git** - Let DVC handle large files
3. **Use meaningful stage names** - Makes pipelines self-documenting
4. **Track parameters separately** - Use `params.yaml` for hyperparameters
5. **Commit .dvc files** - But not the actual data
6. **Use remote storage** - For team collaboration

## 🚀 Advanced Topics (Optional)

### Experiment Tracking

```bash
# Run an experiment
dvc exp run --set-param train.n_est=100

# List experiments
dvc exp list

# Compare experiments
dvc exp diff
```

### Data Versioning

```bash
# Switch to a previous version
git checkout <commit-hash> data/data.xml.dvc
dvc checkout

# Return to latest
git checkout main data/data.xml.dvc
dvc checkout
```

## 🔍 Common Commands Reference

| Command | Description |
|---------|-------------|
| `dvc init` | Initialize DVC in a Git repo |
| `dvc add <file>` | Track a file with DVC |
| `dvc push` | Upload data to remote storage |
| `dvc pull` | Download data from remote storage |
| `dvc repro` | Run/reproduce pipeline |
| `dvc dag` | Visualize pipeline |
| `dvc metrics show` | View metrics |
| `dvc params diff` | Compare parameters |

## 🎯 Practice Exercise

Try this on your own:

1. Create a new ML project with DVC
2. Add a dataset (use any CSV file)
3. Create a simple pipeline with 2 stages:
   - Stage 1: Split data into train/test
   - Stage 2: Train a simple model
4. Track metrics
5. Modify parameters and compare results

## 📚 Additional Resources

- [DVC Official Documentation](https://dvc.org/doc)
- [DVC Tutorial](https://dvc.org/doc/start)
- [DVC Use Cases](https://dvc.org/doc/use-cases)
- [DVC VS Code Extension](https://marketplace.visualstudio.com/items?itemName=Iterative.dvc)

## ❓ Common Questions

**Q: Why not just use Git LFS?**
A: DVC is specifically designed for ML workflows with pipeline management, experiment tracking, and better handling of large datasets.

**Q: Can I use DVC without Git?**
A: No, DVC requires Git as it builds on top of Git's versioning capabilities.

**Q: What storage backends does DVC support?**
A: S3, GCS, Azure Blob, SSH, HDFS, HTTP, and local file systems.

**Q: Is DVC free?**
A: Yes, DVC is open-source and free to use.

---

**Next Module:** [MLflow - Experiment Tracking](../mlflow/README.md)
