#!/bin/bash
# MLOps Scaling Systems - Quick Setup Script
# This script sets up the environment for the MLOps lecture demos

set -e  # Exit on error

echo "=========================================="
echo "MLOps Scaling Systems - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "   * Found Python $PYTHON_VERSION"

# Check pip
echo ""
echo "2. Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip."
    exit 1
fi
echo "   * pip3 is available"

# Create virtual environment
echo ""
echo "3. Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   ⚠️  Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    echo "   * Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "4. Activating virtual environment..."
source venv/bin/activate
echo "   * Virtual environment activated"

# Upgrade pip
echo ""
echo "5. Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "   * pip upgraded"

# Install dependencies
echo ""
echo "6. Installing dependencies..."
echo "   This may take a few minutes..."

# Install core dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter > /dev/null 2>&1
echo "   * Core ML libraries installed"

# Install MLOps tools
pip install dvc mlflow psutil > /dev/null 2>&1
echo "   * DVC and MLflow installed"

# Optional: Install Airflow (can be slow)
read -p "   Install Apache Airflow? (y/n) [n]: " install_airflow
install_airflow=${install_airflow:-n}

if [ "$install_airflow" = "y" ]; then
    echo "   Installing Apache Airflow (this may take several minutes)..."
    AIRFLOW_VERSION=2.7.0
    PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
    CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
    pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}" > /dev/null 2>&1
    echo "   * Apache Airflow installed"
else
    echo "   ⏭️  Skipping Apache Airflow installation"
fi

# Verify installations
echo ""
echo "7. Verifying installations..."
dvc version > /dev/null 2>&1 && echo "   * DVC: $(dvc version)"
mlflow --version > /dev/null 2>&1 && echo "   * MLflow: $(mlflow --version)"

if [ "$install_airflow" = "y" ]; then
    airflow version > /dev/null 2>&1 && echo "   * Airflow: $(airflow version)"
fi

echo ""
echo "=========================================="
echo "* Setup Complete!"
echo "=========================================="
echo ""
echo "To get started:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Choose a module to explore:"
echo "   • DVC:    cd dvc && cat README.md"
echo "   • MLflow: cd mlflow && cat README.md"
if [ "$install_airflow" = "y" ]; then
    echo "   • Airflow: cd airflow && cat README.md"
fi
echo ""
echo "3. Read the Quick Start guide:"
echo "   cat QUICK_START_LECTURE.md"
echo ""
echo "Happy learning! 🚀"
