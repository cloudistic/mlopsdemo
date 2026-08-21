#!/usr/bin/env python3
"""
Master MLOps Lecture Demo Suite & Interactive Launcher
Unified runner for all 10 modules: DVC, MLflow, Airflow, H2O AutoML, and the Grand Finale E2E DAG.
"""
import os
import sys
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXEC = sys.executable

DEMOS = [
    {
        "id": "1",
        "category": "DVC",
        "name": "DVC Data Tracking & Reproducible Pipeline (dvc repro)",
        "script": "dvc/run_dvc_demo.py",
        "description": "Demonstrates data versioning pointers (.dvc), ASCII DAG graph, and instant pipeline caching."
    },
    {
        "id": "2",
        "category": "MLflow",
        "name": "MLflow Demo 1: Core Experiment Tracking",
        "script": "mlflow/demo1_simple.py",
        "description": "Logs hyperparameters, multi-metric business evaluation (Accuracy, F1, ROC-AUC), and model artifacts."
    },
    {
        "id": "3",
        "category": "MLflow",
        "name": "MLflow Demo 2: Systematic Model Comparison",
        "script": "mlflow/demo2_compare.py",
        "description": "Benchmarks 5 distinct model architectures and presents a formatted comparison table."
    },
    {
        "id": "4",
        "category": "MLflow",
        "name": "MLflow Demo 3: Zero-Boilerplate Autologging",
        "script": "mlflow/demo3_autolog.py",
        "description": "Demonstrates 1-line automatic metric, parameter, and model signature logging."
    },
    {
        "id": "5",
        "category": "MLflow",
        "name": "MLflow Demo 4: Model Registry & Deployment",
        "script": "mlflow/demo4_registry.py",
        "description": "Central Model Registry: registers models, assigns '@champion' alias, and runs live inference."
    },
    {
        "id": "6",
        "category": "Airflow",
        "name": "Airflow Demo 1: Hello World Workflow DAG",
        "script": "airflow/dags/hello_world_dag.py",
        "description": "Introduction to Airflow DAGs, operator definitions, and task dependency execution."
    },
    {
        "id": "7",
        "category": "Airflow",
        "name": "Airflow Demo 2: Production ML Pipeline with Quality Gate",
        "script": "airflow/dags/ml_training_pipeline.py",
        "description": "8-stage automated retraining pipeline with automated data & accuracy validation gates."
    },
    {
        "id": "8",
        "category": "Airflow",
        "name": "Airflow Demo 3: Conditional Branching & Dynamic Routing",
        "script": "airflow/dags/ml_pipeline_with_branching.py",
        "description": "Demonstrates runtime data quality scoring and dynamic routing to specialized pipelines."
    },
    {
        "id": "9",
        "category": "AutoML",
        "name": "H2O.ai Enterprise AutoML & Leaderboards",
        "script": "h20ai/demo_h2o.py",
        "description": "Automated feature engineering, multi-model evaluation, leaderboards, and Stacked Ensembles."
    },
    {
        "id": "10",
        "category": "Grand Finale",
        "name": "🚀 Capstone: Unified End-to-End MLOps Retraining DAG",
        "script": "airflow/dags/end_to_end_mlops_pipeline.py",
        "description": "Integrates Ingestion -> Data Gate -> DVC Hash -> MLflow -> Perf Gate -> Registry -> Deployment Probe."
    }
]


def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')


def print_banner():
    print("=" * 80)
    print("      🚀 MODERN MLOPS MASTERCLASS: INTERACTIVE DEMO SUITE")
    print("         End-to-End Reproducibility, Tracking, Governance & Automation")
    print("=" * 80)


def run_script(rel_path: str) -> bool:
    abs_path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(abs_path):
        print(f"\n❌ Error: Script not found at {abs_path}")
        return False
        
    print("\n" + "=" * 80)
    print(f"  EXECUTING: {rel_path}")
    print("=" * 80 + "\n")
    
    t0 = time.time()
    env = os.environ.copy()
    venv_bin = os.path.join(BASE_DIR, "venv", "bin")
    if os.path.exists(venv_bin):
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        
    result = subprocess.run([PYTHON_EXEC, abs_path], cwd=BASE_DIR, env=env)
    elapsed = time.time() - t0
    
    print("\n" + "-" * 80)
    if result.returncode == 0:
        print(f"✓ Execution finished successfully in {elapsed:.2f}s")
        return True
    else:
        print(f"❌ Execution failed with exit code {result.returncode} ({elapsed:.2f}s)")
        return False


def test_all_demos():
    """Run all demos sequentially in automated test mode to guarantee 100% pass rate"""
    print_banner()
    print("  Running pre-flight verification of all 10 demo modules...\n")
    results = []
    
    for demo in DEMOS:
        print(f"\n[{demo['category']}] Running: {demo['name']}...")
        success = run_script(demo["script"])
        results.append((demo["category"], demo["name"], success))
        
    print("\n" + "=" * 80)
    print(f"  PRE-FLIGHT VERIFICATION SUMMARY: {sum(1 for _, _, s in results if s)}/{len(results)} DEMOS PASSED")
    print("=" * 80)
    for cat, name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  [{status}] [{cat:<12}] {name}")
    print("=" * 80)
    
    if all(s for _, _, s in results):
        print("🎉 ALL 10 DEMOS ARE 100% RUNNABLE AND READY FOR THE LIVE LECTURE!\n")
    else:
        print("⚠️ Some demos encountered issues. Please inspect the logs above.\n")


def interactive_menu():
    while True:
        clear_screen()
        print_banner()
        print("\nAvailable Lecture Demonstrations:\n")
        
        current_cat = None
        for demo in DEMOS:
            if demo["category"] != current_cat:
                current_cat = demo["category"]
                print(f"  --- {current_cat.upper()} DEMOS ---")
            print(f"  [{demo['id']:>2}] {demo['name']}")
            print(f"       ↳ {demo['description']}")
            
        print("\n  [A ] Run ALL Demos Sequentially (Pre-Flight Test)")
        print("  [Q ] Quit")
        print("=" * 80)
        
        choice = input("Enter choice (1-10, A, or Q): ").strip()
        
        if choice.lower() == 'q':
            print("\nExiting. Good luck with your lecture! 🎓\n")
            break
        elif choice.lower() == 'a':
            test_all_demos()
            input("\nPress Enter to return to menu...")
        else:
            match = next((d for d in DEMOS if d["id"] == choice), None)
            if match:
                run_script(match["script"])
                input("\nPress Enter to return to menu...")
            else:
                print("\nInvalid selection. Please enter a valid number.")
                time.sleep(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--test-all", "-t"):
        test_all_demos()
    else:
        interactive_menu()
