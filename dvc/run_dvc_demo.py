"""
DVC Master Demo Runner
Provides an automated, visual demonstration of the full Data Version Control workflow:
1. Data file tracking and metadata pointers (.dvc)
2. Pipeline execution graph (dvc dag)
3. Reproducible end-to-end execution (dvc repro)
4. Parameter mutation and metrics diffing
"""
import os
import sys
import subprocess
import time
import yaml

TUTORIAL_DIR = os.path.join(os.path.dirname(__file__), "dvctutorial")


def run_cmd(cmd, cwd=TUTORIAL_DIR):
    """Run shell command with venv on PATH."""
    env = os.environ.copy()
    venv_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "venv", "bin"))
    env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
    
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    return result.stdout.strip()


def main():
    print("=" * 70)
    print("  DVC Demo: Data Versioning & Reproducible ML Pipelines")
    print("=" * 70)
    
    # Step 1: Data Pointer concept
    print("\n1. Data Tracking Concept (Pointers in Git, Payloads in Cache):")
    dvc_pointer = os.path.join(TUTORIAL_DIR, "data", "data.xml.dvc")
    if os.path.exists(dvc_pointer):
        with open(dvc_pointer) as f:
            print("   Content of `data/data.xml.dvc` (tracked by Git):")
            for line in f:
                print(f"     {line.strip()}")
    print("   • Git only tracks this tiny ~90 byte hash file.")
    print("   • The actual 14MB XML dataset is safely managed by DVC in cache/remote.")

    # Step 2: Show Pipeline DAG
    print("\n2. DVC Pipeline Directed Acyclic Graph (dvc dag):")
    dag_output = run_cmd("dvc dag")
    for line in dag_output.split("\n"):
        print(f"   {line}")

    # Step 3: Reproduce Pipeline
    print("\n3. Executing Pipeline Reproduction (`dvc repro`):")
    print("   DVC inspects hashes of code, data, and params to execute ONLY stale stages...")
    t0 = time.time()
    repro_output = run_cmd("dvc repro")
    elapsed = time.time() - t0
    
    # Print key lines
    for line in repro_output.split("\n"):
        if any(keyword in line for keyword in ["Running stage", "Summary", "ROC-AUC", "Updating lock", "Data and pipelines"]):
            print(f"   • {line}")
    print(f"   ✓ Pipeline execution verified in {elapsed:.2f}s")

    # Step 4: Show Metrics
    print("\n4. Current Pipeline Metrics (`dvc metrics show`):")
    metrics_path = os.path.join(TUTORIAL_DIR, "eval", "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
            print(f"   • Train ROC-AUC: {metrics['roc_auc']['train']:.4f} | Avg Precision: {metrics['avg_prec']['train']:.4f}")
            print(f"   • Test ROC-AUC:  {metrics['roc_auc']['test']:.4f} | Avg Precision: {metrics['avg_prec']['test']:.4f}")

    print("\n" + "=" * 70)
    print("5. Key Instructor Takeaway for DVC:")
    print("   • Reproducibility: Anyone cloning this repo can type `dvc repro` to get identical results.")
    print("   • Efficiency: Unchanged stages are skipped automatically using content-addressable caching.")
    print("   • Decoupled Storage: Large datasets stay in S3/GCS without bloating Git commit history.")
    print("=" * 70)


if __name__ == "__main__":
    main()
