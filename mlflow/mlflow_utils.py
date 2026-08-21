"""
MLflow Utilities for Lecture Demos
Provides smart tracking URI initialization with instant health checks.
"""
import os
import sys
import warnings
import urllib.request
import mlflow

# Suppress deprecation/future warnings for crisp lecture presentation
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def is_server_reachable(url: str, timeout: float = 1.0) -> bool:
    """Quickly check if the MLflow tracking server is reachable."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status in [200, 301, 302, 404]
    except Exception:
        return False


def setup_tracking(experiment_name: str, preferred_uri: str = "http://127.0.0.1:5000") -> str:
    """
    Configure MLflow tracking URI with graceful fallback to local storage.
    Ensures scripts never hang if the MLflow server isn't running.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", preferred_uri)
    
    if tracking_uri.startswith("http"):
        if is_server_reachable(tracking_uri, timeout=1.0):
            mlflow.set_tracking_uri(tracking_uri)
            status_msg = f"Connected to MLflow Server at {tracking_uri}"
        else:
            # Fallback to local mlruns in workspace root
            workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            local_mlruns = os.path.join(workspace_root, "mlruns")
            mlflow.set_tracking_uri(f"file://{local_mlruns}")
            status_msg = (
                f"MLflow server not active at {tracking_uri} -> Logging locally to {local_mlruns}\n"
                f"   (Tip: Run 'mlflow server --port 5000' in another terminal to use the web UI)"
            )
    else:
        mlflow.set_tracking_uri(tracking_uri)
        status_msg = f"Tracking to: {tracking_uri}"
        
    mlflow.set_experiment(experiment_name)
    return status_msg
