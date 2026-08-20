import argparse
import json
import platform
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


CANDIDATES = [
    {
        "model_type": "logistic_regression",
        "factory": lambda: LogisticRegression(max_iter=400, solver="lbfgs"),
        "params": {"max_iter": 400, "solver": "lbfgs"},
    },
    {
        "model_type": "random_forest",
        "factory": lambda: RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=4,
            random_state=42,
        ),
        "params": {
            "n_estimators": 150,
            "max_depth": 6,
            "min_samples_split": 4,
            "random_state": 42,
        },
    },
]


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"age", "income", "tenure_months", "num_late_payments", "loan_amount", "has_defaulted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def get_git_commit() -> str:
    try:
        result = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return result.decode().strip()
    except Exception:
        return "unknown"


def build_artifact_report(report_path: Path, payload: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def list_artifacts_recursive(client: MlflowClient, run_id: str, path: str = "") -> list[str]:
    nodes = client.list_artifacts(run_id, path)
    all_paths = []
    for node in nodes:
        all_paths.append(node.path + ("/" if node.is_dir else ""))
        if node.is_dir:
            all_paths.extend(list_artifacts_recursive(client, run_id, node.path))
    return all_paths


def evaluate(model, x_test, y_test) -> dict:
    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(x_test)[:, 1]
    else:
        y_prob = y_pred

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MLflow basic logging + comparison + registry demo")
    parser.add_argument("--tracking-uri", required=True, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="module2-experiment-demo")
    parser.add_argument("--registered-model-name", default="credit-risk-classifier")
    parser.add_argument("--dataset", default="module2/data/sample_training_data.csv")
    parser.add_argument("--dataset-version", default="snapshot-2026-08-20")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    df = load_dataset(args.dataset)
    X = df[["age", "income", "tenure_months", "num_late_payments", "loan_amount"]]
    y = df["has_defaulted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    run_summaries = []

    for idx, candidate in enumerate(CANDIDATES, start=1):
        with mlflow.start_run(run_name=f"candidate-{idx}-{candidate['model_type']}") as run:
            run_id = run.info.run_id

            # Basic/manual logging
            mlflow.log_param("model_type", candidate["model_type"])
            mlflow.log_params(candidate["params"])

            # Reproducibility triad metadata
            mlflow.set_tag("git_commit", get_git_commit())
            mlflow.set_tag("dataset_version", args.dataset_version)
            mlflow.set_tag("python_version", platform.python_version())
            mlflow.set_tag("module", "module2")
            mlflow.set_tag("logging_style", "basic_manual")

            model = candidate["factory"]()
            model.fit(X_train, y_train)

            metrics = evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

            report_payload = {
                "run_id": run_id,
                "candidate": candidate["model_type"],
                "dataset_version": args.dataset_version,
                "params": candidate["params"],
                "metrics": metrics,
            }
            report_file = Path("module2") / "reports" / f"run_report_{run_id}.json"
            build_artifact_report(report_file, report_payload)
            mlflow.log_artifact(str(report_file), artifact_path="reports")

            run_summaries.append({
                "run_id": run_id,
                "model_type": candidate["model_type"],
                **metrics,
            })

            print(f"[INFO] Completed run {run_id} for {candidate['model_type']} with metrics={metrics}")

    comparison_df = pd.DataFrame(run_summaries).sort_values(by=["roc_auc", "f1", "accuracy"], ascending=False)
    best = comparison_df.iloc[0].to_dict()
    best_run_id = best["run_id"]

    # Log comparison artifact under a dedicated summary run
    with mlflow.start_run(run_name="comparison-summary") as summary_run:
        summary_run_id = summary_run.info.run_id
        comparison_csv = Path("module2") / "reports" / "model_comparison.csv"
        comparison_md = Path("module2") / "reports" / "model_comparison.md"
        comparison_csv.parent.mkdir(parents=True, exist_ok=True)

        comparison_df.to_csv(comparison_csv, index=False)
        comparison_md.write_text(comparison_df.to_markdown(index=False), encoding="utf-8")

        mlflow.log_artifact(str(comparison_csv), artifact_path="reports")
        mlflow.log_artifact(str(comparison_md), artifact_path="reports")
        mlflow.log_param("selected_best_run_id", best_run_id)
        mlflow.log_param("selection_metric", "roc_auc")

        print(f"[INFO] Comparison summary run: {summary_run_id}")

    # Register best model from best run
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=args.registered_model_name)

    client = MlflowClient(tracking_uri=args.tracking_uri)
    artifact_tree = list_artifacts_recursive(client, best_run_id)

    print("\n[INFO] Best run selected")
    print(f"[INFO] Run ID: {best_run_id}")
    print(f"[INFO] Model type: {best['model_type']}")
    print(f"[INFO] roc_auc={best['roc_auc']:.4f}, f1={best['f1']:.4f}, accuracy={best['accuracy']:.4f}")

    print("\n[INFO] Best run artifact tree:")
    for p in artifact_tree:
        print(f" - {p}")

    print("\n[INFO] Model registered")
    print(f"[INFO] Registered Model: {args.registered_model_name}")
    print(f"[INFO] Version: {mv.version}")
    print("[INFO] Suggested lifecycle path: Development -> Staging -> Production")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
