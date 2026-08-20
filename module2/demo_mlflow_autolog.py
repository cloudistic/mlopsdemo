import argparse
import platform
import subprocess

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


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


def main() -> int:
    parser = argparse.ArgumentParser(description="MLflow autolog demo")
    parser.add_argument("--tracking-uri", required=True, help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="module2-autolog-demo")
    parser.add_argument("--dataset", default="module2/data/sample_training_data.csv")
    parser.add_argument("--dataset-version", default="snapshot-2026-08-20")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Enables automatic capture of many model/framework artifacts & params
    mlflow.sklearn.autolog(log_input_examples=True, silent=True)

    df = load_dataset(args.dataset)
    X = df[["age", "income", "tenure_months", "num_late_payments", "loan_amount"]]
    y = df["has_defaulted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    with mlflow.start_run(run_name="autolog-rf") as run:
        model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=7)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Optional manual metrics in addition to autologged fields
        mlflow.log_metric("eval_accuracy", float(accuracy_score(y_test, y_pred)))
        mlflow.log_metric("eval_f1", float(f1_score(y_test, y_pred)))
        mlflow.log_metric("eval_roc_auc", float(roc_auc_score(y_test, y_prob)))

        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("python_version", platform.python_version())
        mlflow.set_tag("logging_style", "autolog")

        print("[INFO] Autolog run completed")
        print(f"[INFO] Run ID: {run.info.run_id}")
        print("[INFO] Inspect MLflow UI to compare autolog richness vs manual logging.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
