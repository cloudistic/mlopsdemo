import json
import math
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")


def evaluate(model, matrix, split, live, save_path):
    """
    Dump evaluation metrics and plots for given dataset split.
    """
    labels = matrix[:, 1].toarray().astype(int)
    x = matrix[:, 2:]

    predictions_by_class = model.predict_proba(x)
    predictions = predictions_by_class[:, 1]

    # Calculate metrics
    avg_prec = float(metrics.average_precision_score(labels, predictions))
    roc_auc = float(metrics.roc_auc_score(labels, predictions))

    if live is not None:
        try:
            if not live.summary:
                live.summary = {"avg_prec": {}, "roc_auc": {}}
            live.summary["avg_prec"][split] = avg_prec
            live.summary["roc_auc"][split] = roc_auc

            live.log_sklearn_plot("roc", labels, predictions, name=f"roc/{split}")
            live.log_sklearn_plot(
                "precision_recall",
                labels,
                predictions,
                name=f"prc/{split}",
                drop_intermediate=True,
            )
            live.log_sklearn_plot(
                "confusion_matrix",
                labels.squeeze(),
                predictions_by_class.argmax(-1),
                name=f"cm/{split}",
            )
        except Exception as e:
            pass

    return avg_prec, roc_auc


def save_importance_plot(live, model, feature_names, save_dir):
    """Save feature importance plot."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(figsize=(10, 5), dpi=100)
    fig.subplots_adjust(bottom=0.3, top=0.9)
    axes.set_ylabel("Mean Decrease in Impurity")
    axes.set_title("Top 20 Feature Importances (Random Forest)")

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names[:len(importances)]).nlargest(n=20)
    forest_importances.plot.bar(ax=axes, color="#3498db")
    plt.xticks(rotation=45, ha="right")

    plot_path = os.path.join(save_dir, "importance.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    if live is not None:
        try:
            live.log_image("importance.png", fig)
        except Exception:
            pass


def main():
    EVAL_PATH = "eval"
    os.makedirs(EVAL_PATH, exist_ok=True)

    if len(sys.argv) != 3:
        sys.stderr.write("Usage: python evaluate.py model features\n")
        sys.exit(1)

    model_file = sys.argv[1]
    train_file = os.path.join(sys.argv[2], "train.pkl")
    test_file = os.path.join(sys.argv[2], "test.pkl")

    # Load model and data
    with open(model_file, "rb") as fd:
        model = pickle.load(fd)

    with open(train_file, "rb") as fd:
        train, feature_names = pickle.load(fd)

    with open(test_file, "rb") as fd:
        test, _ = pickle.load(fd)

    # Initialize Live if possible, with fallback
    live = None
    try:
        from dvclive import Live
        # Initialize dvclive without strict git requirements
        live = Live(EVAL_PATH, dvcyaml=True)
    except Exception:
        live = None

    if live is not None:
        with live:
            train_ap, train_auc = evaluate(model, train, "train", live, save_path=EVAL_PATH)
            test_ap, test_auc = evaluate(model, test, "test", live, save_path=EVAL_PATH)
            save_importance_plot(live, model, feature_names, EVAL_PATH)
    else:
        train_ap, train_auc = evaluate(model, train, "train", None, save_path=EVAL_PATH)
        test_ap, test_auc = evaluate(model, test, "test", None, save_path=EVAL_PATH)
        save_importance_plot(None, model, feature_names, EVAL_PATH)

    # Dump standalone metrics.json
    metrics_summary = {
        "avg_prec": {"train": train_ap, "test": test_ap},
        "roc_auc": {"train": train_auc, "test": test_auc}
    }
    with open(os.path.join(EVAL_PATH, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("=" * 60)
    print("  DVC Pipeline Evaluation Summary")
    print("=" * 60)
    print(f"  • Train ROC-AUC:   {train_auc:.4f} | Avg Precision: {train_ap:.4f}")
    print(f"  • Test ROC-AUC:    {test_auc:.4f} | Avg Precision: {test_ap:.4f}")
    print(f"  • Metrics output:  {EVAL_PATH}/metrics.json")
    print(f"  • Plots output:    {EVAL_PATH}/importance.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
