"""
Evaluation utilities: compute and display diagnostic metrics.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[float],
    class_names: List[str] = ("no_tumor", "tumor"),
) -> Dict[str, object]:
    """Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : list of int
        Ground-truth class labels.
    y_pred : list of int
        Predicted class labels.
    y_prob : list of float
        Predicted probability of the positive class (tumor).
    class_names : tuple of str
        Human-readable class names used in the report.

    Returns
    -------
    dict
        Dictionary containing:
        ``accuracy``, ``auc_roc``, ``confusion_matrix``,
        ``classification_report`` (string).
    """
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=list(class_names))

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": acc,
        "auc_roc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def print_metrics(metrics: Dict[str, object]) -> None:
    """Pretty-print evaluation metrics to stdout.

    Parameters
    ----------
    metrics : dict
        Output of :func:`compute_metrics`.
    """
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    if not np.isnan(metrics["auc_roc"]):
        print(f"AUC-ROC  : {metrics['auc_roc']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
