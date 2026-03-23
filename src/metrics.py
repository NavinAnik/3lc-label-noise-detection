"""Classification metrics and confusion matrix computation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1 (macro)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, ...] | list[str] | None = None,
) -> np.ndarray:
    """Compute confusion matrix. Optionally use class_names for labeling."""
    cm = sk_confusion_matrix(y_true, y_pred)
    return cm
