"""Confidence-based noisy label detection and correction suggestions."""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import CIFAR10_CLASSES
from src.model import SimpleCNN


def detect_noisy_labels(
    model: SimpleCNN,
    dataset: Dataset,
    given_labels: np.ndarray,
    confidence_threshold: float,
    device: str | torch.device | None = None,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Flag samples where model confidently predicts a different class.

    A sample is flagged as noisy if:
    - The model's predicted class != given label, AND
    - The model's confidence (max prob) > threshold

    Returns:
        dict with:
            flagged_indices: indices of samples predicted as noisy
            confidences: max prob per sample
            pred_classes: model's predicted class per sample
            is_predicted_noisy: boolean mask
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    n = len(dataset)
    all_confidences = np.zeros(n)
    all_pred_classes = np.zeros(n, dtype=int)

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    idx = 0
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred_classes = probs.argmax(axis=1)
            confidences = probs.max(axis=1)

            b = batch_x.size(0)
            all_confidences[idx : idx + b] = confidences
            all_pred_classes[idx : idx + b] = pred_classes
            idx += b

    given_labels = np.asarray(given_labels).flatten()
    pred_mismatch = all_pred_classes != given_labels
    high_conf_pred = all_confidences > confidence_threshold
    is_predicted_noisy = pred_mismatch & high_conf_pred
    flagged_indices = np.where(is_predicted_noisy)[0].tolist()

    return {
        "flagged_indices": flagged_indices,
        "confidences": all_confidences,
        "pred_classes": all_pred_classes,
        "is_predicted_noisy": is_predicted_noisy,
    }


def detect_noisy_labels_with_probs(
    model: SimpleCNN,
    dataset: Dataset,
    given_labels: np.ndarray,
    confidence_threshold: float,
    device: str | torch.device | None = None,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Same as detect_noisy_labels but stores full probability vectors for corrections."""
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    n = len(dataset)
    all_probs = []
    all_pred_classes = []

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_pred_classes.append(probs.argmax(axis=1))

    all_probs = np.vstack(all_probs)
    all_pred_classes = np.concatenate(all_pred_classes)
    all_confidences = all_probs.max(axis=1)
    given_labels = np.asarray(given_labels).flatten()

    pred_mismatch = all_pred_classes != given_labels
    high_conf_pred = all_confidences > confidence_threshold
    is_predicted_noisy = pred_mismatch & high_conf_pred
    flagged_indices = np.where(is_predicted_noisy)[0].tolist()

    return {
        "flagged_indices": flagged_indices,
        "confidences": all_confidences,
        "pred_classes": all_pred_classes,
        "all_probs": all_probs,
        "is_predicted_noisy": is_predicted_noisy,
    }


def compute_detection_metrics(
    predicted_noisy: np.ndarray,
    actual_noisy: np.ndarray,
) -> dict[str, float]:
    """Precision, recall, F1 of the noise detection itself (binary: noisy vs clean)."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    pred_bin = predicted_noisy.astype(bool)
    actual_bin = actual_noisy.astype(bool)

    return {
        "precision": float(precision_score(actual_bin, pred_bin, zero_division=0)),
        "recall": float(recall_score(actual_bin, pred_bin, zero_division=0)),
        "f1": float(f1_score(actual_bin, pred_bin, zero_division=0)),
    }


def get_corrections(
    pred_classes: np.ndarray,
    flagged_indices: list[int] | np.ndarray,
    all_probs: np.ndarray | None = None,
    min_correction_confidence: float = 0.5,
) -> dict[int, int]:
    """Suggest corrected labels for flagged samples based on model predictions.

    When *all_probs* is provided, only corrections where the model's
    confidence in its predicted class exceeds *min_correction_confidence*
    are included.
    """
    corrections: dict[int, int] = {}
    for i in flagged_indices:
        if all_probs is not None and all_probs[i].max() < min_correction_confidence:
            continue
        corrections[int(i)] = int(pred_classes[i])
    return corrections
