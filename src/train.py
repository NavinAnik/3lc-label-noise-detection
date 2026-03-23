"""Training loop and evaluation for the label noise detection pipeline."""

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import SimpleCNN


def train_model(
    model: SimpleCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    device: str | torch.device | None = None,
    on_epoch_end: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, list[float]]:
    """Train the model and return history of metrics per epoch.

    Returns:
        history: Dict with train_loss, val_loss, train_acc, val_acc,
            val_precision, val_recall, val_f1 (lists per epoch)
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_loss, val_acc, val_precision, val_recall, val_f1 = _evaluate_epoch(
            model, val_loader, criterion, device
        )
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        if on_epoch_end:
            on_epoch_end(epoch, {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            })

    return history


def _evaluate_epoch(
    model: SimpleCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    """Return (val_loss, val_acc, val_precision, val_recall, val_f1)."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.cpu().numpy().tolist())
            total += batch_x.size(0)
    val_loss = total_loss / total
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    val_acc = float(np.mean(y_true == y_pred))
    from src.metrics import compute_classification_metrics
    metrics = compute_classification_metrics(y_true, y_pred)
    return (
        val_loss,
        val_acc,
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )


def evaluate_model(
    model: SimpleCNN,
    loader: DataLoader,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate model; return predictions, probabilities, and metrics."""
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_probs.extend(probs)
            all_labels.extend(batch_y.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    from src.metrics import compute_classification_metrics

    metrics = compute_classification_metrics(y_true, y_pred)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "metrics": metrics,
    }
