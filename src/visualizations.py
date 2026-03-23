"""Visualizations: confusion matrix, confidence distribution, training curves, sample grid."""

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from PIL import Image

from src.data import CIFAR10_CLASSES


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str] | None = None,
) -> Figure:
    """Plot confusion matrix heatmap."""
    if class_names is None:
        class_names = list(CIFAR10_CLASSES)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_confidence_distribution(
    confidences: np.ndarray,
    is_noisy_mask: np.ndarray,
    bins: int = 30,
) -> Figure:
    """Histogram of confidence scores, colored by clean vs noisy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    clean_conf = confidences[~is_noisy_mask]
    noisy_conf = confidences[is_noisy_mask]
    if len(noisy_conf) > 0:
        ax.hist(noisy_conf, bins=bins, alpha=0.7, label="Noisy labels", color="coral", density=True)
    if len(clean_conf) > 0:
        ax.hist(clean_conf, bins=bins, alpha=0.7, label="Clean labels", color="steelblue", density=True)
    ax.set_xlabel("Model Confidence (max prob)")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution: Clean vs Noisy Labels")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_training_curves(history: dict[str, list[float]]) -> Figure:
    """Plot loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", marker="o", markersize=3)
    ax2.plot(epochs, history["val_acc"], label="Val Acc", marker="s", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison_curves(
    history1: dict[str, list[float]],
    history2: dict[str, list[float]],
    metric_key: str,
    label: str,
) -> Figure:
    """Overlay Phase 1 (noisy) vs Phase 2 (corrected) for a given metric across epochs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history1[metric_key]) + 1)
    ax.plot(
        epochs,
        history1[metric_key],
        label="Phase 1 (noisy)",
        marker="o",
        markersize=4,
    )
    ax.plot(
        epochs,
        history2[metric_key],
        label="Phase 2 (corrected)",
        marker="s",
        markersize=4,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.set_title(f"{label} — Phase 1 vs Phase 2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_sample_grid(
    images: list[Image.Image],
    true_labels: list[int] | np.ndarray,
    given_labels: list[int] | np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    cols: int = 5,
    title: str = "Sample Images",
) -> Figure:
    """Grid of sample images with label overlays."""
    if class_names is None:
        class_names = list(CIFAR10_CLASSES)
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(images[i])
            true_str = class_names[int(true_labels[i])]
            if given_labels is not None:
                given_str = class_names[int(given_labels[i])]
                if true_str != given_str:
                    ax.set_title(f"True: {true_str}\nGiven: {given_str}", color="red", fontsize=8)
                else:
                    ax.set_title(f"True: {true_str}", fontsize=8)
            else:
                ax.set_title(true_str, fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return fig
