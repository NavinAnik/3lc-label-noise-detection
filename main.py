#!/usr/bin/env python3
"""CLI entry point for the label noise detection pipeline."""

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import (
    CIFAR10_CLASSES,
    create_noisy_dataset,
    get_sample_images,
    inject_label_noise,
    load_cifar10,
)
from src.metrics import compute_classification_metrics, compute_confusion_matrix
from src.model import create_model
from src.noise_detection import (
    compute_detection_metrics,
    detect_noisy_labels_with_probs,
    get_corrections,
)
from src.train import evaluate_model, train_model
from src.visualizations import (
    plot_confusion_matrix,
    plot_confidence_distribution,
    plot_sample_grid,
    plot_training_curves,
)


def get_device(device: str | None) -> torch.device:
    """Resolve device string to torch device."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_pipeline(
    noise_rate: float = 0.2,
    epochs: int = 10,
    lr: float = 0.001,
    confidence_threshold: float = 0.5,
    data_dir: str = "./data",
    output_dir: str = "./output",
    device: str | None = None,
):
    """Run the full label noise detection pipeline."""
    dev = get_device(device)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading CIFAR-10...")
    train_loader, val_loader, train_dataset, val_dataset = load_cifar10(
        data_dir=data_dir,
        batch_size=128,
    )

    true_labels = np.array(train_dataset.targets)
    noisy_labels, noise_mask = inject_label_noise(
        true_labels, noise_rate=noise_rate, num_classes=10
    )

    # Train on noisy data (phase 1)
    noisy_train_dataset = create_noisy_dataset(train_dataset, noisy_labels)
    noisy_train_loader = torch.utils.data.DataLoader(
        noisy_train_dataset, batch_size=128, shuffle=True
    )

    print("Training model on noisy labels...")
    model = create_model(num_classes=10, device=dev)
    history = train_model(
        model, noisy_train_loader, val_loader, epochs=epochs, lr=lr, device=dev
    )

    # Evaluate before (on test set)
    eval_result = evaluate_model(model, val_loader, device=dev)
    metrics_before = eval_result["metrics"]
    cm = compute_confusion_matrix(
        eval_result["y_true"], eval_result["y_pred"], CIFAR10_CLASSES
    )

    print("\n--- Phase 1: Model trained on noisy data ---")
    print(f"Test Accuracy: {metrics_before['accuracy']:.4f}")
    print(f"Test F1 (macro): {metrics_before['f1']:.4f}")

    # Detect noisy labels
    print("\nDetecting noisy labels...")
    detection = detect_noisy_labels_with_probs(
        model,
        noisy_train_dataset,
        noisy_labels,
        confidence_threshold=confidence_threshold,
        device=dev,
    )
    flagged = detection["flagged_indices"]
    pred_noisy = detection["is_predicted_noisy"]
    det_metrics = compute_detection_metrics(pred_noisy, noise_mask)
    print(f"Detection Precision: {det_metrics['precision']:.4f}")
    print(f"Detection Recall: {det_metrics['recall']:.4f}")
    print(f"Detection F1: {det_metrics['f1']:.4f}")

    # Correct labels and retrain (phase 2)
    corrections = get_corrections(detection["pred_classes"], flagged)
    corrected_labels = noisy_labels.copy()
    for idx, new_label in corrections.items():
        corrected_labels[idx] = new_label

    corrected_dataset = create_noisy_dataset(train_dataset, corrected_labels)
    corrected_train_loader = torch.utils.data.DataLoader(
        corrected_dataset, batch_size=128, shuffle=True
    )

    print("\nTraining model on corrected labels...")
    model2 = create_model(num_classes=10, device=dev)
    history2 = train_model(
        model2, corrected_train_loader, val_loader, epochs=epochs, lr=lr, device=dev
    )

    eval_result2 = evaluate_model(model2, val_loader, device=dev)
    metrics_after = eval_result2["metrics"]
    cm2 = compute_confusion_matrix(
        eval_result2["y_true"], eval_result2["y_pred"], CIFAR10_CLASSES
    )

    print("\n--- Phase 2: Model trained on corrected data ---")
    print(f"Test Accuracy: {metrics_after['accuracy']:.4f}")
    print(f"Test F1 (macro): {metrics_after['f1']:.4f}")

    # Save plots
    fig1 = plot_training_curves(history)
    fig1.savefig(Path(output_dir) / "training_curves_phase1.png", dpi=120)
    fig1.close()

    fig2 = plot_training_curves(history2)
    fig2.savefig(Path(output_dir) / "training_curves_phase2.png", dpi=120)
    fig2.close()

    fig3 = plot_confusion_matrix(cm, CIFAR10_CLASSES)
    fig3.savefig(Path(output_dir) / "confusion_matrix_before.png", dpi=120)
    fig3.close()

    fig4 = plot_confusion_matrix(cm2, CIFAR10_CLASSES)
    fig4.savefig(Path(output_dir) / "confusion_matrix_after.png", dpi=120)
    fig4.close()

    fig5 = plot_confidence_distribution(
        detection["confidences"], noise_mask
    )
    fig5.savefig(Path(output_dir) / "confidence_distribution.png", dpi=120)
    fig5.close()

    # Sample noisy and corrected
    n_show = min(10, len(flagged))
    if n_show > 0:
        sample_idx = flagged[:n_show]
        imgs = get_sample_images(train_dataset, sample_idx)
        true_l = [true_labels[i] for i in sample_idx]
        given_l = [noisy_labels[i] for i in sample_idx]
        fig6 = plot_sample_grid(
            imgs, true_l, given_l, CIFAR10_CLASSES,
            title="Incorrect Labels (True vs Given)",
        )
        fig6.savefig(Path(output_dir) / "incorrect_labels.png", dpi=120)
        fig6.close()

    print(f"\nPlots saved to {output_dir}/")
    return {
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "detection_metrics": det_metrics,
        "history": history,
        "history2": history2,
        "detection": detection,
        "noise_mask": noise_mask,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Label noise detection pipeline on CIFAR-10"
    )
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.2,
        help="Fraction of labels to corrupt (0.0-1.0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs per phase",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for noise detection",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/mps/cpu)",
    )

    args = parser.parse_args()
    run_pipeline(
        noise_rate=args.noise_rate,
        epochs=args.epochs,
        lr=args.lr,
        confidence_threshold=args.confidence_threshold,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
