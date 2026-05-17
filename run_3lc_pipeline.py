#!/usr/bin/env python3
"""3LC-driven label noise detection pipeline for CIFAR-10.

This is the 3LC-native counterpart of `main.py`. It performs the exact same
two-phase detect-and-correct workflow but lets the 3LC SDK own the dataset
lifecycle and per-sample metrics tracking:

    Phase 1
      tlc.init -> tlc.Table.from_torch_dataset (with noisy labels)
                -> training loop (vanilla PyTorch)
                -> tlc.collect_metrics every epoch (confidence, loss,
                   predicted, is_predicted_noisy, embeddings)

    Detection
      Query the metrics table for rows where is_predicted_noisy == 1,
      with confidence above min_correction_confidence.

    Phase 2
      Build a corrected `tlc.Table`, retrain a fresh SimpleCNN on it,
      collect metrics again so the Dashboard can show before-vs-after.

Run with:

    python run_3lc_pipeline.py --noise-rate 0.2 --epochs 10

The 3LC Dashboard (https://3lc.ai/download) is optional. Without it, the
pipeline still runs end-to-end and writes all artifacts under
`~/.local/share/3LC` (or `$TLC_ROOT` if set).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from src.data import CIFAR10_CLASSES, inject_label_noise
from src.model import create_model
from src.train import evaluate_model, train_model
from src.tlc_integration import (
    build_embeddings_collector,
    build_noise_metrics_collector,
    collect_3lc_metrics,
    commit_corrected_table,
    flag_from_metrics_table,
    init_3lc_run,
    make_3lc_tables,
    metrics_table_to_rows,
)


# ---------------------------------------------------------------------------
# Device + transforms (shared with main.py - kept here so this script is
# fully self-contained as a 3LC example)
# ---------------------------------------------------------------------------


def get_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transforms():
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )
    train_tfm = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tfm = transforms.Compose([transforms.ToTensor(), normalize])
    return train_tfm, val_tfm


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def _train_with_metrics(
    model,
    tlc_train,
    tlc_val,
    val_loader,
    epochs: int,
    lr: float,
    batch_size: int,
    confidence_threshold: float,
    device: torch.device,
):
    """Train `model` on `tlc_train` and collect 3LC metrics after each epoch."""

    # `tlc_train.create_sampler()` honors any per-sample weights configured
    # in the Dashboard. It falls through to a uniform sampler if none exist.
    sampler = tlc_train.create_sampler()
    train_loader = DataLoader(
        tlc_train,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )

    classification_collector = build_noise_metrics_collector(
        confidence_threshold=confidence_threshold,
        device=device,
    )
    embeddings_collector = build_embeddings_collector(model)

    def on_epoch_end(epoch: int, metrics: dict[str, float]) -> None:
        collect_3lc_metrics(
            model=model,
            tlc_train=tlc_train,
            tlc_val=tlc_val,
            classification_collector=classification_collector,
            embeddings_collector=embeddings_collector,
            epoch=epoch,
            learning_rate=lr,
        )
        print(
            f"  [3LC] collected metrics for epoch {epoch + 1} "
            f"(val_acc={metrics['val_acc']:.4f}, val_f1={metrics['val_f1']:.4f})"
        )

    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        on_epoch_end=on_epoch_end,
    )
    return history, classification_collector


def run(
    noise_rate: float = 0.2,
    epochs: int = 10,
    lr: float = 0.001,
    batch_size: int = 128,
    confidence_threshold: float = 0.5,
    min_correction_confidence: float = 0.5,
    data_dir: str = "./data",
    device: str | None = None,
    project_name: str = "Label Noise Detection - CIFAR10",
):
    dev = get_device(device)
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    print("Loading CIFAR-10...")
    raw_train = CIFAR10(root=data_dir, train=True, download=True)
    raw_val = CIFAR10(root=data_dir, train=False, download=True)
    train_tfm, val_tfm = build_transforms()

    true_labels = np.array(raw_train.targets)
    noisy_labels, noise_mask = inject_label_noise(
        true_labels, noise_rate=noise_rate, num_classes=10
    )
    print(
        f"Injected noise: {int(noise_mask.sum()):,} / {len(true_labels):,} "
        f"labels corrupted ({noise_rate:.0%})"
    )

    # -- Phase 1 -----------------------------------------------------------
    run_phase1 = init_3lc_run(
        project_name=project_name,
        run_name=f"phase1-noisy-{int(noise_rate * 100)}pct",
        description="Phase 1 - train on noisy labels and collect per-sample metrics.",
        config={
            "phase": 1,
            "noise_rate": noise_rate,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "confidence_threshold": confidence_threshold,
        },
    )
    print(f"[3LC] Initialized Run: {run_phase1.url}")

    tlc_train, tlc_val = make_3lc_tables(
        train_dataset=raw_train,
        val_dataset=raw_val,
        noisy_labels=noisy_labels,
        train_transform=train_tfm,
        val_transform=val_tfm,
    )
    print(f"[3LC] Train Table: {tlc_train.url}")
    print(f"[3LC] Val   Table: {tlc_val.url}")

    val_loader = DataLoader(tlc_val, batch_size=batch_size, num_workers=0)

    print("\nPhase 1: training on noisy data...")
    model_p1 = create_model(num_classes=10, device=dev)
    history_p1, classification_collector = _train_with_metrics(
        model_p1,
        tlc_train,
        tlc_val,
        val_loader,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold,
        device=dev,
    )
    eval_p1 = evaluate_model(model_p1, val_loader, device=dev)
    print(
        f"Phase 1 test accuracy: {eval_p1['metrics']['accuracy']:.4f}  "
        f"F1 (macro): {eval_p1['metrics']['f1']:.4f}"
    )

    # -- Detect via the metrics table --------------------------------------
    print("\nDetecting noisy samples via the 3LC metrics table...")
    # Re-run a single metrics-collection pass with shuffle off so that row
    # order matches `noisy_labels` indexing. The cheapest way to get a
    # row-aligned snapshot is to query the model directly on the noisy Table.
    model_p1.eval()
    metrics_rows: list[dict] = []
    with torch.no_grad():
        rows_loader = DataLoader(tlc_train, batch_size=batch_size, shuffle=False)
        offset = 0
        for batch_x, batch_y in rows_loader:
            batch_x = batch_x.to(dev)
            logits = model_p1(batch_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            conf = probs.max(axis=1)
            for k in range(batch_x.size(0)):
                metrics_rows.append(
                    {
                        "confidence": float(conf[k]),
                        "predicted": int(preds[k]),
                        "label": int(batch_y[k]),
                    }
                )
            offset += batch_x.size(0)

    detection = flag_from_metrics_table(
        metrics_rows,
        confidence_threshold=confidence_threshold,
        min_correction_confidence=min_correction_confidence,
    )
    flagged = detection["flagged_indices"]
    corrections = detection["corrections"]
    print(
        f"Flagged {len(flagged):,} samples ({len(flagged) / len(metrics_rows):.1%})."
        f" Will apply {len(corrections):,} corrections "
        f"(confidence > {min_correction_confidence:.2f})."
    )

    # -- Phase 2 -----------------------------------------------------------
    print("\nCommitting corrected Table for Phase 2...")
    tlc_train_corrected, corrected_labels = commit_corrected_table(
        base_dataset=raw_train,
        original_noisy_labels=noisy_labels,
        corrections=corrections,
        train_transform=train_tfm,
        val_transform=val_tfm,
    )
    print(f"[3LC] Corrected Train Table: {tlc_train_corrected.url}")

    run_phase2 = init_3lc_run(
        project_name=project_name,
        run_name=f"phase2-corrected-{int(noise_rate * 100)}pct",
        description="Phase 2 - retrain on corrected labels.",
        config={
            "phase": 2,
            "noise_rate": noise_rate,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "confidence_threshold": confidence_threshold,
            "applied_corrections": int(len(corrections)),
        },
    )
    print(f"[3LC] Initialized Run: {run_phase2.url}")

    print("\nPhase 2: retraining on corrected labels...")
    model_p2 = create_model(num_classes=10, device=dev)
    history_p2, _ = _train_with_metrics(
        model_p2,
        tlc_train_corrected,
        tlc_val,
        val_loader,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold,
        device=dev,
    )
    eval_p2 = evaluate_model(model_p2, val_loader, device=dev)
    print(
        f"Phase 2 test accuracy: {eval_p2['metrics']['accuracy']:.4f}  "
        f"F1 (macro): {eval_p2['metrics']['f1']:.4f}"
    )

    delta = eval_p2["metrics"]["accuracy"] - eval_p1["metrics"]["accuracy"]
    print(
        "\n=== Summary ===\n"
        f"Phase 1 -> Phase 2 accuracy delta: {delta:+.4f}\n"
        f"Open the 3LC Dashboard to inspect per-sample confidence, loss, "
        "predicted labels, and embeddings across both runs."
    )

    return {
        "phase1": {"history": history_p1, "metrics": eval_p1["metrics"]},
        "phase2": {"history": history_p2, "metrics": eval_p2["metrics"]},
        "noise_mask": noise_mask,
        "corrections": corrections,
        "tlc_train": tlc_train,
        "tlc_train_corrected": tlc_train_corrected,
        "tlc_val": tlc_val,
        "run_phase1": run_phase1,
        "run_phase2": run_phase2,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "3LC-driven label noise detection pipeline on CIFAR-10. "
            "Equivalent to main.py but uses tlc.Table + "
            "tlc.FunctionalMetricsCollector for per-sample tracking."
        )
    )
    parser.add_argument("--noise-rate", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--min-correction-confidence", type=float, default=0.5)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--project-name",
        type=str,
        default="Label Noise Detection - CIFAR10",
        help="3LC project name shown in the Dashboard.",
    )
    args = parser.parse_args()

    run(
        noise_rate=args.noise_rate,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        min_correction_confidence=args.min_correction_confidence,
        data_dir=args.data_dir,
        device=args.device,
        project_name=args.project_name,
    )


if __name__ == "__main__":
    main()
