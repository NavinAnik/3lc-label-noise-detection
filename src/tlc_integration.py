"""3LC SDK integration for the label noise detection pipeline.

This module wraps the existing PyTorch pipeline (CIFAR-10 + SimpleCNN +
confidence-based noise detection) with the 3LC SDK so that:

  - Datasets become first-class 3LC Tables that can be versioned and edited
    from the 3LC Dashboard (https://3lc.ai).
  - Per-sample metrics (confidence, predicted class, loss, and the
    is_predicted_noisy flag) are streamed into a 3LC Run via
    FunctionalMetricsCollector, so detection happens *inside* the 3LC
    metrics pipeline instead of as a one-off script.
  - Penultimate-layer embeddings are collected with
    EmbeddingsMetricsCollector and can be reduced with PaCMAP for
    interactive inspection in the Dashboard.
  - Flagged noisy samples are converted into a corrected Table version that
    Phase 2 trains on, closing the detect -> correct -> retrain loop.

The 3LC SDK is an optional dependency. We import it lazily so the rest of
the package can be used without it installed. If `tlc` is missing, every
helper here raises a clear, actionable error at call time.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data import CIFAR10_CLASSES, NoisyLabelDataset
from src.model import SimpleCNN


_TLC_IMPORT_ERROR = (
    "The 3LC SDK is not installed. Install it with `pip install 3lc` and "
    "(optionally) the desktop Dashboard from https://3lc.ai/download to "
    "enable interactive inspection. The rest of this repository works "
    "without 3LC."
)


def _require_tlc():
    """Lazy import of the `tlc` package so the rest of the repo runs without it."""
    try:
        import tlc  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without tlc
        raise ImportError(_TLC_IMPORT_ERROR) from exc
    return tlc


# ---------------------------------------------------------------------------
# Run + config bootstrap
# ---------------------------------------------------------------------------


def init_3lc_run(
    project_name: str = "Label Noise Detection - CIFAR10",
    run_name: str = "phase1-noisy-baseline",
    description: str = "Detect and correct noisy CIFAR-10 labels with 3LC.",
    config: dict[str, Any] | None = None,
    if_exists: str = "overwrite",
):
    """Initialize a 3LC Run and persist its config parameters.

    Mirrors the official tutorial pattern:
        run = tlc.init(project_name=..., run_name=..., description=...)
        run.set_parameters(config)
    """
    tlc = _require_tlc()
    run = tlc.init(
        project_name=project_name,
        run_name=run_name,
        description=description,
        if_exists=if_exists,
    )
    if config:
        run.set_parameters(dict(config))
    return run


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------


def make_3lc_tables(
    train_dataset: Dataset,
    val_dataset: Dataset,
    noisy_labels: np.ndarray,
    train_transform: Callable | None = None,
    val_transform: Callable | None = None,
    train_table_name: str = "train",
    val_table_name: str = "val",
    train_dataset_name: str = "cifar-10-train-noisy",
    val_dataset_name: str = "cifar-10-val",
    class_names: Iterable[str] = CIFAR10_CLASSES,
):
    """Build 3LC Tables for the (noisy) train and the clean val datasets.

    The training Table wraps a `NoisyLabelDataset` so that the corrupted
    labels are the ones recorded inside the Table. This is what makes the
    "given vs. true vs. predicted" comparison surface naturally in the
    Dashboard.

    Returns:
        (tlc_train_dataset, tlc_val_dataset) - both are `tlc.Table` instances
        with their latest committed revision applied.
    """
    tlc = _require_tlc()

    # Wrap the underlying CIFAR-10 train set with our noisy labels so the
    # Table records the *given* (potentially incorrect) labels.
    noisy_train_dataset = NoisyLabelDataset(train_dataset, noisy_labels)

    structure = (
        tlc.PILImage("image"),
        tlc.CategoricalLabel("label", classes=list(class_names)),
    )

    def _identity(sample):
        return sample

    train_fn = (lambda s: (train_transform(s[0]), s[1])) if train_transform else _identity
    val_fn = (lambda s: (val_transform(s[0]), s[1])) if val_transform else _identity

    tlc_train = (
        tlc.Table.from_torch_dataset(
            dataset=noisy_train_dataset,
            dataset_name=train_dataset_name,
            table_name=train_table_name,
            description="CIFAR-10 training set with injected label noise.",
            structure=structure,
            if_exists="overwrite",
        )
        .map(train_fn)
        .map_collect_metrics(val_fn)
    )

    tlc_val = tlc.Table.from_torch_dataset(
        dataset=val_dataset,
        dataset_name=val_dataset_name,
        table_name=val_table_name,
        description="CIFAR-10 held-out validation set (clean labels).",
        structure=structure,
        if_exists="overwrite",
    ).map(val_fn)

    # Pull the latest revision so any Dashboard-side edits are honored.
    tlc_train = tlc_train.latest()
    tlc_val = tlc_val.latest()

    return tlc_train, tlc_val


# ---------------------------------------------------------------------------
# Metrics collectors
# ---------------------------------------------------------------------------


def build_noise_metrics_collector(
    confidence_threshold: float = 0.5,
    class_names: Iterable[str] = CIFAR10_CLASSES,
    device: str | torch.device | None = None,
):
    """A FunctionalMetricsCollector that records per-sample noise signals.

    For every sample in a metrics-collection pass we emit:
      - confidence:        max softmax probability
      - predicted:         argmax of logits (categorical column)
      - loss:              unreduced cross-entropy loss
      - is_predicted_noisy: 1 if predicted != given label AND
                            confidence > threshold, else 0.

    This is the same algorithm as `src/noise_detection.py:detect_noisy_labels`,
    re-expressed as a 3LC collector so the flag becomes a queryable column on
    the Run's metrics table.
    """
    tlc = _require_tlc()
    names = list(class_names)

    def metrics_fn(batch, predictor_output) -> dict[str, np.ndarray]:
        # `batch` is a (image_tensor, label_tensor) tuple from the DataLoader.
        labels = batch[1]
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
        logits = predictor_output.forward
        labels = labels.to(logits.device)

        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        confidence = probs.gather(1, pred.unsqueeze(1)).squeeze(1)
        loss = F.cross_entropy(logits, labels, reduction="none")

        is_noisy = ((pred != labels) & (confidence > confidence_threshold)).int()

        return {
            "confidence": confidence.detach().cpu().numpy(),
            "predicted": pred.detach().cpu().numpy(),
            "loss": loss.detach().cpu().numpy(),
            "is_predicted_noisy": is_noisy.detach().cpu().numpy(),
        }

    schemas = {
        "loss": tlc.Schema(
            description="Per-sample cross entropy loss",
            value=tlc.Float32Value(),
        ),
        "predicted": tlc.CategoricalLabelSchema(
            display_name="predicted label",
            classes=names,
        ),
        "is_predicted_noisy": tlc.Schema(
            description=(
                f"1 when predicted != given AND confidence > "
                f"{confidence_threshold:.2f}. Used to filter the metrics "
                "table for noisy-candidate samples."
            ),
            value=tlc.Float32Value(),
        ),
    }

    return tlc.FunctionalMetricsCollector(
        collection_fn=metrics_fn,
        column_schemas=schemas,
    )


def build_embeddings_collector(model: SimpleCNN):
    """An EmbeddingsMetricsCollector hooked into SimpleCNN's penultimate FC layer.

    SimpleCNN's classifier is a `nn.Sequential` ending in
    `nn.Linear(128, num_classes)` (see src/model.py). We grab the 128-dim FC
    *before* the final logits layer, which is the natural representation
    space for clustering clean vs. noisy samples in PaCMAP.
    """
    tlc = _require_tlc()

    target_idx = _find_penultimate_linear_index(model)
    return tlc.EmbeddingsMetricsCollector(layers=[target_idx])


def _find_penultimate_linear_index(model: nn.Module) -> int:
    """Return the named_modules index of the second-to-last nn.Linear layer."""
    modules = list(enumerate(model.named_modules()))
    linear_indices = [i for i, (_, m) in modules if isinstance(m, nn.Linear)]
    if len(linear_indices) < 2:
        # Fallback: just use the last module that exists (matches the tutorial
        # pattern). The collector still works, just on a less ideal layer.
        return modules[-1][0]
    return linear_indices[-2]


# ---------------------------------------------------------------------------
# Per-epoch driver
# ---------------------------------------------------------------------------


def collect_3lc_metrics(
    model: SimpleCNN,
    tlc_train,
    tlc_val,
    classification_collector,
    embeddings_collector,
    epoch: int,
    learning_rate: float,
    num_workers: int = 0,
    batch_size: int = 512,
) -> None:
    """Run a metrics-collection pass on both Tables for the current epoch.

    Equivalent to the official tutorial's per-epoch `tlc.collect_metrics(...)`
    block. We use a larger batch size for metrics collection than training
    since no backprop happens.
    """
    tlc = _require_tlc()

    target_idx = _find_penultimate_linear_index(model)
    predictor = tlc.Predictor(model, layers=[target_idx])

    lr_schema = tlc.Schema(
        display_name="LR",
        description="Learning rate at the time of collection",
        value=tlc.Float32Value(),
        default_visible=False,
    )

    common = dict(
        metrics_collectors=[classification_collector, embeddings_collector],
        predictor=predictor,
        constants={"epoch": epoch, "learning_rate": float(learning_rate)},
        constants_schemas={"learning_rate": lr_schema},
        dataloader_args={"num_workers": num_workers, "batch_size": batch_size},
    )

    tlc.collect_metrics(tlc_train, split="train", **common)
    tlc.collect_metrics(tlc_val, split="val", **common)


# ---------------------------------------------------------------------------
# Flagging + correction
# ---------------------------------------------------------------------------


def flag_from_metrics_table(
    metrics_rows: list[dict[str, Any]] | np.ndarray,
    confidence_threshold: float = 0.5,
    min_correction_confidence: float = 0.5,
) -> dict[str, Any]:
    """Read a 3LC metrics table (or in-memory equivalent) and return flags.

    `metrics_rows` is expected to be an iterable of row dicts with at least
    the columns `confidence`, `predicted`, and `label` (the given label).
    This function is intentionally simple-to-mock: in production you'd pass
    in `list(metrics_table.table_rows)`; in tests you can pass a hand-built
    list of dicts.

    Returns:
        dict with:
            flagged_indices: list[int]
            corrections: dict[int, int] (only includes flags whose predicted
                confidence is at least min_correction_confidence)
            confidences: np.ndarray (per-row confidence)
            pred_classes: np.ndarray (per-row predicted class)
    """
    n = len(metrics_rows)
    confidences = np.zeros(n, dtype=np.float32)
    pred_classes = np.zeros(n, dtype=np.int64)
    given_labels = np.zeros(n, dtype=np.int64)

    for i, row in enumerate(metrics_rows):
        confidences[i] = float(row.get("confidence", 0.0))
        pred_classes[i] = int(row.get("predicted", 0))
        given_labels[i] = int(row.get("label", row.get("given_label", 0)))

    mismatch = pred_classes != given_labels
    confident = confidences > confidence_threshold
    flagged_mask = mismatch & confident
    flagged_indices = np.where(flagged_mask)[0].tolist()

    corrections: dict[int, int] = {}
    for idx in flagged_indices:
        if confidences[idx] < min_correction_confidence:
            continue
        corrections[int(idx)] = int(pred_classes[idx])

    return {
        "flagged_indices": flagged_indices,
        "corrections": corrections,
        "confidences": confidences,
        "pred_classes": pred_classes,
    }


def commit_corrected_table(
    base_dataset: Dataset,
    original_noisy_labels: np.ndarray,
    corrections: dict[int, int],
    train_transform: Callable | None = None,
    val_transform: Callable | None = None,
    table_name: str = "train-corrected",
    dataset_name: str = "cifar-10-train-corrected",
    class_names: Iterable[str] = CIFAR10_CLASSES,
):
    """Materialize a new 3LC Table whose labels have the corrections applied.

    Preferred production workflow (not used here because it requires the
    Dashboard): make virtual edits to the original Table via the Dashboard
    UI, then re-read it with `.latest()`. We instead build a fresh Table
    from a `NoisyLabelDataset` carrying the corrected labels, which uses
    the exact same `tlc.Table.from_torch_dataset` primitive shown in the
    official 3LC tutorial. This guarantees compatibility across minor
    versions of the 3LC SDK.
    """
    tlc = _require_tlc()

    corrected_labels = original_noisy_labels.copy()
    for idx, new_label in corrections.items():
        corrected_labels[idx] = new_label
    corrected_dataset = NoisyLabelDataset(base_dataset, corrected_labels)

    structure = (
        tlc.PILImage("image"),
        tlc.CategoricalLabel("label", classes=list(class_names)),
    )

    def _identity(s):
        return s

    train_fn = (lambda s: (train_transform(s[0]), s[1])) if train_transform else _identity
    val_fn = (lambda s: (val_transform(s[0]), s[1])) if val_transform else _identity

    table = (
        tlc.Table.from_torch_dataset(
            dataset=corrected_dataset,
            dataset_name=dataset_name,
            table_name=table_name,
            description=(
                f"CIFAR-10 training set with {len(corrections):,} label "
                "corrections applied based on Phase 1 confidence metrics."
            ),
            structure=structure,
            if_exists="overwrite",
        )
        .map(train_fn)
        .map_collect_metrics(val_fn)
    )

    return table.latest(), corrected_labels


# ---------------------------------------------------------------------------
# Convenience: pull metrics out of a Table for offline analysis
# ---------------------------------------------------------------------------


def metrics_table_to_rows(metrics_table) -> list[dict[str, Any]]:
    """Convert a 3LC metrics table to a list of row dicts.

    This is a small adapter so `flag_from_metrics_table` stays decoupled
    from any specific 3LC version's iteration API. If your installed `tlc`
    version exposes `table_rows`, we use it; otherwise we fall back to
    iterating the table directly.
    """
    _require_tlc()
    if hasattr(metrics_table, "table_rows"):
        return list(metrics_table.table_rows)
    return [dict(row) for row in metrics_table]
