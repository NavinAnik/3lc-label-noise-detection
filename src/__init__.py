"""Label noise detection pipeline for CIFAR-10 with optional 3LC integration."""

from src.data import (
    CIFAR10_CLASSES,
    NoisyLabelDataset,
    create_noisy_dataset,
    get_sample_images,
    inject_label_noise,
    load_cifar10,
)
from src.metrics import compute_classification_metrics, compute_confusion_matrix
from src.model import SimpleCNN, create_model
from src.noise_detection import (
    compute_detection_metrics,
    detect_noisy_labels,
    detect_noisy_labels_with_probs,
    get_corrections,
)
from src.train import evaluate_model, train_model
from src.visualizations import (
    plot_comparison_curves,
    plot_confidence_distribution,
    plot_confusion_matrix,
    plot_sample_grid,
    plot_training_curves,
)

# `tlc_integration` is imported lazily so that the rest of the package keeps
# working even when the optional `tlc` SDK is not installed.
try:  # pragma: no cover - exercised only when tlc is installed
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
except ImportError:
    # `tlc_integration` module itself imports fine; this branch only fires
    # if someone has aggressively stripped optional deps.
    pass

__all__ = [
    "CIFAR10_CLASSES",
    "NoisyLabelDataset",
    "SimpleCNN",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "compute_detection_metrics",
    "create_model",
    "create_noisy_dataset",
    "detect_noisy_labels",
    "detect_noisy_labels_with_probs",
    "evaluate_model",
    "get_corrections",
    "get_sample_images",
    "inject_label_noise",
    "load_cifar10",
    "plot_comparison_curves",
    "plot_confidence_distribution",
    "plot_confusion_matrix",
    "plot_sample_grid",
    "plot_training_curves",
    "train_model",
]
