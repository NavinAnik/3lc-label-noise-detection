"""
Label Noise Detection — SaaS-style Streamlit UI

A production-grade interface for detecting and correcting label noise in CIFAR-10.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (
    CIFAR10_CLASSES,
    create_noisy_dataset,
    get_sample_images,
    inject_label_noise,
    load_cifar10,
)
from src.metrics import compute_confusion_matrix
from src.model import create_model
from src.noise_detection import (
    compute_detection_metrics,
    detect_noisy_labels_with_probs,
    get_corrections,
)
from src.train import evaluate_model, train_model
from src.visualizations import (
    plot_comparison_curves,
    plot_confusion_matrix,
    plot_confidence_distribution,
    plot_sample_grid,
    plot_training_curves,
)


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inject_custom_css() -> None:
    """Inject custom CSS for SaaS-style UI."""
    st.markdown(
        """
        <style>
        /* Hero header */
        .hero {
            padding: 1.5rem 0 2rem 0;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid var(--color-border-default, #e0e0e0);
        }
        .hero h1 {
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .hero .tagline {
            color: var(--text-secondary, #6c757d);
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        .hero .desc {
            font-size: 0.9rem;
            line-height: 1.5;
            color: var(--text-secondary, #6c757d);
        }
        /* Card-like sections */
        .metric-card {
            background: var(--background-secondary, #f8f9fa);
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin: 0.5rem 0;
            border: 1px solid var(--color-border-default, #e0e0e0);
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin: 1.5rem 0 0.75rem 0;
            padding-bottom: 0.25rem;
        }
        /* Delta styling */
        [data-testid="stMetricDelta"] svg {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def cached_load_data(data_dir: str = "./data"):
    """Load CIFAR-10 (cached)."""
    return load_cifar10(data_dir=data_dir, batch_size=128)


class PipelineLogger:
    """Collects timestamped log messages and renders into a Streamlit container."""

    def __init__(self, container):
        self._logs: list[str] = []
        self._container = container

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._logs.append(f"[{timestamp}] {message}")
        self._container.code("\n".join(self._logs), language=None)


# Step weights for progress bar (cumulative)
_STEP_WEIGHTS = [
    0.05,   # 1. Data Loading
    0.10,   # 2. Noise Injection
    0.40,   # 3. Initial Training
    0.50,   # 4. Label Error Detection
    0.55,   # 5. Label Correction
    0.85,   # 6. Retraining
    1.00,   # 7. Final Evaluation
]


def run_pipeline_with_progress(
    noise_rate: float,
    epochs: int,
    lr: float,
    confidence_threshold: float,
    device: torch.device,
    progress_bar,
    status_updater,
    logger: PipelineLogger,
    epoch_placeholder,
) -> dict[str, Any]:
    """Execute the full pipeline with progress tracking and live logs."""
    t_start = time.time()

    # Step 1: Data Loading
    status_updater("Loading CIFAR-10...", "running")
    step_start = time.time()
    train_loader, val_loader, train_dataset, val_dataset = cached_load_data()
    true_labels = np.array(train_dataset.targets)
    n_train, n_val = len(train_dataset), len(val_dataset)
    logger.log(f"Loaded CIFAR-10: {n_train:,} train, {n_val:,} test samples")
    progress_bar(_STEP_WEIGHTS[0], "Step 1/7: Data loaded")
    logger.log(f"Data loading completed ({time.time() - step_start:.1f}s)")

    # Step 2: Noise Injection
    status_updater("Injecting label noise...", "running")
    step_start = time.time()
    noisy_labels, noise_mask = inject_label_noise(
        true_labels, noise_rate=noise_rate, num_classes=10
    )
    n_noisy = int(noise_mask.sum())
    logger.log(f"Injected noise: {n_noisy:,} corrupted labels ({noise_rate*100:.0f}%)")
    progress_bar(_STEP_WEIGHTS[1], "Step 2/7: Noise injected")
    logger.log(f"Noise injection completed ({time.time() - step_start:.1f}s)")

    # Step 3: Initial Training
    status_updater("Training model (Phase 1)...", "running")
    step_start = time.time()
    noisy_train_dataset = create_noisy_dataset(train_dataset, noisy_labels)
    noisy_train_loader = DataLoader(
        noisy_train_dataset, batch_size=128, shuffle=True
    )
    model = create_model(num_classes=10, device=device)

    def on_epoch_end_phase1(epoch: int, metrics: dict[str, float]) -> None:
        frac = (epoch + 1) / epochs
        prog = _STEP_WEIGHTS[2] + (_STEP_WEIGHTS[3] - _STEP_WEIGHTS[2]) * frac
        progress_bar(prog, f"Step 3/7: Training... Epoch {epoch+1}/{epochs}")
        msg = (
            f"Epoch {epoch+1}/{epochs} — "
            f"loss: {metrics['val_loss']:.4f}, acc: {metrics['val_acc']:.4f}, "
            f"P: {metrics['val_precision']:.4f}, R: {metrics['val_recall']:.4f}, F1: {metrics['val_f1']:.4f}"
        )
        logger.log(msg)
        epoch_placeholder.caption(msg)

    history = train_model(
        model, noisy_train_loader, val_loader,
        epochs=epochs, lr=lr, device=device,
        on_epoch_end=on_epoch_end_phase1,
    )
    progress_bar(_STEP_WEIGHTS[3], "Step 3/7: Initial training complete")
    eval_before = evaluate_model(model, val_loader, device=device)
    logger.log(f"Phase 1 training completed ({time.time() - step_start:.1f}s)")
    logger.log(f"Initial test accuracy: {eval_before['metrics']['accuracy']:.4f}")

    # Step 4: Label Error Detection
    status_updater("Detecting label errors...", "running")
    step_start = time.time()
    detection = detect_noisy_labels_with_probs(
        model, noisy_train_dataset, noisy_labels,
        confidence_threshold=confidence_threshold, device=device
    )
    pred_noisy = detection["is_predicted_noisy"]
    flagged = detection["flagged_indices"]
    n_flagged = len(flagged)
    det_metrics = compute_detection_metrics(pred_noisy, noise_mask)
    logger.log(f"Flagged {n_flagged:,} noisy samples out of {n_train:,}")
    logger.log(f"Detection precision: {det_metrics['precision']:.4f}, recall: {det_metrics['recall']:.4f}")
    progress_bar(_STEP_WEIGHTS[4], "Step 4/7: Label errors detected")
    logger.log(f"Label error detection completed ({time.time() - step_start:.1f}s)")

    # Step 5: Label Correction
    status_updater("Applying label corrections...", "running")
    step_start = time.time()
    corrections = get_corrections(
        detection["pred_classes"], flagged, all_probs=detection["all_probs"]
    )
    corrected_labels = noisy_labels.copy()
    for idx, new_label in corrections.items():
        corrected_labels[idx] = new_label
    logger.log(f"Corrected {len(corrections):,} / {n_flagged:,} flagged labels (low-confidence skipped)")
    progress_bar(_STEP_WEIGHTS[5], "Step 5/7: Labels corrected")
    logger.log(f"Label correction completed ({time.time() - step_start:.1f}s)")

    # Step 6: Retraining
    status_updater("Retraining with cleaned data...", "running")
    step_start = time.time()
    corrected_dataset = create_noisy_dataset(train_dataset, corrected_labels)
    corrected_train_loader = DataLoader(
        corrected_dataset, batch_size=128, shuffle=True
    )
    model2 = create_model(num_classes=10, device=device)

    def on_epoch_end_phase2(epoch: int, metrics: dict[str, float]) -> None:
        frac = (epoch + 1) / epochs
        prog = _STEP_WEIGHTS[5] + (_STEP_WEIGHTS[6] - _STEP_WEIGHTS[5]) * frac
        progress_bar(prog, f"Step 6/7: Retraining... Epoch {epoch+1}/{epochs}")
        msg = (
            f"Retrain Epoch {epoch+1}/{epochs} — "
            f"loss: {metrics['val_loss']:.4f}, acc: {metrics['val_acc']:.4f}, "
            f"P: {metrics['val_precision']:.4f}, R: {metrics['val_recall']:.4f}, F1: {metrics['val_f1']:.4f}"
        )
        logger.log(msg)
        epoch_placeholder.caption(msg)

    history2 = train_model(
        model2, corrected_train_loader, val_loader,
        epochs=epochs, lr=lr, device=device,
        on_epoch_end=on_epoch_end_phase2,
    )
    progress_bar(_STEP_WEIGHTS[6], "Step 6/7: Retraining complete")
    logger.log(f"Phase 2 training completed ({time.time() - step_start:.1f}s)")

    # Step 7: Final Evaluation
    status_updater("Running final evaluation...", "running")
    step_start = time.time()
    eval_after = evaluate_model(model2, val_loader, device=device)
    logger.log(f"Final test accuracy: {eval_after['metrics']['accuracy']:.4f}")
    progress_bar(_STEP_WEIGHTS[6], "Step 7/7: Evaluation complete")
    logger.log(f"Evaluation completed ({time.time() - step_start:.1f}s)")

    total_time = time.time() - t_start
    logger.log(f"Pipeline completed in {total_time:.1f}s total")

    return {
        "total_time": total_time,
        "metrics_before": eval_before["metrics"],
        "metrics_after": eval_after["metrics"],
        "detection_metrics": det_metrics,
        "history": history,
        "history2": history2,
        "detection": detection,
        "noise_mask": noise_mask,
        "noisy_labels": noisy_labels,
        "corrected_labels": corrected_labels,
        "true_labels": true_labels,
        "train_dataset": train_dataset,
        "eval_before": eval_before,
        "eval_after": eval_after,
    }


def main() -> None:
    st.set_page_config(
        page_title="Label Noise Detection",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    # Header
    st.markdown(
        """
        <div class="hero">
            <h1>🔬 Label Noise Detection</h1>
            <p class="tagline">Detect and correct noisy labels in CIFAR-10</p>
            <p class="desc">
                This pipeline injects synthetic label noise, trains a CNN, detects mislabeled samples
                using confidence-based heuristics, and retrains with corrected labels for improved accuracy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        st.markdown("---")
        noise_rate = st.slider(
            "Noise level",
            min_value=0.05,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Fraction of training labels to corrupt",
        )
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Flag samples below this confidence as noisy",
        )
        epochs = st.slider(
            "Epochs",
            min_value=2,
            max_value=20,
            value=5,
            help="Training epochs per phase (use 5 for quick demo)",
        )
        lr = st.number_input(
            "Learning rate",
            value=0.001,
            min_value=0.0001,
            max_value=0.1,
            step=0.0005,
            format="%.4f",
        )
        st.markdown("---")
        run_demo = st.button(
            "▶ Run pipeline",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("running", False),
        )
        if st.button("Reset", use_container_width=True):
            st.session_state.pipeline_result = None
            st.session_state.pop("pipeline_error", None)
            st.rerun()
        st.markdown("---")
        st.caption(f"Device: {get_device()}")

    # Overview section
    st.markdown('<p class="section-title">Overview</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Load CIFAR-10** — Download and prepare the dataset")
    with col2:
        st.markdown("**2. Inject noise** — Randomly corrupt a fraction of labels")
    with col3:
        st.markdown("**3. Train & detect** — Train on noisy data, flag low-confidence samples")
    st.markdown(
        "**4. Correct & retrain** — Apply corrections and retrain for better accuracy"
    )
    st.markdown("---")

    # Initialize session state for results and running flag
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None
    if "running" not in st.session_state:
        st.session_state.running = False

    if run_demo:
        st.session_state.running = True
        try:
            progress = st.progress(0.0, text="0.0% — Starting pipeline...")
        except TypeError:
            progress = st.progress(0.0)
        log_expander = st.expander("View Detailed Logs", expanded=True)
        log_placeholder = log_expander.empty()
        logger = PipelineLogger(log_placeholder)
        epoch_placeholder = st.empty()

        def update_progress(value: float, text: str | None = None) -> None:
            pct = f"{value * 100:.1f}%"
            display_text = f"{pct} — {text}" if text else pct
            try:
                progress.progress(value, text=display_text)
            except TypeError:
                progress.progress(value)  # Fallback for older Streamlit

        try:
            with st.status("Running pipeline...", expanded=True) as status:
                def status_updater(label: str, state: str = "running") -> None:
                    status.update(label=label, state=state)

                result = run_pipeline_with_progress(
                    noise_rate=noise_rate,
                    epochs=epochs,
                    lr=lr,
                    confidence_threshold=confidence_threshold,
                    device=get_device(),
                    progress_bar=update_progress,
                    status_updater=status_updater,
                    logger=logger,
                    epoch_placeholder=epoch_placeholder,
                )
                st.session_state.pipeline_result = result
                status.update(label="Pipeline complete!", state="complete")
            total_time = result.get("total_time")
            st.session_state.success_message = (
                f"Pipeline completed successfully in {total_time:.1f}s!"
                if total_time is not None
                else "Pipeline completed successfully!"
            )
        except Exception as e:
            logger.log(f"ERROR: {e}")
            st.session_state.pipeline_error = str(e)
            st.session_state.pipeline_result = None
        finally:
            st.session_state.running = False
            st.rerun()

    result = st.session_state.pipeline_result
    success_msg = st.session_state.pop("success_message", None)
    error_msg = st.session_state.pop("pipeline_error", None)
    if success_msg:
        st.success(success_msg)
    if error_msg:
        st.error(f"Pipeline failed: {error_msg}")
    if result is None:
        st.info("Configure parameters in the sidebar and click **Run pipeline** to start.")
        st.stop()

    # Metrics dashboard
    st.markdown('<p class="section-title">📊 Metrics Dashboard</p>', unsafe_allow_html=True)
    m_before = result["metrics_before"]
    m_after = result["metrics_after"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta_acc = m_after["accuracy"] - m_before["accuracy"]
        st.metric("Accuracy", f"{m_after['accuracy']:.3f}", f"{delta_acc:+.3f}")
    with col2:
        delta_p = m_after["precision"] - m_before["precision"]
        st.metric("Precision", f"{m_after['precision']:.3f}", f"{delta_p:+.3f}")
    with col3:
        delta_r = m_after["recall"] - m_before["recall"]
        st.metric("Recall", f"{m_after['recall']:.3f}", f"{delta_r:+.3f}")
    with col4:
        delta_f1 = m_after["f1"] - m_before["f1"]
        st.metric("F1 (macro)", f"{m_after['f1']:.3f}", f"{delta_f1:+.3f}")

    st.markdown("*Deltas: Before (noisy) → After (corrected)*")
    st.markdown("---")

    # Detection metrics
    dm = result["detection_metrics"]
    st.markdown("**Noise detection performance**")
    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        st.metric("Detection Precision", f"{dm['precision']:.3f}", None)
    with dcol2:
        st.metric("Detection Recall", f"{dm['recall']:.3f}", None)
    with dcol3:
        st.metric("Detection F1", f"{dm['f1']:.3f}", None)
    st.markdown("---")

    # Visualizations
    st.markdown('<p class="section-title">📈 Visualizations</p>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "Confusion matrix",
        "Confidence distribution",
        "Training curves",
        "Phase comparison",
    ])

    with tab1:
        c1, c2 = st.columns(2)
        cm_before = compute_confusion_matrix(
            result["eval_before"]["y_true"],
            result["eval_before"]["y_pred"],
            CIFAR10_CLASSES,
        )
        cm_after = compute_confusion_matrix(
            result["eval_after"]["y_true"],
            result["eval_after"]["y_pred"],
            CIFAR10_CLASSES,
        )
        with c1:
            fig_cm1 = plot_confusion_matrix(cm_before, CIFAR10_CLASSES)
            st.pyplot(fig_cm1)
            st.caption("Before (trained on noisy labels)")
        with c2:
            fig_cm2 = plot_confusion_matrix(cm_after, CIFAR10_CLASSES)
            st.pyplot(fig_cm2)
            st.caption("After (trained on corrected labels)")

    with tab2:
        fig_conf = plot_confidence_distribution(
            result["detection"]["confidences"],
            result["noise_mask"],
        )
        st.pyplot(fig_conf)

    with tab3:
        t1, t2 = st.columns(2)
        with t1:
            fig_h1 = plot_training_curves(result["history"])
            st.pyplot(fig_h1)
            st.caption("Phase 1: Noisy labels")
        with t2:
            fig_h2 = plot_training_curves(result["history2"])
            st.pyplot(fig_h2)
            st.caption("Phase 2: Corrected labels")

    with tab4:
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig_loss = plot_comparison_curves(
                result["history"], result["history2"],
                "val_loss", "Validation Loss",
            )
            st.pyplot(fig_loss)
        with r1c2:
            fig_f1 = plot_comparison_curves(
                result["history"], result["history2"],
                "val_f1", "F1 (macro)",
            )
            st.pyplot(fig_f1)
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            fig_prec = plot_comparison_curves(
                result["history"], result["history2"],
                "val_precision", "Precision (macro)",
            )
            st.pyplot(fig_prec)
        with r2c2:
            fig_rec = plot_comparison_curves(
                result["history"], result["history2"],
                "val_recall", "Recall (macro)",
            )
            st.pyplot(fig_rec)

    st.markdown("---")

    # Data inspection
    st.markdown('<p class="section-title">🔍 Data Inspection</p>', unsafe_allow_html=True)
    flagged = result["detection"]["flagged_indices"]
    noise_mask = result["noise_mask"]
    noisy_indices = np.where(noise_mask)[0]
    train_dataset = result["train_dataset"]
    true_labels = result["true_labels"]
    noisy_labels = result["noisy_labels"]
    corrected_labels = result["corrected_labels"]

    inspect_tab1, inspect_tab2 = st.tabs(["Incorrect labels (noisy)", "Corrected labels"])

    with inspect_tab1:
        n_show = min(10, len(noisy_indices))
        if n_show > 0:
            sample_idx = list(noisy_indices[:n_show])
            imgs = get_sample_images(train_dataset, sample_idx)
            true_l = [int(true_labels[i]) for i in sample_idx]
            given_l = [int(noisy_labels[i]) for i in sample_idx]
            fig_inc = plot_sample_grid(
                imgs, true_l, given_l, CIFAR10_CLASSES,
                title="True vs Given (noisy) labels",
                cols=5,
            )
            st.pyplot(fig_inc)
        else:
            st.info("No noisy labels in this run (noise rate may be 0).")

    with inspect_tab2:
        n_show2 = min(10, len(flagged))
        if n_show2 > 0:
            sample_idx2 = flagged[:n_show2]
            imgs2 = get_sample_images(train_dataset, sample_idx2)
            true_l2 = [int(true_labels[i]) for i in sample_idx2]
            given_l2 = [int(noisy_labels[i]) for i in sample_idx2]
            corr_l2 = [int(corrected_labels[i]) for i in sample_idx2]
            fig_corr = plot_sample_grid(
                imgs2, corr_l2, given_l2, CIFAR10_CLASSES,
                title="Given (noisy) vs Corrected labels",
                cols=5,
            )
            st.pyplot(fig_corr)
        else:
            st.info("No samples were flagged for correction.")


if __name__ == "__main__":
    main()
