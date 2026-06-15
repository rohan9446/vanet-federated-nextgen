"""
Evaluation Metrics for VANET Anomaly Detection.

Computes AUC, Precision, Recall, F1-Score and generates reports.
Reference: Section 7.1 of the paper.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    average: str = "weighted",
    num_classes: int = 3,
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (n_samples, n_classes) for AUC.
        average: Averaging strategy for multi-class metrics.
        num_classes: Number of classes.

    Returns:
        Dict with metric name -> value.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # AUC (requires probability estimates)
    if y_prob is not None:
        try:
            if num_classes == 2:
                metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics["auc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average=average
                )
        except ValueError as e:
            logger.warning(f"AUC computation failed: {e}")
            metrics["auc"] = None
    else:
        metrics["auc"] = None

    return metrics


def print_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target_names: list[str] | None = None,
) -> str:
    """Print and return sklearn classification report."""
    if target_names is None:
        target_names = ["Normal", "Malfunction", "Attack"]

    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    logger.info(f"\n{'='*60}\n{model_name} — Classification Report\n{'='*60}\n{report}")
    logger.info(f"Confusion Matrix:\n{cm}")
    return report


def save_results(
    results: dict,
    output_dir: str = "results",
    filename: str = "metrics.json",
) -> None:
    """Save metrics dict to JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename

    # Convert numpy types for JSON serialization
    clean = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean[k] = {kk: float(vv) if vv is not None else None for kk, vv in v.items()}
        else:
            clean[k] = float(v) if v is not None else None

    with open(filepath, "w") as f:
        json.dump(clean, f, indent=2)

    logger.info(f"Results saved to {filepath}")
