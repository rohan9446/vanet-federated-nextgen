"""
Class Weighting for VANET Anomaly Detection.

Instead of SMOTE (synthetic oversampling), compute inverse-frequency class
weights and apply them via loss functions:
  - XGBoost: sample_weight parameter
  - PyTorch: CrossEntropyLoss(weight=...)

Benefits over SMOTE:
  - No synthetic data (more defensible to reviewers)
  - Preserves natural per-RSU data heterogeneity (critical for FedTrust)
  - Faster (no data inflation)
  - No risk of learning non-physical interpolated patterns
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """
    Compute inverse-frequency class weights.

    Args:
        y: Label array.
        num_classes: Number of classes.

    Returns:
        Array of shape (num_classes,) with weight per class.
        Larger weight = rarer class = model pays more attention.
    """
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(float)
    # Avoid division by zero for missing classes
    counts = np.maximum(counts, 1.0)
    # Inverse frequency, normalized so weights sum to num_classes
    weights = len(y) / (num_classes * counts)
    logger.debug(f"Class counts: {counts.astype(int).tolist()}, weights: {weights.round(3).tolist()}")
    return weights


def compute_sample_weights(y: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """
    Compute per-sample weights for XGBoost's sample_weight parameter.
    Each sample gets the weight of its class.

    Args:
        y: Label array.

    Returns:
        Array of shape (n_samples,) with weight per sample.
    """
    class_weights = compute_class_weights(y, num_classes)
    return class_weights[y.astype(int)]