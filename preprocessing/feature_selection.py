"""
Feature Selection for VANET Anomaly Detection.

Selects the final training features from engineered DataFrame.
Reference: Section 4.2 of the paper.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Default feature set used for training (after engineering)
TRAINING_FEATURES = [
    "posx", "posy",
    "spdx", "spdy",
    "hedx", "hedy",
    "distance", "timedif", "speed", "acceleration", "heading_angle_change",
]


def select_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target_col: str = "triClass",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select features and target from processed DataFrame.

    Args:
        df: Processed DataFrame with engineered features.
        features: List of feature column names. Defaults to TRAINING_FEATURES.
        target_col: Target column name.

    Returns:
        (X, y) tuple.
    """
    if features is None:
        features = TRAINING_FEATURES

    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.warning(f"Missing features (skipped): {missing}")

    X = df[available].copy()
    y = df[target_col].copy()

    logger.info(f"Selected {len(available)} features, {len(y)} samples")
    return X, y
