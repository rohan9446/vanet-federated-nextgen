"""
Class Balancing for VANET Anomaly Detection.

Applies resampling (SMOTE, SMOTETomek) per RSU partition to handle
class imbalance without cross-partition data leakage.

Reference: Section 4.4 of the paper.
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import logging

logger = logging.getLogger(__name__)

# Columns that should NOT be fed to SMOTE (non-feature columns)
_META_COLUMNS = [
    "sendTime", "sender", "class", "binaryClass", "triClass",
    "RSU_x", "RSU_y", "RSU_area", "Sub_RSU_x", "Sub_RSU_y",
]


def balance_partition(
    df: pd.DataFrame,
    target_col: str = "triClass",
    method: str = "smote_tomek",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Balance a single partition's class distribution.

    Args:
        df: Single RSU partition DataFrame.
        target_col: Target column name.
        method: "smote", "smote_tomek", or "none".
        random_state: Seed for reproducibility.

    Returns:
        Balanced DataFrame with features + target_col.
    """
    if method == "none":
        return df

    # Separate features from target and metadata
    meta_cols = [c for c in _META_COLUMNS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in meta_cols and c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    # Check minimum class sizes
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        logger.warning(
            f"Skipping balancing: smallest class has {class_counts.min()} samples"
        )
        result = X.copy()
        result[target_col] = y
        return result

    # Apply resampling
    if method == "smote":
        sampler = SMOTE(random_state=random_state)
    elif method == "smote_tomek":
        sampler = SMOTETomek(random_state=random_state)
    else:
        raise ValueError(f"Unknown balancing method: {method}")

    X_res, y_res = sampler.fit_resample(X, y)

    result = pd.DataFrame(X_res, columns=feature_cols)
    result[target_col] = y_res

    logger.info(
        f"Balanced: {len(df)} -> {len(result)} "
        f"(classes: {dict(y.value_counts())} -> {dict(y_res.value_counts())})"
    )
    return result


def balance_all_partitions(
    partitions: dict[str, pd.DataFrame],
    target_col: str = "triClass",
    method: str = "smote_tomek",
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Balance every RSU partition independently."""
    balanced = {}
    for rsu_id, df in partitions.items():
        logger.info(f"Balancing RSU '{rsu_id}' ({len(df)} records)...")
        try:
            balanced[rsu_id] = balance_partition(df, target_col, method, random_state)
        except Exception as e:
            logger.error(f"Failed to balance RSU '{rsu_id}': {e}. Using original data.")
            # Fallback: return features + target without balancing
            meta_cols = [c for c in _META_COLUMNS if c in df.columns]
            feature_cols = [c for c in df.columns if c not in meta_cols and c != target_col]
            result = df[feature_cols].copy()
            result[target_col] = df[target_col]
            balanced[rsu_id] = result
    return balanced
