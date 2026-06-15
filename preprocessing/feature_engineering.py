"""
Feature Engineering for VANET Anomaly Detection.

Computes derived features from raw VeReMi Extension dataset:
- Distance between consecutive messages (per sender)
- Speed magnitude
- Acceleration magnitude
- Time difference between consecutive messages
- Heading angle change

Reference: Section 4.1 of the paper.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, drop_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: Raw VeReMi Extension dataframe.
        drop_columns: Columns to drop before engineering (z-components, metadata).

    Returns:
        DataFrame with engineered features appended.
    """
    df = df.copy()

    # --- Step 1: Drop irrelevant columns ---
    if drop_columns:
        existing = [c for c in drop_columns if c in df.columns]
        df.drop(columns=existing, inplace=True)
        logger.info(f"Dropped {len(existing)} columns: {existing}")

    # --- Step 2: Sort by sender + time (critical for shift operations) ---
    df.sort_values(by=["sender", "sendTime"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Step 3: Distance between consecutive messages ---
    df["distance"] = _compute_distance(df)

    # --- Step 4: Time difference between consecutive messages ---
    df["timedif"] = df.groupby("sender")["sendTime"].diff().fillna(0).astype(float)

    # --- Step 5: Speed magnitude ---
    df["speed"] = np.sqrt(df["spdx"] ** 2 + df["spdy"] ** 2)

    # --- Step 6: Acceleration magnitude ---
    df["acceleration"] = np.sqrt(df["aclx"] ** 2 + df["acly"] ** 2)

    # --- Step 7: Heading angle change ---
    df["heading_angle_change"] = _compute_heading_angle_change(df)

    logger.info(
        f"Feature engineering complete. Shape: {df.shape}, "
        f"New features: distance, timedif, speed, acceleration, heading_angle_change"
    )
    return df


def _compute_distance(df: pd.DataFrame) -> pd.Series:
    """Euclidean distance between consecutive positions per sender."""
    prev_x = df.groupby("sender")["posx"].shift(1)
    prev_y = df.groupby("sender")["posy"].shift(1)
    dist = np.sqrt((df["posx"] - prev_x) ** 2 + (df["posy"] - prev_y) ** 2)
    return dist.fillna(0)


def _compute_heading_angle_change(df: pd.DataFrame) -> pd.Series:
    """
    Angle change (degrees) between consecutive heading vectors per sender.
    Uses dot product of normalized heading vectors.
    """
    # Normalize heading vectors
    norm = np.sqrt(df["hedx"] ** 2 + df["hedy"] ** 2).replace(0, np.nan)
    hx_norm = df["hedx"] / norm
    hy_norm = df["hedy"] / norm

    # Previous heading (shifted per sender)
    prev_hx = df.groupby("sender")[lambda d: hx_norm].shift(1) if False else None
    # groupby + shift on computed series needs care:
    sender_groups = df["sender"]
    prev_hx = hx_norm.groupby(sender_groups).shift(1)
    prev_hy = hy_norm.groupby(sender_groups).shift(1)

    # Dot product → angle
    dot = (hx_norm * prev_hx + hy_norm * prev_hy).clip(-1, 1)
    angle_change = np.degrees(np.arccos(dot))

    return angle_change.fillna(0)


def create_class_labels(df: pd.DataFrame, malfunction_classes: list[int]) -> pd.DataFrame:
    """
    Create binary and tri-class labels.

    Args:
        df: DataFrame with 'class' column (0-19 anomaly types).
        malfunction_classes: List of class IDs considered malfunctions.

    Returns:
        DataFrame with 'binaryClass' and 'triClass' columns added.
    """
    df = df.copy()

    # Binary: 0 = normal, 1 = anomaly
    df["binaryClass"] = (df["class"] != 0).astype(int)

    # Tri-class: 0 = normal, 1 = malfunction, 2 = attack
    df["triClass"] = df["class"].map(
        lambda x: 0 if x == 0 else (1 if x in malfunction_classes else 2)
    )
    df["triClass"] = df["triClass"].astype(int)

    logger.info(
        f"Labels created. Distribution — "
        f"Normal: {(df['triClass']==0).sum()}, "
        f"Malfunction: {(df['triClass']==1).sum()}, "
        f"Attack: {(df['triClass']==2).sum()}"
    )
    return df
