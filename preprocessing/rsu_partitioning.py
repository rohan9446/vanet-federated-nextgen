"""
RSU-Based Client Partitioning for Federated Learning.

Two strategies:
1. Grid-based: Divide space into grid cells, subdivide dense cells, merge sparse ones.
2. K-Means: Cluster vehicles by (posx, posy) to simulate RSU coverage areas.

Reference: Section 6.2 of the paper.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


# ============================================================
# K-Means Partitioning (paper's claimed approach)
# ============================================================

def partition_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 5,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Partition dataset into RSU regions using K-Means on (posx, posy).

    Args:
        df: DataFrame with 'posx' and 'posy' columns.
        n_clusters: Number of RSU clusters.
        random_state: Seed for reproducibility.

    Returns:
        Dict mapping RSU label -> DataFrame subset.
    """
    coords = df[["posx", "posy"]].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(coords)

    df = df.copy()
    df["RSU_area"] = [f"rsu_{l}" for l in labels]

    partitions = {rsu: group.copy() for rsu, group in df.groupby("RSU_area")}

    logger.info(f"K-Means partitioning: {n_clusters} clusters")
    for rsu, group in partitions.items():
        logger.info(f"  {rsu}: {len(group)} records, classes={sorted(group['triClass'].unique())}")

    return partitions


# ============================================================
# Grid-Based Partitioning (your original code, cleaned up)
# ============================================================

def partition_grid(
    df: pd.DataFrame,
    initial_grid_size: int = 1000,
    dense_threshold: int = 300_000,
    dense_grid_size: int = 500,
    super_dense_threshold: int = 500_000,
    super_dense_grid_size: int = 250,
    sparse_min_classes: int = 3,
    sparse_min_points: int = 50_000,
    sparse_min_points_final: int = 20_000,
) -> dict[str, pd.DataFrame]:
    """
    Grid-based RSU partitioning with adaptive subdivision and merging.

    Steps:
        1. Divide space into initial grid cells.
        2. Subdivide dense cells into smaller grids.
        3. Merge sparse cells into nearest neighbor.
        4. Further subdivide super-dense cells.
        5. Re-merge any newly sparse cells.

    Returns:
        Dict mapping RSU label -> DataFrame subset.
    """
    df = df.copy()

    # --- Step 1: Initial grid ---
    df["RSU_area"] = _assign_grid(df, initial_grid_size)
    _log_partition(df, "Initial grid")

    # --- Step 2: Subdivide dense RSUs ---
    df = _subdivide_dense(df, dense_threshold, dense_grid_size)
    _log_partition(df, "After dense subdivision")

    # --- Step 3: Merge sparse RSUs ---
    df = _merge_sparse(df, sparse_min_classes, sparse_min_points)
    _log_partition(df, "After sparse merge")

    # --- Step 4: Subdivide super-dense RSUs ---
    df = _subdivide_dense(df, super_dense_threshold, super_dense_grid_size)
    _log_partition(df, "After super-dense subdivision")

    # --- Step 5: Re-merge sparse RSUs ---
    df = _merge_sparse(df, sparse_min_classes, sparse_min_points_final)
    _log_partition(df, "Final")

    partitions = {rsu: group.copy() for rsu, group in df.groupby("RSU_area")}
    return partitions


def _assign_grid(df: pd.DataFrame, grid_size: int) -> pd.Series:
    """Assign grid cell labels based on position."""
    gx = (df["posx"] // grid_size).astype(int).astype(str)
    gy = (df["posy"] // grid_size).astype(int).astype(str)
    return gx + "_" + gy


def _subdivide_dense(df: pd.DataFrame, threshold: int, new_grid_size: int) -> pd.DataFrame:
    """Subdivide RSU areas that exceed threshold into finer grids."""
    counts = df.groupby("RSU_area").size()
    dense_rsus = counts[counts > threshold].index.tolist()

    if not dense_rsus:
        return df

    mask = df["RSU_area"].isin(dense_rsus)
    df.loc[mask, "RSU_area"] = _assign_grid(df.loc[mask], new_grid_size)
    logger.info(f"Subdivided {len(dense_rsus)} dense RSUs (threshold={threshold})")
    return df


def _merge_sparse(df: pd.DataFrame, min_classes: int, min_points: int) -> pd.DataFrame:
    """Merge sparse RSU areas into nearest non-sparse RSU by centroid distance."""
    rsu_stats = df.groupby("RSU_area").agg(
        count=("triClass", "size"),
        n_classes=("triClass", "nunique"),
        cx=("posx", "mean"),
        cy=("posy", "mean"),
    )

    sparse_mask = (rsu_stats["n_classes"] < min_classes) | (rsu_stats["count"] < min_points)
    sparse_rsus = rsu_stats[sparse_mask].index.tolist()
    valid_rsus = rsu_stats[~sparse_mask]

    if not sparse_rsus or valid_rsus.empty:
        return df

    for sparse_rsu in sparse_rsus:
        sc = rsu_stats.loc[sparse_rsu]
        # Find nearest valid RSU by centroid distance
        dists = np.sqrt(
            (valid_rsus["cx"] - sc["cx"]) ** 2 + (valid_rsus["cy"] - sc["cy"]) ** 2
        )
        nearest = dists.idxmin()
        df.loc[df["RSU_area"] == sparse_rsu, "RSU_area"] = nearest
        logger.info(
            f"Merged sparse RSU '{sparse_rsu}' ({int(sc['count'])} records) -> '{nearest}'"
        )

    return df


def _log_partition(df: pd.DataFrame, stage: str) -> None:
    """Log partition stats."""
    stats = df.groupby("RSU_area").size()
    logger.info(f"[{stage}] {len(stats)} RSU areas, sizes: {stats.describe().to_dict()}")
