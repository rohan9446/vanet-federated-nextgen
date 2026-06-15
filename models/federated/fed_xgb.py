"""
Federated XGBoost (FedXGB) with Cyclic Training.

Each RSU client trains XGBoost locally and passes the model to the next
client in round-robin fashion. This avoids aggregation overhead and
ensures sequential knowledge transfer.

Reference: Section 6.3 of the paper.
"""

import json
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def train_fedxgb_cyclic(
    partitions: dict[str, tuple[np.ndarray, np.ndarray]],
    num_rounds: int = 10,
    xgb_params: dict | None = None,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """
    Federated XGBoost with cyclic (round-robin) training.

    Each round, every client trains the model on its local data,
    then passes the updated model to the next client.

    Args:
        partitions: Dict of RSU_id -> (X, y) tuples.
        num_rounds: Number of full cycles through all clients.
        xgb_params: XGBoost hyperparameters.
        random_state: Seed.

    Returns:
        Trained XGBClassifier (global model after all rounds).
    """
    if xgb_params is None:
        xgb_params = {}

    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "multi:softprob",
        "num_class": 3,
        "tree_method": "hist",
        "random_state": random_state,
        "eval_metric": "mlogloss",
    }
    params.update(xgb_params)

    client_ids = sorted(partitions.keys())
    n_clients = len(client_ids)
    logger.info(f"FedXGB Cyclic: {n_clients} clients, {num_rounds} rounds")

    # Initialize global model
    global_model = None

    for round_idx in range(num_rounds):
        for client_idx, client_id in enumerate(client_ids):
            X_local, y_local = partitions[client_id]

            # Create fresh classifier with same params
            local_model = xgb.XGBClassifier(**params)

            if global_model is not None:
                # Continue training from global model state
                # Save and reload to transfer tree structure
                tmp_path = f"/tmp/fedxgb_round{round_idx}_client{client_idx}.json"
                global_model.save_model(tmp_path)
                local_model.fit(
                    X_local, y_local,
                    xgb_model=tmp_path,
                    verbose=False,
                )
                Path(tmp_path).unlink(missing_ok=True)
            else:
                # First client, first round: train from scratch
                local_model.fit(X_local, y_local, verbose=False)

            global_model = local_model
            logger.debug(
                f"Round {round_idx+1}/{num_rounds}, "
                f"Client {client_id} trained ({len(X_local)} samples)"
            )

        logger.info(f"Round {round_idx+1}/{num_rounds} complete")

    logger.info("FedXGB Cyclic training finished.")
    return global_model


def train_fedxgb_bagging(
    partitions: dict[str, tuple[np.ndarray, np.ndarray]],
    xgb_params: dict | None = None,
    random_state: int = 42,
) -> list[xgb.XGBClassifier]:
    """
    Federated XGBoost with bagging (independent training + ensemble).

    Each client trains independently; predictions are aggregated via
    majority voting or probability averaging.

    Args:
        partitions: Dict of RSU_id -> (X, y) tuples.
        xgb_params: XGBoost hyperparameters.
        random_state: Seed.

    Returns:
        List of trained XGBClassifier models.
    """
    if xgb_params is None:
        xgb_params = {}

    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "multi:softprob",
        "num_class": 3,
        "tree_method": "hist",
        "random_state": random_state,
        "eval_metric": "mlogloss",
    }
    params.update(xgb_params)

    models = []
    for client_id, (X_local, y_local) in partitions.items():
        model = xgb.XGBClassifier(**params)
        model.fit(X_local, y_local, verbose=False)
        models.append(model)
        logger.info(f"Bagging: Client {client_id} trained ({len(X_local)} samples)")

    return models


def predict_bagging_ensemble(
    models: list[xgb.XGBClassifier],
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average probabilities from bagging ensemble."""
    all_probs = np.array([m.predict_proba(X_test) for m in models])
    avg_probs = all_probs.mean(axis=0)
    preds = avg_probs.argmax(axis=1)
    return preds, avg_probs
