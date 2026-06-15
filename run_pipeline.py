"""
VANET Anomaly Detection — Main Pipeline

Usage:
    python run_pipeline.py --config configs/config.yaml --mode all
    python run_pipeline.py --config configs/config.yaml --mode centralized
    python run_pipeline.py --config configs/config.yaml --mode federated
    python run_pipeline.py --config configs/config.yaml --mode federated --fed-model fed_lstm
"""

import argparse
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from preprocessing.feature_engineering import engineer_features, create_class_labels
from preprocessing.feature_selection import select_features, TRAINING_FEATURES
from preprocessing.rsu_partitioning import partition_kmeans, partition_grid
from preprocessing.balancing import balance_all_partitions
from evaluation.metrics import compute_metrics, print_report, save_results

from datetime import datetime

log_dir = Path("results")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_and_preprocess(cfg: dict) -> pd.DataFrame:
    """Load raw CSV and run full preprocessing pipeline."""
    logger.info(f"Loading data from {cfg['data']['raw_csv_path']}...")
    df = pd.read_csv(cfg["data"]["raw_csv_path"])
    logger.info(f"Raw shape: {df.shape}")

    # Feature engineering
    df = engineer_features(df, drop_columns=cfg["preprocessing"]["drop_columns"])

    # Create labels
    df = create_class_labels(df, malfunction_classes=cfg["preprocessing"]["malfunction_classes"])

    return df


def run_centralized(cfg: dict, df: pd.DataFrame, results: dict) -> dict:
    """Train and evaluate all centralized models."""
    from models.centralized.train_models import (
        train_random_forest, train_extra_trees, train_xgboost,
        LSTMModel, BiLSTMModel, CNNModel, _prepare_dl_data,
    )

    X, y = select_features(df, target_col="triClass")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["random_state"]
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ---- Tree-based models ----
    for name, train_fn in [
        ("RandomForest", train_random_forest),
        ("ExtraTrees", train_extra_trees),
        ("XGBoost", train_xgboost),
    ]:
        logger.info(f"Training {name}...")
        model_cfg = cfg["models"]["centralized"].get(name.lower().replace(" ", "_"), {})
        model = train_fn(X_train.values, y_train.values, **model_cfg)
        y_pred = model.predict(X_test.values)
        y_prob = model.predict_proba(X_test.values) if hasattr(model, "predict_proba") else None
        metrics = compute_metrics(y_test.values, y_pred, y_prob)
        print_report(y_test.values, y_pred, name)
        results[name] = metrics
        logger.info(f"{name}: {metrics}")

    # ---- Deep learning models ----
    dl_cfg = cfg["models"]["centralized"]
    for name, ModelClass, model_key in [
        ("LSTM", LSTMModel, "lstm"),
        ("BiLSTM", BiLSTMModel, "bilstm"),
        ("CNN", CNNModel, "cnn"),
    ]:
        logger.info(f"Training {name}...")
        mcfg = dl_cfg.get(model_key, {})
        batch_size = mcfg.get("batch_size", 256)
        train_loader, test_loader, scaler, input_dim = _prepare_dl_data(
            X_train.values, y_train, X_test.values, y_test, batch_size=batch_size,
        )
        model = ModelClass(
            input_dim=input_dim,
            units=mcfg.get("units", 128),
            dropout=mcfg.get("dropout", 0.3),
        ) if model_key != "cnn" else ModelClass(
            input_dim=input_dim,
            filters=mcfg.get("filters", [64, 128]),
            dropout=mcfg.get("dropout", 0.3),
        )
        model.train(train_loader, epochs=mcfg.get("epochs", 20), lr=mcfg.get("learning_rate", 0.001))
        y_pred, y_prob = model.predict(test_loader)
        metrics = compute_metrics(y_test.values, y_pred, y_prob)
        print_report(y_test.values, y_pred, name)
        results[name] = metrics
        logger.info(f"{name}: {metrics}")

    return results


def run_federated(cfg: dict, df: pd.DataFrame, results: dict, fed_model: str = "all") -> dict:
    """Train and evaluate federated models."""
    from models.federated.fed_xgb import (
        train_fedxgb_cyclic, train_fedxgb_bagging, predict_bagging_ensemble,
    )
    from models.federated.fed_dl import (
        train_federated_cyclic, train_federated_fedavg, predict_federated_dl,
    )

    # --- Partition data into RSU regions ---
    rsu_cfg = cfg["rsu_partitioning"]
    if rsu_cfg["method"] == "kmeans":
        partitions = partition_kmeans(df, **rsu_cfg["kmeans"])
    else:
        partitions = partition_grid(df, **rsu_cfg["grid"])

    logger.info(f"Partitioned into {len(partitions)} RSU regions")

    # --- Balance each partition ---
    bal_cfg = cfg["balancing"]
    balanced = balance_all_partitions(
        partitions, target_col="triClass",
        method=bal_cfg["method"], random_state=bal_cfg["random_state"],
    )

    # --- Prepare (X, y) tuples per partition ---
    feature_cols = [c for c in TRAINING_FEATURES if c in list(balanced.values())[0].columns]
    partition_data = {}
    for rsu_id, bdf in balanced.items():
        X = bdf[feature_cols].values
        y = bdf["triClass"].values
        partition_data[rsu_id] = (X, y)

    # --- Global test set (from original unbalanced data) ---
    X_all, y_all = select_features(df, target_col="triClass")
    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=cfg["data"]["test_size"], random_state=cfg["data"]["random_state"],
    )
    X_test_np = X_test.values
    y_test_np = y_test.values

    fed_cfg = cfg["federated"]

    # ---- FedXGB (Cyclic) ----
    if fed_model in ("all", "fed_xgb"):
        logger.info("Training FedXGB (Cyclic)...")
        model = train_fedxgb_cyclic(
            partition_data,
            num_rounds=fed_cfg["num_rounds"],
            xgb_params=fed_cfg.get("fed_xgb", {}),
        )
        y_pred = model.predict(X_test_np)
        y_prob = model.predict_proba(X_test_np)
        metrics = compute_metrics(y_test_np, y_pred, y_prob)
        print_report(y_test_np, y_pred, "FedXGB_Cyclic")
        results["FedXGB_Cyclic"] = metrics

        # ---- FedXGB (Bagging) ----
        logger.info("Training FedXGB (Bagging)...")
        models = train_fedxgb_bagging(partition_data, xgb_params=fed_cfg.get("fed_xgb", {}))
        y_pred, y_prob = predict_bagging_ensemble(models, X_test_np)
        metrics = compute_metrics(y_test_np, y_pred, y_prob)
        print_report(y_test_np, y_pred, "FedXGB_Bagging")
        results["FedXGB_Bagging"] = metrics

    # ---- Federated DL models ----
    for model_type, model_key in [("lstm", "fed_lstm"), ("bilstm", "fed_bilstm"), ("cnn", "fed_cnn")]:
        if fed_model not in ("all", model_key):
            continue

        name_cyclic = f"Fed{model_type.upper()}_Cyclic"
        logger.info(f"Training {name_cyclic}...")
        mcfg = fed_cfg.get(model_key, fed_cfg.get(f"fed_{model_type}", {}))

        model, scaler, history = train_federated_cyclic(
            model_type=model_type,
            partitions=partition_data,
            num_rounds=fed_cfg["num_rounds"],
            local_epochs=mcfg.get("local_epochs", 3),
            batch_size=mcfg.get("batch_size", 256),
            lr=mcfg.get("learning_rate", 0.001),
            model_kwargs={k: v for k, v in mcfg.items()
                          if k not in ("local_epochs", "batch_size", "learning_rate")},
        )
        y_pred, y_prob = predict_federated_dl(model, scaler, X_test_np)
        metrics = compute_metrics(y_test_np, y_pred, y_prob)
        print_report(y_test_np, y_pred, name_cyclic)
        results[name_cyclic] = metrics

    return results


def main():
    parser = argparse.ArgumentParser(description="VANET Anomaly Detection Pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--mode", choices=["all", "centralized", "federated"], default="all")
    parser.add_argument("--fed-model", default="all",
                        help="Which federated model: all, fed_xgb, fed_lstm, fed_bilstm, fed_cnn")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_and_preprocess(cfg)
    results = {}

    if args.mode in ("all", "centralized"):
        results = run_centralized(cfg, df, results)

    if args.mode in ("all", "federated"):
        results = run_federated(cfg, df, results, fed_model=args.fed_model)

    # Save all results
    save_results(results, output_dir=cfg["evaluation"]["output_dir"])

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    for model_name, metrics in results.items():
        auc = metrics.get("auc", "N/A")
        f1 = metrics.get("f1", "N/A")
        auc_str = f"{auc:.4f}" if isinstance(auc, float) else auc
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else f1
        logger.info(f"  {model_name:25s} | AUC: {auc_str:>8s} | F1: {f1_str:>8s}")


if __name__ == "__main__":
    main()
