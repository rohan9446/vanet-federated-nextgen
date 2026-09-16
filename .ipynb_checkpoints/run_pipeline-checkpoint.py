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
from datetime import datetime
from sklearn.model_selection import train_test_split

from preprocessing.feature_engineering import engineer_features, create_class_labels
from preprocessing.feature_selection import select_features, TRAINING_FEATURES
from preprocessing.rsu_partitioning import partition_kmeans, partition_grid
from preprocessing.balancing import balance_partition, balance_all_partitions
from evaluation.metrics import compute_metrics, print_report, save_results

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

    # Step 1: Split — same 70/10/20 as federated for fair comparison
    X, y = select_features(df, target_col="triClass")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg["data"]["random_state"]
    )
    X_train, _, y_train, _ = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=cfg["data"]["random_state"]
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Step 2: Train models (class weighting handled inside each model)
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
        train_loader, test_loader, scaler, input_dim, class_weights = _prepare_dl_data(
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
        model.train(train_loader, epochs=mcfg.get("epochs", 20),
                    lr=mcfg.get("learning_rate", 0.001), class_weights=class_weights)
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
    from models.federated.fed_trust import (
        train_fedtrust_dl, train_fedtrust_xgb, predict_fedtrust_xgb,
    )
    logger.info("Running federated models with early stopping enabled")

    # Step 1: Three-way split — Train (70%) / Val (10%) / Test (20%)
    X_all, y_all = select_features(df, target_col="triClass")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=cfg["data"]["random_state"],
    )
    # Split trainval into train + val (val = 10% of total = 12.5% of trainval)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=cfg["data"]["random_state"],
    )
    X_test_np = X_test.values
    y_test_np = y_test.values
    X_val_np = X_val.values
    y_val_np = y_val.values
    logger.info(f"Split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Step 2: Reconstruct training DF for partitioning (needs posx, posy)
    train_df = df.loc[X_train.index].copy()

    # Step 3: Partition training data into RSU regions
    rsu_cfg = cfg["rsu_partitioning"]
    if rsu_cfg["method"] == "kmeans":
        partitions = partition_kmeans(train_df, **rsu_cfg["kmeans"])
    else:
        partitions = partition_grid(train_df, **rsu_cfg["grid"])
    logger.info(f"Partitioned into {len(partitions)} RSU regions")

    # Step 4: Balance each partition independently
    bal_cfg = cfg["balancing"]
    balanced = balance_all_partitions(
        partitions, target_col="triClass",
        method=bal_cfg["method"], random_state=bal_cfg["random_state"],
    )

    # Step 5: Prepare (X, y) tuples per partition
    feature_cols = [c for c in TRAINING_FEATURES if c in list(balanced.values())[0].columns]
    partition_data = {}
    for rsu_id, bdf in balanced.items():
        X = bdf[feature_cols].values
        y = bdf["triClass"].values
        partition_data[rsu_id] = (X, y)

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

    # ---- Federated DL models (Cyclic + FedAvg) ----
    for model_type, model_key in [("lstm", "fed_lstm"), ("bilstm", "fed_bilstm"), ("cnn", "fed_cnn")]:
        if fed_model not in ("all", model_key):
            continue

        mcfg = fed_cfg.get(model_key, fed_cfg.get(f"fed_{model_type}", {}))
        common_kwargs = dict(
            model_type=model_type,
            partitions=partition_data,
            num_rounds=fed_cfg["num_rounds"],
            local_epochs=mcfg.get("local_epochs", 5),
            batch_size=mcfg.get("batch_size", 256),
            lr=mcfg.get("learning_rate", 0.001),
            patience=mcfg.get("patience", 3),
            model_kwargs={k: v for k, v in mcfg.items()
                          if k not in ("local_epochs", "batch_size", "learning_rate", "patience")},
        )

        # Cyclic
        name_cyclic = f"Fed{model_type.upper()}_Cyclic"
        logger.info(f"Training {name_cyclic}...")
        model, scaler, history = train_federated_cyclic(**common_kwargs)
        y_pred, y_prob = predict_federated_dl(model, scaler, X_test_np)
        metrics = compute_metrics(y_test_np, y_pred, y_prob)
        print_report(y_test_np, y_pred, name_cyclic)
        results[name_cyclic] = metrics

        # FedAvg
        name_fedavg = f"Fed{model_type.upper()}_FedAvg"
        logger.info(f"Training {name_fedavg}...")
        model, scaler, history = train_federated_fedavg(**common_kwargs)
        y_pred, y_prob = predict_federated_dl(model, scaler, X_test_np)
        metrics = compute_metrics(y_test_np, y_pred, y_prob)
        print_report(y_test_np, y_pred, name_fedavg)
        results[name_fedavg] = metrics

    # ---- FedTrust XGBoost ----
    if fed_model in ("all", "fed_xgb", "fed_trust"):
        logger.info("Training FedTrust XGBoost...")
        ft_models, ft_weights, ft_history = train_fedtrust_xgb(
            partition_data, X_val_np, y_val_np,
            num_rounds=3,
            xgb_params=fed_cfg.get("fed_xgb", {}),
        )
        y_pred, y_prob = predict_fedtrust_xgb(ft_models, ft_weights, X_test_np)
        metrics = compute_metrics(y_test_np, y_pred, y_prob)
        print_report(y_test_np, y_pred, "FedTrust_XGB")
        results["FedTrust_XGB"] = metrics

    # ---- FedTrust DL models ----
    for model_type, model_key in [("lstm", "fed_lstm"), ("bilstm", "fed_bilstm"), ("cnn", "fed_cnn")]:
        if fed_model not in ("all", model_key, "fed_trust"):
            continue

        name_trust = f"FedTrust_{model_type.upper()}"
        logger.info(f"Training {name_trust}...")
        mcfg = fed_cfg.get(model_key, fed_cfg.get(f"fed_{model_type}", {}))

        model, scaler, history = train_fedtrust_dl(
            model_type=model_type,
            partitions=partition_data,
            num_rounds=fed_cfg["num_rounds"],
            local_epochs=mcfg.get("local_epochs", 5),
            batch_size=mcfg.get("batch_size", 256),
            lr=mcfg.get("learning_rate", 0.001),
            patience=mcfg.get("patience", 3),
            global_val_data=(X_val_np, y_val_np),
            model_kwargs={k: v for k, v in mcfg.items()
                          if k not in ("local_epochs", "batch_size", "learning_rate", "patience")},
        )
        y_pred, y_prob = predict_federated_dl(model, scaler, X_test_np)
        metrics = compute_metrics(y_test_np, y_pred, y_prob)
        print_report(y_test_np, y_pred, name_trust)
        results[name_trust] = metrics

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