"""
Automated Experiment Sweep for VANET Anomaly Detection.

Runs all combinations of:
  - n_clusters: [4, 8, 16, 32, 64]
  - num_rounds: [5, 10, 20, 30]
  - models: FedXGB, FedLSTM, FedBiLSTM, FedCNN (Cyclic + FedAvg + FedTrust)

Outputs:
  - results/experiments/exp_<id>/metrics.json   (per experiment)
  - results/experiments/exp_<id>/run.log         (per experiment)
  - results/experiments/summary.csv              (all results in one table)

Usage:
    python run_experiments.py
    python run_experiments.py --clusters 4 8 16 --rounds 10 20
    python run_experiments.py --mode federated_only
    python run_experiments.py --mode all
"""

import argparse
import itertools
import json
import logging
import os
import sys
import time
import traceback
import yaml
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split

from preprocessing.feature_engineering import engineer_features, create_class_labels
from preprocessing.feature_selection import select_features, TRAINING_FEATURES
from preprocessing.rsu_partitioning import partition_kmeans, partition_grid
from preprocessing.balancing import balance_partition, balance_all_partitions
from evaluation.metrics import compute_metrics, print_report, save_results


# ============================================================
# Experiment Grid
# ============================================================

DEFAULT_CLUSTERS = [4, 8, 16, 32, 64]
DEFAULT_ROUNDS = [5, 10, 20, 30]
DEFAULT_LOCAL_EPOCHS = [3, 5]


def load_base_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def setup_experiment_logger(exp_dir: Path) -> logging.Logger:
    """Create a logger that writes to both console and experiment log file."""
    logger = logging.getLogger(f"exp_{exp_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(exp_dir / "run.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    return logger


# ============================================================
# Single Experiment Runner
# ============================================================

def run_single_experiment(
    cfg: dict,
    df: pd.DataFrame,
    n_clusters: int,
    num_rounds: int,
    local_epochs: int,
    exp_dir: Path,
    run_centralized: bool = False,
) -> dict:
    """Run one experiment with specific hyperparameters."""

    from models.centralized.train_models import (
        train_random_forest, train_extra_trees, train_xgboost,
        LSTMModel, BiLSTMModel, CNNModel, _prepare_dl_data,
    )
    from models.federated.fed_xgb import (
        train_fedxgb_cyclic, train_fedxgb_bagging, predict_bagging_ensemble,
    )
    from models.federated.fed_dl import (
        train_federated_cyclic, train_federated_fedavg, predict_federated_dl,
    )
    from models.federated.fed_trust import (
        train_fedtrust_dl, train_fedtrust_xgb, predict_fedtrust_xgb,
    )

    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_experiment_logger(exp_dir)

    logger.info(f"=" * 70)
    logger.info(f"EXPERIMENT: clusters={n_clusters}, rounds={num_rounds}, local_epochs={local_epochs}")
    logger.info(f"Output: {exp_dir}")
    logger.info(f"=" * 70)

    results = {
        "config": {
            "n_clusters": n_clusters,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
        }
    }

    # --- Three-way split: Train (70%) / Val (10%) / Test (20%) ---
    X_all, y_all = select_features(df, target_col="triClass")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=cfg["data"]["random_state"],
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=cfg["data"]["random_state"],
    )
    X_test_np = X_test.values
    y_test_np = y_test.values
    X_val_np = X_val.values
    y_val_np = y_val.values
    logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # --- Centralized (only once, skip if not requested) ---
    if run_centralized:
        logger.info("--- CENTRALIZED MODELS ---")

        for name, train_fn in [
            ("RandomForest", train_random_forest),
            ("ExtraTrees", train_extra_trees),
            ("XGBoost", train_xgboost),
        ]:
            try:
                logger.info(f"Training {name}...")
                model = train_fn(X_train.values, y_train.values)
                y_pred = model.predict(X_test_np)
                y_prob = model.predict_proba(X_test_np)
                metrics = compute_metrics(y_test_np, y_pred, y_prob)
                results[name] = metrics
                logger.info(f"{name}: AUC={metrics.get('auc', 'N/A')}, F1={metrics['f1']:.4f}")
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                results[name] = {"error": str(e)}

        dl_cfg = cfg["models"]["centralized"]
        for name, ModelClass, model_key in [
            ("LSTM", LSTMModel, "lstm"),
            ("BiLSTM", BiLSTMModel, "bilstm"),
            ("CNN", CNNModel, "cnn"),
        ]:
            try:
                logger.info(f"Training {name}...")
                mcfg = dl_cfg.get(model_key, {})
                train_loader, test_loader, scaler, input_dim, class_weights = _prepare_dl_data(
                    X_train.values, y_train, X_test_np, y_test, batch_size=mcfg.get("batch_size", 256),
                )
                if model_key != "cnn":
                    model = ModelClass(input_dim=input_dim, units=mcfg.get("units", 128), dropout=mcfg.get("dropout", 0.3))
                else:
                    model = ModelClass(input_dim=input_dim, filters=mcfg.get("filters", [64, 128]), dropout=mcfg.get("dropout", 0.3))
                model.train(train_loader, epochs=mcfg.get("epochs", 50),
                            lr=mcfg.get("learning_rate", 0.001), class_weights=class_weights)
                y_pred, y_prob = model.predict(test_loader)
                metrics = compute_metrics(y_test_np, y_pred, y_prob)
                results[name] = metrics
                logger.info(f"{name}: AUC={metrics.get('auc', 'N/A')}, F1={metrics['f1']:.4f}")
            except Exception as e:
                logger.error(f"{name} failed: {e}")
                results[name] = {"error": str(e)}

    # --- Partition training data ---
    logger.info(f"--- FEDERATED MODELS (clusters={n_clusters}, rounds={num_rounds}) ---")
    train_df = df.loc[X_train.index].copy()
    partitions = partition_kmeans(train_df, n_clusters=n_clusters, random_state=cfg["rsu_partitioning"]["kmeans"]["random_state"])

    bal_cfg = cfg["balancing"]
    balanced = balance_all_partitions(
        partitions, target_col="triClass",
        method=bal_cfg["method"], random_state=bal_cfg["random_state"],
    )

    feature_cols = [c for c in TRAINING_FEATURES if c in list(balanced.values())[0].columns]
    partition_data = {}
    for rsu_id, bdf in balanced.items():
        partition_data[rsu_id] = (bdf[feature_cols].values, bdf["triClass"].values)

    fed_cfg = cfg["federated"]

    # --- FedXGB Cyclic ---
    # --- FedXGB Bagging ---
    try:
        logger.info("Training FedXGB_Bagging...")
        models = train_fedxgb_bagging(partition_data, xgb_params=fed_cfg.get("fed_xgb", {}))
        y_pred, y_prob = predict_bagging_ensemble(models, X_test_np)
        metrics = compute_metrics(y_test_np, y_pred, y_prob)
        results["FedXGB_Bagging"] = metrics
        logger.info(f"FedXGB_Bagging: AUC={metrics.get('auc', 'N/A')}, F1={metrics['f1']:.4f}")
    except Exception as e:
        logger.error(f"FedXGB_Bagging failed: {e}")
        results["FedXGB_Bagging"] = {"error": str(e)}

    # --- FedTrust XGB ---
    try:
        logger.info("Training FedTrust_XGB...")
        ft_models, ft_weights, _ = train_fedtrust_xgb(
            partition_data, X_val_np, y_val_np, num_rounds=3, xgb_params=fed_cfg.get("fed_xgb", {}),
        )
        y_pred, y_prob = predict_fedtrust_xgb(ft_models, ft_weights, X_test_np)
        metrics = compute_metrics(y_test_np, y_pred, y_prob)
        results["FedTrust_XGB"] = metrics
        logger.info(f"FedTrust_XGB: AUC={metrics.get('auc', 'N/A')}, F1={metrics['f1']:.4f}")
    except Exception as e:
        logger.error(f"FedTrust_XGB failed: {e}")
        results["FedTrust_XGB"] = {"error": str(e)}

    # --- Federated DL: FedAvg + FedTrust ---
    for model_type, model_key in [("lstm", "fed_lstm"), ("bilstm", "fed_bilstm"), ("cnn", "fed_cnn")]:
        mcfg = fed_cfg.get(model_key, fed_cfg.get(f"fed_{model_type}", {}))
        common = dict(
            model_type=model_type,
            partitions=partition_data,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            batch_size=mcfg.get("batch_size", 256),
            lr=mcfg.get("learning_rate", 0.001),
            patience=mcfg.get("patience", 3),
            model_kwargs={k: v for k, v in mcfg.items()
                          if k not in ("local_epochs", "batch_size", "learning_rate", "patience")},
        )

        for strategy, train_fn in [
            ("FedAvg", train_federated_fedavg),
            ("FedTrust", train_fedtrust_dl),
        ]:
            name = f"Fed{model_type.upper()}_{strategy}"
            try:
                logger.info(f"Training {name}...")
                t0 = time.time()
                kwargs = dict(common)
                if strategy == "FedTrust":
                    kwargs["global_val_data"] = (X_val_np, y_val_np)
                model, scaler, history = train_fn(**kwargs)
                train_time = time.time() - t0
                y_pred, y_prob = predict_federated_dl(model, scaler, X_test_np)
                metrics = compute_metrics(y_test_np, y_pred, y_prob)
                metrics["train_time_sec"] = round(train_time, 2)
                results[name] = metrics
                logger.info(f"{name}: AUC={metrics.get('auc', 'N/A')}, F1={metrics['f1']:.4f}, Time={train_time:.1f}s")
            except Exception as e:
                logger.error(f"{name} failed: {e}\n{traceback.format_exc()}")
                results[name] = {"error": str(e)}

    # Save experiment results
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Experiment complete. Results saved to {exp_dir / 'metrics.json'}")
    return results


# ============================================================
# Sweep Runner
# ============================================================

def run_sweep(
    clusters_list: list[int],
    rounds_list: list[int],
    local_epochs_list: list[int],
    run_centralized_once: bool = True,
    config_path: str = "configs/config.yaml",
):
    """Run full experiment sweep."""
    base_cfg = load_base_config(config_path)
    output_root = Path("results/experiments")
    output_root.mkdir(parents=True, exist_ok=True)

    # --- Load and preprocess data once ---
    print(f"Loading and preprocessing data...")
    df = pd.read_csv(base_cfg["data"]["raw_csv_path"])
    df = engineer_features(df, drop_columns=base_cfg["preprocessing"]["drop_columns"])
    df = create_class_labels(df, malfunction_classes=base_cfg["preprocessing"]["malfunction_classes"])
    print(f"Data ready: {df.shape}")

    # --- Build experiment grid ---
    experiments = list(itertools.product(clusters_list, rounds_list, local_epochs_list))
    total = len(experiments)
    print(f"\nTotal experiments: {total}")
    print(f"Grid: clusters={clusters_list}, rounds={rounds_list}, local_epochs={local_epochs_list}")
    print(f"Output: {output_root}\n")

    all_results = []

    for idx, (n_clusters, num_rounds, local_epochs) in enumerate(experiments):
        exp_id = f"c{n_clusters}_r{num_rounds}_e{local_epochs}"
        exp_dir = output_root / exp_id

        # Skip if already completed
        if (exp_dir / "metrics.json").exists():
            print(f"[{idx+1}/{total}] Skipping {exp_id} (already done)")
            with open(exp_dir / "metrics.json") as f:
                result = json.load(f)
            all_results.append(result)
            continue

        print(f"\n[{idx+1}/{total}] Running {exp_id}...")
        cfg = deepcopy(base_cfg)
        cfg["rsu_partitioning"]["kmeans"]["n_clusters"] = n_clusters
        cfg["federated"]["num_rounds"] = num_rounds

        try:
            result = run_single_experiment(
                cfg=cfg,
                df=df,
                n_clusters=n_clusters,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                exp_dir=exp_dir,
                run_centralized=(run_centralized_once and idx == 0),
            )
            all_results.append(result)
        except Exception as e:
            print(f"  EXPERIMENT FAILED: {e}")
            traceback.print_exc()
            all_results.append({"config": {"n_clusters": n_clusters, "num_rounds": num_rounds, "local_epochs": local_epochs}, "error": str(e)})

    # --- Build summary CSV ---
    build_summary_csv(all_results, output_root)
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE. Summary: {output_root / 'summary.csv'}")
    print(f"{'='*70}")


def build_summary_csv(all_results: list[dict], output_root: Path):
    """Flatten all experiment results into a single CSV."""
    rows = []
    for result in all_results:
        cfg = result.get("config", {})
        base = {
            "n_clusters": cfg.get("n_clusters"),
            "num_rounds": cfg.get("num_rounds"),
            "local_epochs": cfg.get("local_epochs"),
        }
        for model_name, metrics in result.items():
            if model_name == "config" or not isinstance(metrics, dict) or "error" in metrics:
                continue
            row = {**base, "model": model_name}
            for metric_name in ["accuracy", "precision", "recall", "f1", "auc", "train_time_sec"]:
                row[metric_name] = metrics.get(metric_name)
            rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["model", "n_clusters", "num_rounds"])
    summary.to_csv(output_root / "summary.csv", index=False)

    # Also print top results
    if not summary.empty and "auc" in summary.columns:
        print("\n--- TOP 10 BY AUC ---")
        top = summary.dropna(subset=["auc"]).nlargest(10, "auc")
        print(top[["model", "n_clusters", "num_rounds", "local_epochs", "auc", "f1"]].to_string(index=False))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VANET Experiment Sweep")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--clusters", nargs="+", type=int, default=DEFAULT_CLUSTERS)
    parser.add_argument("--rounds", nargs="+", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--local-epochs", nargs="+", type=int, default=DEFAULT_LOCAL_EPOCHS)
    parser.add_argument("--mode", choices=["all", "federated_only"], default="all",
                        help="'all' runs centralized once + all federated. 'federated_only' skips centralized.")
    args = parser.parse_args()

    run_sweep(
        clusters_list=args.clusters,
        rounds_list=args.rounds,
        local_epochs_list=args.local_epochs,
        run_centralized_once=(args.mode == "all"),
        config_path=args.config,
    )


if __name__ == "__main__":
    main()