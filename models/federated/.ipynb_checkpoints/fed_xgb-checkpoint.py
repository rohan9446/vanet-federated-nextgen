"""
fl_xgb.py — Federated XGBoost using Flower's FedXgbBagging.

Uses Flower's Client interface (not NumPyClient) to avoid msgpack
corruption of XGBoost model bytes during serialization.

Usage:
    python -m models.federated.fl_xgb --data_dir data/processed --output_dir results/fl_xgb
    python -m models.federated.fl_xgb --data_dir data/processed --output_dir results/fl_xgb \
        --clusters 8 --classifications binary --rounds 5 --seeds 42
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json as json_lib
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import flwr as fl
from flwr.client import Client
from flwr.common import (
    Parameters, Status, Code,
    FitRes, FitIns,
    EvaluateRes, EvaluateIns,
    GetParametersRes, GetParametersIns,
    GetPropertiesRes, GetPropertiesIns,
)
from flwr.server.strategy import FedXgbBagging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score

# -----------------------------------------------------------------------
# Features
# -----------------------------------------------------------------------
CLASSIFICATION_FEATURES = [
    "sender_spd", "sender_acl",
    "time_delta", "distance", "heading_change",
    "speed_consistency", "accel_consistency",
    "heading_trajectory_consistency",
    "tx_delay", "mean_sender_receiver_dist", "n_receivers",
    "sender_dist_to_road_edge",
]

# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_partitions(data_dir, k):
    parts = []
    for i in range(k):
        path = Path(data_dir) / "partitions" / f"k{k}" / f"partition_{i}.parquet"
        parts.append(pd.read_parquet(path))
    return parts


def get_Xy(df, classification, label_encoder=None):
    feat_cols = [c for c in CLASSIFICATION_FEATURES if c in df.columns]
    X = df[feat_cols].fillna(0.0).values.astype(np.float32)
    if classification == "binary":
        y = df["attacker"].values.astype(np.int64)
        n_classes = 2
        le = None
    elif classification == "triclass":
        if label_encoder is None:
            le = LabelEncoder()
            le.fit(sorted(df["attack_triclass"].unique()))
        else:
            le = label_encoder
        y = le.transform(df["attack_triclass"]).astype(np.int64)
        n_classes = len(le.classes_)
    elif classification == "7class":
        if label_encoder is None:
            le = LabelEncoder()
            le.fit(sorted(df["attack_category"].unique()))
        else:
            le = label_encoder
        y = le.transform(df["attack_category"]).astype(np.int64)
        n_classes = len(le.classes_)
    else:
        raise ValueError(classification)
    return X, y, n_classes, le


# -----------------------------------------------------------------------
# Model serialization (JSON text, required by FedXgbBagging)
# -----------------------------------------------------------------------

def save_model_bytes(bst):
    """Save XGBoost model as JSON text bytes."""
    fname = tempfile.mktemp(suffix=".json")
    bst.save_model(fname)
    with open(fname, "r", encoding="utf-8") as f:
        json_str = f.read()
    os.unlink(fname)
    return json_str.encode("utf-8")


def load_model_bytes(model_bytes, params):
    """Load XGBoost model from JSON text bytes."""
    fname = tempfile.mktemp(suffix=".json")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(model_bytes.decode("utf-8") if isinstance(model_bytes, bytes) else model_bytes)
    bst = xgb.Booster(params=params)
    bst.load_model(fname)
    os.unlink(fname)
    return bst


# -----------------------------------------------------------------------
# XGBoost Client (uses Client interface, not NumPyClient)
# -----------------------------------------------------------------------

class XgbClient(Client):
    def __init__(self, X_train, y_train, X_val, y_val, num_train, params):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_train = num_train
        self.params = params

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=Status(code=Code.OK, message=""),
            properties={},
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(
            status=Status(code=Code.OK, message=""),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def fit(self, ins: FitIns) -> FitRes:
        global_tensors = ins.parameters.tensors
        train_dmatrix = xgb.DMatrix(self.X_train, label=self.y_train)

        if not global_tensors:
            # First round: init model
            bst = xgb.Booster(self.params, [train_dmatrix])
        else:
            bst = load_model_bytes(global_tensors[0], self.params)

        # Train one tree
        bst.update(train_dmatrix, bst.num_boosted_rounds())

        model_bytes = save_model_bytes(bst)

        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=Parameters(tensor_type="", tensors=[model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        global_tensors = ins.parameters.tensors
        if not global_tensors:
            return EvaluateRes(
                status=Status(code=Code.OK, message=""),
                loss=0.0, num_examples=self.num_train,
                metrics={"AUC": 0.0},
            )

        bst = load_model_bytes(global_tensors[0], self.params)
        val_dmatrix = xgb.DMatrix(self.X_val, label=self.y_val)
        preds = bst.predict(val_dmatrix)

        try:
            n_cls = self.params.get("num_class", 2)
            if n_cls > 2 and preds.ndim == 1:
                preds = preds.reshape(-1, n_cls)
                auc = roc_auc_score(self.y_val, preds, multi_class="ovr", average="macro")
            elif preds.ndim == 1:
                auc = roc_auc_score(self.y_val, preds)
            else:
                auc = roc_auc_score(self.y_val, preds, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0

        return EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=0.0, num_examples=self.num_train,
            metrics={"AUC": float(auc)},
        )


# -----------------------------------------------------------------------
# Centralized eval (server-side)
# -----------------------------------------------------------------------

def get_evaluate_fn(X_test, y_test, n_classes, params):
    def evaluate_fn(server_round, parameters, config):
        if not parameters.tensors:
            return 0.0, {"AUC": 0.0, "F1": 0.0}

        bst = load_model_bytes(parameters.tensors[0], params)
        test_dmatrix = xgb.DMatrix(X_test, label=y_test)
        preds = bst.predict(test_dmatrix)

        try:
            if n_classes == 2:
                auc = roc_auc_score(y_test, preds)
                y_pred = (preds > 0.5).astype(int)
                f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
            else:
                if preds.ndim == 1:
                    preds = preds.reshape(-1, n_classes)
                auc = roc_auc_score(y_test, preds, multi_class="ovr", average="macro")
                y_pred = preds.argmax(axis=1)
                f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        except Exception as e:
            print(f"    Eval error: {e}")
            auc, f1 = 0.0, 0.0

        print(f"    Round {server_round}: AUC={auc:.4f}, F1={f1:.4f}")
        return 0.0, {"AUC": float(auc), "F1": float(f1)}

    return evaluate_fn


# -----------------------------------------------------------------------
# Run experiment
# -----------------------------------------------------------------------

def run_experiment(data_dir, k, classification, n_rounds, seed):
    np.random.seed(seed)

    partitions = load_partitions(data_dir, k)
    val_df = pd.read_parquet(Path(data_dir) / "val.parquet")
    test_df = pd.read_parquet(Path(data_dir) / "test.parquet")

    _, y_val_raw, n_classes, le = get_Xy(val_df, classification)
    _, y_test_raw, _, _ = get_Xy(test_df, classification, le)

    # Scale
    all_train_X = np.vstack([
        p[CLASSIFICATION_FEATURES].fillna(0).values.astype(np.float32)
        for p in partitions
    ])
    scaler = StandardScaler().fit(all_train_X)
    del all_train_X

    partition_data = []
    for p in partitions:
        X_p, y_p, _, _ = get_Xy(p, classification, le)
        X_p = scaler.transform(X_p).astype(np.float32)
        partition_data.append((X_p, y_p))

    X_val = scaler.transform(
        val_df[CLASSIFICATION_FEATURES].fillna(0).values.astype(np.float32))
    X_test = scaler.transform(
        test_df[CLASSIFICATION_FEATURES].fillna(0).values.astype(np.float32))

    params = {
        "eta": 0.1,
        "max_depth": 8,
        "subsample": 1.0,
        "tree_method": "hist",
        "nthread": 2,
        "device": "cpu",          # <-- ADD THIS
    }
    if n_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "auc"
    else:
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
        params["num_class"] = n_classes

    # Client factory — returns Client (not NumPyClient)
    def client_fn(cid):
        cid_int = int(cid)
        X_c, y_c = partition_data[cid_int]
        return XgbClient(
            X_train=X_c, y_train=y_c,
            X_val=X_val, y_val=y_val_raw,
            num_train=len(y_c),
            params=params,
        )

    strategy = FedXgbBagging(
        fraction_fit=1.0,
        min_fit_clients=k,
        min_available_clients=k,
        fraction_evaluate=1.0,
        min_evaluate_clients=k,
        evaluate_function=get_evaluate_fn(X_test, y_test_raw, n_classes, params),
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=k,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0},
        ray_init_args={"runtime_env": {"env_vars": {"CUDA_VISIBLE_DEVICES": ""}}},
    )

    # Extract results
    results = {"rounds": []}
    if history.metrics_centralized:
        for metric_name, values in history.metrics_centralized.items():
            for rnd, val in values:
                existing = [r for r in results["rounds"] if r["round"] == rnd]
                if existing:
                    existing[0][metric_name] = val
                else:
                    results["rounds"].append({"round": rnd, metric_name: val})

    final_auc, final_f1 = 0.0, 0.0
    if results["rounds"]:
        last = results["rounds"][-1]
        final_auc = last.get("AUC", 0.0)
        final_f1 = last.get("F1", 0.0)

    return final_auc, final_f1, results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--clusters", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--classifications", nargs="+", default=["binary", "triclass", "7class"])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for classification in args.classifications:
        for k in args.clusters:
            for seed in args.seeds:
                rid = f"xgb_bagging__k{k}__{classification}__seed{seed}"
                rp = Path(args.output_dir) / f"{rid}.json"

                if rp.exists():
                    print(f"[SKIP] {rid}")
                    all_results.append(json_lib.loads(rp.read_text()))
                    continue

                print(f"\n{'='*60}")
                print(f"XGB FedBagging | k={k} | {classification} | seed={seed}")
                print(f"{'='*60}")

                t0 = time.time()
                auc, f1, history = run_experiment(
                    args.data_dir, k, classification, args.rounds, seed)
                elapsed = round(time.time() - t0, 1)

                result = {
                    "run_id": rid, "model": "xgb", "strategy": "FedXgbBagging",
                    "k": k, "classification": classification,
                    "seed": seed, "rounds": args.rounds,
                    "auc": round(auc, 6), "f1": round(f1, 6),
                    "runtime_s": elapsed, "history": history,
                }
                rp.write_text(json_lib.dumps(result, indent=2, default=str))
                all_results.append(result)

                print(f"  Final: AUC={auc:.4f} F1={f1:.4f} | {elapsed}s")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for classification in args.classifications:
        cr = [r for r in all_results if r.get("classification") == classification]
        if not cr:
            continue
        print(f"\n  {classification}:")
        print(f"  {'K':>3} {'AUC':>14} {'F1':>14}")
        print(f"  {'-'*35}")
        for k in args.clusters:
            mr = [r for r in cr if r.get("k") == k]
            if not mr:
                continue
            aucs = [r["auc"] for r in mr]
            f1s = [r["f1"] for r in mr]
            fmt = lambda v: f"{np.mean(v):.4f}\u00B1{np.std(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"
            print(f"  {k:>3} {fmt(aucs):>14} {fmt(f1s):>14}")

    pd.DataFrame([{
        "k": r.get("k"), "classification": r.get("classification"),
        "seed": r.get("seed"), "auc": r.get("auc"), "f1": r.get("f1")
    } for r in all_results]).to_csv(
        Path(args.output_dir) / "summary.csv", index=False)

    print(f"\nDone. Results -> {args.output_dir}/")


if __name__ == "__main__":
    main()