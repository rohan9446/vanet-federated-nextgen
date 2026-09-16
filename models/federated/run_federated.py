"""
run_federated.py — Federated learning experiments.

Custom FL simulation loop (not Flower) for full control over TrustAgg.

Strategies: FedAvg, FedProx, TrustAgg
Models: XGBoost (federated ensemble), LSTM, BiLSTM
Partitions: k=4, 8, 16

Usage:
    python -m models.federated.run_federated --data_dir data/processed --output_dir results/federated

    # Quick test
    python -m models.federated.run_federated --data_dir data/processed --output_dir results/federated \
        --models xgb --strategies fedavg --clusters 8 --seeds 42 --classifications binary --rounds 5
"""

import argparse
import copy
import gc
import json
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report,
)

# -----------------------------------------------------------------------
# Features (must match preprocess.py)
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
    """Load k partition parquets."""
    parts = []
    for i in range(k):
        path = Path(data_dir) / "partitions" / f"k{k}" / f"partition_{i}.parquet"
        parts.append(pd.read_parquet(path))
    return parts


def load_split(data_dir, split):
    return pd.read_parquet(Path(data_dir) / f"{split}.parquet")


def get_Xy(df, classification, label_encoder=None):
    feat_cols = [c for c in CLASSIFICATION_FEATURES if c in df.columns]
    X = df[feat_cols].fillna(0.0).values.astype(np.float32)
    if classification == "binary":
        y = df["attacker"].values.astype(np.int64)
        names = ["Normal", "Attack"]
        le = None
    elif classification == "triclass":
        if label_encoder is None:
            le = LabelEncoder()
            le.fit(sorted(df["attack_triclass"].unique()))
        else:
            le = label_encoder
        y = le.transform(df["attack_triclass"]).astype(np.int64)
        names = le.classes_.tolist()
    elif classification == "7class":
        if label_encoder is None:
            le = LabelEncoder()
            le.fit(sorted(df["attack_category"].unique()))
        else:
            le = label_encoder
        y = le.transform(df["attack_category"]).astype(np.int64)
        names = le.classes_.tolist()
    else:
        raise ValueError(classification)
    return X, y, feat_cols, names, le


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def evaluate_predictions(y_true, y_pred, y_prob, class_names):
    nc = len(class_names)
    avg = "binary" if nc == 2 else "macro"
    f1 = float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    prec = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob[:, 1])) if nc == 2 else \
              float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except:
        auc = float("nan")
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    pcf1 = {n: round(report[n]["f1-score"], 4) for n in class_names if n in report}
    return {"auc": round(auc, 6), "f1": round(f1, 6),
            "precision": round(prec, 6), "recall": round(rec, 6),
            "per_class_f1": pcf1}


def compute_class_entropy(y, n_classes):
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def compute_min_recall(y_true, y_pred, n_classes):
    recalls = recall_score(y_true, y_pred, labels=range(n_classes),
                           average=None, zero_division=0)
    return float(recalls.min())


# -----------------------------------------------------------------------
# DL Model
# -----------------------------------------------------------------------

def build_dl_model(model_name, n_features, n_classes):
    bidir = model_name == "bilstm"

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, 128, 2, dropout=0.3,
                                batch_first=True, bidirectional=bidir)
            h = 256 if bidir else 128
            self.fc = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(h, 64),
                nn.ReLU(), nn.Linear(64, n_classes))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return LSTMModel()


def get_model_weights(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def set_model_weights(model, weights):
    model.load_state_dict(weights)


# -----------------------------------------------------------------------
# DL Local Training
# -----------------------------------------------------------------------

def train_local_dl(model, X_train, y_train, n_classes, local_epochs,
                   global_weights=None, mu=0.0, seed=42):
    """Train model locally. Returns updated weights + metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Class weights
    cc = np.bincount(y_train, minlength=n_classes).astype(float)
    cc = np.maximum(cc, 1)
    cw = torch.FloatTensor(len(y_train) / (n_classes * cc)).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=512, shuffle=True)

    # Store global params for FedProx
    if global_weights is not None and mu > 0:
        global_params = [v.to(device) for v in global_weights.values()]

    for epoch in range(local_epochs):
        for xb, yb in loader:
            xb, yb = xb.unsqueeze(1).to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)

            # FedProx proximal term
            if global_weights is not None and mu > 0:
                prox = 0.0
                for p, gp in zip(model.parameters(), global_params):
                    prox += torch.sum((p - gp.detach()) ** 2)
                loss += (mu / 2.0) * prox

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return get_model_weights(model)


def predict_dl(model, X, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X), torch.zeros(len(X)).long()),
                        batch_size=2048, shuffle=False)
    preds, probs = [], []
    with torch.no_grad():
        for xb, _ in loader:
            out = model(xb.unsqueeze(1).to(device))
            probs.append(torch.softmax(out, -1).cpu().numpy())
            preds.extend(out.argmax(-1).cpu().numpy())
    return np.array(preds), np.vstack(probs)


# -----------------------------------------------------------------------
# XGBoost Local Training
# -----------------------------------------------------------------------

def train_local_xgb(X_train, y_train, n_classes, seed=42):
    import xgboost as xgb
    cls, cnts = np.unique(y_train, return_counts=True)
    wmap = {c: len(y_train) / (n_classes * max(n, 1)) for c, n in zip(cls, cnts)}
    # Fill missing classes
    for c in range(n_classes):
        if c not in wmap:
            wmap[c] = 1.0
    sw = np.array([wmap.get(y, 1.0) for y in y_train], dtype=np.float32)

    params = {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1,
              "tree_method": "hist", "random_state": seed, "n_jobs": -1,
              "eval_metric": "logloss" if n_classes == 2 else "mlogloss"}
    if n_classes == 2:
        params["objective"] = "binary:logistic"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sw, verbose=False)
    return model


# -----------------------------------------------------------------------
# Aggregation Strategies
# -----------------------------------------------------------------------

def fedavg_aggregate(client_weights_list, client_sizes):
    """Weighted average by sample count."""
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]
    return _weighted_avg_weights(client_weights_list, weights)


def trustagg_aggregate(client_weights_list, client_trust_scores, temperature):
    """Trust-weighted aggregation with temperature-scaled softmax."""
    agg_weights = _softmax(client_trust_scores, temperature)
    return _weighted_avg_weights(client_weights_list, agg_weights), agg_weights


def _weighted_avg_weights(weights_list, agg_weights):
    """Weighted average of model state dicts."""
    avg = OrderedDict()
    for key in weights_list[0].keys():
        avg[key] = sum(w * weights_list[i][key].float()
                       for i, w in enumerate(agg_weights))
    return avg


def _softmax(scores, temperature):
    s = np.array(scores, dtype=np.float64) / temperature
    s -= s.max()
    exp_s = np.exp(s)
    return (exp_s / exp_s.sum()).tolist()


def compute_trust_score(val_f1, min_recall, class_entropy, ema_trust,
                        alpha=0.35, beta=0.30, gamma=0.15, delta=0.20,
                        max_entropy=1.0):
    """Composite trust score."""
    entropy_norm = min(class_entropy / max_entropy, 1.0) if max_entropy > 0 else 0
    return (alpha * val_f1 + beta * min_recall +
            gamma * entropy_norm + delta * ema_trust)


def get_temperature(round_num, total_rounds, temp_start=2.0, temp_end=0.3):
    """Linear cooling schedule."""
    frac = min((round_num - 1) / max(total_rounds - 1, 1), 1.0)
    return temp_start + frac * (temp_end - temp_start)


# -----------------------------------------------------------------------
# FL Simulation: DL Models
# -----------------------------------------------------------------------

def run_fl_dl(model_name, partitions_Xy, X_val, y_val, X_test, y_test,
              n_classes, class_names, strategy, n_rounds, local_epochs,
              seed, mu=0.1):
    """Run federated DL experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = partitions_Xy[0][0].shape[1]
    n_clients = len(partitions_Xy)

    # Initialize global model
    global_model = build_dl_model(model_name, n_features, n_classes)
    global_weights = get_model_weights(global_model)

    ema_trust = {i: 0.5 for i in range(n_clients)}
    max_entropy = math.log(n_classes) if n_classes > 1 else 1.0
    history = []

    for rnd in range(1, n_rounds + 1):
        client_weights = []
        client_sizes = []
        client_trust_scores = []

        for i, (X_part, y_part) in enumerate(partitions_Xy):
            # Clone global model
            local_model = build_dl_model(model_name, n_features, n_classes)
            set_model_weights(local_model, copy.deepcopy(global_weights))

            # Local training
            gw = global_weights if strategy == "fedprox" else None
            local_w = train_local_dl(
                local_model, X_part, y_part, n_classes, local_epochs,
                global_weights=gw, mu=mu if strategy == "fedprox" else 0,
                seed=seed + rnd)

            client_weights.append(local_w)
            client_sizes.append(len(y_part))

            # Compute trust metrics for TrustAgg
            if strategy == "trustagg":
                set_model_weights(local_model, local_w)
                y_pred_val, y_prob_val = predict_dl(local_model, X_val)
                avg = "binary" if n_classes == 2 else "macro"
                val_f1 = float(f1_score(y_val, y_pred_val, average=avg, zero_division=0))
                min_rec = compute_min_recall(y_val, y_pred_val, n_classes)
                cls_ent = compute_class_entropy(y_part, n_classes)

                ema_trust[i] = 0.7 * ema_trust[i] + 0.3 * val_f1
                trust = compute_trust_score(
                    val_f1, min_rec, cls_ent, ema_trust[i],
                    max_entropy=max_entropy)
                client_trust_scores.append(trust)

            del local_model
            gc.collect()

        # Aggregate
        if strategy in ("fedavg", "fedprox"):
            global_weights = fedavg_aggregate(client_weights, client_sizes)
        elif strategy == "trustagg":
            temp = get_temperature(rnd, n_rounds)
            global_weights, agg_w = trustagg_aggregate(
                client_weights, client_trust_scores, temp)

        # Evaluate global model
        set_model_weights(global_model, global_weights)
        y_pred_test, y_prob_test = predict_dl(global_model, X_test)
        test_metrics = evaluate_predictions(y_test, y_pred_test, y_prob_test, class_names)

        y_pred_val, y_prob_val = predict_dl(global_model, X_val)
        val_metrics = evaluate_predictions(y_val, y_pred_val, y_prob_val, class_names)

        round_info = {
            "round": rnd,
            "val_auc": val_metrics["auc"], "val_f1": val_metrics["f1"],
            "test_auc": test_metrics["auc"], "test_f1": test_metrics["f1"],
        }
        if strategy == "trustagg":
            round_info["trust_scores"] = client_trust_scores
            round_info["agg_weights"] = agg_w
            round_info["temperature"] = temp

        history.append(round_info)
        print(f"      R{rnd:>2}: val_auc={val_metrics['auc']:.4f} "
              f"test_auc={test_metrics['auc']:.4f} "
              f"test_f1={test_metrics['f1']:.4f}")

        del client_weights
        gc.collect()

    return test_metrics, history


# -----------------------------------------------------------------------
# FL Simulation: XGBoost (Federated Ensemble)
# -----------------------------------------------------------------------

def run_fl_xgb(partitions_Xy, X_val, y_val, X_test, y_test,
               n_classes, class_names, strategy, seed):
    """Run federated XGBoost ensemble."""
    np.random.seed(seed)
    n_clients = len(partitions_Xy)

    # Train local models
    client_models = []
    client_sizes = []
    client_trust_scores = []

    for i, (X_part, y_part) in enumerate(partitions_Xy):
        model = train_local_xgb(X_part, y_part, n_classes, seed=seed)
        client_models.append(model)
        client_sizes.append(len(y_part))

        # Compute trust for TrustAgg
        if strategy == "trustagg":
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)
            avg = "binary" if n_classes == 2 else "macro"
            val_f1 = float(f1_score(y_val, y_pred, average=avg, zero_division=0))
            min_rec = compute_min_recall(y_val, y_pred, n_classes)
            cls_ent = compute_class_entropy(y_part, n_classes)
            max_entropy = math.log(n_classes) if n_classes > 1 else 1.0
            trust = compute_trust_score(
                val_f1, min_rec, cls_ent, 0.5,
                max_entropy=max_entropy)
            client_trust_scores.append(trust)

        print(f"      Client {i}: {len(y_part):,} samples trained")

    # Compute aggregation weights
    if strategy in ("fedavg", "fedprox"):
        total = sum(client_sizes)
        weights = [s / total for s in client_sizes]
    elif strategy == "trustagg":
        weights = _softmax(client_trust_scores, temperature=1.0)

    # Ensemble predictions (weighted average of probabilities)
    all_probs = []
    for model in client_models:
        prob = model.predict_proba(X_test)
        # Handle missing classes
        if prob.shape[1] < n_classes:
            full_prob = np.zeros((len(X_test), n_classes))
            for j, cls in enumerate(model.classes_):
                full_prob[:, int(cls)] = prob[:, j]
            prob = full_prob
        all_probs.append(prob)

    # Weighted ensemble
    y_prob = np.zeros_like(all_probs[0])
    for w, prob in zip(weights, all_probs):
        y_prob += w * prob
    y_pred = y_prob.argmax(axis=1)

    test_metrics = evaluate_predictions(y_test, y_pred, y_prob, class_names)

    history = [{
        "round": 1,
        "test_auc": test_metrics["auc"],
        "test_f1": test_metrics["f1"],
        "weights": weights,
    }]
    if strategy == "trustagg":
        history[0]["trust_scores"] = client_trust_scores

    print(f"      Ensemble: AUC={test_metrics['auc']:.4f} F1={test_metrics['f1']:.4f}")
    print(f"      Weights: {[f'{w:.3f}' for w in weights]}")

    return test_metrics, history


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

MODEL_NAMES = {"xgb": "XGBoost", "lstm": "LSTM", "bilstm": "BiLSTM"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models", nargs="+", default=["xgb", "lstm", "bilstm"],
                        choices=list(MODEL_NAMES.keys()))
    parser.add_argument("--strategies", nargs="+", default=["fedavg", "fedprox", "trustagg"])
    parser.add_argument("--clusters", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--classifications", nargs="+", default=["binary", "triclass", "7class"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--mu", type=float, default=0.1, help="FedProx mu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load val/test once
    print("Loading val/test data...")
    val_df = load_split(args.data_dir, "val")
    test_df = load_split(args.data_dir, "test")

    all_results = []

    total_exps = (len(args.classifications) * len(args.models) *
                  len(args.strategies) * len(args.clusters) * len(args.seeds))
    exp_num = 0

    for classification in args.classifications:
        print(f"\n{'='*60}")
        print(f"Classification: {classification}")
        print(f"{'='*60}")

        _, y_val, _, class_names, le = get_Xy(val_df, classification)
        _, y_test, _, _, _ = get_Xy(test_df, classification, le)
        n_classes = len(class_names)

        for k in args.clusters:
            # Load partitions
            partitions = load_partitions(args.data_dir, k)

            # Scale features (fit on all training data combined)
            all_train_X = np.vstack([
                p[CLASSIFICATION_FEATURES].fillna(0).values.astype(np.float32)
                for p in partitions])
            scaler = StandardScaler().fit(all_train_X)
            del all_train_X

            # Prepare scaled partition data
            partitions_Xy = []
            for p in partitions:
                X_p, y_p, _, _, _ = get_Xy(p, classification, le)
                X_p = scaler.transform(X_p).astype(np.float32)
                partitions_Xy.append((X_p, y_p))

            X_val_s = scaler.transform(
                val_df[CLASSIFICATION_FEATURES].fillna(0).values.astype(np.float32)
            ).astype(np.float32)
            X_test_s = scaler.transform(
                test_df[CLASSIFICATION_FEATURES].fillna(0).values.astype(np.float32)
            ).astype(np.float32)

            for model_key in args.models:
                for strategy in args.strategies:
                    # XGB doesn't benefit from FedProx (no gradient-based training)
                    if model_key == "xgb" and strategy == "fedprox":
                        continue

                    for seed in args.seeds:
                        exp_num += 1
                        rid = f"{model_key}__{strategy}__k{k}__{classification}__seed{seed}"
                        rp = Path(args.output_dir) / f"{rid}.json"

                        if rp.exists():
                            print(f"\n  [{exp_num}] [SKIP] {rid}")
                            all_results.append(json.loads(rp.read_text()))
                            continue

                        print(f"\n  [{exp_num}/{total_exps}] {MODEL_NAMES[model_key]} | "
                              f"{strategy} | k={k} | {classification} | seed={seed}")
                        t0 = time.time()

                        try:
                            if model_key == "xgb":
                                metrics, history = run_fl_xgb(
                                    partitions_Xy, X_val_s, y_val,
                                    X_test_s, y_test, n_classes,
                                    class_names, strategy, seed)
                            else:
                                metrics, history = run_fl_dl(
                                    model_key, partitions_Xy,
                                    X_val_s, y_val, X_test_s, y_test,
                                    n_classes, class_names, strategy,
                                    args.rounds, args.local_epochs,
                                    seed, mu=args.mu)

                            elapsed = round(time.time() - t0, 1)
                            result = {
                                "run_id": rid, "model": model_key,
                                "strategy": strategy, "k": k,
                                "classification": classification,
                                "seed": seed, "runtime_s": elapsed,
                                "class_names": class_names,
                                "rounds": args.rounds,
                                "local_epochs": args.local_epochs,
                                **metrics,
                                "history": history,
                            }
                            rp.write_text(json.dumps(result, indent=2, default=str))
                            all_results.append(result)

                            print(f"    AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} | {elapsed}s")

                        except Exception as e:
                            print(f"    [FAIL] {e}")
                            import traceback
                            traceback.print_exc()

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            del partitions, partitions_Xy
            gc.collect()

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for classification in args.classifications:
        cr = [r for r in all_results if r.get("classification") == classification]
        if not cr:
            continue

        print(f"\n  {classification}:")
        print(f"  {'Model':<8} {'Strategy':<10} {'K':>3} "
              f"{'AUC':>14} {'F1':>14}")
        print(f"  {'-'*55}")

        for model_key in args.models:
            for strategy in args.strategies:
                if model_key == "xgb" and strategy == "fedprox":
                    continue
                for k in args.clusters:
                    mr = [r for r in cr if r.get("model") == model_key
                          and r.get("strategy") == strategy and r.get("k") == k]
                    if not mr:
                        continue
                    aucs = [r["auc"] for r in mr if not np.isnan(r.get("auc", float("nan")))]
                    f1s = [r["f1"] for r in mr]
                    fmt = lambda v: f"{np.mean(v):.4f}±{np.std(v):.4f}" if len(v) > 1 else f"{v[0]:.4f}"
                    print(f"  {model_key:<8} {strategy:<10} {k:>3} "
                          f"{fmt(aucs):>14} {fmt(f1s):>14}")

    # Save summary CSV
    rows = []
    for r in all_results:
        rows.append({
            "model": r.get("model"), "strategy": r.get("strategy"),
            "k": r.get("k"), "classification": r.get("classification"),
            "seed": r.get("seed"), "auc": r.get("auc"), "f1": r.get("f1"),
            "precision": r.get("precision"), "recall": r.get("recall"),
        })
    pd.DataFrame(rows).to_csv(Path(args.output_dir) / "summary.csv", index=False)
    print(f"\n  Summary -> {args.output_dir}/summary.csv")
    print("Done.")


if __name__ == "__main__":
    main()