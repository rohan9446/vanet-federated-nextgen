"""
FedTrust: Trust-Weighted Federated Aggregation for VANET Anomaly Detection.

Novel trust mechanism combining:
  1. Global validation performance — each client's model evaluated on a
     held-out global validation set (not just local val). Clients whose
     models generalize poorly to diverse data get lower trust.
  2. Per-class detection balance — rewards clients that detect ALL anomaly
     classes, not just the majority class. Critical for VANET security.
  3. Data quality score — class distribution entropy.
  4. Historical trust — EMA across rounds for stability.
  5. Adaptive temperature — starts warm (uniform), cools down as trust
     estimates become reliable over rounds.

Trust score:
  T_i = α * global_val_f1 + β * per_class_balance + γ * data_quality + δ * historical_trust

Aggregation weights derived via temperature-scaled softmax of trust scores.
"""

import numpy as np
from collections import OrderedDict
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score
import xgboost as xgb
from models.federated.fed_xgb import _ensure_all_classes
from preprocessing.class_weights import compute_sample_weights
import logging

logger = logging.getLogger(__name__)


def _import_torch():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    return torch, nn, DataLoader, TensorDataset


# ============================================================
# Trust Score Components
# ============================================================

def compute_data_quality_score(y: np.ndarray, num_classes: int = 3) -> float:
    """
    Class distribution entropy normalized to [0, 1].
    Uniform = 1.0, single class = 0.0.
    """
    counts = np.bincount(y.astype(int), minlength=num_classes)
    probs = counts / counts.sum()
    max_ent = np.log(num_classes)
    if max_ent == 0:
        return 0.0
    return float(entropy(probs) / max_ent)


def compute_per_class_balance(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> float:
    """
    Per-class detection balance: minimum per-class recall.
    A model that detects all classes equally well scores 1.0.
    A model that misses one class entirely scores 0.0.
    
    This penalizes clients that only learn majority-class patterns.
    """
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(num_classes)))
    # Use min recall — if any class is missed, score drops hard
    min_recall = float(np.min(recalls))
    # Blend with mean to avoid being too harsh
    mean_recall = float(np.mean(recalls))
    return 0.6 * min_recall + 0.4 * mean_recall


def compute_trust_score(
    global_val_f1: float,
    per_class_balance: float,
    data_quality: float,
    historical_trust: float,
    alpha: float = 0.35,
    beta: float = 0.30,
    gamma: float = 0.15,
    delta: float = 0.20,
) -> float:
    """
    Composite trust score for an RSU client.

    Args:
        global_val_f1: F1 score on global validation set [0, 1].
        per_class_balance: Per-class detection balance [0, 1].
        data_quality: Class distribution entropy [0, 1].
        historical_trust: EMA of past trust scores [0, 1].
        alpha: Weight for global validation F1.
        beta: Weight for per-class balance.
        gamma: Weight for data quality.
        delta: Weight for historical trust.
    """
    return (
        alpha * global_val_f1
        + beta * per_class_balance
        + gamma * data_quality
        + delta * historical_trust
    )


def adaptive_temperature(round_idx: int, num_rounds: int, t_start: float = 2.0, t_end: float = 0.3) -> float:
    """
    Linearly decay temperature from t_start to t_end over rounds.
    Early rounds: high temperature → near-uniform weights (exploration).
    Late rounds: low temperature → sharp trust-based weights (exploitation).
    """
    progress = round_idx / max(num_rounds - 1, 1)
    return t_start + (t_end - t_start) * progress


def normalize_trust_scores(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Temperature-scaled softmax normalization."""
    ids = list(scores.keys())
    vals = np.array([scores[k] for k in ids])
    vals = vals / max(temperature, 1e-8)
    exp_vals = np.exp(vals - vals.max())
    weights = exp_vals / exp_vals.sum()
    return {k: float(w) for k, w in zip(ids, weights)}


# ============================================================
# FedTrust DL Client
# ============================================================

class FedTrustClient:
    """FL client with trust-aware training and evaluation."""

    def __init__(self, model, train_loader, val_loader, device,
                 local_epochs=5, lr=0.001, patience=3, class_weights=None):
        torch, nn, _, _ = _import_torch()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.patience = patience
        self.torch = torch

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            {k: self.torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def train_local(self):
        """Train locally with early stopping. Returns best val loss."""
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = self.torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(self.local_epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            val_loss = self._val_loss()
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_loss

    def _val_loss(self):
        self.model.eval()
        total_loss, n = 0, 0
        with self.torch.no_grad():
            for X_b, y_b in self.val_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                total_loss += self.criterion(self.model(X_b), y_b).item()
                n += 1
        self.model.train()
        return total_loss / max(n, 1)

    def predict_on_data(self, X_tensor):
        """Run inference, return (preds, probs)."""
        self.model.eval()
        with self.torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            output = self.model(X_tensor)
            probs = self.torch.softmax(output, dim=1)
            preds = output.argmax(1)
        return preds.cpu().numpy(), probs.cpu().numpy()


# ============================================================
# FedTrust DL Training
# ============================================================

def train_fedtrust_dl(
    model_type: str,
    partitions: dict[str, tuple[np.ndarray, np.ndarray]],
    num_rounds: int = 20,
    local_epochs: int = 5,
    batch_size: int = 256,
    lr: float = 0.001,
    patience: int = 3,
    t_start: float = 2.0,
    t_end: float = 0.3,
    alpha: float = 0.35,
    beta: float = 0.30,
    gamma: float = 0.15,
    delta: float = 0.20,
    ema_decay: float = 0.7,
    global_val_data: tuple | None = None,
    model_kwargs: dict | None = None,
) -> tuple:
    """
    FedTrust v2: Trust-weighted federated averaging with:
    - Global validation trust (not just local)
    - Per-class detection balance
    - Adaptive temperature scheduling

    Args:
        global_val_data: (X_val, y_val) numpy arrays for global trust evaluation.
                         If None, falls back to local validation only.

    Returns:
        (trained_model, scaler, history)
    """
    from models.federated.fed_dl import MODEL_BUILDERS

    torch, nn, DataLoader, TensorDataset = _import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"FedTrust {model_type.upper()} | Device: {device} | Rounds: {num_rounds}")

    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs = {k: v for k, v in model_kwargs.items()
                    if k not in ("local_epochs", "batch_size", "learning_rate", "patience",
                                 "t_start", "t_end", "alpha", "beta", "gamma", "delta", "ema_decay")}

    # Fit scaler on all training data
    scaler = StandardScaler()
    client_ids = sorted(partitions.keys())
    all_X = np.vstack([partitions[c][0] for c in client_ids])
    scaler.fit(all_X)
    del all_X

    # Prepare client train/val loaders
    from preprocessing.class_weights import compute_class_weights
    train_loaders, val_loaders = {}, {}
    data_quality_scores = {}
    client_class_weights = {}
    for cid in client_ids:
        X, y = partitions[cid]
        X_s = scaler.transform(X)
        n_val = max(int(len(X_s) * 0.1), 1)
        n_train = len(X_s) - n_val

        train_loaders[cid] = DataLoader(
            TensorDataset(torch.FloatTensor(X_s[:n_train]).unsqueeze(1), torch.LongTensor(y[:n_train])),
            batch_size=batch_size, shuffle=True,
        )
        val_loaders[cid] = DataLoader(
            TensorDataset(torch.FloatTensor(X_s[n_train:]).unsqueeze(1), torch.LongTensor(y[n_train:])),
            batch_size=batch_size, shuffle=False,
        )
        data_quality_scores[cid] = compute_data_quality_score(y)
        client_class_weights[cid] = compute_class_weights(y[:n_train])

    # Prepare global validation tensor
    global_val_tensor = None
    global_val_labels = None
    if global_val_data is not None:
        X_gval, y_gval = global_val_data
        X_gval_s = scaler.transform(X_gval)
        global_val_tensor = torch.FloatTensor(X_gval_s).unsqueeze(1)
        global_val_labels = y_gval
        logger.info(f"Global validation set: {len(y_gval)} samples")

    input_dim = list(partitions.values())[0][0].shape[1]
    global_model = MODEL_BUILDERS[model_type](input_dim, **model_kwargs).to(device)

    # Historical trust
    historical_trust = {cid: 0.5 for cid in client_ids}

    history = []
    for round_idx in range(num_rounds):
        temp = adaptive_temperature(round_idx, num_rounds, t_start, t_end)
        client_params = []
        trust_scores = {}

        for cid in client_ids:
            # Clone global model
            local_model = MODEL_BUILDERS[model_type](input_dim, **model_kwargs).to(device)
            local_model.load_state_dict(global_model.state_dict())

            client = FedTrustClient(
                model=local_model,
                train_loader=train_loaders[cid],
                val_loader=val_loaders[cid],
                device=device,
                local_epochs=local_epochs,
                lr=lr,
                patience=patience,
                class_weights=client_class_weights[cid],
            )
            client.train_local()
            client_params.append(client.get_parameters())

            # --- Compute trust components ---
            if global_val_tensor is not None:
                # Global validation: evaluate client model on global val set
                y_pred_global, _ = client.predict_on_data(global_val_tensor)
                gval_f1 = f1_score(global_val_labels, y_pred_global, average='weighted', zero_division=0)
                pcb = compute_per_class_balance(global_val_labels, y_pred_global)
            else:
                # Fallback to local validation
                all_preds, all_labels = [], []
                for X_b, y_b in val_loaders[cid]:
                    preds, _ = client.predict_on_data(X_b)
                    all_preds.extend(preds)
                    all_labels.extend(y_b.numpy())
                gval_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                pcb = compute_per_class_balance(np.array(all_labels), np.array(all_preds))

            trust = compute_trust_score(
                global_val_f1=gval_f1,
                per_class_balance=pcb,
                data_quality=data_quality_scores[cid],
                historical_trust=historical_trust[cid],
                alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            )
            trust_scores[cid] = trust
            historical_trust[cid] = ema_decay * historical_trust[cid] + (1 - ema_decay) * gval_f1

            logger.debug(
                f"R{round_idx+1} {cid}: gval_f1={gval_f1:.4f}, pcb={pcb:.4f}, "
                f"dq={data_quality_scores[cid]:.4f}, trust={trust:.4f}"
            )

        # Trust-weighted aggregation
        weights = normalize_trust_scores(trust_scores, temperature=temp)
        weight_list = [weights[cid] for cid in client_ids]

        avg_params = []
        for pidx in range(len(client_params[0])):
            weighted = sum(w * client_params[i][pidx] for i, w in enumerate(weight_list))
            avg_params.append(weighted)

        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), avg_params)}
        )
        global_model.load_state_dict(state_dict)

        logger.info(
            f"Round {round_idx+1}/{num_rounds} | temp={temp:.2f} | "
            f"weights: {{{', '.join(f'{c}: {w:.3f}' for c, w in weights.items())}}}"
        )
        history.append({
            "round": round_idx + 1,
            "temperature": round(temp, 3),
            "trust_scores": {k: round(v, 4) for k, v in trust_scores.items()},
            "weights": {k: round(v, 4) for k, v in weights.items()},
        })

    logger.info("FedTrust DL training complete.")
    return global_model, scaler, history


# ============================================================
# FedTrust XGBoost
# ============================================================

def train_fedtrust_xgb(
    partitions: dict[str, tuple[np.ndarray, np.ndarray]],
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_rounds: int = 3,
    xgb_params: dict | None = None,
    t_start: float = 2.0,
    t_end: float = 0.3,
    alpha: float = 0.35,
    beta: float = 0.30,
    gamma: float = 0.15,
    delta: float = 0.20,
    ema_decay: float = 0.7,
    random_state: int = 42,
) -> tuple:
    """
    FedTrust XGBoost with global validation trust scoring.

    Returns:
        (models_list, trust_weights, history)
    """
    if xgb_params is None:
        xgb_params = {}

    params = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "objective": "multi:softprob",
        "num_class": 3,
        "tree_method": "hist",
        "random_state": random_state,
        "eval_metric": "mlogloss",
    }
    params.update(xgb_params)

    client_ids = sorted(partitions.keys())
    historical_trust = {cid: 0.5 for cid in client_ids}
    data_quality_scores = {
        cid: compute_data_quality_score(partitions[cid][1]) for cid in client_ids
    }

    history = []
    final_models = []
    final_weights = {}

    for round_idx in range(num_rounds):
        temp = adaptive_temperature(round_idx, num_rounds, t_start, t_end)
        models = []
        trust_scores = {}

        for cid in client_ids:
            X_local, y_local = partitions[cid]
            X_local, y_local = _ensure_all_classes(X_local, y_local)

            model = xgb.XGBClassifier(**params)
            model.fit(X_local, y_local,
                      sample_weight=compute_sample_weights(y_local),
                      verbose=False)
            models.append(model)

            # Global validation trust
            y_pred = model.predict(X_val)
            gval_f1 = float(f1_score(y_val, y_pred, average='weighted', zero_division=0))
            pcb = compute_per_class_balance(y_val, y_pred)

            trust = compute_trust_score(
                global_val_f1=gval_f1,
                per_class_balance=pcb,
                data_quality=data_quality_scores[cid],
                historical_trust=historical_trust[cid],
                alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            )
            trust_scores[cid] = trust
            historical_trust[cid] = ema_decay * historical_trust[cid] + (1 - ema_decay) * gval_f1

            logger.debug(f"R{round_idx+1} XGB {cid}: gval_f1={gval_f1:.4f}, pcb={pcb:.4f}, trust={trust:.4f}")

        final_weights = normalize_trust_scores(trust_scores, temperature=temp)
        final_models = models

        logger.info(
            f"FedTrust XGB R{round_idx+1}/{num_rounds} | temp={temp:.2f} | "
            f"weights: {{{', '.join(f'{c}: {w:.3f}' for c, w in final_weights.items())}}}"
        )
        history.append({
            "round": round_idx + 1,
            "temperature": round(temp, 3),
            "trust_scores": {k: round(v, 4) for k, v in trust_scores.items()},
            "weights": {k: round(v, 4) for k, v in final_weights.items()},
        })

    return final_models, final_weights, history


def predict_fedtrust_xgb(
    models: list,
    weights: dict[str, float],
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Trust-weighted probability averaging."""
    client_ids = sorted(weights.keys())
    weight_list = [weights[cid] for cid in client_ids]
    all_probs = np.array([m.predict_proba(X_test) for m in models])
    weighted_probs = np.average(all_probs, axis=0, weights=weight_list)
    preds = weighted_probs.argmax(axis=1)
    return preds, weighted_probs