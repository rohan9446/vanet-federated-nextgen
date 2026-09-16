"""
Flower-Based Federated Deep Learning for VANET Anomaly Detection.

Implements FedLSTM, FedBiLSTM, and FedCNN with:
- Cyclic training (round-robin)
- FedAvg (weighted parameter averaging)
- Early stopping per client
- Class-weighted loss (replaces SMOTE when balancing method is "class_weight")
"""

import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


def _import_deps():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    return torch, nn, DataLoader, TensorDataset


# ============================================================
# Model Definitions
# ============================================================

def _build_lstm(input_dim, num_classes=3, units=128, dropout=0.3, **kwargs):
    torch, nn, _, _ = _import_deps()

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, units, batch_first=True, num_layers=2, dropout=dropout)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(units, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

    return LSTM()


def _build_bilstm(input_dim, num_classes=3, units=128, dropout=0.3, **kwargs):
    torch, nn, _, _ = _import_deps()

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, units, batch_first=True,
                bidirectional=True, num_layers=2, dropout=dropout,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(units * 2, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

    return BiLSTM()


def _build_cnn(input_dim, num_classes=3, filters=None, kernel_size=3, dropout=0.3, **kwargs):
    torch, nn, _, _ = _import_deps()
    if filters is None:
        filters = [64, 128]

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, filters[0], kernel_size, padding=1)
            self.bn1 = nn.BatchNorm1d(filters[0])
            self.conv2 = nn.Conv1d(filters[0], filters[1], kernel_size, padding=1)
            self.bn2 = nn.BatchNorm1d(filters[1])
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(filters[1], num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x).squeeze(-1)
            x = self.dropout(x)
            return self.fc(x)

    return CNN()


MODEL_BUILDERS = {
    "lstm": _build_lstm,
    "bilstm": _build_bilstm,
    "cnn": _build_cnn,
}


# ============================================================
# Flower Client with Early Stopping + Class Weights
# ============================================================

class VANETFlowerClient:
    def __init__(self, model, train_loader, val_loader, device,
                 local_epochs=5, lr=0.001, patience=3, class_weights=None):
        torch, nn, _, _ = _import_deps()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.patience = patience
        self.torch = torch

        # Class-weighted loss
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
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = self.torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(self.local_epochs):
            train_loss = 0
            n_batches = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            val_loss = self._validate()
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.debug(f"Early stopping at local epoch {epoch+1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_loss, len(self.train_loader.dataset)

    def _validate(self):
        self.model.eval()
        total_loss, n = 0, 0
        with self.torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                total_loss += self.criterion(self.model(X_batch), y_batch).item()
                n += 1
        self.model.train()
        return total_loss / max(n, 1)


# ============================================================
# Data Preparation Helper
# ============================================================

def _prepare_client_loaders(partitions, scaler, batch_size=256, val_split=0.1):
    """Create train+val loaders and class weights per client."""
    torch, _, DataLoader, TensorDataset = _import_deps()
    from preprocessing.class_weights import compute_class_weights

    client_ids = sorted(partitions.keys())
    client_train_loaders = {}
    client_val_loaders = {}
    client_sizes = {}
    client_class_weights = {}

    for cid in client_ids:
        X, y = partitions[cid]
        X_s = scaler.transform(X)

        n_val = max(int(len(X_s) * val_split), 1)
        n_train = len(X_s) - n_val

        X_train_t = torch.FloatTensor(X_s[:n_train]).unsqueeze(1)
        y_train_t = torch.LongTensor(y[:n_train])
        X_val_t = torch.FloatTensor(X_s[n_train:]).unsqueeze(1)
        y_val_t = torch.LongTensor(y[n_train:])

        client_train_loaders[cid] = DataLoader(
            TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
        )
        client_val_loaders[cid] = DataLoader(
            TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False
        )
        client_sizes[cid] = n_train
        client_class_weights[cid] = compute_class_weights(y[:n_train])

    return client_ids, client_train_loaders, client_val_loaders, client_sizes, client_class_weights


# ============================================================
# Cyclic Federated Training
# ============================================================

def train_federated_cyclic(
    model_type: str,
    partitions: dict[str, tuple[np.ndarray, np.ndarray]],
    num_rounds: int = 20,
    local_epochs: int = 5,
    batch_size: int = 256,
    lr: float = 0.001,
    patience: int = 3,
    use_class_weights: bool = True,
    model_kwargs: dict | None = None,
) -> tuple:
    torch, _, _, _ = _import_deps()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Fed{model_type.upper()} Cyclic | Device: {device} | Rounds: {num_rounds}")

    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs = {k: v for k, v in model_kwargs.items()
                    if k not in ("local_epochs", "batch_size", "learning_rate", "patience",
                                 "use_class_weights")}

    scaler = StandardScaler()
    client_ids = sorted(partitions.keys())
    all_X = np.vstack([partitions[c][0] for c in client_ids])
    scaler.fit(all_X)
    del all_X

    client_ids, train_loaders, val_loaders, client_sizes, client_cw = _prepare_client_loaders(
        partitions, scaler, batch_size
    )

    input_dim = list(partitions.values())[0][0].shape[1]
    model = MODEL_BUILDERS[model_type](input_dim, **model_kwargs).to(device)

    history = []
    for round_idx in range(num_rounds):
        round_losses = []
        for cid in client_ids:
            client = VANETFlowerClient(
                model=model,
                train_loader=train_loaders[cid],
                val_loader=val_loaders[cid],
                device=device,
                local_epochs=local_epochs,
                lr=lr,
                patience=patience,
                class_weights=client_cw[cid] if use_class_weights else None,
            )
            loss, _ = client.train_local()
            round_losses.append(loss)

        avg_loss = np.mean(round_losses)
        history.append({"round": round_idx + 1, "avg_loss": avg_loss})
        logger.info(f"Cyclic Round {round_idx+1}/{num_rounds} — Avg Loss: {avg_loss:.4f}")

    return model, scaler, history


# ============================================================
# FedAvg Training
# ============================================================

def train_federated_fedavg(
    model_type: str,
    partitions: dict[str, tuple[np.ndarray, np.ndarray]],
    num_rounds: int = 20,
    local_epochs: int = 5,
    batch_size: int = 256,
    lr: float = 0.001,
    patience: int = 3,
    use_class_weights: bool = True,
    model_kwargs: dict | None = None,
) -> tuple:
    torch, _, _, _ = _import_deps()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Fed{model_type.upper()} FedAvg | Device: {device} | Rounds: {num_rounds}")

    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs = {k: v for k, v in model_kwargs.items()
                    if k not in ("local_epochs", "batch_size", "learning_rate", "patience",
                                 "use_class_weights")}

    scaler = StandardScaler()
    client_ids = sorted(partitions.keys())
    all_X = np.vstack([partitions[c][0] for c in client_ids])
    scaler.fit(all_X)
    del all_X

    client_ids, train_loaders, val_loaders, client_sizes, client_cw = _prepare_client_loaders(
        partitions, scaler, batch_size
    )
    total_samples = sum(client_sizes.values())

    input_dim = list(partitions.values())[0][0].shape[1]
    global_model = MODEL_BUILDERS[model_type](input_dim, **model_kwargs).to(device)

    history = []
    for round_idx in range(num_rounds):
        client_params = []
        client_weights = []

        for cid in client_ids:
            local_model = MODEL_BUILDERS[model_type](input_dim, **model_kwargs).to(device)
            local_model.load_state_dict(global_model.state_dict())

            client = VANETFlowerClient(
                model=local_model,
                train_loader=train_loaders[cid],
                val_loader=val_loaders[cid],
                device=device,
                local_epochs=local_epochs,
                lr=lr,
                patience=patience,
                class_weights=client_cw[cid] if use_class_weights else None,
            )
            loss, _ = client.train_local()
            client_params.append(client.get_parameters())
            client_weights.append(client_sizes[cid] / total_samples)

        avg_params = []
        for param_idx in range(len(client_params[0])):
            weighted = sum(
                w * client_params[i][param_idx]
                for i, w in enumerate(client_weights)
            )
            avg_params.append(weighted)

        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), avg_params)}
        )
        global_model.load_state_dict(state_dict)

        history.append({"round": round_idx + 1})
        logger.info(f"FedAvg Round {round_idx+1}/{num_rounds} complete")

    return global_model, scaler, history


# ============================================================
# Prediction
# ============================================================

def predict_federated_dl(model, scaler, X_test, batch_size=256):
    torch, _, DataLoader, TensorDataset = _import_deps()
    device = next(model.parameters()).device

    X_s = scaler.transform(X_test)
    X_t = torch.FloatTensor(X_s).unsqueeze(1)
    y_dummy = torch.zeros(len(X_t), dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_dummy), batch_size=batch_size, shuffle=False)

    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            prob = torch.softmax(output, dim=1)
            preds.append(output.argmax(1).cpu().numpy())
            probs.append(prob.cpu().numpy())

    return np.concatenate(preds), np.concatenate(probs)