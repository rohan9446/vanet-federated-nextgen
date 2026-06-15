"""
Flower-Based Federated Deep Learning for VANET Anomaly Detection.

Implements FedLSTM, FedBiLSTM, and FedCNN using the Flower framework
with cyclic training strategy.

New contribution: extends the paper's FedXGB approach to deep learning
models, enabled by GPU access.
"""

import numpy as np
from collections import OrderedDict
from typing import Callable
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


def _import_deps():
    """Lazy imports for Flower + PyTorch."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import flwr as fl
    return torch, nn, DataLoader, TensorDataset, fl


# ============================================================
# PyTorch Model Definitions
# ============================================================

def _build_lstm(input_dim, num_classes=3, units=128, dropout=0.3):
    torch, nn, _, _, _ = _import_deps()

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, units, batch_first=True, dropout=dropout)
            self.fc = nn.Linear(units, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return LSTM()


def _build_bilstm(input_dim, num_classes=3, units=128, dropout=0.3):
    torch, nn, _, _, _ = _import_deps()

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, units, batch_first=True,
                bidirectional=True, dropout=dropout,
            )
            self.fc = nn.Linear(units * 2, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    return BiLSTM()


def _build_cnn(input_dim, num_classes=3, filters=None, kernel_size=3, dropout=0.3):
    torch, nn, _, _, _ = _import_deps()
    if filters is None:
        filters = [64, 128]

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, filters[0], kernel_size, padding=1)
            self.conv2 = nn.Conv1d(filters[0], filters[1], kernel_size, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(filters[1], num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
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
# Flower Client
# ============================================================

class VANETFlowerClient:
    """
    Flower client for a single RSU.
    Handles local training and parameter exchange.
    """

    def __init__(self, model, train_loader, test_loader, device, local_epochs=3, lr=0.001):
        torch, nn, _, _, _ = _import_deps()
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self):
        """Extract model parameters as numpy arrays."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Load model parameters from numpy arrays."""
        torch, _, _, _, _ = _import_deps()
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def train_local(self):
        """Train model on local data for local_epochs."""
        torch, _, _, _, _ = _import_deps()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        total_loss = 0
        n_batches = 0
        for epoch in range(self.local_epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, len(self.train_loader.dataset)

    def evaluate_local(self):
        """Evaluate model on local test data."""
        torch, _, _, _, _ = _import_deps()
        self.model.eval()
        correct, total, total_loss = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                output = self.model(X_batch)
                total_loss += self.criterion(output, y_batch).item()
                correct += (output.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
        accuracy = correct / max(total, 1)
        return total_loss / max(len(self.test_loader), 1), accuracy, total


# ============================================================
# Cyclic Federated Training (simulated, no Flower server needed)
# ============================================================

def train_federated_cyclic(
    model_type: str,
    partitions: dict[str, tuple[np.ndarray, np.ndarray]],
    num_rounds: int = 10,
    local_epochs: int = 3,
    batch_size: int = 256,
    lr: float = 0.001,
    model_kwargs: dict | None = None,
) -> tuple:
    """
    Simulated cyclic federated training for deep learning models.

    The model is passed from one client to the next in round-robin order.
    Each client trains locally for local_epochs before passing it on.

    Args:
        model_type: "lstm", "bilstm", or "cnn".
        partitions: Dict of RSU_id -> (X, y).
        num_rounds: Number of full cycles.
        local_epochs: Epochs per client per round.
        batch_size: Training batch size.
        lr: Learning rate.
        model_kwargs: Extra kwargs for model builder.

    Returns:
        (trained_model, history) where history is list of per-round metrics.
    """
    torch, nn, DataLoader, TensorDataset, _ = _import_deps()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Federated {model_type.upper()} Cyclic | Device: {device}")

    if model_kwargs is None:
        model_kwargs = {}

    # Prepare data loaders for each client
    scaler = StandardScaler()
    client_ids = sorted(partitions.keys())

    # Fit scaler on all data combined
    all_X = np.vstack([partitions[c][0] for c in client_ids])
    scaler.fit(all_X)
    del all_X

    client_loaders = {}
    for cid in client_ids:
        X, y = partitions[cid]
        X_s = scaler.transform(X)
        X_t = torch.FloatTensor(X_s).unsqueeze(1)  # (N, 1, features)
        y_t = torch.LongTensor(y)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        client_loaders[cid] = loader

    # Get input dimension
    input_dim = list(partitions.values())[0][0].shape[1]

    # Build model
    builder = MODEL_BUILDERS[model_type]
    model = builder(input_dim, **model_kwargs).to(device)

    # Cyclic training
    history = []
    for round_idx in range(num_rounds):
        round_losses = []
        for cid in client_ids:
            client = VANETFlowerClient(
                model=model,
                train_loader=client_loaders[cid],
                test_loader=client_loaders[cid],  # using same for simplicity
                device=device,
                local_epochs=local_epochs,
                lr=lr,
            )
            loss, n_samples = client.train_local()
            round_losses.append(loss)
            logger.debug(f"Round {round_idx+1}, Client {cid}: loss={loss:.4f}")

        avg_loss = np.mean(round_losses)
        history.append({"round": round_idx + 1, "avg_loss": avg_loss})
        logger.info(f"Round {round_idx+1}/{num_rounds} — Avg Loss: {avg_loss:.4f}")

    return model, scaler, history


def train_federated_fedavg(
    model_type: str,
    partitions: dict[str, tuple[np.ndarray, np.ndarray]],
    num_rounds: int = 10,
    local_epochs: int = 3,
    batch_size: int = 256,
    lr: float = 0.001,
    model_kwargs: dict | None = None,
) -> tuple:
    """
    Simulated FedAvg training for deep learning models.

    Each round: all clients train locally, then parameters are averaged.

    Returns:
        (trained_model, scaler, history)
    """
    torch, nn, DataLoader, TensorDataset, _ = _import_deps()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Federated {model_type.upper()} FedAvg | Device: {device}")

    if model_kwargs is None:
        model_kwargs = {}

    scaler = StandardScaler()
    client_ids = sorted(partitions.keys())
    all_X = np.vstack([partitions[c][0] for c in client_ids])
    scaler.fit(all_X)
    del all_X

    client_loaders = {}
    client_sizes = {}
    for cid in client_ids:
        X, y = partitions[cid]
        X_s = scaler.transform(X)
        X_t = torch.FloatTensor(X_s).unsqueeze(1)
        y_t = torch.LongTensor(y)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        client_loaders[cid] = loader
        client_sizes[cid] = len(X)

    input_dim = list(partitions.values())[0][0].shape[1]
    builder = MODEL_BUILDERS[model_type]
    global_model = builder(input_dim, **model_kwargs).to(device)

    history = []
    total_samples = sum(client_sizes.values())

    for round_idx in range(num_rounds):
        client_params = []
        client_weights = []

        for cid in client_ids:
            # Clone global model for local training
            local_model = builder(input_dim, **model_kwargs).to(device)
            local_model.load_state_dict(global_model.state_dict())

            client = VANETFlowerClient(
                model=local_model,
                train_loader=client_loaders[cid],
                test_loader=client_loaders[cid],
                device=device,
                local_epochs=local_epochs,
                lr=lr,
            )
            loss, n_samples = client.train_local()

            params = client.get_parameters()
            client_params.append(params)
            client_weights.append(client_sizes[cid] / total_samples)

        # Weighted average of parameters
        avg_params = []
        for param_idx in range(len(client_params[0])):
            weighted = sum(
                w * client_params[i][param_idx]
                for i, w in enumerate(client_weights)
            )
            avg_params.append(weighted)

        # Update global model
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), avg_params)}
        )
        global_model.load_state_dict(state_dict)

        history.append({"round": round_idx + 1})
        logger.info(f"FedAvg Round {round_idx+1}/{num_rounds} complete")

    return global_model, scaler, history


def predict_federated_dl(model, scaler, X_test, batch_size=256):
    """Generate predictions from a federated DL model."""
    torch, _, DataLoader, TensorDataset, _ = _import_deps()
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
