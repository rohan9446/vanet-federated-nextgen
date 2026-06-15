"""
Centralized Model Training for VANET Anomaly Detection.

Models: Random Forest, Extra Trees, XGBoost, LSTM, Bi-LSTM, CNN.
All trained on the full (non-partitioned) dataset for baseline comparison.

Reference: Section 5 of the paper.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Scikit-learn / XGBoost models
# ============================================================

def train_random_forest(X_train, y_train, **kwargs) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    params = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
    params.update(kwargs)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    logger.info("Random Forest trained.")
    return model


def train_extra_trees(X_train, y_train, **kwargs) -> ExtraTreesClassifier:
    """Train Extra Trees classifier."""
    params = {"n_estimators": 100, "random_state": 42, "n_jobs": -1}
    params.update(kwargs)
    model = ExtraTreesClassifier(**params)
    model.fit(X_train, y_train)
    logger.info("Extra Trees trained.")
    return model


def train_xgboost(X_train, y_train, **kwargs) -> xgb.XGBClassifier:
    """Train XGBoost classifier."""
    params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "multi:softprob",
        "num_class": 3,
        "tree_method": "hist",
        "random_state": 42,
        "eval_metric": "mlogloss",
    }
    params.update(kwargs)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    logger.info("XGBoost trained.")
    return model


# ============================================================
# Deep learning models (PyTorch)
# ============================================================

def _try_import_torch():
    """Import torch lazily so non-GPU machines don't fail."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    return torch, nn, DataLoader, TensorDataset


def _prepare_dl_data(X_train, y_train, X_test, y_test, batch_size=256):
    """Convert numpy arrays to PyTorch DataLoaders."""
    torch, nn, DataLoader, TensorDataset = _try_import_torch()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Reshape to (batch, seq_len=1, features) for LSTM/CNN
    X_train_t = torch.FloatTensor(X_train_s).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test_s).unsqueeze(1)
    y_train_t = torch.LongTensor(y_train.values if hasattr(y_train, "values") else y_train)
    y_test_t = torch.LongTensor(y_test.values if hasattr(y_test, "values") else y_test)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, scaler, X_train_t.shape[2]


class LSTMModel:
    """LSTM for multi-class VANET anomaly detection."""

    def __init__(self, input_dim, num_classes=3, units=128, dropout=0.3):
        torch, nn, _, _ = _try_import_torch()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, units, batch_first=True, dropout=dropout)
                self.fc = nn.Linear(units, num_classes)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self.model = _LSTM().to(self.device)
        self.torch = torch
        self.nn = nn

    def train(self, train_loader, epochs=20, lr=0.001):
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = self.nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                logger.info(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    def predict(self, test_loader):
        self.model.eval()
        preds, probs = [], []
        with self.torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                output = self.model(X_batch)
                prob = self.torch.softmax(output, dim=1)
                preds.append(output.argmax(dim=1).cpu().numpy())
                probs.append(prob.cpu().numpy())
        return np.concatenate(preds), np.concatenate(probs)

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class BiLSTMModel:
    """Bidirectional LSTM for VANET anomaly detection."""

    def __init__(self, input_dim, num_classes=3, units=128, dropout=0.3):
        torch, nn, _, _ = _try_import_torch()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _BiLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, units, batch_first=True,
                    bidirectional=True, dropout=dropout
                )
                self.fc = nn.Linear(units * 2, num_classes)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        self.model = _BiLSTM().to(self.device)
        self.torch = torch
        self.nn = nn

    def train(self, train_loader, epochs=20, lr=0.001):
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = self.nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                logger.info(f"BiLSTM Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    def predict(self, test_loader):
        self.model.eval()
        preds, probs = [], []
        with self.torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                output = self.model(X_batch)
                prob = self.torch.softmax(output, dim=1)
                preds.append(output.argmax(dim=1).cpu().numpy())
                probs.append(prob.cpu().numpy())
        return np.concatenate(preds), np.concatenate(probs)

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


class CNNModel:
    """1D-CNN for VANET anomaly detection."""

    def __init__(self, input_dim, num_classes=3, filters=None, kernel_size=3, dropout=0.3):
        torch, nn, _, _ = _try_import_torch()
        if filters is None:
            filters = [64, 128]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(1, filters[0], kernel_size, padding=1)
                self.conv2 = nn.Conv1d(filters[0], filters[1], kernel_size, padding=1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(filters[1], num_classes)
                self.relu = nn.ReLU()

            def forward(self, x):
                # x shape: (batch, 1, features)
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.pool(x).squeeze(-1)
                x = self.dropout(x)
                return self.fc(x)

        self.model = _CNN().to(self.device)
        self.torch = torch
        self.nn = nn

    def train(self, train_loader, epochs=20, lr=0.001):
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = self.nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                logger.info(f"CNN Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    def predict(self, test_loader):
        self.model.eval()
        preds, probs = [], []
        with self.torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(self.device)
                output = self.model(X_batch)
                prob = self.torch.softmax(output, dim=1)
                preds.append(output.argmax(dim=1).cpu().numpy())
                probs.append(prob.cpu().numpy())
        return np.concatenate(preds), np.concatenate(probs)

    def get_state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
