"""
models.py — Model definitions, training, and persistence.

Implements four regressors:
  1. Baseline          (DummyRegressor — training-set mean)
  2. Random Forest     (MultiOutputRegressor wrapper)
  3. XGBoost           (MultiOutputRegressor wrapper)
  4. Deep Neural Net   (PyTorch — shared trunk + per-target heads)
"""

import os
import logging
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    TARGET_COLUMNS,
    RF_PARAMS,
    XGB_PARAMS,
    NN_CONFIG,
    RANDOM_STATE,
    MODEL_DIR,
)

logger = logging.getLogger(__name__)

# Reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════
# Persistence helpers
# ═══════════════════════════════════════════════════════════════════════

def _save_pkl(obj, name: str) -> str:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, name)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    logger.info("Saved %s", path)
    return path


def _load_pkl(name: str):
    path = os.path.join(MODEL_DIR, name)
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ═══════════════════════════════════════════════════════════════════════
# 1.  BASELINE
# ═══════════════════════════════════════════════════════════════════════

def train_baseline(X_train: np.ndarray, y_train: np.ndarray) -> MultiOutputRegressor:
    """Dummy regressor — always predicts training-set mean."""
    model = MultiOutputRegressor(DummyRegressor(strategy="mean"))
    model.fit(X_train, y_train)
    _save_pkl(model, "baseline.pkl")
    logger.info("Baseline model trained.")
    return model


# ═══════════════════════════════════════════════════════════════════════
# 2.  RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> MultiOutputRegressor:
    """Random Forest wrapped in MultiOutputRegressor."""
    base = RandomForestRegressor(**RF_PARAMS)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    _save_pkl(model, "random_forest.pkl")
    logger.info("Random Forest trained (%d trees).", RF_PARAMS["n_estimators"])
    return model


# ═══════════════════════════════════════════════════════════════════════
# 3.  XGBOOST
# ═══════════════════════════════════════════════════════════════════════

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> MultiOutputRegressor:
    """XGBoost via MultiOutputRegressor."""
    base = XGBRegressor(**XGB_PARAMS, verbosity=0)
    model = MultiOutputRegressor(base)
    model.fit(X_train, y_train)
    _save_pkl(model, "xgboost.pkl")
    logger.info("XGBoost trained (%d rounds).", XGB_PARAMS["n_estimators"])
    return model


# ═══════════════════════════════════════════════════════════════════════
# 4.  PYTORCH NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════

class FootballNet(nn.Module):
    """
    Shared dense trunk → per-target regression heads.

    Architecture:
        Input → Dense(128,ReLU) → BN → Drop → Dense(64,ReLU) → BN → Drop
              ↘ Head_i: Dense(32,ReLU) → Dense(1)  ×6 targets
    """

    def __init__(self, input_dim: int, n_targets: int = 6):
        super().__init__()

        layers = []
        in_features = input_dim
        for units in NN_CONFIG.shared_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.Dropout(NN_CONFIG.dropout_rate))
            in_features = units

        self.trunk = nn.Sequential(*layers)

        # Individual output heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            for _ in range(n_targets)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(x)
        outputs = [head(shared) for head in self.heads]
        return torch.cat(outputs, dim=1)  # (batch, n_targets)


class _EarlyStopping:
    """Simple early-stopping tracker."""

    def __init__(self, patience: int = 15):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class NeuralNetWrapper:
    """
    Scikit-learn–compatible wrapper around the PyTorch model so it plugs
    into the same ``evaluate()`` / ``predict()`` API as the tree models.
    """

    def __init__(self):
        self.model: Optional[FootballNet] = None
        self.scaler = StandardScaler()

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "NeuralNetWrapper":
        X_scaled = self.scaler.fit_transform(X_train)

        # Validation split
        n_val = int(len(X_scaled) * NN_CONFIG.validation_split)
        idx = np.random.permutation(len(X_scaled))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_tr = torch.FloatTensor(X_scaled[train_idx]).to(DEVICE)
        y_tr = torch.FloatTensor(y_train[train_idx]).to(DEVICE)
        X_va = torch.FloatTensor(X_scaled[val_idx]).to(DEVICE)
        y_va = torch.FloatTensor(y_train[val_idx]).to(DEVICE)

        train_loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=NN_CONFIG.batch_size,
            shuffle=True,
        )

        self.model = FootballNet(X_scaled.shape[1], y_train.shape[1]).to(DEVICE)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=NN_CONFIG.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            factor=NN_CONFIG.lr_reduce_factor,
            patience=NN_CONFIG.patience_reduce_lr,
            min_lr=NN_CONFIG.min_lr,
        )
        criterion = nn.MSELoss()
        stopper = _EarlyStopping(patience=NN_CONFIG.patience_early_stop)

        logger.info(
            "Training PyTorch NN  |  device=%s  |  params=%s",
            DEVICE,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

        for epoch in range(1, NN_CONFIG.epochs + 1):
            # --- Train ---
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                optimiser.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimiser.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(X_tr)

            # --- Validate ---
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_va)
                val_loss = criterion(val_preds, y_va).item()

            scheduler.step(val_loss)
            current_lr = optimiser.param_groups[0]["lr"]

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "  Epoch %3d/%d  train_mse=%.4f  val_mse=%.4f  lr=%.2e",
                    epoch, NN_CONFIG.epochs, train_loss, val_loss, current_lr,
                )

            if stopper.step(val_loss, self.model):
                logger.info("  Early stopping at epoch %d (best val_mse=%.4f)", epoch, stopper.best_loss)
                break

        stopper.restore(self.model)

        # Save artefacts
        _save_pkl(self.scaler, "nn_scaler.pkl")
        torch.save(self.model.state_dict(), os.path.join(MODEL_DIR, "neural_net.pt"))
        logger.info("Neural network saved.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X_scaled).to(DEVICE)
            preds = self.model(tensor).cpu().numpy()
        return preds


def train_neural_net(X_train: np.ndarray, y_train: np.ndarray) -> NeuralNetWrapper:
    """Train the PyTorch neural network."""
    wrapper = NeuralNetWrapper()
    wrapper.fit(X_train, y_train)
    return wrapper


# ═══════════════════════════════════════════════════════════════════════
# 5.  TRAIN ALL MODELS
# ═══════════════════════════════════════════════════════════════════════

def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    """Train every model and return a dict keyed by model name."""
    models = {}

    logger.info("=" * 60)
    logger.info("Training Baseline …")
    models["Baseline"] = train_baseline(X_train, y_train)

    logger.info("=" * 60)
    logger.info("Training Random Forest …")
    models["Random Forest"] = train_random_forest(X_train, y_train)

    logger.info("=" * 60)
    logger.info("Training XGBoost …")
    models["XGBoost"] = train_xgboost(X_train, y_train)

    logger.info("=" * 60)
    logger.info("Training Neural Network (PyTorch) …")
    models["Neural Network"] = train_neural_net(X_train, y_train)

    return models
