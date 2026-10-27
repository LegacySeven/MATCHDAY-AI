"""
config.py — Central configuration for the Football Match Prediction System.

All constants, URL patterns, league mappings, season codes, and
hyperparameters live here so every other module stays DRY.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────
# Project root (two levels up from src/)
# ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────
# Data source configuration
# ─────────────────────────────────────────────────────────────────────
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"

LEAGUE_CODES: Dict[str, str] = {
    "EPL": "E0",
    "La Liga": "SP1",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "Ligue 1": "F1",
    "Eredivisie": "N1",
    "Liga Portugal": "P1",
    "Süper Lig": "T1",
    "Belgian Pro League": "B1",
    "Super League (Greece)": "G1",
    "Scottish Premiership": "SC0",
    "English Championship": "E1",
    "Czech First League": "C1",
    "Ekstraklasa": "PL1",
    "Danish Superliga": "DK1",
}

SEASON_CODES: List[str] = ["1920", "2021", "2122", "2223", "2324", "2425"]

SEASON_LABELS: Dict[str, str] = {
    "1920": "2019/2020",
    "2021": "2020/2021",
    "2122": "2021/2022",
    "2223": "2022/2023",
    "2324": "2023/2024",
    "2425": "2024/2025",
}

# ─────────────────────────────────────────────────────────────────────
# Feature engineering parameters
# ─────────────────────────────────────────────────────────────────────
INITIAL_ELO: float = 1500.0
HOME_ADVANTAGE: float = 100.0
K_FACTOR: float = 30.0

ROLLING_WINDOW: int = 5
DEFAULT_REST_DAYS: int = 14

RAW_COLUMNS: List[str] = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST",
    "HC", "AC",
]

TARGET_COLUMNS: List[str] = ["FTHG", "FTAG", "HST", "AST", "HC", "AC"]

TARGET_LABELS: Dict[str, str] = {
    "FTHG": "Home Goals",
    "FTAG": "Away Goals",
    "HST": "Home Shots on Target",
    "AST": "Away Shots on Target",
    "HC": "Home Corners",
    "AC": "Away Corners",
}

# ─────────────────────────────────────────────────────────────────────
# Model / training parameters
# ─────────────────────────────────────────────────────────────────────
TRAIN_SEASONS: List[str] = ["1920", "2021", "2122", "2223", "2324"]
TEST_SEASON: str = "2425"

RANDOM_STATE: int = 42

RF_PARAMS: Dict = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS: Dict = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


@dataclass
class NNConfig:
    """PyTorch neural network hyperparameters."""
    shared_units: Tuple[int, ...] = (128, 64)
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    patience_early_stop: int = 15
    patience_reduce_lr: int = 7
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6
    validation_split: float = 0.15


NN_CONFIG = NNConfig()

# ─────────────────────────────────────────────────────────────────────
# Paths (relative to project root)
# ─────────────────────────────────────────────────────────────────────
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
