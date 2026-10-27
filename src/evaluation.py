"""
evaluation.py - Model evaluation utilities.

Computes per-target MAE for every model and prints a formatted
comparison table highlighting the best performer.
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from tabulate import tabulate

from src.config import TARGET_COLUMNS, TARGET_LABELS

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_cols: List[str] = TARGET_COLUMNS,
) -> Dict[str, float]:
    """Evaluate a single model -> dict of per-target MAE + overall mean MAE."""
    y_pred = model.predict(X_test)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    metrics = {}
    for i, col in enumerate(target_cols):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        metrics[col] = round(mae, 4)

    metrics["Mean MAE"] = round(np.mean(list(metrics.values())), 4)
    return metrics


def evaluate_all(
    models: Dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Evaluate every model -> tidy DataFrame."""
    rows = []
    for name, model in models.items():
        logger.info("Evaluating %s …", name)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"Model": name, **metrics})

    results_df = pd.DataFrame(rows)
    rename_map = {c: TARGET_LABELS.get(c, c) for c in TARGET_COLUMNS}
    results_df.rename(columns=rename_map, inplace=True)
    return results_df


def print_comparison_table(results_df: pd.DataFrame) -> str:
    """Pretty-print results and return the name of the best model."""
    print("\n" + "=" * 94)
    print("  MODEL COMPARISON - Mean Absolute Error (MAE) per Target")
    print("=" * 94)
    print(
        tabulate(
            results_df,
            headers="keys",
            tablefmt="pipe",
            showindex=False,
            floatfmt=".4f",
        )
    )
    print()

    best_idx = results_df["Mean MAE"].idxmin()
    best_model = results_df.loc[best_idx, "Model"]
    best_mae = results_df.loc[best_idx, "Mean MAE"]
    print(f"  *  Best model: {best_model}  (Mean MAE = {best_mae:.4f})")
    print("=" * 94 + "\n")

    return best_model
