"""
main.py — Entry point for the Football Match Prediction System.

Orchestrates the full pipeline:
  1. Data ingestion   (download + cache)
  2. Feature engineering
  3. Temporal train / test split
  4. Model training   (Baseline, RF, XGBoost, PyTorch NN)
  5. Evaluation & comparison table
  6. Demo inference
"""

import logging
import sys
import warnings

import numpy as np

from src.config import TRAIN_SEASONS, TEST_SEASON, TARGET_COLUMNS, SEASON_LABELS
from src.data_ingestion import load_all_data
from src.feature_engineering import build_features
from src.models import train_all_models
from src.evaluation import evaluate_all, print_comparison_table
from src.inference import init_predictor, predict_fixture

# ─────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")
warnings.filterwarnings("ignore", category=FutureWarning)


def main() -> None:
    # ==================================================================
    # 1. DATA INGESTION
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STEP 1 - Data Ingestion")
    logger.info("=" * 70)

    raw_df = load_all_data()
    logger.info(
        "Raw dataset: %d matches, columns: %s",
        len(raw_df), list(raw_df.columns),
    )

    # ==================================================================
    # 2. FEATURE ENGINEERING
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STEP 2 - Feature Engineering")
    logger.info("=" * 70)

    enriched_df, elo_engine, feature_cols = build_features(raw_df)
    logger.info(
        "Enriched dataset: %d matches, %d features.",
        len(enriched_df), len(feature_cols),
    )

    # ==================================================================
    # 3. TEMPORAL TRAIN / TEST SPLIT
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STEP 3 - Temporal Split")
    logger.info("=" * 70)

    train_mask = enriched_df["Season"].isin(TRAIN_SEASONS)
    test_mask = enriched_df["Season"] == TEST_SEASON

    X_train = enriched_df.loc[train_mask, feature_cols].values.astype(np.float32)
    y_train = enriched_df.loc[train_mask, TARGET_COLUMNS].values.astype(np.float32)
    X_test = enriched_df.loc[test_mask, feature_cols].values.astype(np.float32)
    y_test = enriched_df.loc[test_mask, TARGET_COLUMNS].values.astype(np.float32)

    logger.info(
        "Train: %d samples (%s)  |  Test: %d samples (%s)",
        len(X_train),
        ", ".join(SEASON_LABELS[s] for s in TRAIN_SEASONS),
        len(X_test),
        SEASON_LABELS[TEST_SEASON],
    )

    if len(X_test) == 0:
        logger.error(
            "Test set is empty - verify SEASON_CODES / TEST_SEASON in config."
        )
        sys.exit(1)

    # ==================================================================
    # 4. MODEL TRAINING
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STEP 4 - Model Training")
    logger.info("=" * 70)

    models = train_all_models(X_train, y_train)

    # ==================================================================
    # 5. EVALUATION & COMPARISON
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STEP 5 - Evaluation")
    logger.info("=" * 70)

    results_df = evaluate_all(models, X_test, y_test)
    best_model_name = print_comparison_table(results_df)

    # ==================================================================
    # 6. INFERENCE DEMO
    # ==================================================================
    logger.info("=" * 70)
    logger.info("STEP 6 - Inference Demo (using %s)", best_model_name)
    logger.info("=" * 70)

    best_model = models[best_model_name]
    init_predictor(enriched_df, elo_engine, feature_cols, best_model, best_model_name)

    demo_fixtures = [
        ("Arsenal", "Chelsea", "EPL"),
        ("Barcelona", "Real Madrid", "La Liga"),
        ("Bayern Munich", "Dortmund", "Bundesliga"),
        ("Juventus", "Inter", "Serie A"),
    ]

    for home, away, league in demo_fixtures:
        try:
            preds = predict_fixture(home, away, league)
            print(f"\n{'-' * 55}")
            print(f"  {league}: {home} vs {away}")
            print(f"{'-' * 55}")
            for target, value in preds.items():
                print(f"    {target:<25s} ->  {value:>6.2f}")
        except Exception as exc:
            logger.warning("Could not predict %s vs %s: %s", home, away, exc)

    print("\n[DONE]  Pipeline complete.\n")


if __name__ == "__main__":
    main()
