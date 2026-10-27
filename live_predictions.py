"""
live_predictions.py — Master automation script.

Full flow:
  1. Load historical CSV data
  2. Sync recent fixtures from API-Football (fill the gap)
  3. Re-compute features (Elo, form, rest days) on current data
  4. Pull today's upcoming fixtures
  5. Map team names (fuzzy matching)
  6. Predict via trained Neural Network
  7. Convert to Poisson probabilities
  8. Compare against de-vigged bookmaker odds
  9. Write ranked value-bet report
"""

import json
import logging
import pickle
import os

import torch

from src.data_ingestion import load_all_data
from src.api_sync import sync_recent_fixtures
from src.feature_engineering import build_features
from src.models import NeuralNetWrapper, FootballNet
from src.inference import init_predictor, predict_fixture, get_known_teams
from src.api_football import get_fixtures_today, get_odds_for_fixture, ODDS_LEAGUE_IDS
from src.name_matching import match_team_name
from src.devig import average_bookmaker_probs
from src.probabilities import compute_poisson_grid, get_match_odds, get_over_under, get_btts

logger = logging.getLogger(__name__)
OUTPUT_PATH = "data/processed/value_bets_today.json"


def _extract_market_odds(bookmakers, market_name, outcome_names):
    """
    Pull one market's odds from each bookmaker's response.
    """
    results = []
    for bm in bookmakers:
        for bet in bm.get("bets", []):
            if bet["name"] != market_name:
                continue
            odds = {v["value"]: float(v["odd"]) for v in bet["values"] if v["value"] in outcome_names}
            if len(odds) == len(outcome_names):
                results.append(odds)
    return results


def generate_value_bets() -> list:
    """
    Full pipeline: sync data → build features → load model → predict → compare odds.
    """
    # ── Step 1: Load CSV data ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Loading historical CSV data...")
    logger.info("=" * 60)
    raw_df = load_all_data()
    logger.info("CSV data: %d matches, last date: %s", len(raw_df), raw_df["Date"].max().strftime("%Y-%m-%d"))

    # ── Step 2: Sync recent fixtures from API-Football ────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Syncing recent fixtures from API-Football...")
    logger.info("=" * 60)
    synced_df = sync_recent_fixtures(raw_df)
    logger.info("After sync: %d matches, last date: %s", len(synced_df), synced_df["Date"].max().strftime("%Y-%m-%d"))

    # ── Step 3: Build features on current data ────────────────────────
    logger.info("=" * 60)
    logger.info("Step 3: Building features on synced data...")
    logger.info("=" * 60)
    enriched_df, elo_engine, feature_cols = build_features(synced_df)

    # ── Step 4: Load trained model ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 4: Loading Neural Network model...")
    logger.info("=" * 60)
    model = NeuralNetWrapper()
    with open("models/nn_scaler.pkl", "rb") as f:
        model.scaler = pickle.load(f)
    model.model = FootballNet(len(feature_cols), 6)
    model.model.load_state_dict(
        torch.load("models/neural_net.pt", map_location="cpu", weights_only=True)
    )
    init_predictor(enriched_df, elo_engine, feature_cols, model, "Neural Network")

    # ── Step 5: Fetch today's fixtures and compare odds ───────────────
    logger.info("=" * 60)
    logger.info("Step 5: Fetching today's fixtures & odds (top 5 leagues)...")
    logger.info("=" * 60)
    known_teams = get_known_teams()
    report = []

    for league_name in ODDS_LEAGUE_IDS:
        fixtures = get_fixtures_today(league_name)
        logger.info("%s: %d fixtures today", league_name, len(fixtures))

        for fx in fixtures:
            api_home = fx["teams"]["home"]["name"]
            api_away = fx["teams"]["away"]["name"]
            fixture_id = fx["fixture"]["id"]

            home = match_team_name(api_home, known_teams)
            away = match_team_name(api_away, known_teams)
            if home is None or away is None:
                logger.warning("Skipping %s vs %s — no confident name match", api_home, api_away)
                continue

            preds = predict_fixture(home, away, league=league_name)
            grid = compute_poisson_grid(preds["Home Goals"], preds["Away Goals"])
            model_hw, model_d, model_aw = get_match_odds(grid)
            model_over, model_under = get_over_under(grid, 2.5)
            model_btts_yes, model_btts_no = get_btts(grid)

            bookmakers = get_odds_for_fixture(fixture_id)
            if not bookmakers:
                logger.info("  No odds available for %s vs %s", home, away)
                continue

            wdl_odds = _extract_market_odds(bookmakers, "Match Winner", ["Home", "Draw", "Away"])
            if not wdl_odds:
                continue
            market_wdl = average_bookmaker_probs(wdl_odds)

            edges = {
                "Home Win": model_hw - market_wdl.get("Home", 0),
                "Draw": model_d - market_wdl.get("Draw", 0),
                "Away Win": model_aw - market_wdl.get("Away", 0),
            }

            ou_odds = _extract_market_odds(bookmakers, "Goals Over/Under", ["Over 2.5", "Under 2.5"])
            if ou_odds:
                market_ou = average_bookmaker_probs(ou_odds)
                edges["Over 2.5"] = model_over - market_ou.get("Over 2.5", 0)
                edges["Under 2.5"] = model_under - market_ou.get("Under 2.5", 0)

            btts_odds = _extract_market_odds(bookmakers, "Both Teams Score", ["Yes", "No"])
            if btts_odds:
                market_btts = average_bookmaker_probs(btts_odds)
                edges["BTTS Yes"] = model_btts_yes - market_btts.get("Yes", 0)
                edges["BTTS No"] = model_btts_no - market_btts.get("No", 0)

            best_bet = max(edges, key=edges.get)
            report.append({
                "league": league_name,
                "fixture": f"{home} vs {away}",
                "kickoff": fx["fixture"]["date"],
                "model_probs": {"Home Win": round(model_hw, 4), "Draw": round(model_d, 4), "Away Win": round(model_aw, 4)},
                "market_probs": {k: round(v, 4) for k, v in market_wdl.items()},
                "edges": {k: round(v, 4) for k, v in edges.items()},
                "best_bet": best_bet,
                "best_edge": round(edges[best_bet], 4),
            })

    report.sort(key=lambda r: r["best_edge"], reverse=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  VALUE BETS — TOP OPPORTUNITIES TODAY")
    print(f"{'=' * 80}")
    if report:
        for row in report[:10]:
            edge_pct = row["best_edge"] * 100
            print(f"  {row['fixture']:<35} | {row['league']:<12} | {row['best_bet']:<12} | edge={edge_pct:+.1f}%")
    else:
        print("  No fixtures with odds found for today.")
    print(f"{'=' * 80}")
    print(f"  Full report: {OUTPUT_PATH}")
    print(f"{'=' * 80}\n")

    logger.info("Wrote %d value-bet rows to %s", len(report), OUTPUT_PATH)
    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    generate_value_bets()
