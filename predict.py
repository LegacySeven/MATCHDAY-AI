import sys
import logging
import pickle
import warnings
from datetime import datetime

from src.data_ingestion import load_all_data
from src.feature_engineering import build_features
from src.inference import init_predictor, predict_fixture
from src.probabilities import compute_poisson_grid, get_match_odds, get_over_under, get_btts
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")


def get_user_choice(prompt: str, options: list) -> str:
    """Helper to print a menu and get a valid choice from the user."""
    print(f"\n{'-' * 40}")
    print(prompt)
    print(f"{'-' * 40}")
    for idx, opt in enumerate(options, 1):
        print(f"  {idx}. {opt}")
    
    while True:
        choice = input(f"\nEnter a number (1-{len(options)}): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Invalid choice, please try again.")


def select_league(leagues: list) -> str:
    """Helper to select a league, sorted by the user's defined European Tiers."""
    tier_1 = ["EPL", "Serie A", "La Liga", "Bundesliga", "Ligue 1"]
    tier_2 = ["Eredivisie", "Liga Portugal", "Belgian Pro League", "Süper Lig", "Czech First League"]
    tier_3 = ["Super League (Greece)", "Ekstraklasa", "Danish Superliga", "Scottish Premiership"]
    
    # Get available leagues in each tier
    avail_t1 = [l for l in tier_1 if l in leagues]
    avail_t2 = [l for l in tier_2 if l in leagues]
    avail_t3 = [l for l in tier_3 if l in leagues]
    
    # Any other leagues not explicitly ranked
    ranked = set(tier_1 + tier_2 + tier_3)
    avail_other = [l for l in sorted(leagues) if l not in ranked]
    
    options = avail_t1 + avail_t2 + avail_t3 + avail_other
    
    print(f"\n{'-' * 50}")
    print("Select a League:")
    print(f"{'-' * 50}")
    
    idx = 1
    if avail_t1:
        print("\n  The Elite Tier (Top 5):")
        for opt in avail_t1:
            print(f"  {idx}. {opt}")
            idx += 1
            
    if avail_t2:
        print("\n  Strong Contenders (Ranks 6\u201310):")
        for opt in avail_t2:
            print(f"  {idx}. {opt}")
            idx += 1
            
    if avail_t3:
        print("\n  Rising European Competitions (Ranks 11\u201320):")
        for opt in avail_t3:
            print(f"  {idx}. {opt}")
            idx += 1
            
    if avail_other:
        print("\n  Other Leagues:")
        for opt in avail_other:
            print(f"  {idx}. {opt}")
            idx += 1
            
    while True:
        choice = input(f"\nEnter a number (1-{len(options)}): ").strip()
        if choice.isdigit():
            c = int(choice)
            if 1 <= c <= len(options):
                return options[c - 1]
        print("Invalid choice, please try again.")


def main():
    print("Fetching latest data to build the menu...")
    
    # 1. Load data
    raw_df = load_all_data()
    
    # 2. Interactive Selection
    leagues = raw_df["League"].unique().tolist()
    if not leagues:
        print("Error: No data available. Please check data_ingestion.")
        sys.exit(1)
        
    selected_league = select_league(leagues)
    
    teams = sorted(raw_df[raw_df["League"] == selected_league]["HomeTeam"].unique())
    selected_home = get_user_choice("Select the Home Team:", teams)
    
    # Remove the selected home team so they can't be chosen twice
    away_teams = [t for t in teams if t != selected_home]
    selected_away = get_user_choice("Select the Away Team:", away_teams)
    
    print(f"\nBuilding latest features for {selected_home} vs {selected_away}...")
    enriched_df, elo_engine, feature_cols = build_features(raw_df)

    # 3. Load the best model (Neural Network)
    from src.models import NeuralNetWrapper, FootballNet
    import torch
    import os
    
    model_path = "models/nn_scaler.pkl"
    try:
        best_model = NeuralNetWrapper()
        with open(model_path, "rb") as f:
            best_model.scaler = pickle.load(f)
            
        best_model.model = FootballNet(len(feature_cols), 6)
        best_model.model.load_state_dict(torch.load("models/neural_net.pt", map_location=torch.device('cpu'), weights_only=True))
    except FileNotFoundError:
        print(f"\nError: Could not find trained model at {model_path}.")
        print("Please run `python main.py` first to train the models.")
        sys.exit(1)

    # 4. Initialize predictor
    init_predictor(enriched_df, elo_engine, feature_cols, best_model, "Neural Network")

    # 5. Predict
    try:
        preds = predict_fixture(selected_home, selected_away, selected_league)
        print(f"\n{'=' * 55}")
        print(f"  {selected_league}: {selected_home} vs {selected_away}")
        print(f"{'=' * 55}")
        for target, value in preds.items():
            print(f"    {target:<25s} ->  {value:>6.2f}")
        
        # Calculate Betting Probabilities
        home_goals_pred = preds.get("Home Goals", 0)
        away_goals_pred = preds.get("Away Goals", 0)
        
        grid = compute_poisson_grid(home_goals_pred, away_goals_pred)
        hw, d, aw = get_match_odds(grid)
        over25, under25 = get_over_under(grid, 2.5)
        btts_yes, btts_no = get_btts(grid)

        print(f"\n{'-' * 55}")
        print("  Match Odds:")
        print(f"    Home Win: {hw*100:>5.1f}%  |  Draw: {d*100:>5.1f}%  |  Away Win: {aw*100:>5.1f}%")
        print("\n  Markets:")
        print(f"    Over 2.5 Goals: {over25*100:>5.1f}%   |  Under 2.5 Goals: {under25*100:>5.1f}%")
        print(f"    BTTS (Yes):     {btts_yes*100:>5.1f}%   |  BTTS (No):       {btts_no*100:>5.1f}%")
        print(f"{'=' * 55}\n")
    except Exception as exc:
        print(f"\nPrediction failed: {exc}")


if __name__ == "__main__":
    main()
