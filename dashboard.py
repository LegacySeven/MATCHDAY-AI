import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import torch
import warnings
warnings.filterwarnings('ignore')

from src.data_ingestion import load_all_data
from src.api_sync import sync_recent_fixtures
from src.feature_engineering import build_features
from src.inference import init_predictor, predict_fixture
from src.probabilities import compute_poisson_grid, get_match_odds, get_over_under, get_btts
from src.models import NeuralNetWrapper, FootballNet

st.set_page_config(page_title="Football Predictor AI", layout="wide")

@st.cache_resource
def load_system():
    df = load_all_data()
    synced_df = sync_recent_fixtures(df)
    enriched_df, elo_engine, feature_cols = build_features(synced_df)
    
    # Load model
    model = NeuralNetWrapper()
    with open("models/nn_scaler.pkl", "rb") as f:
        model.scaler = pickle.load(f)
    model.model = FootballNet(len(feature_cols), 6)
    model.model.load_state_dict(torch.load("models/neural_net.pt", map_location='cpu', weights_only=True))
    
    init_predictor(enriched_df, elo_engine, feature_cols, model, "Neural Network")
    return enriched_df

st.title("⚽ Agentic Football Predictor")
st.markdown("Powered by **Neural Networks** and **Poisson Distribution**")

with st.spinner("Initializing AI Engine... (This may take a moment if geocoding travel distances for the first time)"):
    try:
        enriched_df = load_system()
    except Exception as e:
        st.error(f"System not ready. Run main.py first. Error: {e}")
        st.stop()

# ═══════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════
tab_predict, tab_value = st.tabs(["🎯 Match Predictor", "📡 Live Value Bets"])

# ── Tab 1: Match Predictor ────────────────────────────────────────────
with tab_predict:
    # Get available leagues and teams
    leagues = sorted(enriched_df["League"].unique().tolist())

    st.sidebar.header("Match Selection")
    selected_league = st.sidebar.selectbox("League", leagues)

    # Filter teams by league
    league_df = enriched_df[enriched_df["League"] == selected_league]
    teams = sorted(set(league_df["HomeTeam"]).union(set(league_df["AwayTeam"])))

    home_team = st.sidebar.selectbox("Home Team", teams)
    away_team = st.sidebar.selectbox("Away Team", teams, index=min(1, len(teams)-1))

    if st.sidebar.button("Predict Match"):
        if home_team == away_team:
            st.error("Home and Away teams must be different!")
        else:
            with st.spinner("Calculating probabilities..."):
                preds = predict_fixture(home_team, away_team, selected_league)
                
                home_goals = preds.get("Home Goals", 0)
                away_goals = preds.get("Away Goals", 0)
                
                # Layout
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Expected Home Goals", f"{home_goals:.2f}")
                col2.metric("vs", "-")
                col3.metric("Expected Away Goals", f"{away_goals:.2f}")
                
                st.divider()
                
                # Poisson probabilities
                grid = compute_poisson_grid(home_goals, away_goals)
                hw, d, aw = get_match_odds(grid)
                over25, under25 = get_over_under(grid, 2.5)
                btts_yes, btts_no = get_btts(grid)
                
                st.subheader("Betting Markets")
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"**Home Win:** {hw*100:.1f}%")
                with m2:
                    st.markdown(f"**Draw:** {d*100:.1f}%")
                with m3:
                    st.markdown(f"**Away Win:** {aw*100:.1f}%")
                    
                st.divider()
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### Goals Market")
                    st.write(f"**Over 2.5 Goals:** {over25*100:.1f}%")
                    st.write(f"**Under 2.5 Goals:** {under25*100:.1f}%")
                with c2:
                    st.markdown("### BTTS Market")
                    st.write(f"**Both Teams to Score (Yes):** {btts_yes*100:.1f}%")
                    st.write(f"**Both Teams to Score (No):** {btts_no*100:.1f}%")
                    
                # Radar Chart
                st.subheader("Team Matchup Profile")
                
                categories = ['Goals', 'Shots on Target', 'Corners']
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                      r=[preds['Home Goals'], preds['Home Shots on Target'], preds['Home Corners']],
                      theta=categories,
                      fill='toself',
                      name=home_team
                ))
                fig.add_trace(go.Scatterpolar(
                      r=[preds['Away Goals'], preds['Away Shots on Target'], preds['Away Corners']],
                      theta=categories,
                      fill='toself',
                      name=away_team
                ))
                
                fig.update_layout(
                  polar=dict(
                    radialaxis=dict(
                      visible=True,
                      range=[0, max(preds['Home Corners'], preds['Away Corners'], 8)]
                    )),
                  showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Live Value Bets ────────────────────────────────────────────
with tab_value:
    st.subheader("📡 Live Value Bets")
    try:
        with open("data/processed/value_bets_today.json") as fh:
            report = json.load(fh)
    except FileNotFoundError:
        st.warning("No live predictions yet — run `python live_predictions.py` first.")
        report = None

    if report is not None:
        if not report:
            st.info("No value bets found for today's fixtures.")
        else:
            df_vb = pd.DataFrame(report)
            df_vb["edge_pct"] = (df_vb["best_edge"] * 100).round(1)
            st.dataframe(
                df_vb[["league", "fixture", "kickoff", "best_bet", "edge_pct"]]
                  .style.background_gradient(subset=["edge_pct"], cmap="Greens"),
                use_container_width=True,
            )
