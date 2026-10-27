"""
feature_engineering.py — Chronological feature-extraction pipeline with
zero data-leakage guarantees.

Components
----------
1. Dynamic Elo Rating Engine   (sequential, home advantage)
2. Rolling Form Stats          (5-match window, home/away split, shifted)
3. Rest Days                   (days since previous match per team)
4. Head-to-Head History        (rolling H2H record)
"""

import logging
from typing import Dict, Tuple

from src.elo import EloEngine
from src.geography import calculate_travel_distance

import numpy as np
import pandas as pd

from src.config import (
    INITIAL_ELO,
    HOME_ADVANTAGE,
    K_FACTOR,
    ROLLING_WINDOW,
    DEFAULT_REST_DAYS,
    TARGET_COLUMNS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1.  ELO RATING ENGINE
# ═══════════════════════════════════════════════════════════════════════


def compute_elo_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, EloEngine]:
    """
    Walk through *df* chronologically and attach pre-match Elo ratings
    as ``Home_Elo``, ``Away_Elo``, and ``Elo_Diff``.
    """
    df = df.copy()
    # FIX: previously called EloEngine() with no args, silently ignoring
    # config's K_FACTOR / HOME_ADVANTAGE / INITIAL_ELO in favour of the
    # class's own hardcoded defaults (k=25, home_adv=75).
    engine = EloEngine(k=K_FACTOR, home_adv=HOME_ADVANTAGE, initial=INITIAL_ELO)

    home_elos, away_elos = [], []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        hg, ag = int(row["FTHG"]), int(row["FTAG"])

        r_h_before, r_a_before, _, _ = engine.update(home, away, hg, ag)
        home_elos.append(r_h_before)
        away_elos.append(r_a_before)

    df["Home_Elo"] = home_elos
    df["Away_Elo"] = away_elos
    df["Elo_Diff"] = df["Home_Elo"] - df["Away_Elo"]

    logger.info("Elo features computed for %d matches.", len(df))
    return df, engine


# ═══════════════════════════════════════════════════════════════════════
# 2.  ROLLING FORM FEATURES
# ═══════════════════════════════════════════════════════════════════════

def _team_rolling_stats(
    df: pd.DataFrame,
    team_col: str,
    stat_cols: Dict[str, str],
    window: int = ROLLING_WINDOW,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Per-team shifted rolling means.

    ``shift(1)`` is applied before the rolling window to guarantee
    that the current match's outcome never leaks into its own features.

    NOTE: grouping by HomeTeam/AwayTeam already makes these venue-specific
    (e.g. Home_GF_roll5 is "this team's goals scored, in matches where
    they played at home"). Do not recompute these under a different name
    elsewhere — see compute_venue_form(), which used to duplicate this.
    """
    records = []
    for _team, grp in df.groupby(team_col):
        grp = grp.sort_values("Date")
        for src_col, suffix in stat_cols.items():
            feat_name = f"{prefix}{suffix}_roll{window}"
            grp[feat_name] = (
                grp[src_col]
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
            )
        records.append(grp)

    return pd.concat(records).sort_values("Date")


def compute_rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    """Attach home-specific and away-specific rolling form features."""
    df = df.copy()

    home_stats = {
        "FTHG": "GF", "FTAG": "GA",
        "HST": "SoT_For", "AST": "SoT_Against",
        "HC": "Corners_For", "AC": "Corners_Against",
    }
    df = _team_rolling_stats(df, "HomeTeam", home_stats, prefix="Home_")

    away_stats = {
        "FTAG": "GF", "FTHG": "GA",
        "AST": "SoT_For", "HST": "SoT_Against",
        "AC": "Corners_For", "HC": "Corners_Against",
    }
    df = _team_rolling_stats(df, "AwayTeam", away_stats, prefix="Away_")

    logger.info("Rolling form features (window=%d) attached.", ROLLING_WINDOW)
    return df


def compute_ppg_form(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Attach rolling Points Per Game (PPG) for both home and away teams."""
    df = df.copy()

    home_ppg, away_ppg = [], []
    team_history: Dict[str, list] = {}

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        h_past = team_history.get(home, [])
        home_ppg.append(np.mean([p for _, p in h_past[-window:]]) if h_past else 1.0)

        a_past = team_history.get(away, [])
        away_ppg.append(np.mean([p for _, p in a_past[-window:]]) if a_past else 1.0)

        h_pts = 3 if row["FTR"] == "H" else 1 if row["FTR"] == "D" else 0
        a_pts = 3 if row["FTR"] == "A" else 1 if row["FTR"] == "D" else 0

        h_past.append((row["Date"], h_pts))
        a_past.append((row["Date"], a_pts))
        team_history[home] = h_past
        team_history[away] = a_past

    df[f"Home_PPG_roll{window}"] = home_ppg
    df[f"Away_PPG_roll{window}"] = away_ppg

    logger.info("Points Per Game (PPG) features computed.")
    return df


def compute_venue_form(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Attach venue-specific PPG only.

    FIX: this function previously ALSO recomputed Home_GF_Home_roll5,
    Home_GA_Home_roll5, Away_GF_Away_roll5, Away_GA_Away_roll5 — but
    those are mathematically identical to Home_GF_roll5/Home_GA_roll5/
    Away_GF_roll5/Away_GA_roll5 from compute_rolling_form(), since
    grouping by HomeTeam/AwayTeam already restricts to that venue.
    Feeding duplicate/perfectly-correlated columns into the models
    wastes capacity and destabilizes weight estimates. Only PPG (which
    was genuinely new) is kept here.
    """
    df = df.copy()

    h_records = []
    for team, grp in df.groupby("HomeTeam"):
        grp = grp.sort_values("Date")
        pts = grp["FTR"].map({"H": 3, "D": 1, "A": 0})
        grp[f"Home_PPG_Home_roll{window}"] = pts.shift(1).rolling(window=window, min_periods=1).mean()
        h_records.append(grp)
    df = pd.concat(h_records).sort_values("Date")

    a_records = []
    for team, grp in df.groupby("AwayTeam"):
        grp = grp.sort_values("Date")
        pts = grp["FTR"].map({"H": 0, "D": 1, "A": 3})
        grp[f"Away_PPG_Away_roll{window}"] = pts.shift(1).rolling(window=window, min_periods=1).mean()
        a_records.append(grp)
    df = pd.concat(a_records).sort_values("Date")

    for col in [f"Home_PPG_Home_roll{window}", f"Away_PPG_Away_roll{window}"]:
        df[col] = df[col].fillna(1.0)

    logger.info("Venue-specific PPG features computed.")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3.  REST DAYS
# ═══════════════════════════════════════════════════════════════════════

def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Days elapsed since each team's previous match.
    Imputes ``DEFAULT_REST_DAYS`` for season openers.
    """
    df = df.copy()
    last_match: Dict[str, pd.Timestamp] = {}
    home_rest, away_rest = [], []

    for _, row in df.iterrows():
        match_date = row["Date"]

        if row["HomeTeam"] in last_match:
            delta = (match_date - last_match[row["HomeTeam"]]).days
            home_rest.append(delta if delta > 0 else DEFAULT_REST_DAYS)
        else:
            home_rest.append(DEFAULT_REST_DAYS)

        if row["AwayTeam"] in last_match:
            delta = (match_date - last_match[row["AwayTeam"]]).days
            away_rest.append(delta if delta > 0 else DEFAULT_REST_DAYS)
        else:
            away_rest.append(DEFAULT_REST_DAYS)

        last_match[row["HomeTeam"]] = match_date
        last_match[row["AwayTeam"]] = match_date

    df["Home_RestDays"] = home_rest
    df["Away_RestDays"] = away_rest

    logger.info("Rest-day features computed.")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 4.  HEAD-TO-HEAD HISTORY
# ═══════════════════════════════════════════════════════════════════════

def compute_h2h_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling H2H win-rates and goals between the fixture's two teams."""
    df = df.copy()
    h2h_hw, h2h_dr, h2h_aw = [], [], []
    h2h_hg, h2h_ag = [], []
    h2h_history: Dict[frozenset, list] = {}

    for _, row in df.iterrows():
        key = frozenset([row["HomeTeam"], row["AwayTeam"]])
        past = h2h_history.get(key, [])

        if past:
            recent = past[-window:]
            total = len(recent)
            hw = sum(
                1 for ht, r, hg, ag in recent
                if (ht == row["HomeTeam"] and r == "H")
                or (ht != row["HomeTeam"] and r == "A")
            )
            aw = sum(
                1 for ht, r, hg, ag in recent
                if (ht == row["HomeTeam"] and r == "A")
                or (ht != row["HomeTeam"] and r == "H")
            )

            home_goals = sum(
                hg if ht == row["HomeTeam"] else ag for ht, r, hg, ag in recent
            )
            away_goals = sum(
                ag if ht == row["HomeTeam"] else hg for ht, r, hg, ag in recent
            )

            dr = total - hw - aw
            h2h_hw.append(hw / total)
            h2h_dr.append(dr / total)
            h2h_aw.append(aw / total)
            h2h_hg.append(home_goals / total)
            h2h_ag.append(away_goals / total)
        else:
            h2h_hw.append(0.5)
            h2h_dr.append(0.0)
            h2h_aw.append(0.5)
            h2h_hg.append(1.0)
            h2h_ag.append(1.0)

        past.append((row["HomeTeam"], row.get("FTR", "D"), row.get("FTHG", 1), row.get("FTAG", 1)))
        h2h_history[key] = past

    df["H2H_HomeWinRate"] = h2h_hw
    df["H2H_DrawRate"] = h2h_dr
    df["H2H_AwayWinRate"] = h2h_aw
    df["H2H_HomeGoals_Mean"] = h2h_hg
    df["H2H_AwayGoals_Mean"] = h2h_ag

    logger.info("Head-to-head features computed (window=%d).", window)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 5.  MASTER PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of model-input feature columns present in *df*."""
    candidates = [
        "Home_Elo", "Away_Elo", "Elo_Diff",
        "Home_RestDays", "Away_RestDays",
        "H2H_HomeWinRate", "H2H_DrawRate", "H2H_AwayWinRate",
        "H2H_HomeGoals_Mean", "H2H_AwayGoals_Mean",
    ]
    roll_cols = sorted(c for c in df.columns if "_roll" in c)
    candidates.extend(roll_cols)
    candidates.append("Travel_Distance")
    return [c for c in candidates if c in df.columns]


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, EloEngine, list]:
    """
    Run the full feature-engineering pipeline.

    Returns (enriched_df, elo_engine, feature_columns).
    """
    logger.info("Starting feature engineering pipeline …")
    df = df.sort_values("Date").reset_index(drop=True)

    df, elo_engine = compute_elo_features(df)
    df = compute_rolling_form(df)
    df = compute_ppg_form(df)
    df = compute_venue_form(df)
    df = compute_rest_days(df)
    df = compute_h2h_features(df)

    logger.info("Computing travel distance...")
    distances = {}
    def get_dist(home, away):
        pair = (home, away)
        if pair not in distances:
            distances[pair] = calculate_travel_distance(home, away)
        return distances[pair]

    df['Travel_Distance'] = df.apply(lambda row: get_dist(row['HomeTeam'], row['AwayTeam']), axis=1)

    # FIX: geocode failures now come back as real NaN (see geography.py)
    # instead of a fake 0.0 km that's indistinguishable from a genuine
    # local derby. Fill with the dataset median rather than letting
    # dropna() silently remove those rows.
    n_missing = df['Travel_Distance'].isna().sum()
    if n_missing:
        median_dist = df['Travel_Distance'].median()
        logger.warning(
            "%d rows missing travel distance (geocode failures) — filled with median %.1f km",
            n_missing, median_dist,
        )
        df['Travel_Distance'] = df['Travel_Distance'].fillna(median_dist)

    feature_cols = get_feature_columns(df)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    before = len(df)
    df.dropna(subset=feature_cols + TARGET_COLUMNS, inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("Dropped %d incomplete rows → %d remain.", before - len(df), len(df))

    return df, elo_engine, feature_cols