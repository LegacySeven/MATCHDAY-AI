"""
inference.py — Standalone inference module.

``predict_fixture(home_team, away_team, league, date)`` computes live
Elo / form / rest / H2H features from the historical data and returns
predictions for all 6 targets via the best-performing model.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import (
    TARGET_COLUMNS,
    TARGET_LABELS,
    ROLLING_WINDOW,
    DEFAULT_REST_DAYS,
)
from src.feature_engineering import EloEngine

logger = logging.getLogger(__name__)


class MatchPredictor:
    """
    Encapsulates inference state: enriched dataset, Elo engine,
    feature columns, and the trained model.
    """

    def __init__(
        self,
        enriched_df: pd.DataFrame,
        elo_engine: EloEngine,
        feature_cols: list,
        model,
        model_name: str = "Best",
    ):
        self.df = enriched_df.copy()
        self.elo = elo_engine
        self.feature_cols = feature_cols
        self.model = model
        self.model_name = model_name

    # ── Rolling form ──────────────────────────────────────────────────

    def _rolling_form(self, team: str, venue: str) -> Dict[str, float]:
        form = {}

        mask_all = (self.df["HomeTeam"] == team) | (self.df["AwayTeam"] == team)
        subset_all = self.df.loc[mask_all].sort_values("Date").tail(ROLLING_WINDOW)

        pts_all = []
        for _, row in subset_all.iterrows():
            if row["HomeTeam"] == team:
                pts_all.append(3 if row["FTR"] == "H" else 1 if row["FTR"] == "D" else 0)
            else:
                pts_all.append(3 if row["FTR"] == "A" else 1 if row["FTR"] == "D" else 0)

        ppg_overall = np.mean(pts_all) if pts_all else 1.0

        if venue == "home":
            form[f"Home_PPG_roll{ROLLING_WINDOW}"] = ppg_overall
            mask = self.df["HomeTeam"] == team
            stat_map = {
                "FTHG": "Home_GF", "FTAG": "Home_GA",
                "HST": "Home_SoT_For", "AST": "Home_SoT_Against",
                "HC": "Home_Corners_For", "AC": "Home_Corners_Against",
            }
        else:
            form[f"Away_PPG_roll{ROLLING_WINDOW}"] = ppg_overall
            mask = self.df["AwayTeam"] == team
            stat_map = {
                "FTAG": "Away_GF", "FTHG": "Away_GA",
                "AST": "Away_SoT_For", "HST": "Away_SoT_Against",
                "AC": "Away_Corners_For", "HC": "Away_Corners_Against",
            }

        subset_venue = self.df.loc[mask].sort_values("Date").tail(ROLLING_WINDOW)

        for src_col, feat_prefix in stat_map.items():
            col_name = f"{feat_prefix}_roll{ROLLING_WINDOW}"
            if src_col in subset_venue.columns and not subset_venue[src_col].isna().all():
                form[col_name] = subset_venue[src_col].mean()
            else:
                form[col_name] = 0.0

        # Venue-specific PPG only — matches the fixed compute_venue_form(),
        # which no longer duplicates GF/GA under a second name.
        pts_venue = []
        for _, row in subset_venue.iterrows():
            if venue == "home":
                pts_venue.append(3 if row["FTR"] == "H" else 1 if row["FTR"] == "D" else 0)
            else:
                pts_venue.append(3 if row["FTR"] == "A" else 1 if row["FTR"] == "D" else 0)

        if venue == "home":
            form[f"Home_PPG_Home_roll{ROLLING_WINDOW}"] = np.mean(pts_venue) if pts_venue else 1.0
        else:
            form[f"Away_PPG_Away_roll{ROLLING_WINDOW}"] = np.mean(pts_venue) if pts_venue else 1.0

        return form

    # ── Rest days ─────────────────────────────────────────────────────

    def _rest_days(self, team: str, match_date: datetime) -> int:
        mask = (self.df["HomeTeam"] == team) | (self.df["AwayTeam"] == team)
        past = self.df.loc[mask & (self.df["Date"] < pd.Timestamp(match_date))]
        if past.empty:
            return DEFAULT_REST_DAYS
        last = past["Date"].max()
        delta = (pd.Timestamp(match_date) - last).days
        return delta if delta > 0 else DEFAULT_REST_DAYS

    # ── Head-to-head ──────────────────────────────────────────────────

    def _h2h(self, home: str, away: str) -> Dict[str, float]:
        window = 10
        mask = (
            ((self.df["HomeTeam"] == home) & (self.df["AwayTeam"] == away))
            | ((self.df["HomeTeam"] == away) & (self.df["AwayTeam"] == home))
        )
        past = self.df.loc[mask].tail(window)
        if past.empty:
            return {
                "H2H_HomeWinRate": 0.5,
                "H2H_DrawRate": 0.0,
                "H2H_AwayWinRate": 0.5,
                "H2H_HomeGoals_Mean": 1.0,
                "H2H_AwayGoals_Mean": 1.0,
            }

        hw, aw, dr = 0, 0, 0
        h_goals, a_goals = 0, 0
        for _, row in past.iterrows():
            ftr = row.get("FTR", "D")
            hg = row.get("FTHG", 1)
            ag = row.get("FTAG", 1)

            if row["HomeTeam"] == home:
                if ftr == "H": hw += 1
                elif ftr == "A": aw += 1
                else: dr += 1
                h_goals += hg
                a_goals += ag
            else:
                if ftr == "H": aw += 1
                elif ftr == "A": hw += 1
                else: dr += 1
                h_goals += ag
                a_goals += hg

        total = hw + aw + dr
        return {
            "H2H_HomeWinRate": hw / total,
            "H2H_DrawRate": dr / total,
            "H2H_AwayWinRate": aw / total,
            "H2H_HomeGoals_Mean": h_goals / total,
            "H2H_AwayGoals_Mean": a_goals / total,
        }

    # ── Predict ───────────────────────────────────────────────────────

    def predict_fixture(
        self,
        home_team: str,
        away_team: str,
        league: str = "",
        date: Optional[str] = None,
    ) -> Dict[str, float]:
        match_date = datetime.now() if date is None else datetime.fromisoformat(date)

        features: Dict[str, float] = {}

        features["Home_Elo"] = self.elo.get_rating(home_team)
        features["Away_Elo"] = self.elo.get_rating(away_team)
        features["Elo_Diff"] = features["Home_Elo"] - features["Away_Elo"]

        features["Home_RestDays"] = self._rest_days(home_team, match_date)
        features["Away_RestDays"] = self._rest_days(away_team, match_date)

        features.update(self._rolling_form(home_team, "home"))
        features.update(self._rolling_form(away_team, "away"))

        features.update(self._h2h(home_team, away_team))

        X = np.array([[features.get(c, 0.0) for c in self.feature_cols]])

        y_pred = self.model.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)

        result = {}
        for i, col in enumerate(TARGET_COLUMNS):
            label = TARGET_LABELS.get(col, col)
            result[label] = round(float(y_pred[0, i]), 2)

        return result

    def get_known_teams(self) -> list:
        """All team names present in the historical dataset — used to
        validate/fuzzy-match incoming API-Football fixture team names."""
        return sorted(set(self.df["HomeTeam"]).union(self.df["AwayTeam"]))


# ═══════════════════════════════════════════════════════════════════════
# Module-level convenience API
# ═══════════════════════════════════════════════════════════════════════

_predictor: Optional[MatchPredictor] = None


def init_predictor(
    enriched_df: pd.DataFrame,
    elo_engine: EloEngine,
    feature_cols: list,
    model,
    model_name: str = "Best",
) -> None:
    """Initialise the module-level predictor singleton."""
    global _predictor
    _predictor = MatchPredictor(
        enriched_df, elo_engine, feature_cols, model, model_name
    )


def predict_fixture(
    home_team: str,
    away_team: str,
    league: str = "",
    date: Optional[str] = None,
) -> Dict[str, float]:
    """
    Public entry point.

    Requires ``init_predictor()`` to have been called first
    (done automatically by ``main.py``).
    """
    if _predictor is None:
        raise RuntimeError(
            "Predictor not initialised. Call init_predictor() or run main.py first."
        )
    return _predictor.predict_fixture(home_team, away_team, league, date)


def get_known_teams() -> list:
    """Public entry point mirroring predict_fixture() — used by live_predictions.py."""
    if _predictor is None:
        raise RuntimeError(
            "Predictor not initialised. Call init_predictor() or run main.py first."
        )
    return _predictor.get_known_teams()
