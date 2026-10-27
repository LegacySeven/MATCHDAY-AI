"""
api_sync.py — Keeps the historical dataset current by fetching recent
completed fixtures from API-Football.

The football-data.co.uk CSVs we train on can lag days or weeks behind.
This module bridges that gap:

  1. Detects the last date in our existing dataset
  2. Fetches all completed fixtures since then from API-Football
  3. Fetches per-match statistics (shots, corners) for each fixture
  4. Converts everything into the exact same DataFrame format as our CSVs
  5. Appends the new rows so Elo / form / rest days are truly current

Typical cost: ~15-30 API requests per sync (1 per league for fixtures,
plus 1 per new match for statistics). Well within the 100/day free quota
if run once daily.
"""

import logging
from datetime import date as date_cls, timedelta
from typing import Optional

import pandas as pd

from src.api_football import (
    get_completed_fixtures,
    get_fixture_statistics,
    LEAGUE_IDS,
)
from src.config import RAW_COLUMNS

logger = logging.getLogger(__name__)

# Map API-Football league IDs back to our internal league names
_API_ID_TO_NAME = {v: k for k, v in LEAGUE_IDS.items()}

# Map API-Football season year to our season code format
def _season_to_code(year: int) -> str:
    """Convert API-Football season year (e.g. 2024) to our code (e.g. '2425')."""
    short_start = str(year)[-2:]
    short_end = str(year + 1)[-2:]
    return f"{short_start}{short_end}"


def _extract_stat(team_stats: list, stat_name: str) -> Optional[int]:
    """
    Extract a single stat value from API-Football's statistics response.

    The stats come as a list of {"type": "...", "value": ...} dicts.
    Returns None if the stat isn't found.
    """
    for item in team_stats:
        if item.get("type") == stat_name:
            val = item.get("value")
            if val is not None:
                return int(val) if isinstance(val, (int, float)) else None
            return None
    return None


def _fixture_to_row(fixture: dict, match_stats: list, league_name: str) -> Optional[dict]:
    """
    Convert a single API-Football fixture + its statistics into a dict
    matching our CSV column format.

    Returns None if the fixture can't be converted (e.g. missing data).
    """
    try:
        teams = fixture["teams"]
        goals = fixture["goals"]
        score = fixture["score"]
        fixture_info = fixture["fixture"]

        home_team = teams["home"]["name"]
        away_team = teams["away"]["name"]
        home_goals = goals["home"]
        away_goals = goals["away"]

        if home_goals is None or away_goals is None:
            return None

        # Full-time result
        if home_goals > away_goals:
            ftr = "H"
        elif home_goals < away_goals:
            ftr = "A"
        else:
            ftr = "D"

        # Parse date
        match_date = fixture_info["date"][:10]  # "2025-03-15T20:00:00+00:00" -> "2025-03-15"

        # Season code
        season_year = fixture["league"]["season"]
        season_code = _season_to_code(season_year)

        # Extract statistics (shots, corners, etc.)
        home_stats = []
        away_stats = []
        for block in match_stats:
            if block["team"]["id"] == teams["home"]["id"]:
                home_stats = block.get("statistics", [])
            elif block["team"]["id"] == teams["away"]["id"]:
                away_stats = block.get("statistics", [])

        row = {
            "Date": pd.Timestamp(match_date),
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "FTHG": int(home_goals),
            "FTAG": int(away_goals),
            "FTR": ftr,
            "HS": _extract_stat(home_stats, "Total Shots"),
            "AS": _extract_stat(away_stats, "Total Shots"),
            "HST": _extract_stat(home_stats, "Shots on Goal"),
            "AST": _extract_stat(away_stats, "Shots on Goal"),
            "HC": _extract_stat(home_stats, "Corner Kicks"),
            "AC": _extract_stat(away_stats, "Corner Kicks"),
            "League": league_name,
            "Season": season_code,
        }
        return row

    except (KeyError, TypeError) as e:
        logger.warning("Could not parse fixture: %s", e)
        return None


def sync_recent_fixtures(
    existing_df: pd.DataFrame,
    max_days_back: int = 30,
) -> pd.DataFrame:
    """
    Fetch completed fixtures since the last date in existing_df and
    append them. Returns the updated DataFrame.

    Args:
        existing_df: The current dataset (from load_all_data()).
        max_days_back: Safety cap — don't look further back than this.

    Returns:
        Updated DataFrame with new rows appended and sorted by date.
    """
    last_date = existing_df["Date"].max()
    # Start from the day after the last match we have
    from_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    to_date = date_cls.today().isoformat()

    logger.info(
        "Data sync: last match in dataset = %s, fetching %s → %s",
        last_date.strftime("%Y-%m-%d"), from_date, to_date,
    )

    if pd.Timestamp(from_date) > pd.Timestamp(to_date):
        logger.info("Dataset is already up to date!")
        return existing_df

    new_rows = []
    total_api_calls = 0

    for league_name in LEAGUE_IDS:
        try:
            fixtures = get_completed_fixtures(league_name, from_date, to_date)
            total_api_calls += 1
            logger.info(
                "%s: %d completed fixtures since %s",
                league_name, len(fixtures), from_date,
            )

            for fx in fixtures:
                fixture_id = fx["fixture"]["id"]

                # Fetch per-match statistics (shots, corners, etc.)
                try:
                    stats = get_fixture_statistics(fixture_id)
                    total_api_calls += 1
                except Exception as e:
                    logger.warning("Could not fetch stats for fixture %d: %s", fixture_id, e)
                    stats = []

                row = _fixture_to_row(fx, stats, league_name)
                if row is not None:
                    new_rows.append(row)

        except Exception as e:
            logger.error("Failed to sync %s: %s", league_name, e)

    logger.info(
        "Data sync complete: %d new matches fetched (%d API calls used)",
        len(new_rows), total_api_calls,
    )

    if not new_rows:
        return existing_df

    new_df = pd.DataFrame(new_rows)

    # Deduplicate: avoid adding matches that might already exist
    # (in case CSV was partially updated)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    before = len(combined)
    combined.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam"],
        keep="first",
        inplace=True,
    )
    dupes = before - len(combined)
    if dupes:
        logger.info("Removed %d duplicate rows after merge.", dupes)

    combined.sort_values("Date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info(
        "Updated dataset: %d matches (was %d, +%d new)",
        len(combined), len(existing_df), len(combined) - len(existing_df),
    )
    return combined
