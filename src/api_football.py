"""
api_football.py — API-Football client with aggressive local JSON caching.

Handles:
  - Fetching today's/upcoming fixtures for value-bet comparisons
  - Fetching completed fixtures to keep the dataset current (data sync)
  - Fetching pre-match bookmaker odds
  - Protecting the 100 req/day free-tier quota via file-based caching
"""

import os
import json
import logging
from datetime import date as date_cls, timedelta
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_FOOTBALL_KEY")
API_HOST = os.getenv("API_FOOTBALL_HOST", "v3.football.api-sports.io")
BASE_URL = f"https://{API_HOST}"
HEADERS = {"x-apisports-key": API_KEY}

CACHE_DIR = "data/api_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# League ID mapping — covers all 15 leagues we model, plus the top 5
# are prioritised for live odds (free-tier quota protection).
# ─────────────────────────────────────────────────────────────────────

LEAGUE_IDS: Dict[str, int] = {
    # Top 5
    "EPL": 39,
    "La Liga": 140,
    "Bundesliga": 78,
    "Serie A": 135,
    "Ligue 1": 61,
    # Additional modelled leagues
    "Eredivisie": 88,
    "Liga Portugal": 94,
    "Süper Lig": 203,
    "Belgian Pro League": 144,
    "Super League (Greece)": 197,
    "Scottish Premiership": 179,
    "English Championship": 40,
    "Czech First League": 345,
    "Ekstraklasa": 106,
    "Danish Superliga": 119,
}

# Subset for live odds (top 5 only — protects free quota)
ODDS_LEAGUE_IDS: Dict[str, int] = {
    k: v for k, v in LEAGUE_IDS.items()
    if k in ("EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1")
}


def _season_year(iso_date: str) -> int:
    """European seasons run Aug-May; Jan-Jul dates belong to the previous year's season code."""
    y, m, _ = map(int, iso_date.split("-"))
    return y if m >= 8 else y - 1


def _cached_get(url: str, params: dict, cache_key: str, force_refresh: bool = False) -> dict:
    """GET with file-based JSON cache. Set force_refresh=True to bypass cache."""
    path = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(path) and not force_refresh:
        logger.info("Cache hit: %s", path)
        with open(path) as fh:
            return json.load(fh)

    if not API_KEY:
        raise RuntimeError("API_FOOTBALL_KEY not set — check your .env file")

    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Log remaining quota
    remaining = resp.headers.get("x-ratelimit-requests-remaining", "?")
    logger.info("API request OK — %s remaining today", remaining)

    if data.get("errors"):
        logger.warning("API-Football returned errors: %s", data["errors"])

    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    return data


# ═══════════════════════════════════════════════════════════════════════
# 1. Upcoming fixtures (for value-bet predictions)
# ═══════════════════════════════════════════════════════════════════════

def get_fixtures_today(league_name: str, target_date: Optional[str] = None) -> List[dict]:
    """Fetch fixtures for a specific date (defaults to today)."""
    league_id = LEAGUE_IDS.get(league_name)
    if league_id is None:
        raise ValueError(f"Unsupported league: {league_name}")

    target_date = target_date or date_cls.today().isoformat()
    cache_key = f"fixtures_{league_id}_{target_date}"
    data = _cached_get(
        f"{BASE_URL}/fixtures",
        params={"league": league_id, "date": target_date, "season": _season_year(target_date)},
        cache_key=cache_key,
    )
    return data.get("response", [])


# ═══════════════════════════════════════════════════════════════════════
# 2. Completed fixtures (for data sync — keeping dataset current)
# ═══════════════════════════════════════════════════════════════════════

def get_completed_fixtures(
    league_name: str,
    from_date: str,
    to_date: Optional[str] = None,
) -> List[dict]:
    """
    Fetch all FINISHED fixtures for a league between from_date and to_date.

    Uses the /fixtures endpoint with status=FT-AET-PEN to only get
    completed matches (Full Time, After Extra Time, Penalties).
    Results are cached per league+date-range.
    """
    league_id = LEAGUE_IDS.get(league_name)
    if league_id is None:
        raise ValueError(f"Unsupported league: {league_name}")

    to_date = to_date or date_cls.today().isoformat()
    season = _season_year(from_date)
    cache_key = f"completed_{league_id}_{from_date}_{to_date}"

    data = _cached_get(
        f"{BASE_URL}/fixtures",
        params={
            "league": league_id,
            "season": season,
            "from": from_date,
            "to": to_date,
            "status": "FT-AET-PEN",
        },
        cache_key=cache_key,
        force_refresh=True,  # Always fetch fresh for sync
    )
    return data.get("response", [])


def get_fixture_statistics(fixture_id: int) -> List[dict]:
    """
    Fetch per-team match statistics (shots, corners, etc.) for a
    completed fixture. Returns list of team stat blocks.
    """
    data = _cached_get(
        f"{BASE_URL}/fixtures/statistics",
        params={"fixture": fixture_id},
        cache_key=f"stats_{fixture_id}",
    )
    return data.get("response", [])


# ═══════════════════════════════════════════════════════════════════════
# 3. Odds (for value-bet comparison)
# ═══════════════════════════════════════════════════════════════════════

def get_odds_for_fixture(fixture_id: int) -> List[dict]:
    """Fetch bookmaker pre-match odds for a fixture."""
    data = _cached_get(
        f"{BASE_URL}/odds",
        params={"fixture": fixture_id},
        cache_key=f"odds_{fixture_id}",
    )
    response = data.get("response", [])
    return response[0].get("bookmakers", []) if response else []
