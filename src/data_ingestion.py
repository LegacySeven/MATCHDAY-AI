"""
data_ingestion.py — Automated download and loading of historical match data
from Football-Data.co.uk.

Downloads CSV files for each (season, league) pair, caches them in
``data/raw/``, and returns a single consolidated DataFrame.
"""

import io
import os
import logging
from typing import Optional

import pandas as pd
import requests

from src.config import (
    BASE_URL,
    LEAGUE_CODES,
    SEASON_CODES,
    SEASON_LABELS,
    RAW_COLUMNS,
    DATA_RAW_DIR,
)

logger = logging.getLogger(__name__)


def _build_url(season: str, league_code: str) -> str:
    """Construct the download URL for a given season and league."""
    return BASE_URL.format(season=season, league_code=league_code)


def _download_csv(url: str, cache_path: str) -> pd.DataFrame:
    """
    Download a CSV from *url*.  If *cache_path* already exists on disk,
    load from cache instead.
    """
    if os.path.exists(cache_path):
        logger.info("Cache hit: %s", cache_path)
        return pd.read_csv(cache_path, encoding="utf-8", encoding_errors="replace")

    logger.info("Downloading %s …", url)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as fh:
        fh.write(response.content)

    return pd.read_csv(
        io.StringIO(response.content.decode("utf-8", errors="replace"))
    )


def _parse_date(date_series: pd.Series) -> pd.Series:
    """Robustly parse dates in dd/mm/yyyy or dd/mm/yy format."""
    return pd.to_datetime(date_series, dayfirst=True, format="mixed", errors="coerce")


def _clean_dataframe(df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """
    Keep only needed columns, parse dates, coerce numerics, drop
    incomplete rows, and attach league / season metadata.
    """
    available = [c for c in RAW_COLUMNS if c in df.columns]
    missing = set(RAW_COLUMNS) - set(available)
    if missing:
        logger.warning("Missing columns %s in %s / %s — will be NaN", missing, league, season)

    df = df[available].copy()
    df["Date"] = _parse_date(df["Date"])

    numeric_cols = ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    essential = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    df.dropna(subset=[c for c in essential if c in df.columns], inplace=True)

    df["League"] = league
    df["Season"] = season
    return df


def load_all_data(
    leagues: Optional[dict] = None,
    seasons: Optional[list] = None,
) -> pd.DataFrame:
    """
    Download and consolidate match data across all leagues and seasons.

    Returns a single DataFrame sorted chronologically with a clean index.
    """
    leagues = leagues or LEAGUE_CODES
    seasons = seasons or SEASON_CODES

    frames = []
    for league_name, league_code in leagues.items():
        for season in seasons:
            url = _build_url(season, league_code)
            cache_path = os.path.join(DATA_RAW_DIR, f"{league_code}_{season}.csv")

            try:
                raw = _download_csv(url, cache_path)
                cleaned = _clean_dataframe(raw, league_name, season)
                frames.append(cleaned)
                logger.info(
                    "Loaded %d matches — %s %s",
                    len(cleaned), league_name, SEASON_LABELS.get(season, season),
                )
            except requests.HTTPError as exc:
                logger.error("HTTP error for %s %s: %s", league_name, season, exc)
            except Exception as exc:
                logger.error("Failed to load %s %s: %s", league_name, season, exc)

    if not frames:
        raise RuntimeError("No data loaded — check network and URLs.")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("Date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    logger.info(
        "Total dataset: %d matches  |  %d leagues  |  %d seasons",
        len(combined), len(leagues), len(seasons),
    )
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_all_data()
    print(f"\nShape: {df.shape}")
    print(df.head(10).to_string(index=False))
