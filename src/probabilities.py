import math
from typing import Tuple, List

def poisson_prob(lam: float, k: int) -> float:
    """Calculates the Poisson probability of exactly k events given a rate of lam."""
    # Handle negative lambdas gracefully just in case model predicts slightly negative
    lam = max(0.01, lam)
    return (math.pow(lam, k) * math.exp(-lam)) / math.factorial(k)

def compute_poisson_grid(home_lambda: float, away_lambda: float, max_goals: int = 10) -> List[List[float]]:
    """
    Computes a 2D grid of probabilities for scorelines from 0-0 up to max_goals-max_goals.
    grid[h][a] gives the probability of Home scoring h goals and Away scoring a goals.
    """
    grid = []
    for h in range(max_goals + 1):
        row = []
        for a in range(max_goals + 1):
            prob = poisson_prob(home_lambda, h) * poisson_prob(away_lambda, a)
            row.append(prob)
        grid.append(row)
    return grid

def get_match_odds(grid: List[List[float]]) -> Tuple[float, float, float]:
    """Returns probabilities for (Home Win, Draw, Away Win)."""
    hw, d, aw = 0.0, 0.0, 0.0
    for h in range(len(grid)):
        for a in range(len(grid[h])):
            prob = grid[h][a]
            if h > a:
                hw += prob
            elif h == a:
                d += prob
            else:
                aw += prob
    return hw, d, aw

def get_over_under(grid: List[List[float]], threshold: float = 2.5) -> Tuple[float, float]:
    """Returns probabilities for (Over, Under) a specific goals threshold."""
    over, under = 0.0, 0.0
    for h in range(len(grid)):
        for a in range(len(grid[h])):
            prob = grid[h][a]
            if h + a > threshold:
                over += prob
            else:
                under += prob
    return over, under

def get_btts(grid: List[List[float]]) -> Tuple[float, float]:
    """Returns probabilities for Both Teams to Score (Yes, No)."""
    btts_yes, btts_no = 0.0, 0.0
    for h in range(len(grid)):
        for a in range(len(grid[h])):
            prob = grid[h][a]
            if h > 0 and a > 0:
                btts_yes += prob
            else:
                btts_no += prob
    return btts_yes, btts_no
