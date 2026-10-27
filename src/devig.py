from typing import Dict, List


def odds_to_implied_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def devig_market(decimal_odds: Dict[str, float]) -> Dict[str, float]:
    """
    Bookmaker odds always imply >100% total probability (their margin,
    the 'overround'). Rescale so the outcomes sum to exactly 100% before
    comparing against model probabilities — otherwise the model looks
    artificially better than it is on every single market.
    """
    raw = {k: odds_to_implied_prob(v) for k, v in decimal_odds.items()}
    overround = sum(raw.values())
    return {k: v / overround for k, v in raw.items()}


def average_bookmaker_probs(per_bookmaker_odds: List[Dict[str, float]]) -> Dict[str, float]:
    """Average de-vigged probabilities across bookmakers so one stale/skewed line doesn't dominate."""
    if not per_bookmaker_odds:
        raise ValueError("No bookmaker odds provided")
    devigged = [devig_market(o) for o in per_bookmaker_odds]
    keys = devigged[0].keys()
    return {k: sum(d[k] for d in devigged) / len(devigged) for k in keys}
