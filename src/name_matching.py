from typing import Dict, List, Optional
from thefuzz import process

MANUAL_OVERRIDES: Dict[str, str] = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Brighton & Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest",
    "West Ham United": "West Ham",
    "Paris Saint Germain": "Paris SG",
}

FUZZY_THRESHOLD = 85  # below this, return None rather than guess — a wrong match
                      # silently feeds the wrong team's Elo/form into the prediction


def match_team_name(api_name: str, known_names: List[str]) -> Optional[str]:
    if api_name in MANUAL_OVERRIDES:
        mapped = MANUAL_OVERRIDES[api_name]
        return mapped if mapped in known_names else None

    if not known_names:
        return None

    best, score = process.extractOne(api_name, known_names)
    return best if score >= FUZZY_THRESHOLD else None
