import math
from typing import Dict, Tuple

class EloEngine:
    """
    Chronological Elo rating tracker with Margin of Victory (MoV) adjustments.

    Home advantage is applied only when computing expected scores;
    the stored rating is the *neutral* rating.
    """

    def __init__(
        self,
        k: float = 25.0,
        home_adv: float = 75.0,
        initial: float = 1500.0,
    ):
        self.k = k
        self.home_adv = home_adv
        self.initial = initial
        self._ratings: Dict[str, float] = {}

    def _get(self, team: str) -> float:
        return self._ratings.setdefault(team, self.initial)

    @staticmethod
    def _expected(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def _actual_score(goals_a: int, goals_b: int) -> float:
        if goals_a > goals_b:
            return 1.0
        if goals_a < goals_b:
            return 0.0
        return 0.5

    @staticmethod
    def _mov_multiplier(goals_a: int, goals_b: int) -> float:
        """
        Margin of Victory multiplier based on World Football Elo Ratings.
        - 0 or 1 goal difference: multiplier = 1
        - 2 goals difference: multiplier = 1.5
        - 3 or more: multiplier = (11 + diff) / 8
        """
        diff = abs(goals_a - goals_b)
        if diff <= 1:
            return 1.0
        elif diff == 2:
            return 1.5
        else:
            return (11.0 + diff) / 8.0

    def update(
        self, home: str, away: str, home_goals: int, away_goals: int
    ) -> Tuple[float, float, float, float]:
        """
        Update ratings after one match, factoring in goal difference.

        Returns (home_elo_before, away_elo_before, home_elo_after, away_elo_after).
        """
        r_home = self._get(home)
        r_away = self._get(away)

        e_home = self._expected(r_home + self.home_adv, r_away)
        e_away = 1.0 - e_home

        s_home = self._actual_score(home_goals, away_goals)
        s_away = 1.0 - s_home
        
        mov = self._mov_multiplier(home_goals, away_goals)

        # Elo update logic: Rating_new = Rating_old + K * MoV * (Result - Expected)
        new_home = r_home + self.k * mov * (s_home - e_home)
        new_away = r_away + self.k * mov * (s_away - e_away)

        self._ratings[home] = new_home
        self._ratings[away] = new_away

        return r_home, r_away, new_home, new_away

    def get_rating(self, team: str) -> float:
        """Return current neutral Elo for *team*."""
        return self._get(team)

    @property
    def ratings(self) -> Dict[str, float]:
        return dict(self._ratings)