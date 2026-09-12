from dataclasses import dataclass

from src.features.league_baseline import LeagueBaseline
from src.features.strength_engine import TeamStrength


@dataclass
class ExpectedValue:
    home: float
    away: float


class ExpectedValueEngine:
    """
    Generic point-in-time expected-value model: works for goals,
    shots, or any other count statistic, given the matching
    TeamStrength/LeagueBaseline pair.

    Formula: baseline_rate * attack_strength * opponent_defense_strength
    """

    VERSION = "3.0.0"

    def __init__(self, strength_weight: float = 1.0) -> None:
        if not 0.0 <= strength_weight <= 1.0:
            raise ValueError("strength_weight must be between 0 and 1")
        self.strength_weight = strength_weight

    @staticmethod
    def _blend(strength: float, weight: float) -> float:
        return 1.0 + ((strength - 1.0) * weight)

    def calculate(
        self,
        home_strength: TeamStrength,
        away_strength: TeamStrength,
        baseline: LeagueBaseline,
    ) -> ExpectedValue:

        home_attack = self._blend(home_strength.home_attack_strength, self.strength_weight)
        away_defense = self._blend(away_strength.away_defense_strength, self.strength_weight)

        away_attack = self._blend(away_strength.away_attack_strength, self.strength_weight)
        home_defense = self._blend(home_strength.home_defense_strength, self.strength_weight)

        home_expected = baseline.home_per_match * home_attack * away_defense
        away_expected = baseline.away_per_match * away_attack * home_defense

        return ExpectedValue(
            home=max(home_expected, 0.0),
            away=max(away_expected, 0.0),
        )
