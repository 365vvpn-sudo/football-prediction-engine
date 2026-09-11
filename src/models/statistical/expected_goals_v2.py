from dataclasses import dataclass

from src.features.league_baseline import LeagueBaseline
from src.features.strength_engine import TeamStrength


@dataclass
class ExpectedGoalsV2:
    home: float
    away: float


class ExpectedGoalsEngineV2:
    VERSION = "2.0.1"

    def __init__(self, strength_weight: float = 1.0) -> None:
        if not 0.0 <= strength_weight <= 1.0:
            raise ValueError(
                "strength_weight must be between 0 and 1"
            )

        self.strength_weight = strength_weight

    @staticmethod
    def _blend_strength(
        strength: float,
        weight: float,
    ) -> float:
        """
        Controls how strongly team strength affects the baseline.

        weight = 0 -> neutral strength (1.0)
        weight = 1 -> full strength effect
        """
        return 1.0 + ((strength - 1.0) * weight)

    def calculate(
        self,
        home_strength: TeamStrength,
        away_strength: TeamStrength,
        baseline: LeagueBaseline,
    ) -> ExpectedGoalsV2:

        home_attack = self._blend_strength(
            home_strength.home_attack_strength,
            self.strength_weight,
        )

        away_defense = self._blend_strength(
            away_strength.away_defense_strength,
            self.strength_weight,
        )

        away_attack = self._blend_strength(
            away_strength.away_attack_strength,
            self.strength_weight,
        )

        home_defense = self._blend_strength(
            home_strength.home_defense_strength,
            self.strength_weight,
        )

        home_expected = (
            baseline.home_goals_per_match
            * home_attack
            * away_defense
        )

        away_expected = (
            baseline.away_goals_per_match
            * away_attack
            * home_defense
        )

        return ExpectedGoalsV2(
            home=max(home_expected, 0.0),
            away=max(away_expected, 0.0),
        )
