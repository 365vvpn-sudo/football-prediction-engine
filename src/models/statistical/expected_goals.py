from dataclasses import dataclass

from src.features.match_features import MatchFeatures


@dataclass
class ExpectedGoals:
    home: float
    away: float


class ExpectedGoalsEngine:
    """
    Calculates expected goals from point-in-time team features.
    """

    VERSION = "1.1.0"

    @staticmethod
    def _safe_average(values: list[float]) -> float:
        valid = [
            value for value in values
            if value is not None
        ]

        if not valid:
            return 0.0

        return sum(valid) / len(valid)

    def calculate(
        self,
        features: MatchFeatures,
    ) -> ExpectedGoals:

        home_expected = self._safe_average(
            [
                features.home_goals_for,
                features.away_goals_against,
            ]
        )

        away_expected = self._safe_average(
            [
                features.away_goals_for,
                features.home_goals_against,
            ]
        )

        return ExpectedGoals(
            home=max(home_expected, 0.0),
            away=max(away_expected, 0.0),
        )
