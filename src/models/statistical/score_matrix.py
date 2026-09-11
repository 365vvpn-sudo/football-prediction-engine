from typing import Dict, Tuple

from src.models.statistical.poisson_math import PoissonMath


class ScoreMatrix:
    """
    Builds the joint probability matrix for home and away goals.
    """

    @classmethod
    def build(
        cls,
        home_expected_goals: float,
        away_expected_goals: float,
        max_goals: int = 10,
    ) -> Dict[Tuple[int, int], float]:

        if home_expected_goals < 0:
            raise ValueError("home_expected_goals must be >= 0")

        if away_expected_goals < 0:
            raise ValueError("away_expected_goals must be >= 0")

        if max_goals < 0:
            raise ValueError("max_goals must be >= 0")

        home_distribution = PoissonMath.distribution(
            home_expected_goals,
            max_goals,
        )

        away_distribution = PoissonMath.distribution(
            away_expected_goals,
            max_goals,
        )

        matrix = {}

        for home_goals, home_probability in home_distribution.items():
            for away_goals, away_probability in away_distribution.items():
                matrix[(home_goals, away_goals)] = (
                    home_probability * away_probability
                )

        return matrix
