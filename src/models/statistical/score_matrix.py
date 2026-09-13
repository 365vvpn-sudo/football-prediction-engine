from typing import Dict, Tuple

from src.models.statistical.poisson_math import PoissonMath


class ScoreMatrix:
    """
    Builds the joint probability matrix for two count statistics
    (e.g. home/away goals or shots), assuming independent Poisson
    distributions, with an optional Dixon-Coles correction for the
    slight negative correlation observed in real low-scoring
    outcomes (0-0, 1-0, 0-1, 1-1). dixon_coles_rho=0.0 (default)
    disables the correction and reproduces plain independent
    Poisson behaviour.
    """

    @staticmethod
    def _dixon_coles_tau(
        home_count: int,
        away_count: int,
        home_expected: float,
        away_expected: float,
        rho: float,
    ) -> float:
        if home_count == 0 and away_count == 0:
            return 1.0 - (home_expected * away_expected * rho)
        if home_count == 0 and away_count == 1:
            return 1.0 + (home_expected * rho)
        if home_count == 1 and away_count == 0:
            return 1.0 + (away_expected * rho)
        if home_count == 1 and away_count == 1:
            return 1.0 - rho
        return 1.0

    @classmethod
    def build(
        cls,
        home_expected_goals: float,
        away_expected_goals: float,
        max_goals: int = 10,
        dixon_coles_rho: float = 0.0,
    ) -> Dict[Tuple[int, int], float]:

        if home_expected_goals < 0:
            raise ValueError("home_expected_goals must be >= 0")
        if away_expected_goals < 0:
            raise ValueError("away_expected_goals must be >= 0")
        if max_goals < 0:
            raise ValueError("max_goals must be >= 0")
        if not -1.0 <= dixon_coles_rho <= 1.0:
            raise ValueError("dixon_coles_rho must be between -1 and 1")

        home_distribution = PoissonMath.distribution(home_expected_goals, max_goals)
        away_distribution = PoissonMath.distribution(away_expected_goals, max_goals)

        matrix: Dict[Tuple[int, int], float] = {}

        for home_goals, home_probability in home_distribution.items():
            for away_goals, away_probability in away_distribution.items():
                probability = home_probability * away_probability

                if dixon_coles_rho != 0.0:
                    tau = cls._dixon_coles_tau(
                        home_goals, away_goals,
                        home_expected_goals, away_expected_goals,
                        dixon_coles_rho,
                    )
                    probability *= tau

                matrix[(home_goals, away_goals)] = probability

        if dixon_coles_rho != 0.0:
            total = sum(matrix.values())
            if total > 0:
                matrix = {key: value / total for key, value in matrix.items()}

        return matrix
