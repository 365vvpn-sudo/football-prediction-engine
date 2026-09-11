import math
from typing import Dict


class PoissonMath:
    """
    Mathematical engine for Poisson goal probabilities.
    """

    @staticmethod
    def probability(goals: int, expected_goals: float) -> float:
        if goals < 0:
            raise ValueError("goals must be >= 0")

        if expected_goals < 0:
            raise ValueError("expected_goals must be >= 0")

        return (
            math.exp(-expected_goals)
            * (expected_goals ** goals)
            / math.factorial(goals)
        )

    @classmethod
    def distribution(
        cls,
        expected_goals: float,
        max_goals: int = 10,
    ) -> Dict[int, float]:

        if max_goals < 0:
            raise ValueError("max_goals must be >= 0")

        return {
            goals: cls.probability(goals, expected_goals)
            for goals in range(max_goals + 1)
        }
