from typing import Dict, List, Optional

from src.models.statistical.score_matrix import ScoreMatrix


class ShotMarketEngine:
    """
    Converts expected home/away shots into shot-total market
    probabilities (Over/Under), using the same independent-Poisson
    joint-matrix approach as goals.

    NOTE (known limitation): real shot counts are typically more
    over-dispersed than a Poisson distribution assumes. This is a
    reasonable first approximation; a Negative Binomial model is a
    natural future upgrade specifically for this market.
    """

    @staticmethod
    def _default_lines(expected_total: float) -> List[float]:
        base = round(expected_total * 2) / 2  # nearest 0.5
        return [base - 1.5, base - 0.5, base + 0.5, base + 1.5]

    @classmethod
    def calculate_all(
        cls,
        home_expected_shots: float,
        away_expected_shots: float,
        lines: Optional[List[float]] = None,
        max_shots: int = 30,
        dixon_coles_rho: float = 0.0,
    ) -> Dict[str, float]:

        matrix = ScoreMatrix.build(
            home_expected_shots, away_expected_shots, max_shots,
            dixon_coles_rho=dixon_coles_rho,
        )

        expected_total = home_expected_shots + away_expected_shots

        if lines is None:
            lines = cls._default_lines(expected_total)

        markets: Dict[str, float] = {
            "EXPECTED_HOME_SHOTS": home_expected_shots,
            "EXPECTED_AWAY_SHOTS": away_expected_shots,
        }

        for line in lines:
            over = sum(p for (h, a), p in matrix.items() if h + a > line)
            under = sum(p for (h, a), p in matrix.items() if h + a < line)
            markets[f"SHOTS_OVER_{line}"] = over
            markets[f"SHOTS_UNDER_{line}"] = under

        return markets
