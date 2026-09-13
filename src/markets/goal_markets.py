from typing import Dict

from src.models.statistical.score_matrix import ScoreMatrix


class GoalMarketEngine:
    """
    Converts a score probability matrix into goal-related
    market probabilities.
    """

    @staticmethod
    def over_under(matrix: Dict[tuple[int, int], float], line: float) -> Dict[str, float]:
        over = sum(p for (h, a), p in matrix.items() if h + a > line)
        under = sum(p for (h, a), p in matrix.items() if h + a < line)
        return {"OVER": over, "UNDER": under}

    @staticmethod
    def btts(matrix: Dict[tuple[int, int], float]) -> Dict[str, float]:
        yes = sum(p for (h, a), p in matrix.items() if h > 0 and a > 0)
        no = sum(p for (h, a), p in matrix.items() if h == 0 or a == 0)
        return {"YES": yes, "NO": no}

    @staticmethod
    def result_1x2(matrix: Dict[tuple[int, int], float]) -> Dict[str, float]:
        home_win = sum(p for (h, a), p in matrix.items() if h > a)
        draw = sum(p for (h, a), p in matrix.items() if h == a)
        away_win = sum(p for (h, a), p in matrix.items() if h < a)
        return {"HOME_WIN": home_win, "DRAW": draw, "AWAY_WIN": away_win}

    @classmethod
    def calculate_all(
        cls,
        home_expected_goals: float,
        away_expected_goals: float,
        max_goals: int = 10,
        dixon_coles_rho: float = 0.0,
    ) -> Dict[str, float]:

        matrix = ScoreMatrix.build(
            home_expected_goals, away_expected_goals, max_goals,
            dixon_coles_rho=dixon_coles_rho,
        )

        markets: Dict[str, float] = {}

        for line in (0.5, 1.5, 2.5, 3.5):
            result = cls.over_under(matrix, line)
            markets[f"OVER_{line}"] = result["OVER"]
            markets[f"UNDER_{line}"] = result["UNDER"]

        btts = cls.btts(matrix)
        markets["BTTS_YES"] = btts["YES"]
        markets["BTTS_NO"] = btts["NO"]

        result = cls.result_1x2(matrix)
        markets["HOME_WIN"] = result["HOME_WIN"]
        markets["DRAW"] = result["DRAW"]
        markets["AWAY_WIN"] = result["AWAY_WIN"]

        return markets
