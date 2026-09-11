from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ValueResult:
    probability: float
    odds: float
    fair_odds: float
    value: float
    edge: float


class ValueEngine:
    VERSION = "1.1.0"

    @staticmethod
    def calculate(
        probability: float,
        odds: float,
    ) -> ValueResult:

        probability = max(0.0, min(1.0, probability))

        if odds <= 1.0:
            raise ValueError("Odds must be greater than 1.0")

        fair_odds = 1.0 / probability if probability > 0 else float("inf")
        value = (probability * odds) - 1.0
        edge = probability - (1.0 / odds)

        return ValueResult(
            probability=probability,
            odds=odds,
            fair_odds=fair_odds,
            value=value,
            edge=edge,
        )

    @classmethod
    def evaluate_market(
        cls,
        probability: float,
        market_odds: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        results = []

        for item in market_odds:
            odds = item.get("odd")

            if odds is None:
                continue

            result = cls.calculate(
                probability=probability,
                odds=float(odds),
            )

            results.append(
                {
                    **item,
                    "probability": result.probability,
                    "fair_odds": result.fair_odds,
                    "value": result.value,
                    "edge": result.edge,
                }
            )

        return results
