from typing import Any, Dict, List

from src.contracts.decision import DecisionResult
from src.contracts.enums import IntegrityStatus
from src.contracts.match import MatchData
from src.data.api_football_odds_parser import ApiFootballOddsParser
from src.decision.engine import DecisionEngine
from src.predictions.goal_prediction_engine import GoalPredictionEngine


class PredictionPipeline:
    VERSION = "1.2.0"

    def __init__(self) -> None:
        self.decision_engine = DecisionEngine()
        self.prediction_engine = GoalPredictionEngine()

    def predict_match(
        self,
        match: MatchData,
    ):
        return self.prediction_engine.predict_match(match)

    def evaluate_market(
        self,
        match_id: str,
        market: str,
        probability: float,
        confidence: float = 0.0,
        data_quality: float = 0.0,
        league_reliability: float = 0.0,
        model_agreement: float = 0.0,
        value: float = 0.0,
        odds: float | None = None,
        integrity_status: str = IntegrityStatus.LOW.value,
    ) -> DecisionResult:

        return self.decision_engine.evaluate(
            match_id=match_id,
            market=market,
            probability=probability,
            confidence=confidence,
            data_quality=data_quality,
            league_reliability=league_reliability,
            model_agreement=model_agreement,
            value=value,
            odds=odds,
            integrity_status=integrity_status,
        )

    @staticmethod
    def _market_probability(
        prediction_markets: Dict[str, float],
        market: str,
        label: str,
    ) -> float | None:

        if market == "MATCH_WINNER":
            mapping = {
                "Home": "HOME_WIN",
                "Draw": "DRAW",
                "Away": "AWAY_WIN",
            }
            key = mapping.get(label)
            return prediction_markets.get(key) if key else None

        if market == "GOALS_OVER_UNDER":
            key = label.upper().replace(" ", "_")
            return prediction_markets.get(key)

        if market == "BTTS":
            mapping = {
                "Yes": "BTTS_YES",
                "No": "BTTS_NO",
            }
            key = mapping.get(label)
            return prediction_markets.get(key) if key else None

        return None

    def evaluate_odds(
        self,
        match_id: str,
        prediction_markets: Dict[str, float],
        bookmakers: List[Dict[str, Any]],
        confidence: float = 0.0,
        data_quality: float = 0.0,
        league_reliability: float = 0.0,
        model_agreement: float = 0.0,
        integrity_status: str = IntegrityStatus.LOW.value,
    ) -> List[DecisionResult]:

        parsed_markets = ApiFootballOddsParser.parse(bookmakers)

        decisions: List[DecisionResult] = []

        for market, odds_list in parsed_markets.items():

            for item in odds_list:
                label = item.get("label")
                odds = item.get("odd")

                if label is None or odds is None:
                    continue

                probability = self._market_probability(
                    prediction_markets=prediction_markets,
                    market=market,
                    label=str(label),
                )

                if probability is None:
                    continue

                decision = self.evaluate_market(
                    match_id=match_id,
                    market=f"{market}:{label}",
                    probability=probability,
                    confidence=confidence,
                    data_quality=data_quality,
                    league_reliability=league_reliability,
                    model_agreement=model_agreement,
                    odds=float(odds),
                    integrity_status=integrity_status,
                )

                decision.bookmaker = item.get("bookmaker")

                value_result = None

                if odds > 1.0:
                    from src.markets.value_engine import ValueEngine

                    value_result = ValueEngine.calculate(
                        probability=probability,
                        odds=float(odds),
                    )

                    decision.fair_odds = value_result.fair_odds
                    decision.value = value_result.value
                    decision.edge = value_result.edge

                decisions.append(decision)

        return decisions
    @staticmethod
    def select_best_odds(
        decisions: List[DecisionResult],
    ) -> List[DecisionResult]:

        best: Dict[str, DecisionResult] = {}

        for decision in decisions:
            market_key = decision.market

            current = best.get(market_key)

            if current is None:
                best[market_key] = decision
                continue

            current_value = current.value or 0.0
            new_value = decision.value or 0.0

            if new_value > current_value:
                best[market_key] = decision

        return list(best.values())

    def rank(
        self,
        decisions: List[DecisionResult],
    ) -> List[DecisionResult]:

                best_decisions = self.select_best_odds(decisions)

        return self.decision_engine.rank(best_decisions)
