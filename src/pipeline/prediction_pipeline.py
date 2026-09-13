from typing import Any, Dict, List, Optional

from src.contracts.decision import DecisionResult
from src.contracts.enums import IntegrityStatus
from src.contracts.match import MatchData
from src.data.api_football_odds_parser import ApiFootballOddsParser
from src.decision.engine import DecisionEngine
from src.markets.value_engine import ValueEngine
from src.predictions.match_prediction_engine import MatchPrediction, MatchPredictionEngine


class PredictionPipeline:
    VERSION = "3.0.0"

    def __init__(self) -> None:
        self.decision_engine = DecisionEngine()
        self.prediction_engine = MatchPredictionEngine()

    def predict_match(self, match: MatchData) -> MatchPrediction:
        return self.prediction_engine.predict_match(match)

    def evaluate_market(
        self,
        match_id: str,
        market: str,
        probability: float,
        prediction: Optional[MatchPrediction] = None,
        confidence: Optional[float] = None,
        odds: Optional[float] = None,
        integrity_status: str = IntegrityStatus.LOW.value,
    ) -> DecisionResult:

        if confidence is None:
            confidence = abs((2 * probability) - 1)

        data_quality = prediction.data_quality if prediction else 0.0
        league_reliability = prediction.league_reliability if prediction else 0.0
        model_agreement = prediction.model_agreement if prediction else 0.0

        return self.decision_engine.evaluate(
            match_id=match_id,
            market=market,
            probability=probability,
            confidence=confidence,
            data_quality=data_quality,
            league_reliability=league_reliability,
            model_agreement=model_agreement,
            odds=odds,
            integrity_status=integrity_status,
        )

    @staticmethod
    def _market_probability(prediction: MatchPrediction, market: str, label: str) -> Optional[float]:

        if market == "MATCH_WINNER":
            mapping = {"Home": "HOME_WIN", "Draw": "DRAW", "Away": "AWAY_WIN"}
            key = mapping.get(label)
            return prediction.goal_markets.get(key) if key else None

        if market == "GOALS_OVER_UNDER":
            key = label.upper().replace(" ", "_")
            return prediction.goal_markets.get(key)

        if market == "BTTS":
            mapping = {"Yes": "BTTS_YES", "No": "BTTS_NO"}
            key = mapping.get(label)
            return prediction.goal_markets.get(key) if key else None

        if market == "SHOTS_OVER_UNDER":
            key = label.upper().replace(" ", "_")
            return prediction.shot_markets.get(key)

        return None

    def evaluate_odds(
        self,
        prediction: MatchPrediction,
        bookmakers: List[Dict[str, Any]],
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

                probability = self._market_probability(prediction, market, str(label))

                if probability is None:
                    continue

                decision = self.evaluate_market(
                    match_id=prediction.match_id,
                    market=f"{market}:{label}",
                    probability=probability,
                    prediction=prediction,
                    odds=float(odds),
                    integrity_status=integrity_status,
                )

                decision.bookmaker = item.get("bookmaker")

                if odds > 1.0:
                    value_result = ValueEngine.calculate(probability=probability, odds=float(odds))
                    decision.fair_odds = value_result.fair_odds
                    decision.value = value_result.value
                    decision.edge = value_result.edge

                decisions.append(decision)

        return decisions

    @staticmethod
    def select_best_odds(decisions: List[DecisionResult]) -> List[DecisionResult]:
        best: Dict[str, DecisionResult] = {}

        for decision in decisions:
            current = best.get(decision.market)

            if current is None:
                best[decision.market] = decision
                continue

            if (decision.value or 0.0) > (current.value or 0.0):
                best[decision.market] = decision

        return list(best.values())

    def rank(self, decisions: List[DecisionResult]) -> List[DecisionResult]:
        best_decisions = self.select_best_odds(decisions)
        return self.decision_engine.rank(best_decisions)
