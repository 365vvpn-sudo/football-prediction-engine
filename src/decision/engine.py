from typing import List

from src.contracts.decision import DecisionResult
from src.contracts.enums import (
    DecisionStatus,
    IntegrityStatus,
    RiskLevel,
)
from src.markets.value_engine import ValueEngine


class DecisionEngine:
    VERSION = "1.1.0"

    def evaluate(
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

        reasons: List[str] = []

        probability = max(0.0, min(1.0, probability))
        confidence = max(0.0, min(1.0, confidence))
        data_quality = max(0.0, min(1.0, data_quality))
        league_reliability = max(0.0, min(1.0, league_reliability))
        model_agreement = max(0.0, min(1.0, model_agreement))

        if odds is not None:
            value_result = ValueEngine.calculate(
                probability=probability,
                odds=odds,
            )
            value = value_result.value

        if integrity_status in (
            IntegrityStatus.HIGH.value,
            IntegrityStatus.EXCLUDED.value,
        ):
            return DecisionResult(
                match_id=match_id,
                market=market,
                decision=DecisionStatus.NO_BET.value,
                probability=probability,
                confidence=confidence,
                data_quality=data_quality,
                league_reliability=league_reliability,
                model_agreement=model_agreement,
                value=value,
                risk_level=RiskLevel.HIGH.value,
                reasons=["Integrity risk is too high"],
            )

        selection_score = (
            probability * 0.35
            + confidence * 0.15
            + data_quality * 0.15
            + league_reliability * 0.10
            + model_agreement * 0.15
            + max(0.0, min(1.0, value)) * 0.10
        )

        if probability < 0.50:
            decision = DecisionStatus.REJECT.value
            risk = RiskLevel.HIGH.value
            reasons.append("Probability below minimum threshold")

        elif selection_score >= 0.78 and value > 0:
            decision = DecisionStatus.ACCEPT.value
            risk = RiskLevel.LOW.value
            reasons.append("Strong combined opportunity")

        elif selection_score >= 0.68:
            decision = DecisionStatus.CAUTION.value
            risk = RiskLevel.MEDIUM.value
            reasons.append("Moderate opportunity with uncertainty")

        else:
            decision = DecisionStatus.NO_BET.value
            risk = RiskLevel.HIGH.value
            reasons.append("Overall evidence is insufficient")

        if value <= 0:
            reasons.append("No positive value detected")

        if data_quality < 0.50:
            reasons.append("Data quality is weak")

        if model_agreement < 0.50:
            reasons.append("Model agreement is weak")

        return DecisionResult(
            match_id=match_id,
            market=market,
            decision=decision,
            selection_score=selection_score,
            probability=probability,
            confidence=confidence,
            data_quality=data_quality,
            league_reliability=league_reliability,
            model_agreement=model_agreement,
            value=value,
            risk_level=risk,
            reasons=reasons,
        )

    def rank(
        self,
        decisions: List[DecisionResult],
    ) -> List[DecisionResult]:

        ranked = sorted(
            decisions,
            key=lambda item: item.selection_score or 0.0,
            reverse=True,
        )

        for index, item in enumerate(ranked, start=1):
            item.rank = index

        accepted = [
            item
            for item in ranked
            if item.decision == DecisionStatus.ACCEPT.value
        ]

        if accepted:
            accepted[0].decision = DecisionStatus.BEST.value

        return ranked
