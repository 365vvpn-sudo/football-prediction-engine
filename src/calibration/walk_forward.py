from dataclasses import dataclass
from typing import Dict, List, Sequence

from src.calibration.calibrator import ProbabilityCalibrator


@dataclass
class MarketEvalResult:
    market: str
    n_test: int
    raw_brier: float
    calibrated_brier: float
    raw_ece: float
    calibrated_ece: float
    improvement: float


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(probabilities)


def expected_calibration_error(probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> float:
    total = len(probabilities)
    result = 0.0

    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins

        bucket = [
            (p, y) for p, y in zip(probabilities, outcomes)
            if lower <= p < upper or (index == bins - 1 and p == upper)
        ]

        if not bucket:
            continue

        mean_p = sum(p for p, _ in bucket) / len(bucket)
        mean_y = sum(y for _, y in bucket) / len(bucket)

        result += (len(bucket) / total) * abs(mean_p - mean_y)

    return result


def run_walk_forward_calibration(
    market_names: List[str],
    predictions_by_market: Dict[str, Dict[str, list]],
    train_ratio: float = 0.70,
    bins: int = 10,
    smoothing: float = 5.0,
) -> List[MarketEvalResult]:
    """
    predictions_by_market: {market: {"p": [...], "y": [...]}}, already
    sorted chronologically by the caller (point-in-time order).
    """
    results: List[MarketEvalResult] = []

    for market in market_names:
        data = predictions_by_market.get(market)

        if not data or not data["p"]:
            continue

        split = int(len(data["p"]) * train_ratio)
        train_p, train_y = data["p"][:split], data["y"][:split]
        test_p, test_y = data["p"][split:], data["y"][split:]

        if not train_p or not test_p:
            continue

        calibrator = ProbabilityCalibrator(bins=bins, smoothing=smoothing)
        calibrator.fit(train_p, train_y)

        calibrated = [calibrator.calibrate(p).calibrated_probability for p in test_p]

        results.append(
            MarketEvalResult(
                market=market,
                n_test=len(test_p),
                raw_brier=brier_score(test_p, test_y),
                calibrated_brier=brier_score(calibrated, test_y),
                raw_ece=expected_calibration_error(test_p, test_y, bins),
                calibrated_ece=expected_calibration_error(calibrated, test_y, bins),
                improvement=brier_score(test_p, test_y) - brier_score(calibrated, test_y),
            )
        )

    return results
