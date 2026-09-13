from dataclasses import dataclass
from typing import List


@dataclass
class CalibrationResult:
    raw_probability: float
    calibrated_probability: float
    sample_size: int
    confidence: float


class ProbabilityCalibrator:
    """
    Leakage-safe empirical probability calibrator.
    Calibration bins are learned only from training data.
    Bin probabilities are smoothed and forced to be monotonic
    (Pool Adjacent Violators Algorithm).
    """

    VERSION = "2.1.0"

    def __init__(self, bins: int = 10, smoothing: float = 5.0) -> None:
        if bins < 2:
            raise ValueError("bins must be at least 2")
        if smoothing < 0.0:
            raise ValueError("smoothing must be non-negative")

        self.bins = bins
        self.smoothing = smoothing
        self._boundaries: List[float] = []
        self._centers: List[float] = []
        self._probabilities: List[float] = []
        self._counts: List[int] = []
        self._fitted = False

    @staticmethod
    def _pava(values: List[float]) -> List[float]:
        blocks = []

        for index, value in enumerate(values):
            blocks.append({"start": index, "end": index, "value": value})

            while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
                left = blocks[-2]
                right = blocks[-1]

                size_left = left["end"] - left["start"] + 1
                size_right = right["end"] - right["start"] + 1
                total_size = size_left + size_right

                merged_value = (left["value"] * size_left + right["value"] * size_right) / total_size

                blocks[-2] = {"start": left["start"], "end": right["end"], "value": merged_value}
                blocks.pop()

        result = [0.0] * len(values)
        for block in blocks:
            for index in range(block["start"], block["end"] + 1):
                result[index] = block["value"]

        return result

    def fit(self, probabilities: List[float], outcomes: List[int]) -> None:
        if len(probabilities) != len(outcomes):
            raise ValueError("probabilities and outcomes must have the same length")
        if not probabilities:
            raise ValueError("calibration data cannot be empty")

        pairs = []
        for probability, outcome in zip(probabilities, outcomes):
            if not 0.0 <= probability <= 1.0:
                raise ValueError("probability must be between 0 and 1")
            if outcome not in (0, 1):
                raise ValueError("outcome must be 0 or 1")
            pairs.append((float(probability), int(outcome)))

        pairs.sort(key=lambda item: item[0])

        n = len(pairs)
        actual_bins = min(self.bins, n)
        base_size = n // actual_bins
        remainder = n % actual_bins

        raw_rates, centers, counts = [], [], []
        position = 0

        for bin_index in range(actual_bins):
            size = base_size + (1 if bin_index < remainder else 0)
            current = pairs[position:position + size]
            position += size

            count = len(current)
            successes = sum(outcome for _, outcome in current)
            center = sum(probability for probability, _ in current) / count

            smoothed_rate = (successes + self.smoothing * center) / (count + self.smoothing)

            centers.append(center)
            counts.append(count)
            raw_rates.append(smoothed_rate)

        calibrated_rates = self._pava(raw_rates)

        boundaries = [
            (centers[index] + centers[index + 1]) / 2.0
            for index in range(actual_bins - 1)
        ]

        self._boundaries = boundaries
        self._centers = centers
        self._probabilities = calibrated_rates
        self._counts = counts
        self._fitted = True

    def _find_bin(self, probability: float) -> int:
        for index, boundary in enumerate(self._boundaries):
            if probability < boundary:
                return index
        return len(self._centers) - 1

    def calibrate(self, probability: float) -> CalibrationResult:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if not self._fitted:
            raise RuntimeError("calibrator must be fitted before calibration")

        index = self._find_bin(probability)
        calibrated = self._probabilities[index]
        sample_size = self._counts[index]
        confidence = min(sample_size / (sample_size + 20.0), 1.0)

        return CalibrationResult(
            raw_probability=probability,
            calibrated_probability=calibrated,
            sample_size=sample_size,
            confidence=confidence,
        )
