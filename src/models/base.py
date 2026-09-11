from abc import ABC, abstractmethod

from src.features.feature_set import FeatureSet
from src.contracts import PredictionResult


class PredictionModel(ABC):
    """
    Standard interface for all prediction models.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        features: FeatureSet,
        market: str,
    ) -> PredictionResult:
        """
        Generate a prediction for a specific market.
        """
        raise NotImplementedError
