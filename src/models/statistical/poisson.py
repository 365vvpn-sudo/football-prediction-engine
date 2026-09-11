from src.contracts import PredictionResult
from src.features.feature_set import FeatureSet
from src.models.base import PredictionModel


class PoissonModel(PredictionModel):
    """
    Initial Poisson model implementation.
    Mathematical prediction logic will be added
    after the historical data pipeline is ready.
    """

    @property
    def name(self) -> str:
        return "Poisson"

    @property
    def version(self) -> str:
        return "1.0.0"

    def predict(
        self,
        features: FeatureSet,
        market: str,
    ) -> PredictionResult:
        return PredictionResult(
            match_id=features.match_id,
            market=market,
            probability=0.0,
            model_name=self.name,
            model_version=self.version,
        )
