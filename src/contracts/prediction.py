from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PredictionResult:
    match_id: str
    market: str

    probability: float

    model_name: str
    model_version: str

    confidence: Optional[float] = None
    fair_odds: Optional[float] = None

    features_version: Optional[str] = None

    metadata: Optional[Dict] = None
