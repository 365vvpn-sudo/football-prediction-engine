from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class FeatureSet:
    match_id: str

    values: Dict[str, float] = field(default_factory=dict)

    feature_version: str = "1.0.0"

    metadata: Dict[str, Any] = field(default_factory=dict)
