from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DecisionResult:
    match_id: str
    market: str
    decision: str
    rank: Optional[int] = None
    selection_score: Optional[float] = None
    probability: Optional[float] = None
    confidence: Optional[float] = None
    data_quality: Optional[float] = None
    league_reliability: Optional[float] = None
    model_agreement: Optional[float] = None
    odds: Optional[float] = None
    fair_odds: Optional[float] = None
    value: Optional[float] = None
    edge: Optional[float] = None
    bookmaker: Optional[str] = None
    risk_level: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
