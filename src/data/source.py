from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class DataSourceInfo:
    source_name: str
    source_type: str

    collected_at: datetime

    reliability_score: Optional[float] = None
    freshness_score: Optional[float] = None

    version: Optional[str] = None
    notes: Optional[str] = None
