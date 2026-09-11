from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TeamForm:
    team_id: str
    team_name: str

    matches_played: int = 0

    goals_for: float = 0.0
    goals_against: float = 0.0

    xg_for: Optional[float] = None
    xg_against: Optional[float] = None

    home_matches: int = 0
    away_matches: int = 0

    recent_results: List[str] = field(default_factory=list)
