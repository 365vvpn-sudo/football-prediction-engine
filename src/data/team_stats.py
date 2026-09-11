from dataclasses import dataclass
from typing import Optional


@dataclass
class TeamStats:
    team_id: str
    team_name: str

    matches_played: int = 0

    goals_for_per_match: float = 0.0
    goals_against_per_match: float = 0.0

    home_goals_for_per_match: Optional[float] = None
    home_goals_against_per_match: Optional[float] = None

    away_goals_for_per_match: Optional[float] = None
    away_goals_against_per_match: Optional[float] = None

    xg_for_per_match: Optional[float] = None
    xg_against_per_match: Optional[float] = None
