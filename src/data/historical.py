from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class HistoricalMatch:
    match_id: str
    league_id: str
    season: str

    kickoff_time: datetime

    home_team_id: str
    home_team_name: str

    away_team_id: str
    away_team_name: str

    home_goals: Optional[int] = None
    away_goals: Optional[int] = None

    home_xg: Optional[float] = None
    away_xg: Optional[float] = None

    home_shots: Optional[int] = None
    away_shots: Optional[int] = None

    home_shots_on_target: Optional[int] = None
    away_shots_on_target: Optional[int] = None

    home_corners: Optional[int] = None
    away_corners: Optional[int] = None

    home_cards: Optional[int] = None
    away_cards: Optional[int] = None

    source: Optional[str] = None
