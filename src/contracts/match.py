from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MatchData:
    match_id: str
    league_id: str
    league_name: str

    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str

    kickoff_time: datetime

    season: Optional[str] = None
    venue: Optional[str] = None

    data_timestamp: Optional[datetime] = None
    data_quality: Optional[float] = None
