from dataclasses import dataclass
from typing import Optional


@dataclass
class MatchFeatures:
    """
    Model-ready features for a single football match.
    """

    match_id: str

    home_team_id: str
    away_team_id: str

    home_matches: int = 0
    away_matches: int = 0

    home_goals_for: float = 0.0
    home_goals_against: float = 0.0

    away_goals_for: float = 0.0
    away_goals_against: float = 0.0

    home_attack_strength: Optional[float] = None
    home_defense_strength: Optional[float] = None

    away_attack_strength: Optional[float] = None
    away_defense_strength: Optional[float] = None

    feature_version: str = "1.0.0"
