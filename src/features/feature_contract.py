from dataclasses import dataclass
from typing import Optional


@dataclass
class TeamFeature:
    """
    Standard feature set for one team before a match.
    """

    matches_played: int = 0

    goals_for_per_match: float = 0.0
    goals_against_per_match: float = 0.0

    home_goals_for_per_match: Optional[float] = None
    home_goals_against_per_match: Optional[float] = None

    away_goals_for_per_match: Optional[float] = None
    away_goals_against_per_match: Optional[float] = None

    recent_goals_for_per_match: Optional[float] = None
    recent_goals_against_per_match: Optional[float] = None


@dataclass
class MatchFeatureContract:
    """
    Versioned feature contract passed from Feature Engineering
    to prediction models.
    """

    match_id: str

    home: TeamFeature
    away: TeamFeature

    feature_version: str = "1.0.0"

    # Data sufficiency / reliability information.
    historical_matches_available: int = 0
    minimum_history_met: bool = False

    # Optional league/context features reserved for later versions.
    league_strength: Optional[float] = None
    home_advantage: Optional[float] = None
