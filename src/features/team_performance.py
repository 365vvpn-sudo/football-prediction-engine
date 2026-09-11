from dataclasses import dataclass
from typing import List

from src.data.historical import HistoricalMatch


@dataclass
class TeamPerformance:
    team_id: str
    matches_used: int = 0

    goals_for_per_match: float = 0.0
    goals_against_per_match: float = 0.0

    home_matches: int = 0
    home_goals_for_per_match: float = 0.0
    home_goals_against_per_match: float = 0.0

    away_matches: int = 0
    away_goals_for_per_match: float = 0.0
    away_goals_against_per_match: float = 0.0


class TeamPerformanceEngine:
    """
    Calculates a team's historical attacking and defensive
    performance using only completed matches supplied to it.
    """

    VERSION = "1.0.0"

    def calculate(
        self,
        team_id: str,
        matches: List[HistoricalMatch],
    ) -> TeamPerformance:

        valid_matches = [
            match
            for match in matches
            if (
                (match.home_team_id == team_id
                 or match.away_team_id == team_id)
                and match.home_goals is not None
                and match.away_goals is not None
            )
        ]

        if not valid_matches:
            return TeamPerformance(team_id=team_id)

        goals_for = 0
        goals_against = 0

        home_matches = 0
        home_goals_for = 0
        home_goals_against = 0

        away_matches = 0
        away_goals_for = 0
        away_goals_against = 0

        for match in valid_matches:

            if match.home_team_id == team_id:
                goals_for += match.home_goals
                goals_against += match.away_goals

                home_matches += 1
                home_goals_for += match.home_goals
                home_goals_against += match.away_goals

            else:
                goals_for += match.away_goals
                goals_against += match.home_goals

                away_matches += 1
                away_goals_for += match.away_goals
                away_goals_against += match.home_goals

        total = len(valid_matches)

        return TeamPerformance(
            team_id=team_id,
            matches_used=total,

            goals_for_per_match=goals_for / total,
            goals_against_per_match=goals_against / total,

            home_matches=home_matches,
            home_goals_for_per_match=(
                home_goals_for / home_matches
                if home_matches else 0.0
            ),
            home_goals_against_per_match=(
                home_goals_against / home_matches
                if home_matches else 0.0
            ),

            away_matches=away_matches,
            away_goals_for_per_match=(
                away_goals_for / away_matches
                if away_matches else 0.0
            ),
            away_goals_against_per_match=(
                away_goals_against / away_matches
                if away_matches else 0.0
            ),
        )
