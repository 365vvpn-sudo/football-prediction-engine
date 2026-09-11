from typing import List

from src.data.historical import HistoricalMatch
from src.data.team_stats import TeamStats


class TeamStatsEngine:
    """
    Calculates team statistics from historical matches.
    """

    def calculate(
        self,
        team_id: str,
        team_name: str,
        matches: List[HistoricalMatch],
    ) -> TeamStats:

        valid_matches = [
            match
            for match in matches
            if (
                (match.home_team_id == team_id or match.away_team_id == team_id)
                and match.home_goals is not None
                and match.away_goals is not None
            )
        ]

        if not valid_matches:
            return TeamStats(
                team_id=team_id,
                team_name=team_name,
            )

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
                home_matches += 1
                goals_for += match.home_goals
                goals_against += match.away_goals
                home_goals_for += match.home_goals
                home_goals_against += match.away_goals

            elif match.away_team_id == team_id:
                away_matches += 1
                goals_for += match.away_goals
                goals_against += match.home_goals
                away_goals_for += match.away_goals
                away_goals_against += match.home_goals

        total_matches = len(valid_matches)

        return TeamStats(
            team_id=team_id,
            team_name=team_name,
            matches_played=total_matches,

            goals_for_per_match=goals_for / total_matches,
            goals_against_per_match=goals_against / total_matches,

            home_goals_for_per_match=(
                home_goals_for / home_matches
                if home_matches > 0 else None
            ),

            home_goals_against_per_match=(
                home_goals_against / home_matches
                if home_matches > 0 else None
            ),

            away_goals_for_per_match=(
                away_goals_for / away_matches
                if away_matches > 0 else None
            ),

            away_goals_against_per_match=(
                away_goals_against / away_matches
                if away_matches > 0 else None
            ),
        )
