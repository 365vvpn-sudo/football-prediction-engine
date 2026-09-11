from dataclasses import dataclass
from typing import List

from src.data.historical import HistoricalMatch


@dataclass
class LeagueBaseline:
    league_id: str
    matches_used: int = 0

    goals_per_match: float = 0.0
    home_goals_per_match: float = 0.0
    away_goals_per_match: float = 0.0


class LeagueBaselineEngine:
    VERSION = "1.0.0"

    def calculate(
        self,
        league_id: str,
        matches: List[HistoricalMatch],
    ) -> LeagueBaseline:

        valid_matches = [
            match
            for match in matches
            if (
                match.league_id == league_id
                and match.home_goals is not None
                and match.away_goals is not None
            )
        ]

        if not valid_matches:
            return LeagueBaseline(league_id=league_id)

        total_goals = 0
        home_goals = 0
        away_goals = 0

        for match in valid_matches:
            home_goals += match.home_goals
            away_goals += match.away_goals
            total_goals += match.home_goals + match.away_goals

        count = len(valid_matches)

        return LeagueBaseline(
            league_id=league_id,
            matches_used=count,
            goals_per_match=total_goals / count,
            home_goals_per_match=home_goals / count,
            away_goals_per_match=away_goals / count,
        )
