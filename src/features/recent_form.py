from dataclasses import dataclass
from typing import List

from src.data.historical import HistoricalMatch


@dataclass
class RecentForm:
    matches_used: int = 0

    goals_for_per_match: float = 0.0
    goals_against_per_match: float = 0.0

    results: List[str] = None

    def __post_init__(self) -> None:
        if self.results is None:
            self.results = []


class RecentFormEngine:
    """
    Calculates a team's recent form using only completed
    historical matches available before the prediction time.
    """

    VERSION = "1.0.0"

    def calculate(
        self,
        team_id: str,
        matches: List[HistoricalMatch],
        limit: int = 5,
    ) -> RecentForm:

        if limit <= 0:
            raise ValueError("limit must be greater than zero")

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

        valid_matches.sort(
            key=lambda match: match.kickoff_time,
            reverse=True,
        )

        recent_matches = valid_matches[:limit]

        if not recent_matches:
            return RecentForm()

        goals_for = 0
        goals_against = 0
        results: List[str] = []

        for match in recent_matches:

            if match.home_team_id == team_id:
                scored = match.home_goals
                conceded = match.away_goals

            else:
                scored = match.away_goals
                conceded = match.home_goals

            goals_for += scored
            goals_against += conceded

            if scored > conceded:
                results.append("W")
            elif scored < conceded:
                results.append("L")
            else:
                results.append("D")

        count = len(recent_matches)

        return RecentForm(
            matches_used=count,
            goals_for_per_match=goals_for / count,
            goals_against_per_match=goals_against / count,
            results=results,
        )
