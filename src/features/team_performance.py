from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

from src.data.historical import HistoricalMatch

StatExtractor = Callable[[HistoricalMatch, bool], Optional[float]]


@dataclass
class TeamPerformance:
    team_id: str
    matches_used: int = 0

    for_per_match: float = 0.0
    against_per_match: float = 0.0

    home_matches: int = 0
    home_for_per_match: float = 0.0
    home_against_per_match: float = 0.0

    away_matches: int = 0
    away_for_per_match: float = 0.0
    away_against_per_match: float = 0.0


class TeamPerformanceEngine:
    """
    Calculates a team's historical performance for an arbitrary
    match statistic (goals, shots, ...), using only matches
    supplied to it (caller is responsible for point-in-time
    filtering).

    Recency weighting: matches are weighted with exponential decay
    based on days between the match and `reference_time`, controlled
    by `half_life_days`. Pass half_life_days=None to disable decay
    (equal weighting, old behaviour).
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        for_extractor: StatExtractor,
        against_extractor: StatExtractor,
        half_life_days: Optional[float] = 180.0,
    ) -> None:
        self.for_extractor = for_extractor
        self.against_extractor = against_extractor
        self.half_life_days = half_life_days

    def _weight(self, match: HistoricalMatch, reference_time: datetime) -> float:
        if self.half_life_days is None:
            return 1.0
        age_days = max((reference_time - match.kickoff_time).days, 0)
        return 0.5 ** (age_days / self.half_life_days)

    def _accumulate(self, matches: List[HistoricalMatch], team_id: str, reference_time: datetime):
        total_for = 0.0
        total_against = 0.0
        total_weight = 0.0
        count = 0

        for match in matches:
            is_home = match.home_team_id == team_id
            for_value = self.for_extractor(match, is_home)
            against_value = self.against_extractor(match, is_home)

            if for_value is None or against_value is None:
                continue

            weight = self._weight(match, reference_time)
            total_for += for_value * weight
            total_against += against_value * weight
            total_weight += weight
            count += 1

        if total_weight == 0:
            return 0.0, 0.0, 0

        return total_for / total_weight, total_against / total_weight, count

    def calculate(
        self,
        team_id: str,
        matches: List[HistoricalMatch],
        reference_time: Optional[datetime] = None,
    ) -> TeamPerformance:

        valid_matches = [
            match for match in matches
            if match.home_team_id == team_id or match.away_team_id == team_id
        ]

        if not valid_matches:
            return TeamPerformance(team_id=team_id)

        if reference_time is None:
            reference_time = max(m.kickoff_time for m in valid_matches)

        overall_for, overall_against, overall_count = self._accumulate(
            valid_matches, team_id, reference_time,
        )

        home_matches = [m for m in valid_matches if m.home_team_id == team_id]
        away_matches = [m for m in valid_matches if m.away_team_id == team_id]

        home_for, home_against, home_count = self._accumulate(home_matches, team_id, reference_time)
        away_for, away_against, away_count = self._accumulate(away_matches, team_id, reference_time)

        return TeamPerformance(
            team_id=team_id,
            matches_used=overall_count,
            for_per_match=overall_for,
            against_per_match=overall_against,
            home_matches=home_count,
            home_for_per_match=home_for,
            home_against_per_match=home_against,
            away_matches=away_count,
            away_for_per_match=away_for,
            away_against_per_match=away_against,
        )
