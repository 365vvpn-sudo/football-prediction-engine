from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

from src.data.historical import HistoricalMatch

BaselineExtractor = Callable[[HistoricalMatch], Optional[float]]


@dataclass
class LeagueBaseline:
    league_id: str
    matches_used: int = 0

    per_match: float = 0.0
    home_per_match: float = 0.0
    away_per_match: float = 0.0


class LeagueBaselineEngine:
    """
    Calculates a league's baseline rate for an arbitrary statistic
    (goals, shots, ...), with the same recency-weighting behaviour
    as TeamPerformanceEngine.
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        home_extractor: BaselineExtractor,
        away_extractor: BaselineExtractor,
        half_life_days: Optional[float] = 180.0,
    ) -> None:
        self.home_extractor = home_extractor
        self.away_extractor = away_extractor
        self.half_life_days = half_life_days

    def _weight(self, match: HistoricalMatch, reference_time: datetime) -> float:
        if self.half_life_days is None:
            return 1.0
        age_days = max((reference_time - match.kickoff_time).days, 0)
        return 0.5 ** (age_days / self.half_life_days)

    def calculate(
        self,
        league_id: str,
        matches: List[HistoricalMatch],
        reference_time: Optional[datetime] = None,
    ) -> LeagueBaseline:

        valid_matches = [
            match for match in matches
            if match.league_id == league_id
            and self.home_extractor(match) is not None
            and self.away_extractor(match) is not None
        ]

        if not valid_matches:
            return LeagueBaseline(league_id=league_id)

        if reference_time is None:
            reference_time = max(m.kickoff_time for m in valid_matches)

        total_home = 0.0
        total_away = 0.0
        total_weight = 0.0

        for match in valid_matches:
            weight = self._weight(match, reference_time)
            total_home += self.home_extractor(match) * weight
            total_away += self.away_extractor(match) * weight
            total_weight += weight

        if total_weight == 0:
            return LeagueBaseline(league_id=league_id, matches_used=len(valid_matches))

        return LeagueBaseline(
            league_id=league_id,
            matches_used=len(valid_matches),
            per_match=(total_home + total_away) / (2 * total_weight),
            home_per_match=total_home / total_weight,
            away_per_match=total_away / total_weight,
        )
