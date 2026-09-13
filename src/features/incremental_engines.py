"""
O(1)-per-event replacement for repeatedly calling
TeamPerformanceEngine.calculate() / LeagueBaselineEngine.calculate()
from scratch on a growing history list (which is O(n) per call,
O(n^2) total across a walk-forward loop over n matches).

Feed matches strictly in chronological order via add(). Call
snapshot_team()/snapshot_league() BEFORE add()-ing a given match, so
that match is not counted as its own history (point-in-time safe).

Produces the exact same TeamPerformance / LeagueBaseline dataclasses
the original engines return, so StrengthEngine and
ExpectedValueEngine need no changes to consume them.

NOTE: uses continuous exponential decay (fractional days) instead of
the original engines' per-call truncated-integer-day decay. Results
are numerically very close (identical decay curve shape) but not
bit-for-bit identical -- an acceptable, documented approximation
made for O(n) performance.
"""
from datetime import datetime
from typing import Callable, Dict, Optional

from src.data.historical import HistoricalMatch
from src.features.league_baseline import LeagueBaseline
from src.features.team_performance import TeamPerformance

StatExtractor = Callable[[HistoricalMatch, bool], Optional[float]]
BaselineExtractor = Callable[[HistoricalMatch], Optional[float]]


class _Accumulator:
    __slots__ = ("weighted_value", "weight_sum", "count", "last_time")

    def __init__(self) -> None:
        self.weighted_value = 0.0
        self.weight_sum = 0.0
        self.count = 0
        self.last_time: Optional[datetime] = None

    def snapshot(self, query_time: datetime, half_life_days: Optional[float]):
        if self.count == 0:
            return 0.0, 0

        if half_life_days is None or self.last_time is None:
            weight = self.weight_sum
            value = self.weighted_value
        else:
            age_days = max((query_time - self.last_time).days, 0)
            factor = 0.5 ** (age_days / half_life_days)
            weight = self.weight_sum * factor
            value = self.weighted_value * factor

        if weight == 0:
            return 0.0, self.count

        return value / weight, self.count

    def add(self, event_time: datetime, value: float, half_life_days: Optional[float]) -> None:
        if self.last_time is not None and half_life_days is not None:
            age_days = max((event_time - self.last_time).days, 0)
            factor = 0.5 ** (age_days / half_life_days)
            self.weighted_value *= factor
            self.weight_sum *= factor

        self.weighted_value += value
        self.weight_sum += 1.0
        self.count += 1
        self.last_time = event_time


class _PairAccumulator:
    __slots__ = ("for_acc", "against_acc")

    def __init__(self) -> None:
        self.for_acc = _Accumulator()
        self.against_acc = _Accumulator()

    def snapshot(self, query_time: datetime, half_life_days: Optional[float]):
        for_value, count = self.for_acc.snapshot(query_time, half_life_days)
        against_value, _ = self.against_acc.snapshot(query_time, half_life_days)
        return for_value, against_value, count

    def add(self, event_time: datetime, for_value: float, against_value: float, half_life_days: Optional[float]) -> None:
        self.for_acc.add(event_time, for_value, half_life_days)
        self.against_acc.add(event_time, against_value, half_life_days)


class IncrementalStatTracker:
    """
    Tracks a single statistic (goals, shots, ...) for every team and
    for the league as a whole, incrementally, in O(1) per match.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        for_extractor: StatExtractor,
        against_extractor: StatExtractor,
        league_home_extractor: BaselineExtractor,
        league_away_extractor: BaselineExtractor,
        half_life_days: Optional[float] = 180.0,
    ) -> None:
        self.for_extractor = for_extractor
        self.against_extractor = against_extractor
        self.league_home_extractor = league_home_extractor
        self.league_away_extractor = league_away_extractor
        self.half_life_days = half_life_days

        self._team_overall: Dict[str, _PairAccumulator] = {}
        self._team_home: Dict[str, _PairAccumulator] = {}
        self._team_away: Dict[str, _PairAccumulator] = {}

        self._league_home = _Accumulator()
        self._league_away = _Accumulator()
        self._league_matches_used = 0

    def _get(self, store: Dict[str, _PairAccumulator], team_id: str) -> _PairAccumulator:
        acc = store.get(team_id)
        if acc is None:
            acc = _PairAccumulator()
            store[team_id] = acc
        return acc

    def snapshot_team(self, team_id: str, query_time: datetime) -> TeamPerformance:
        overall_for, overall_against, overall_n = self._get(self._team_overall, team_id).snapshot(
            query_time, self.half_life_days
        )
        home_for, home_against, home_n = self._get(self._team_home, team_id).snapshot(
            query_time, self.half_life_days
        )
        away_for, away_against, away_n = self._get(self._team_away, team_id).snapshot(
            query_time, self.half_life_days
        )

        return TeamPerformance(
            team_id=team_id,
            matches_used=overall_n,
            for_per_match=overall_for,
            against_per_match=overall_against,
            home_matches=home_n,
            home_for_per_match=home_for,
            home_against_per_match=home_against,
            away_matches=away_n,
            away_for_per_match=away_for,
            away_against_per_match=away_against,
        )

    def snapshot_league(self, league_id: str, query_time: datetime) -> LeagueBaseline:
        home_value, _ = self._league_home.snapshot(query_time, self.half_life_days)
        away_value, _ = self._league_away.snapshot(query_time, self.half_life_days)

        per_match = (home_value + away_value) / 2.0 if self._league_matches_used else 0.0

        return LeagueBaseline(
            league_id=league_id,
            matches_used=self._league_matches_used,
            per_match=per_match,
            home_per_match=home_value,
            away_per_match=away_value,
        )

    def add(self, match: HistoricalMatch) -> None:
        home_for = self.for_extractor(match, True)
        home_against = self.against_extractor(match, True)
        away_for = self.for_extractor(match, False)
        away_against = self.against_extractor(match, False)

        if home_for is not None and home_against is not None:
            self._get(self._team_overall, match.home_team_id).add(
                match.kickoff_time, home_for, home_against, self.half_life_days
            )
            self._get(self._team_home, match.home_team_id).add(
                match.kickoff_time, home_for, home_against, self.half_life_days
            )

        if away_for is not None and away_against is not None:
            self._get(self._team_overall, match.away_team_id).add(
                match.kickoff_time, away_for, away_against, self.half_life_days
            )
            self._get(self._team_away, match.away_team_id).add(
                match.kickoff_time, away_for, away_against, self.half_life_days
            )

        league_home_value = self.league_home_extractor(match)
        league_away_value = self.league_away_extractor(match)

        if league_home_value is not None and league_away_value is not None:
            self._league_home.add(match.kickoff_time, league_home_value, self.half_life_days)
            self._league_away.add(match.kickoff_time, league_away_value, self.half_life_days)
            self._league_matches_used += 1
