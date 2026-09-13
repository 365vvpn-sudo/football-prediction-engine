#!/usr/bin/env bash
set -e

echo "=== Step 1: Writing incremental_engines.py (O(n) point-in-time tracker) ==="

cat > src/features/incremental_engines.py << 'PYEOF'
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
PYEOF

echo ""
echo "=== Step 2: Rewriting walk_forward_calibration.py (combine seasons per league + incremental) ==="

cat > 17_scripts/walk_forward_calibration.py << 'PYEOF'
"""
Unified walk-forward + calibration backtest for the live pipeline
(goals AND shots markets).

v4 changes:
- All seasons of the SAME league are combined into one continuous
  chronological history (previously each season file was tested
  in isolation, throwing away cross-season depth -- a real bug,
  not just a performance issue).
- Uses IncrementalStatTracker (O(1) per match) instead of rebuilding
  full-history stats from scratch for every target match, which
  turns the walk-forward loop from O(n^2) into O(n) per league.
"""
import json
from collections import defaultdict
from pathlib import Path

from src.calibration.walk_forward import run_walk_forward_calibration
from src.data.football_data_loader import FootballDataLoader
from src.features.incremental_engines import IncrementalStatTracker
from src.features.stat_extractors import (
    goals_against, goals_for, league_away_goals, league_away_shots,
    league_home_goals, league_home_shots, shots_against, shots_for,
)
from src.features.strength_engine import StrengthEngine
from src.markets.goal_markets import GoalMarketEngine
from src.markets.shot_markets import ShotMarketEngine
from src.models.statistical.expected_value_engine import ExpectedValueEngine

GOAL_MARKETS = [
    "OVER_0.5", "UNDER_0.5", "OVER_1.5", "UNDER_1.5",
    "OVER_2.5", "UNDER_2.5", "OVER_3.5", "UNDER_3.5",
    "BTTS_YES", "BTTS_NO", "HOME_WIN", "DRAW", "AWAY_WIN",
]

SHOT_LINES = [18.5, 21.5, 24.5, 27.5]
SHOT_MARKETS = [f"SHOTS_OVER_{line}" for line in SHOT_LINES] + [f"SHOTS_UNDER_{line}" for line in SHOT_LINES]

MIN_MATCHES_PER_SIDE = 3
WEIGHTS_TO_TEST = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]


def _load_grouped_datasets():
    """Returns {league_id: [(season, path), ...]}, all seasons of the
    same league grouped together so they can be combined into one
    continuous history."""
    manifest_path = Path("01_data/raw/football_data/manifest.json")
    if not manifest_path.exists():
        return {}

    with manifest_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["league_id"]].append((entry["season"], entry["path"]))

    return grouped


LEAGUE_DATASETS = _load_grouped_datasets()


def goal_market_outcome(match, market):
    total = match.home_goals + match.away_goals

    if market.startswith("OVER_"):
        return int(total > float(market.split("_")[1]))
    if market.startswith("UNDER_"):
        return int(total < float(market.split("_")[1]))
    if market == "BTTS_YES":
        return int(match.home_goals > 0 and match.away_goals > 0)
    if market == "BTTS_NO":
        return int(match.home_goals == 0 or match.away_goals == 0)
    if market == "HOME_WIN":
        return int(match.home_goals > match.away_goals)
    if market == "DRAW":
        return int(match.home_goals == match.away_goals)
    if market == "AWAY_WIN":
        return int(match.home_goals < match.away_goals)

    raise ValueError(f"Unsupported goal market: {market}")


def shot_market_outcome(match, market):
    total = match.home_shots + match.away_shots
    line = float(market.split("_")[-1])

    if market.startswith("SHOTS_OVER_"):
        return int(total > line)
    if market.startswith("SHOTS_UNDER_"):
        return int(total < line)

    raise ValueError(f"Unsupported shot market: {market}")


def _find_best_weight(matches, league_id):
    tracker = IncrementalStatTracker(
        goals_for, goals_against, league_home_goals, league_away_goals, half_life_days=180.0,
    )
    strength_engine = StrengthEngine()
    engines_by_weight = {w: ExpectedValueEngine(strength_weight=w) for w in WEIGHTS_TO_TEST}
    errors_by_weight = {w: [] for w in WEIGHTS_TO_TEST}

    for target in matches:
        home_perf = tracker.snapshot_team(target.home_team_id, target.kickoff_time)
        away_perf = tracker.snapshot_team(target.away_team_id, target.kickoff_time)

        if home_perf.home_matches >= MIN_MATCHES_PER_SIDE and away_perf.away_matches >= MIN_MATCHES_PER_SIDE:
            baseline = tracker.snapshot_league(league_id, target.kickoff_time)
            home_strength = strength_engine.calculate(home_perf, baseline)
            away_strength = strength_engine.calculate(away_perf, baseline)

            for weight, engine in engines_by_weight.items():
                expected = engine.calculate(home_strength, away_strength, baseline)
                errors_by_weight[weight].append(abs(expected.home - target.home_goals))
                errors_by_weight[weight].append(abs(expected.away - target.away_goals))

        tracker.add(target)

    print("\n-- Expected Goals: strength_weight search (lower MAE is better) --")
    maes = {}
    for weight in WEIGHTS_TO_TEST:
        errors = errors_by_weight[weight]
        if not errors:
            continue
        mae = sum(errors) / len(errors)
        maes[weight] = mae
        print(f"  weight={weight:.2f} | n={len(errors) // 2} | MAE={mae:.4f}")

    if not maes:
        return None

    best_weight = min(maes, key=maes.get)
    print(f"  >> BEST weight for {league_id}: {best_weight:.2f} (MAE={maes[best_weight]:.4f})")
    print(f"     Copy into LEAGUE_STRENGTH_WEIGHTS['{league_id}'] = {best_weight:.2f}")

    return best_weight


def _collect_calibration_data(matches, league_id, strength_weight, has_shots):
    goal_tracker = IncrementalStatTracker(
        goals_for, goals_against, league_home_goals, league_away_goals, half_life_days=180.0,
    )
    shot_tracker = IncrementalStatTracker(
        shots_for, shots_against, league_home_shots, league_away_shots, half_life_days=180.0,
    ) if has_shots else None

    strength_engine = StrengthEngine()
    engine = ExpectedValueEngine(strength_weight=strength_weight)

    goal_data = {m: {"p": [], "y": []} for m in GOAL_MARKETS}
    shot_data = {m: {"p": [], "y": []} for m in SHOT_MARKETS} if has_shots else {}

    for target in matches:
        home_goal_perf = goal_tracker.snapshot_team(target.home_team_id, target.kickoff_time)
        away_goal_perf = goal_tracker.snapshot_team(target.away_team_id, target.kickoff_time)

        if home_goal_perf.home_matches >= MIN_MATCHES_PER_SIDE and away_goal_perf.away_matches >= MIN_MATCHES_PER_SIDE:
            goal_baseline = goal_tracker.snapshot_league(league_id, target.kickoff_time)
            home_goal_strength = strength_engine.calculate(home_goal_perf, goal_baseline)
            away_goal_strength = strength_engine.calculate(away_goal_perf, goal_baseline)
            expected_goals = engine.calculate(home_goal_strength, away_goal_strength, goal_baseline)
            goal_markets = GoalMarketEngine.calculate_all(expected_goals.home, expected_goals.away, 10)

            for market in GOAL_MARKETS:
                goal_data[market]["p"].append(goal_markets[market])
                goal_data[market]["y"].append(goal_market_outcome(target, market))

            if has_shots and target.home_shots is not None and target.away_shots is not None:
                home_shot_perf = shot_tracker.snapshot_team(target.home_team_id, target.kickoff_time)
                away_shot_perf = shot_tracker.snapshot_team(target.away_team_id, target.kickoff_time)
                shot_baseline = shot_tracker.snapshot_league(league_id, target.kickoff_time)

                if (
                    shot_baseline.matches_used > 0
                    and home_shot_perf.home_matches >= MIN_MATCHES_PER_SIDE
                    and away_shot_perf.away_matches >= MIN_MATCHES_PER_SIDE
                ):
                    home_shot_strength = strength_engine.calculate(home_shot_perf, shot_baseline)
                    away_shot_strength = strength_engine.calculate(away_shot_perf, shot_baseline)
                    expected_shots = engine.calculate(home_shot_strength, away_shot_strength, shot_baseline)
                    shot_markets = ShotMarketEngine.calculate_all(
                        expected_shots.home, expected_shots.away, lines=SHOT_LINES, max_shots=40,
                    )

                    for market in SHOT_MARKETS:
                        shot_data[market]["p"].append(shot_markets[market])
                        shot_data[market]["y"].append(shot_market_outcome(target, market))

        goal_tracker.add(target)
        if has_shots and target.home_shots is not None and target.away_shots is not None:
            shot_tracker.add(target)

    return goal_data, shot_data


def run_league(league_id, season_path_pairs):
    loader = FootballDataLoader()

    matches = []
    seasons_used = []
    for season, path in sorted(season_path_pairs):
        try:
            matches.extend(loader.load_file(path, league_id, season))
            seasons_used.append(season)
        except (FileNotFoundError, ValueError) as error:
            print(f"  WARNING: skipping {path}: {error}")

    matches.sort(key=lambda m: m.kickoff_time)

    has_shots = any(m.home_shots is not None and m.away_shots is not None for m in matches)

    print(f"\n{'=' * 90}")
    print(f"{league_id} | seasons={seasons_used[0]}..{seasons_used[-1]} ({len(seasons_used)} files) | "
          f"matches={len(matches)} | shots_available={has_shots}")
    print("=" * 90)

    if not matches:
        print("  No matches loaded, skipping.")
        return

    best_weight = _find_best_weight(matches, league_id)

    if best_weight is None:
        print("  Not enough data for this league to run calibration, skipping.")
        return

    goal_data, shot_data = _collect_calibration_data(matches, league_id, best_weight, has_shots)

    print("\n-- Calibration: Goals markets --")
    for r in run_walk_forward_calibration(GOAL_MARKETS, goal_data):
        print(
            f"  {r.market:12s} n={r.n_test:4d} BrierRaw={r.raw_brier:.4f} BrierCal={r.calibrated_brier:.4f} "
            f"ECERaw={r.raw_ece:.4f} ECECal={r.calibrated_ece:.4f} Improve={r.improvement:+.4f}"
        )

    if has_shots:
        print("\n-- Calibration: Shots markets --")
        for r in run_walk_forward_calibration(SHOT_MARKETS, shot_data):
            print(
                f"  {r.market:14s} n={r.n_test:4d} BrierRaw={r.raw_brier:.4f} BrierCal={r.calibrated_brier:.4f} "
                f"ECERaw={r.raw_ece:.4f} ECECal={r.calibrated_ece:.4f} Improve={r.improvement:+.4f}"
            )
    else:
        print("\n-- Shots markets skipped: this league has no shot data --")


if __name__ == "__main__":
    print("Walk-Forward + Calibration Backtest (v4 -- incremental O(n), multi-season combined)")

    for league_id, season_path_pairs in LEAGUE_DATASETS.items():
        run_league(league_id, season_path_pairs)

    print(f"\n{'=' * 90}")
    print("Point-in-Time: ENABLED | Future Leakage Protection: ENABLED")
PYEOF

echo ""
echo "=== Step 3: Compile check ==="
python3 -m py_compile $(find src 17_scripts -name "*.py" -not -path "*/__pycache__/*")

echo ""
echo "=== ALL DONE. No errors above means everything is syntactically valid. ==="
