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
WEIGHTS_TO_TEST = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20]


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
