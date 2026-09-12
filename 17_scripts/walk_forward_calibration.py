"""
Unified walk-forward + calibration backtest for the live V2/V3
pipeline (goals AND shots markets). Replaces the old
walk_forward_xg.py, multi_league_xg_backtest.py and
calibration_backtest.py, which tested the disabled V1 goals model.

Usage:
    python 17_scripts/walk_forward_calibration.py
"""
from src.calibration.walk_forward import run_walk_forward_calibration
from src.data.football_data_loader import FootballDataLoader
from src.features.league_baseline import LeagueBaselineEngine
from src.features.stat_extractors import (
    goals_against, goals_for, league_away_goals, league_away_shots,
    league_home_goals, league_home_shots, shots_against, shots_for,
)
from src.features.strength_engine import StrengthEngine
from src.features.team_performance import TeamPerformanceEngine
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
WEIGHTS_TO_TEST = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

DATASETS = [
    ("ENGLAND_PREMIER_LEAGUE", "2025/2026", "01_data/raw/football_data/E0_2025_2026.csv"),
    ("SPAIN_LALIGA", "2020/2021", "01_data/raw/laliga/laliga_2020_2021.csv"),
    # Add more (league_id, season, csv_path) rows here as more
    # historical data is collected.
]


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


def run_league(league_id, season, path):
    loader = FootballDataLoader()
    matches = sorted(loader.load_file(path, league_id, season), key=lambda m: m.kickoff_time)

    has_shots = any(m.home_shots is not None and m.away_shots is not None for m in matches)

    goal_baseline_engine = LeagueBaselineEngine(league_home_goals, league_away_goals)
    goal_performance_engine = TeamPerformanceEngine(goals_for, goals_against)

    shot_baseline_engine = LeagueBaselineEngine(league_home_shots, league_away_shots)
    shot_performance_engine = TeamPerformanceEngine(shots_for, shots_against)

    strength_engine = StrengthEngine()

    print(f"\n{'=' * 90}\n{league_id} | {season} | matches={len(matches)} | shots_available={has_shots}\n{'=' * 90}")

    print("\n-- Expected Goals: strength_weight search (lower MAE is better) --")
    best_weight, best_mae = None, None

    for weight in WEIGHTS_TO_TEST:
        engine = ExpectedValueEngine(strength_weight=weight)
        errors = []

        for index, target in enumerate(matches):
            history = matches[:index]

            baseline = goal_baseline_engine.calculate(league_id, history, target.kickoff_time)
            home_perf = goal_performance_engine.calculate(target.home_team_id, history, target.kickoff_time)
            away_perf = goal_performance_engine.calculate(target.away_team_id, history, target.kickoff_time)

            if home_perf.home_matches < MIN_MATCHES_PER_SIDE or away_perf.away_matches < MIN_MATCHES_PER_SIDE:
                continue

            home_strength = strength_engine.calculate(home_perf, baseline)
            away_strength = strength_engine.calculate(away_perf, baseline)
            expected = engine.calculate(home_strength, away_strength, baseline)

            errors.append(abs(expected.home - target.home_goals))
            errors.append(abs(expected.away - target.away_goals))

        if not errors:
            continue

        mae = sum(errors) / len(errors)
        print(f"  weight={weight:.2f} | n={len(errors) // 2} | MAE={mae:.4f}")

        if best_mae is None or mae < best_mae:
            best_mae, best_weight = mae, weight

    if best_weight is not None:
        print(f"  >> BEST weight for {league_id}: {best_weight:.2f} (MAE={best_mae:.4f})")
        print(f"     Copy into LEAGUE_STRENGTH_WEIGHTS['{league_id}'] = {best_weight:.2f}")

    calibration_weight = best_weight or 0.20
    goal_engine = ExpectedValueEngine(strength_weight=calibration_weight)

    goal_data = {m: {"p": [], "y": []} for m in GOAL_MARKETS}
    shot_data = {m: {"p": [], "y": []} for m in SHOT_MARKETS} if has_shots else {}

    for index, target in enumerate(matches):
        history = matches[:index]

        goal_baseline = goal_baseline_engine.calculate(league_id, history, target.kickoff_time)
        home_goal_perf = goal_performance_engine.calculate(target.home_team_id, history, target.kickoff_time)
        away_goal_perf = goal_performance_engine.calculate(target.away_team_id, history, target.kickoff_time)

        if home_goal_perf.home_matches < MIN_MATCHES_PER_SIDE or away_goal_perf.away_matches < MIN_MATCHES_PER_SIDE:
            continue

        home_goal_strength = strength_engine.calculate(home_goal_perf, goal_baseline)
        away_goal_strength = strength_engine.calculate(away_goal_perf, goal_baseline)
        expected_goals = goal_engine.calculate(home_goal_strength, away_goal_strength, goal_baseline)
        goal_markets = GoalMarketEngine.calculate_all(expected_goals.home, expected_goals.away, 10)

        for market in GOAL_MARKETS:
            goal_data[market]["p"].append(goal_markets[market])
            goal_data[market]["y"].append(goal_market_outcome(target, market))

        if has_shots and target.home_shots is not None and target.away_shots is not None:
            shot_baseline = shot_baseline_engine.calculate(league_id, history, target.kickoff_time)
            home_shot_perf = shot_performance_engine.calculate(target.home_team_id, history, target.kickoff_time)
            away_shot_perf = shot_performance_engine.calculate(target.away_team_id, history, target.kickoff_time)

            if (
                shot_baseline.matches_used > 0
                and home_shot_perf.home_matches >= MIN_MATCHES_PER_SIDE
                and away_shot_perf.away_matches >= MIN_MATCHES_PER_SIDE
            ):
                home_shot_strength = strength_engine.calculate(home_shot_perf, shot_baseline)
                away_shot_strength = strength_engine.calculate(away_shot_perf, shot_baseline)
                expected_shots = goal_engine.calculate(home_shot_strength, away_shot_strength, shot_baseline)
                shot_markets = ShotMarketEngine.calculate_all(
                    expected_shots.home, expected_shots.away, lines=SHOT_LINES, max_shots=40,
                )

                for market in SHOT_MARKETS:
                    shot_data[market]["p"].append(shot_markets[market])
                    shot_data[market]["y"].append(shot_market_outcome(target, market))

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
        print("\n-- Shots markets skipped: this dataset has no shot data --")


if __name__ == "__main__":
    print("Walk-Forward + Calibration Backtest (v3 — Goals & Shots, live pipeline)")
    for league_id, season, path in DATASETS:
        run_league(league_id, season, path)

    print(f"\n{'=' * 90}")
    print("Point-in-Time: ENABLED | Future Leakage Protection: ENABLED")
