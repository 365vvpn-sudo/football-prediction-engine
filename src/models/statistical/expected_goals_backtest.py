from math import sqrt

from src.data.historical import HistoricalMatch
from src.features.league_baseline import LeagueBaselineEngine
from src.features.team_performance import TeamPerformanceEngine
from src.features.strength_engine import StrengthEngine
from src.models.statistical.expected_goals_v2 import ExpectedGoalsEngineV2
from src.data.football_data_loader import FootballDataLoader


class ExpectedGoalsBacktester:

    def run(self, matches: list[HistoricalMatch], league_id: str, weight: float, min_matches: int = 3):

        matches = sorted(matches, key=lambda x: x.kickoff_time)

        abs_home = []
        abs_away = []
        baseline_abs_home = []
        baseline_abs_away = []
        sq_home = []
        sq_away = []

        baseline_engine = LeagueBaselineEngine()
        performance_engine = TeamPerformanceEngine()
        strength_engine = StrengthEngine()

        for target in matches:

            history = [
                x for x in matches
                if x.league_id == league_id
                and x.kickoff_time < target.kickoff_time
            ]

            baseline = baseline_engine.calculate(league_id, history)

            home_perf = performance_engine.calculate(
                target.home_team_id, history
            )

            away_perf = performance_engine.calculate(
                target.away_team_id, history
            )

            if (
                home_perf.matches_used < min_matches
                or away_perf.matches_used < min_matches
            ):
                continue

            home_strength = strength_engine.calculate(
                home_perf, baseline
            )

            away_strength = strength_engine.calculate(
                away_perf, baseline
            )

            baseline_abs_home.append(abs(baseline.home_goals_per_match - target.home_goals))
            baseline_abs_away.append(abs(baseline.away_goals_per_match - target.away_goals))
            prediction = ExpectedGoalsEngineV2(
                strength_weight=weight
            ).calculate(
                home_strength,
                away_strength,
                baseline,
            )

            eh = prediction.home - target.home_goals
            ea = prediction.away - target.away_goals

            abs_home.append(abs(eh))
            abs_away.append(abs(ea))
            sq_home.append(eh ** 2)
            sq_away.append(ea ** 2)

        n = len(abs_home)

        if n == 0:
            return weight, 0, 0.0, 0.0, 0.0

        mae = (
            sum(abs_home) + sum(abs_away)
        ) / (2 * n)
        baseline_mae = (
            sum(baseline_abs_home) + sum(baseline_abs_away)
        ) / (2 * n)

        rmse = sqrt(
            (sum(sq_home) + sum(sq_away)) / (2 * n)
        )

        return weight, n, mae, rmse, baseline_mae


if __name__ == "__main__":

    path = "01_data/raw/football_data/E0_2025_2026.csv"



    data = type("Dataset", (), {"matches": FootballDataLoader().load_file(path, "ENGLAND_PREMIER_LEAGUE", "2025/2026")})()

    backtester = ExpectedGoalsBacktester()

    print("Expected Goals Backtest v1")
    print("=" * 50)

    results = []

    for weight in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:

        result = backtester.run(
            data.matches,
            "ENGLAND_PREMIER_LEAGUE",
            weight,
            3,
        )

        results.append(result)

        print(
            f"Weight: {result[0]:.2f} | "
            f"Matches: {result[1]} | "
            f"MAE: {result[2]:.4f} | "
            f"RMSE: {result[3]:.4f} | "
            f"Baseline MAE: {result[4]:.4f}"
        )

    valid = [x for x in results if x[1] > 0]

    if valid:
        best_mae = min(valid, key=lambda x: x[2])
        best_rmse = min(valid, key=lambda x: x[3])

        print("=" * 50)
        print(
            f"Best MAE weight: {best_mae[0]:.2f}"
        )
        print(
            f"Best RMSE weight: {best_rmse[0]:.2f}"
        )
        print(
            f"Baseline MAE: {best_mae[4]:.4f}"
        )

    print("=" * 50)
    print("Point-in-Time: ENABLED")
    print("Future Leakage Protection: ENABLED")
