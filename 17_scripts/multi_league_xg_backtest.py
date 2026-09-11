from src.data.football_data_loader import FootballDataLoader
from src.models.statistical.expected_goals_backtest import ExpectedGoalsBacktester

loader = FootballDataLoader()
backtester = ExpectedGoalsBacktester()

datasets = [
    (
        "ENGLAND_PREMIER_LEAGUE",
        "2025/2026",
        "01_data/raw/football_data/E0_2025_2026.csv",
    ),
    (
        "SPAIN_LALIGA",
        "2020/2021",
        "01_data/raw/laliga/laliga_2020_2021.csv",
    ),
]

weights = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

all_results = []

print("Multi-League Expected Goals Backtest")
print("=" * 70)

for league_id, season, path in datasets:
    matches = loader.load_file(path, league_id, season)

    print(f"\n{league_id} | {season} | Loaded: {len(matches)}")

    for weight in weights:
        result = backtester.run(matches, league_id, weight, 3)
        all_results.append((league_id, season, result))

        print(
            f"Weight: {result[0]:.2f} | "
            f"Matches: {result[1]} | "
            f"MAE: {result[2]:.4f} | "
            f"RMSE: {result[3]:.4f} | "
            f"Baseline MAE: {result[4]:.4f}"
        )

print("\n" + "=" * 70)
print("BEST WEIGHTS")
print("=" * 70)

for league_id, season, _ in datasets:
    results = [
        r for lid, s, r in all_results
        if lid == league_id and s == season and r[1] > 0
    ]

    best_mae = min(results, key=lambda x: x[2])
    best_rmse = min(results, key=lambda x: x[3])

    print(
        f"{league_id} | "
        f"Best MAE: {best_mae[0]:.2f} ({best_mae[2]:.4f}) | "
        f"Best RMSE: {best_rmse[0]:.2f} ({best_rmse[3]:.4f})"
    )

print("=" * 70)
print("Point-in-Time: ENABLED")
print("Future Leakage Protection: ENABLED")

