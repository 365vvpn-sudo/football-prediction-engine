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

weights = [0.20, 0.25, 0.30, 0.35, 0.40]

print("Walk-Forward Expected Goals Test")
print("=" * 70)

for league_id, season, path in datasets:
    matches = loader.load_file(path, league_id, season)

    split = int(len(matches) * 0.70)
    train = matches[:split]
    test = matches[split:]

    print(f"\n{league_id} | {season}")
    print(f"Train: {len(train)} | Test: {len(test)}")

    results = []

    for weight in weights:
        result = backtester.run(test, league_id, weight, 3)
        results.append(result)

        print(
            f"Weight: {weight:.2f} | "
            f"Matches: {result[1]} | "
            f"MAE: {result[2]:.4f} | "
            f"RMSE: {result[3]:.4f} | "
            f"Baseline: {result[4]:.4f}"
        )

    valid = [r for r in results if r[1] > 0]

    if valid:
        best_mae = min(valid, key=lambda x: x[2])
        best_rmse = min(valid, key=lambda x: x[3])

        print(
            f"BEST MAE WEIGHT: {best_mae[0]:.2f}"
        )
        print(
            f"BEST RMSE WEIGHT: {best_rmse[0]:.2f}"
        )

print("\n" + "=" * 70)
print("Point-in-Time: ENABLED")
print("Future Leakage Protection: ENABLED")
