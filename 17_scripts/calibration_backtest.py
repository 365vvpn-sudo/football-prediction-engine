from src.data.football_data_loader import FootballDataLoader
from src.features.match_feature_engine import MatchFeatureEngine
from src.models.statistical.expected_goals import ExpectedGoalsEngine
from src.markets.goal_markets import GoalMarketEngine
from importlib.machinery import SourceFileLoader


calibrator_module = SourceFileLoader(
    "calibrator",
    "05_backtest/calibration/calibrator.py",
).load_module()

ProbabilityCalibrator = (
    calibrator_module.ProbabilityCalibrator
)


MARKETS = [
    "OVER_0.5",
    "UNDER_0.5",
    "OVER_1.5",
    "UNDER_1.5",
    "OVER_2.5",
    "UNDER_2.5",
    "OVER_3.5",
    "UNDER_3.5",
    "BTTS_YES",
    "BTTS_NO",
    "HOME_WIN",
    "DRAW",
    "AWAY_WIN",
]


def market_outcome(match, market):
    total_goals = (
        match.home_goals
        + match.away_goals
    )

    if market.startswith("OVER_"):
        line = float(
            market.split("_")[1]
        )
        return int(total_goals > line)

    if market.startswith("UNDER_"):
        line = float(
            market.split("_")[1]
        )
        return int(total_goals < line)

    if market == "BTTS_YES":
        return int(
            match.home_goals > 0
            and match.away_goals > 0
        )

    if market == "BTTS_NO":
        return int(
            match.home_goals == 0
            or match.away_goals == 0
        )

    if market == "HOME_WIN":
        return int(
            match.home_goals
            > match.away_goals
        )

    if market == "DRAW":
        return int(
            match.home_goals
            == match.away_goals
        )

    if market == "AWAY_WIN":
        return int(
            match.home_goals
            < match.away_goals
        )

    raise ValueError(
        f"Unsupported market: {market}"
    )


def brier(probabilities, outcomes):
    return sum(
        (p - y) ** 2
        for p, y in zip(
            probabilities,
            outcomes,
        )
    ) / len(probabilities)


def ece(probabilities, outcomes, bins=10):
    total = len(probabilities)
    result = 0.0

    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins

        bucket = [
            (p, y)
            for p, y in zip(
                probabilities,
                outcomes,
            )
            if (
                lower <= p < upper
                or (
                    index == bins - 1
                    and p == upper
                )
            )
        ]

        if not bucket:
            continue

        mean_probability = sum(
            p for p, _ in bucket
        ) / len(bucket)

        mean_outcome = sum(
            y for _, y in bucket
        ) / len(bucket)

        result += (
            len(bucket) / total
        ) * abs(
            mean_probability
            - mean_outcome
        )

    return result


def collect_predictions(
    matches,
    train_ratio=0.70,
):
    feature_engine = MatchFeatureEngine()
    prediction_engine = ExpectedGoalsEngine()

    matches = sorted(
        matches,
        key=lambda match: match.kickoff_time,
    )

    split = int(
        len(matches) * train_ratio
    )

    train_matches = matches[:split]
    test_matches = matches[split:]

    train_data = {
        market: {"p": [], "y": []}
        for market in MARKETS
    }

    test_data = {
        market: {"p": [], "y": []}
        for market in MARKETS
    }

    for target_index, target in enumerate(
        matches
    ):
        history = [
            item
            for item in matches
            if item.kickoff_time
            < target.kickoff_time
        ]

        home_history = [
            item
            for item in history
            if (
                item.home_team_id
                == target.home_team_id
                or item.away_team_id
                == target.home_team_id
            )
        ]

        away_history = [
            item
            for item in history
            if (
                item.home_team_id
                == target.away_team_id
                or item.away_team_id
                == target.away_team_id
            )
        ]

        if (
            len(home_history) < 3
            or len(away_history) < 3
        ):
            continue

        features = feature_engine.build(
            target,
            home_history,
            away_history,
        )

        expected_goals = (
            prediction_engine.calculate(
                features
            )
        )

        markets = (
            GoalMarketEngine.calculate_all(
                expected_goals.home,
                expected_goals.away,
                10,
            )
        )

        destination = (
            train_data
            if target_index < split
            else test_data
        )

        for market in MARKETS:
            destination[market]["p"].append(
                markets[market]
            )

            destination[market]["y"].append(
                market_outcome(
                    target,
                    market,
                )
            )

    return train_data, test_data


def run_league(
    league_id,
    season,
    path,
):
    loader = FootballDataLoader()

    matches = loader.load_file(
        path,
        league_id,
        season,
    )

    train_data, test_data = (
        collect_predictions(matches)
    )

    print(
        f"\n{league_id} | {season}"
    )
    print("=" * 105)

    print(
        "MARKET        N    "
        "Brier Raw  Brier Cal  "
        "ECE Raw    ECE Cal   "
        "Improvement"
    )

    for market in MARKETS:
        train_p = train_data[market]["p"]
        train_y = train_data[market]["y"]

        test_p = test_data[market]["p"]
        test_y = test_data[market]["y"]

        if not train_p or not test_p:
            continue

        calibrator = ProbabilityCalibrator(
            bins=10,
            smoothing=5.0,
        )

        calibrator.fit(
            train_p,
            train_y,
        )

        calibrated = [
            calibrator.calibrate(
                probability
            ).calibrated_probability
            for probability in test_p
        ]

        raw_brier = brier(
            test_p,
            test_y,
        )

        calibrated_brier = brier(
            calibrated,
            test_y,
        )

        raw_ece = ece(
            test_p,
            test_y,
        )

        calibrated_ece = ece(
            calibrated,
            test_y,
        )

        improvement = (
            raw_brier
            - calibrated_brier
        )

        print(
            f"{market:12s} "
            f"{len(test_p):4d} "
            f"{raw_brier:10.4f} "
            f"{calibrated_brier:10.4f} "
            f"{raw_ece:9.4f} "
            f"{calibrated_ece:9.4f} "
            f"{improvement:11.4f}"
        )


if __name__ == "__main__":
    datasets = [
        (
            "ENGLAND_PREMIER_LEAGUE",
            "2025/2026",
            "01_data/raw/football_data/"
            "E0_2025_2026.csv",
        ),
        (
            "SPAIN_LALIGA",
            "2020/2021",
            "01_data/raw/laliga/"
            "laliga_2020_2021.csv",
        ),
    ]

    print("Calibration Backtest v2")
    print("=" * 105)
    print(
        "Train: 70% | Test: 30% | "
        "Quantile Bins | Smoothing | Monotonic"
    )

    for league_id, season, path in datasets:
        run_league(
            league_id,
            season,
            path,
        )

    print("\n" + "=" * 105)
    print("Point-in-Time: ENABLED")
    print("Future Leakage Protection: ENABLED")
