from dataclasses import dataclass
from typing import Dict, Tuple

from src.contracts.match import MatchData
from src.features.engine import FeatureEngine
from src.features.match_features import MatchFeatures
from src.features.league_baseline import LeagueBaselineEngine
from src.features.team_performance import TeamPerformanceEngine
from src.features.strength_engine import StrengthEngine
from src.models.statistical.expected_goals_v2 import ExpectedGoalsEngineV2
from src.models.statistical.score_matrix import ScoreMatrix
from src.markets.goal_markets import GoalMarketEngine


@dataclass
class GoalPrediction:
    match_id: str
    expected_home_goals: float
    expected_away_goals: float
    score_matrix: Dict[Tuple[int, int], float]
    markets: Dict[str, float]
    model_version: str = "2.0.1"


class GoalPredictionEngine:
    """
    Main prediction pipeline for Phase 1 goal markets.

    Uses the point-in-time V2 expected-goals model.
    """

    VERSION = "2.0.1"

    def __init__(self) -> None:
        self.feature_engine = FeatureEngine()
        self.baseline_engine = LeagueBaselineEngine()
        self.performance_engine = TeamPerformanceEngine()
        self.strength_engine = StrengthEngine()
        self.expected_goals_engine = ExpectedGoalsEngineV2(
            strength_weight=0.20
        )

    def predict(
        self,
        features: MatchFeatures,
        max_goals: int = 10,
    ) -> GoalPrediction:
        raise ValueError(
            "predict(features) is not supported by V2. "
            "Use predict_match(match) so the model can build "
            "point-in-time performance and league baseline."
        )

    def predict_match(
        self,
        match: MatchData,
        max_goals: int = 10,
    ) -> GoalPrediction:

        target, history = self.feature_engine.build_v2_context(match)

        baseline = self.baseline_engine.calculate(
            league_id=target.league_id,
            matches=history,
        )

        home_performance = self.performance_engine.calculate(
            team_id=target.home_team_id,
            matches=history,
        )

        away_performance = self.performance_engine.calculate(
            team_id=target.away_team_id,
            matches=history,
        )

        if (
            home_performance.matches_used < 3
            or away_performance.matches_used < 3
        ):
            raise ValueError(
                "Insufficient point-in-time history for V2 prediction."
            )

        home_strength = self.strength_engine.calculate(
            performance=home_performance,
            baseline=baseline,
        )

        away_strength = self.strength_engine.calculate(
            performance=away_performance,
            baseline=baseline,
        )

        expected_goals = self.expected_goals_engine.calculate(
            home_strength=home_strength,
            away_strength=away_strength,
            baseline=baseline,
        )

        score_matrix = ScoreMatrix.build(
            home_expected_goals=expected_goals.home,
            away_expected_goals=expected_goals.away,
            max_goals=max_goals,
        )

        markets = GoalMarketEngine.calculate_all(
            home_expected_goals=expected_goals.home,
            away_expected_goals=expected_goals.away,
            max_goals=max_goals,
        )

        return GoalPrediction(
            match_id=target.match_id,
            expected_home_goals=expected_goals.home,
            expected_away_goals=expected_goals.away,
            score_matrix=score_matrix,
            markets=markets,
            model_version=self.VERSION,
        )
