from dataclasses import dataclass
from typing import Dict, List, Optional

from src.contracts.match import MatchData
from src.data.quality_validator import DataQualityValidator
from src.features.engine import FeatureEngine
from src.features.league_baseline import LeagueBaselineEngine
from src.features.stat_extractors import (
    goals_against, goals_for, league_away_goals, league_away_shots,
    league_home_goals, league_home_shots, shots_against, shots_for,
)
from src.features.strength_engine import StrengthEngine
from src.features.team_performance import TeamPerformanceEngine
from src.markets.goal_markets import GoalMarketEngine
from src.markets.shot_markets import ShotMarketEngine
from src.models.statistical.elo import EloEngine
from src.models.statistical.expected_value_engine import ExpectedValueEngine

# Per-league calibrated strength weights, produced by
# 17_scripts/walk_forward_calibration.py. Update this table after
# each recalibration run. Falls back to DEFAULT_STRENGTH_WEIGHT for
# leagues not yet calibrated.
LEAGUE_STRENGTH_WEIGHTS: Dict[str, float] = {
    "ENGLAND_PREMIER_LEAGUE": 0.80,
    "SPAIN_LALIGA": 0.80,
}
DEFAULT_STRENGTH_WEIGHT = 0.20
DIXON_COLES_RHO = -0.05
MIN_MATCHES_PER_SIDE = 3
LEAGUE_RELIABILITY_TARGET_MATCHES = 200
DATA_QUALITY_TARGET_MATCHES = 10


@dataclass
class MatchPrediction:
    match_id: str

    expected_home_goals: float
    expected_away_goals: float
    goal_markets: Dict[str, float]

    expected_home_shots: Optional[float]
    expected_away_shots: Optional[float]
    shot_markets: Dict[str, float]

    data_quality: float
    league_reliability: float
    model_agreement: float

    model_version: str = "3.0.0"


class MatchPredictionEngine:
    """
    Unified point-in-time prediction engine: goals + shots markets,
    plus the signals the Decision Engine needs (data_quality,
    league_reliability, model_agreement).
    """

    VERSION = "3.0.0"

    def __init__(self) -> None:
        self.feature_engine = FeatureEngine()

        self.goal_baseline_engine = LeagueBaselineEngine(league_home_goals, league_away_goals)
        self.shot_baseline_engine = LeagueBaselineEngine(league_home_shots, league_away_shots)

        self.goal_performance_engine = TeamPerformanceEngine(goals_for, goals_against)
        self.shot_performance_engine = TeamPerformanceEngine(shots_for, shots_against)

        self.strength_engine = StrengthEngine()
        self.elo_engine = EloEngine()
        self.validator = DataQualityValidator()

    def _strength_weight(self, league_id: str) -> float:
        return LEAGUE_STRENGTH_WEIGHTS.get(league_id, DEFAULT_STRENGTH_WEIGHT)

    def _data_quality(self, history: List, home_matches: int, away_matches: int) -> float:
        if not history:
            return 0.0

        results = self.validator.validate_dataset(history)
        valid_fraction = sum(1 for r in results if r.valid) / len(results)

        sample_adequacy = min(1.0, min(home_matches, away_matches) / DATA_QUALITY_TARGET_MATCHES)

        return round((0.5 * valid_fraction) + (0.5 * sample_adequacy), 4)

    def predict_match(self, match: MatchData, max_goals: int = 10, max_shots: int = 30) -> MatchPrediction:

        target, history = self.feature_engine.build_v2_context(match)
        reference_time = target.kickoff_time

        strength_weight = self._strength_weight(target.league_id)
        expected_value_engine = ExpectedValueEngine(strength_weight=strength_weight)

        # --- Goals ---
        goal_baseline = self.goal_baseline_engine.calculate(target.league_id, history, reference_time)
        home_goal_perf = self.goal_performance_engine.calculate(target.home_team_id, history, reference_time)
        away_goal_perf = self.goal_performance_engine.calculate(target.away_team_id, history, reference_time)

        if home_goal_perf.home_matches < MIN_MATCHES_PER_SIDE or away_goal_perf.away_matches < MIN_MATCHES_PER_SIDE:
            raise ValueError("Insufficient point-in-time home/away history for prediction.")

        home_goal_strength = self.strength_engine.calculate(home_goal_perf, goal_baseline)
        away_goal_strength = self.strength_engine.calculate(away_goal_perf, goal_baseline)

        expected_goals = expected_value_engine.calculate(home_goal_strength, away_goal_strength, goal_baseline)

        goal_markets = GoalMarketEngine.calculate_all(
            home_expected_goals=expected_goals.home,
            away_expected_goals=expected_goals.away,
            max_goals=max_goals,
            dixon_coles_rho=DIXON_COLES_RHO,
        )

        # --- Shots (best-effort: skipped gracefully if dataset has no shot data) ---
        expected_home_shots = None
        expected_away_shots = None
        shot_markets: Dict[str, float] = {}

        shot_baseline = self.shot_baseline_engine.calculate(target.league_id, history, reference_time)
        home_shot_perf = self.shot_performance_engine.calculate(target.home_team_id, history, reference_time)
        away_shot_perf = self.shot_performance_engine.calculate(target.away_team_id, history, reference_time)

        if (
            shot_baseline.matches_used > 0
            and home_shot_perf.home_matches >= MIN_MATCHES_PER_SIDE
            and away_shot_perf.away_matches >= MIN_MATCHES_PER_SIDE
        ):
            home_shot_strength = self.strength_engine.calculate(home_shot_perf, shot_baseline)
            away_shot_strength = self.strength_engine.calculate(away_shot_perf, shot_baseline)

            expected_shots = expected_value_engine.calculate(home_shot_strength, away_shot_strength, shot_baseline)

            expected_home_shots = expected_shots.home
            expected_away_shots = expected_shots.away

            shot_markets = ShotMarketEngine.calculate_all(
                home_expected_shots=expected_shots.home,
                away_expected_shots=expected_shots.away,
                max_shots=max_shots,
            )

        # --- Cross-check signal (ELO) for model_agreement ---
        elo_ratings = self.elo_engine.build_ratings(history)
        elo_home_win = self.elo_engine.home_win_probability(
            elo_ratings, target.home_team_id, target.away_team_id,
        )
        poisson_home_win = goal_markets.get("HOME_WIN", 0.0)
        model_agreement = round(1.0 - abs(poisson_home_win - elo_home_win), 4)

        data_quality = self._data_quality(history, home_goal_perf.home_matches, away_goal_perf.away_matches)

        league_reliability = round(
            min(1.0, goal_baseline.matches_used / LEAGUE_RELIABILITY_TARGET_MATCHES), 4,
        )

        return MatchPrediction(
            match_id=target.match_id,
            expected_home_goals=expected_goals.home,
            expected_away_goals=expected_goals.away,
            goal_markets=goal_markets,
            expected_home_shots=expected_home_shots,
            expected_away_shots=expected_away_shots,
            shot_markets=shot_markets,
            data_quality=data_quality,
            league_reliability=league_reliability,
            model_agreement=model_agreement,
        )
