#!/usr/bin/env bash
set -e

echo "=== Step 1: Removing dead/duplicate files ==="

rm -f src/models/statistical/expected_goals.py
rm -f src/models/statistical/expected_goals_v2.py
rm -f src/models/statistical/expected_goals_backtest.py
rm -f src/models/base.py
rm -f src/models/statistical/poisson.py
rm -f src/features/feature_builder.py
rm -f src/features/team_stats_engine.py
rm -f src/features/match_feature_engine.py
rm -f src/features/recent_form.py
rm -f src/features/feature_set.py
rm -f src/features/feature_contract.py
rm -f src/calibration/calibrator.py
rm -f src/calibration/walk_forward.py
rm -f 05_backtest/calibration/calibrator.py
rm -f 17_scripts/calibration_backtest.py
rm -f 17_scripts/multi_league_xg_backtest.py
rm -f 17_scripts/walk_forward_xg.py
rm -f src/predictions/goal_prediction_engine.py

echo "=== Step 2: Writing new/rewritten files ==="

cat > src/features/stat_extractors.py << 'PYEOF'
"""
Central definitions of how to extract a given statistic (goals,
shots, ...) from a HistoricalMatch, for a given team's perspective.

Adding a new statistic (e.g. corners, cards) later means adding
4 small functions here — no changes needed in the engines that
consume them (TeamPerformanceEngine, LeagueBaselineEngine).
"""
from src.data.historical import HistoricalMatch


def goals_for(match: HistoricalMatch, is_home: bool):
    if match.home_goals is None or match.away_goals is None:
        return None
    return float(match.home_goals if is_home else match.away_goals)


def goals_against(match: HistoricalMatch, is_home: bool):
    if match.home_goals is None or match.away_goals is None:
        return None
    return float(match.away_goals if is_home else match.home_goals)


def shots_for(match: HistoricalMatch, is_home: bool):
    if match.home_shots is None or match.away_shots is None:
        return None
    return float(match.home_shots if is_home else match.away_shots)


def shots_against(match: HistoricalMatch, is_home: bool):
    if match.home_shots is None or match.away_shots is None:
        return None
    return float(match.away_shots if is_home else match.home_shots)


def league_home_goals(match: HistoricalMatch):
    return float(match.home_goals) if match.home_goals is not None else None


def league_away_goals(match: HistoricalMatch):
    return float(match.away_goals) if match.away_goals is not None else None


def league_home_shots(match: HistoricalMatch):
    return float(match.home_shots) if match.home_shots is not None else None


def league_away_shots(match: HistoricalMatch):
    return float(match.away_shots) if match.away_shots is not None else None
PYEOF

cat > src/features/team_performance.py << 'PYEOF'
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

from src.data.historical import HistoricalMatch

StatExtractor = Callable[[HistoricalMatch, bool], Optional[float]]


@dataclass
class TeamPerformance:
    team_id: str
    matches_used: int = 0

    for_per_match: float = 0.0
    against_per_match: float = 0.0

    home_matches: int = 0
    home_for_per_match: float = 0.0
    home_against_per_match: float = 0.0

    away_matches: int = 0
    away_for_per_match: float = 0.0
    away_against_per_match: float = 0.0


class TeamPerformanceEngine:
    """
    Calculates a team's historical performance for an arbitrary
    match statistic (goals, shots, ...), using only matches
    supplied to it (caller is responsible for point-in-time
    filtering).

    Recency weighting: matches are weighted with exponential decay
    based on days between the match and `reference_time`, controlled
    by `half_life_days`. Pass half_life_days=None to disable decay
    (equal weighting, old behaviour).
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        for_extractor: StatExtractor,
        against_extractor: StatExtractor,
        half_life_days: Optional[float] = 180.0,
    ) -> None:
        self.for_extractor = for_extractor
        self.against_extractor = against_extractor
        self.half_life_days = half_life_days

    def _weight(self, match: HistoricalMatch, reference_time: datetime) -> float:
        if self.half_life_days is None:
            return 1.0
        age_days = max((reference_time - match.kickoff_time).days, 0)
        return 0.5 ** (age_days / self.half_life_days)

    def _accumulate(self, matches: List[HistoricalMatch], team_id: str, reference_time: datetime):
        total_for = 0.0
        total_against = 0.0
        total_weight = 0.0
        count = 0

        for match in matches:
            is_home = match.home_team_id == team_id
            for_value = self.for_extractor(match, is_home)
            against_value = self.against_extractor(match, is_home)

            if for_value is None or against_value is None:
                continue

            weight = self._weight(match, reference_time)
            total_for += for_value * weight
            total_against += against_value * weight
            total_weight += weight
            count += 1

        if total_weight == 0:
            return 0.0, 0.0, 0

        return total_for / total_weight, total_against / total_weight, count

    def calculate(
        self,
        team_id: str,
        matches: List[HistoricalMatch],
        reference_time: Optional[datetime] = None,
    ) -> TeamPerformance:

        valid_matches = [
            match for match in matches
            if match.home_team_id == team_id or match.away_team_id == team_id
        ]

        if not valid_matches:
            return TeamPerformance(team_id=team_id)

        if reference_time is None:
            reference_time = max(m.kickoff_time for m in valid_matches)

        overall_for, overall_against, overall_count = self._accumulate(
            valid_matches, team_id, reference_time,
        )

        home_matches = [m for m in valid_matches if m.home_team_id == team_id]
        away_matches = [m for m in valid_matches if m.away_team_id == team_id]

        home_for, home_against, home_count = self._accumulate(home_matches, team_id, reference_time)
        away_for, away_against, away_count = self._accumulate(away_matches, team_id, reference_time)

        return TeamPerformance(
            team_id=team_id,
            matches_used=overall_count,
            for_per_match=overall_for,
            against_per_match=overall_against,
            home_matches=home_count,
            home_for_per_match=home_for,
            home_against_per_match=home_against,
            away_matches=away_count,
            away_for_per_match=away_for,
            away_against_per_match=away_against,
        )
PYEOF

cat > src/features/league_baseline.py << 'PYEOF'
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

from src.data.historical import HistoricalMatch

BaselineExtractor = Callable[[HistoricalMatch], Optional[float]]


@dataclass
class LeagueBaseline:
    league_id: str
    matches_used: int = 0

    per_match: float = 0.0
    home_per_match: float = 0.0
    away_per_match: float = 0.0


class LeagueBaselineEngine:
    """
    Calculates a league's baseline rate for an arbitrary statistic
    (goals, shots, ...), with the same recency-weighting behaviour
    as TeamPerformanceEngine.
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        home_extractor: BaselineExtractor,
        away_extractor: BaselineExtractor,
        half_life_days: Optional[float] = 180.0,
    ) -> None:
        self.home_extractor = home_extractor
        self.away_extractor = away_extractor
        self.half_life_days = half_life_days

    def _weight(self, match: HistoricalMatch, reference_time: datetime) -> float:
        if self.half_life_days is None:
            return 1.0
        age_days = max((reference_time - match.kickoff_time).days, 0)
        return 0.5 ** (age_days / self.half_life_days)

    def calculate(
        self,
        league_id: str,
        matches: List[HistoricalMatch],
        reference_time: Optional[datetime] = None,
    ) -> LeagueBaseline:

        valid_matches = [
            match for match in matches
            if match.league_id == league_id
            and self.home_extractor(match) is not None
            and self.away_extractor(match) is not None
        ]

        if not valid_matches:
            return LeagueBaseline(league_id=league_id)

        if reference_time is None:
            reference_time = max(m.kickoff_time for m in valid_matches)

        total_home = 0.0
        total_away = 0.0
        total_weight = 0.0

        for match in valid_matches:
            weight = self._weight(match, reference_time)
            total_home += self.home_extractor(match) * weight
            total_away += self.away_extractor(match) * weight
            total_weight += weight

        if total_weight == 0:
            return LeagueBaseline(league_id=league_id, matches_used=len(valid_matches))

        return LeagueBaseline(
            league_id=league_id,
            matches_used=len(valid_matches),
            per_match=(total_home + total_away) / (2 * total_weight),
            home_per_match=total_home / total_weight,
            away_per_match=total_away / total_weight,
        )
PYEOF

cat > src/features/strength_engine.py << 'PYEOF'
from dataclasses import dataclass

from src.features.league_baseline import LeagueBaseline
from src.features.team_performance import TeamPerformance


@dataclass
class TeamStrength:
    team_id: str

    attack_strength: float = 1.0
    defense_strength: float = 1.0

    home_attack_strength: float = 1.0
    home_defense_strength: float = 1.0

    away_attack_strength: float = 1.0
    away_defense_strength: float = 1.0


class StrengthEngine:
    """
    Converts raw team performance into attack/defense strength
    ratios relative to the league baseline, with Bayesian-style
    shrinkage toward 1.0 (neutral/average) for teams with little
    history — so a team with 2 matches doesn't get an extreme,
    unstable strength value.

    shrinkage_k: pseudo-count of "average" matches to blend
    toward neutral. Higher k = more caution with small samples.
    """

    VERSION = "2.0.0"

    def __init__(self, shrinkage_k: float = 8.0) -> None:
        self.shrinkage_k = shrinkage_k

    def _shrunk_ratio(self, value: float, baseline: float, n_matches: int) -> float:
        if baseline <= 0:
            return 1.0

        raw_ratio = value / baseline
        weight = n_matches / (n_matches + self.shrinkage_k)

        return 1.0 + ((raw_ratio - 1.0) * weight)

    def calculate(
        self,
        performance: TeamPerformance,
        baseline: LeagueBaseline,
    ) -> TeamStrength:

        return TeamStrength(
            team_id=performance.team_id,

            attack_strength=self._shrunk_ratio(
                performance.for_per_match, baseline.per_match, performance.matches_used,
            ),
            defense_strength=self._shrunk_ratio(
                performance.against_per_match, baseline.per_match, performance.matches_used,
            ),

            home_attack_strength=self._shrunk_ratio(
                performance.home_for_per_match, baseline.home_per_match, performance.home_matches,
            ),
            home_defense_strength=self._shrunk_ratio(
                performance.home_against_per_match, baseline.away_per_match, performance.home_matches,
            ),

            away_attack_strength=self._shrunk_ratio(
                performance.away_for_per_match, baseline.away_per_match, performance.away_matches,
            ),
            away_defense_strength=self._shrunk_ratio(
                performance.away_against_per_match, baseline.home_per_match, performance.away_matches,
            ),
        )
PYEOF

cat > src/models/statistical/expected_value_engine.py << 'PYEOF'
from dataclasses import dataclass

from src.features.league_baseline import LeagueBaseline
from src.features.strength_engine import TeamStrength


@dataclass
class ExpectedValue:
    home: float
    away: float


class ExpectedValueEngine:
    """
    Generic point-in-time expected-value model: works for goals,
    shots, or any other count statistic, given the matching
    TeamStrength/LeagueBaseline pair.

    Formula: baseline_rate * attack_strength * opponent_defense_strength
    """

    VERSION = "3.0.0"

    def __init__(self, strength_weight: float = 1.0) -> None:
        if not 0.0 <= strength_weight <= 1.0:
            raise ValueError("strength_weight must be between 0 and 1")
        self.strength_weight = strength_weight

    @staticmethod
    def _blend(strength: float, weight: float) -> float:
        return 1.0 + ((strength - 1.0) * weight)

    def calculate(
        self,
        home_strength: TeamStrength,
        away_strength: TeamStrength,
        baseline: LeagueBaseline,
    ) -> ExpectedValue:

        home_attack = self._blend(home_strength.home_attack_strength, self.strength_weight)
        away_defense = self._blend(away_strength.away_defense_strength, self.strength_weight)

        away_attack = self._blend(away_strength.away_attack_strength, self.strength_weight)
        home_defense = self._blend(home_strength.home_defense_strength, self.strength_weight)

        home_expected = baseline.home_per_match * home_attack * away_defense
        away_expected = baseline.away_per_match * away_attack * home_defense

        return ExpectedValue(
            home=max(home_expected, 0.0),
            away=max(away_expected, 0.0),
        )
PYEOF

cat > src/models/statistical/score_matrix.py << 'PYEOF'
from typing import Dict, Tuple

from src.models.statistical.poisson_math import PoissonMath


class ScoreMatrix:
    """
    Builds the joint probability matrix for two count statistics
    (e.g. home/away goals or shots), assuming independent Poisson
    distributions, with an optional Dixon-Coles correction for the
    slight negative correlation observed in real low-scoring
    outcomes (0-0, 1-0, 0-1, 1-1). dixon_coles_rho=0.0 (default)
    disables the correction and reproduces plain independent
    Poisson behaviour.
    """

    @staticmethod
    def _dixon_coles_tau(
        home_count: int,
        away_count: int,
        home_expected: float,
        away_expected: float,
        rho: float,
    ) -> float:
        if home_count == 0 and away_count == 0:
            return 1.0 - (home_expected * away_expected * rho)
        if home_count == 0 and away_count == 1:
            return 1.0 + (home_expected * rho)
        if home_count == 1 and away_count == 0:
            return 1.0 + (away_expected * rho)
        if home_count == 1 and away_count == 1:
            return 1.0 - rho
        return 1.0

    @classmethod
    def build(
        cls,
        home_expected_goals: float,
        away_expected_goals: float,
        max_goals: int = 10,
        dixon_coles_rho: float = 0.0,
    ) -> Dict[Tuple[int, int], float]:

        if home_expected_goals < 0:
            raise ValueError("home_expected_goals must be >= 0")
        if away_expected_goals < 0:
            raise ValueError("away_expected_goals must be >= 0")
        if max_goals < 0:
            raise ValueError("max_goals must be >= 0")
        if not -1.0 <= dixon_coles_rho <= 1.0:
            raise ValueError("dixon_coles_rho must be between -1 and 1")

        home_distribution = PoissonMath.distribution(home_expected_goals, max_goals)
        away_distribution = PoissonMath.distribution(away_expected_goals, max_goals)

        matrix: Dict[Tuple[int, int], float] = {}

        for home_goals, home_probability in home_distribution.items():
            for away_goals, away_probability in away_distribution.items():
                probability = home_probability * away_probability

                if dixon_coles_rho != 0.0:
                    tau = cls._dixon_coles_tau(
                        home_goals, away_goals,
                        home_expected_goals, away_expected_goals,
                        dixon_coles_rho,
                    )
                    probability *= tau

                matrix[(home_goals, away_goals)] = probability

        if dixon_coles_rho != 0.0:
            total = sum(matrix.values())
            if total > 0:
                matrix = {key: value / total for key, value in matrix.items()}

        return matrix
PYEOF

cat > src/markets/goal_markets.py << 'PYEOF'
from typing import Dict

from src.models.statistical.score_matrix import ScoreMatrix


class GoalMarketEngine:
    """
    Converts a score probability matrix into goal-related
    market probabilities.
    """

    @staticmethod
    def over_under(matrix: Dict[tuple[int, int], float], line: float) -> Dict[str, float]:
        over = sum(p for (h, a), p in matrix.items() if h + a > line)
        under = sum(p for (h, a), p in matrix.items() if h + a < line)
        return {"OVER": over, "UNDER": under}

    @staticmethod
    def btts(matrix: Dict[tuple[int, int], float]) -> Dict[str, float]:
        yes = sum(p for (h, a), p in matrix.items() if h > 0 and a > 0)
        no = sum(p for (h, a), p in matrix.items() if h == 0 or a == 0)
        return {"YES": yes, "NO": no}

    @staticmethod
    def result_1x2(matrix: Dict[tuple[int, int], float]) -> Dict[str, float]:
        home_win = sum(p for (h, a), p in matrix.items() if h > a)
        draw = sum(p for (h, a), p in matrix.items() if h == a)
        away_win = sum(p for (h, a), p in matrix.items() if h < a)
        return {"HOME_WIN": home_win, "DRAW": draw, "AWAY_WIN": away_win}

    @classmethod
    def calculate_all(
        cls,
        home_expected_goals: float,
        away_expected_goals: float,
        max_goals: int = 10,
        dixon_coles_rho: float = 0.0,
    ) -> Dict[str, float]:

        matrix = ScoreMatrix.build(
            home_expected_goals, away_expected_goals, max_goals,
            dixon_coles_rho=dixon_coles_rho,
        )

        markets: Dict[str, float] = {}

        for line in (0.5, 1.5, 2.5, 3.5):
            result = cls.over_under(matrix, line)
            markets[f"OVER_{line}"] = result["OVER"]
            markets[f"UNDER_{line}"] = result["UNDER"]

        btts = cls.btts(matrix)
        markets["BTTS_YES"] = btts["YES"]
        markets["BTTS_NO"] = btts["NO"]

        result = cls.result_1x2(matrix)
        markets["HOME_WIN"] = result["HOME_WIN"]
        markets["DRAW"] = result["DRAW"]
        markets["AWAY_WIN"] = result["AWAY_WIN"]

        return markets
PYEOF

cat > src/markets/shot_markets.py << 'PYEOF'
from typing import Dict, List, Optional

from src.models.statistical.score_matrix import ScoreMatrix


class ShotMarketEngine:
    """
    Converts expected home/away shots into shot-total market
    probabilities (Over/Under), using the same independent-Poisson
    joint-matrix approach as goals.

    NOTE (known limitation): real shot counts are typically more
    over-dispersed than a Poisson distribution assumes. This is a
    reasonable first approximation; a Negative Binomial model is a
    natural future upgrade specifically for this market.
    """

    @staticmethod
    def _default_lines(expected_total: float) -> List[float]:
        base = round(expected_total * 2) / 2  # nearest 0.5
        return [base - 1.5, base - 0.5, base + 0.5, base + 1.5]

    @classmethod
    def calculate_all(
        cls,
        home_expected_shots: float,
        away_expected_shots: float,
        lines: Optional[List[float]] = None,
        max_shots: int = 30,
        dixon_coles_rho: float = 0.0,
    ) -> Dict[str, float]:

        matrix = ScoreMatrix.build(
            home_expected_shots, away_expected_shots, max_shots,
            dixon_coles_rho=dixon_coles_rho,
        )

        expected_total = home_expected_shots + away_expected_shots

        if lines is None:
            lines = cls._default_lines(expected_total)

        markets: Dict[str, float] = {
            "EXPECTED_HOME_SHOTS": home_expected_shots,
            "EXPECTED_AWAY_SHOTS": away_expected_shots,
        }

        for line in lines:
            over = sum(p for (h, a), p in matrix.items() if h + a > line)
            under = sum(p for (h, a), p in matrix.items() if h + a < line)
            markets[f"SHOTS_OVER_{line}"] = over
            markets[f"SHOTS_UNDER_{line}"] = under

        return markets
PYEOF

cat > src/models/statistical/elo.py << 'PYEOF'
from dataclasses import dataclass, field
from typing import Dict, List

from src.data.historical import HistoricalMatch


@dataclass
class EloRatings:
    ratings: Dict[str, float] = field(default_factory=dict)
    default_rating: float = 1500.0

    def get(self, team_id: str) -> float:
        return self.ratings.get(team_id, self.default_rating)


class EloEngine:
    """
    Simple point-in-time ELO rating system, used ONLY as an
    independent cross-check against the statistical (Poisson)
    model to compute model_agreement for the Decision Engine.
    It is not a full prediction model on its own.
    """

    VERSION = "1.0.0"

    def __init__(self, k_factor: float = 20.0, home_advantage: float = 60.0) -> None:
        self.k_factor = k_factor
        self.home_advantage = home_advantage

    def build_ratings(self, matches: List[HistoricalMatch]) -> EloRatings:
        ratings = EloRatings()

        for match in sorted(matches, key=lambda m: m.kickoff_time):
            if match.home_goals is None or match.away_goals is None:
                continue

            home_rating = ratings.get(match.home_team_id)
            away_rating = ratings.get(match.away_team_id)

            expected_home = 1.0 / (
                1.0 + 10 ** (-(home_rating + self.home_advantage - away_rating) / 400.0)
            )

            if match.home_goals > match.away_goals:
                actual_home = 1.0
            elif match.home_goals < match.away_goals:
                actual_home = 0.0
            else:
                actual_home = 0.5

            delta = self.k_factor * (actual_home - expected_home)

            ratings.ratings[match.home_team_id] = home_rating + delta
            ratings.ratings[match.away_team_id] = away_rating - delta

        return ratings

    def home_win_probability(
        self, ratings: EloRatings, home_team_id: str, away_team_id: str,
    ) -> float:
        home_rating = ratings.get(home_team_id)
        away_rating = ratings.get(away_team_id)

        return 1.0 / (
            1.0 + 10 ** (-(home_rating + self.home_advantage - away_rating) / 400.0)
        )
PYEOF

cat > src/predictions/match_prediction_engine.py << 'PYEOF'
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
    # "ENGLAND_PREMIER_LEAGUE": 0.25,
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
PYEOF

cat > src/features/engine.py << 'PYEOF'
from src.contracts import MatchData
from src.data.api_football_collector import ApiFootballCollector
from src.data.api_football_loader import ApiFootballLoader
from src.data.historical import HistoricalMatch


class FeatureEngine:
    """
    Resolves point-in-time historical context needed by the
    prediction pipeline for a given match (live API match or
    an already-known historical match).
    """

    VERSION = "2.0.0"

    def __init__(self) -> None:
        self.loader = ApiFootballLoader()
        self.collector = ApiFootballCollector()
        self.historical_matches = self.loader.load_all().matches

    def _current_team_leagues(self, team_id: str) -> list[str]:
        data = self.collector._request("leagues", {"team": team_id, "current": "true"})
        return [
            str(item["league"]["id"])
            for item in data.get("response", [])
            if item.get("league", {}).get("id") is not None
        ]

    def _get_current_team_history(self, team_id: str, season: str):
        matches = []
        for league_id in self._current_team_leagues(team_id):
            matches.extend(
                self.collector.get_team_history(league_id=league_id, season=season, team_id=team_id)
            )
        return matches

    def build_v2_context(self, match: MatchData) -> tuple[HistoricalMatch, list[HistoricalMatch]]:
        historical_match = next(
            (item for item in self.historical_matches if item.match_id == match.match_id), None,
        )

        if historical_match is not None:
            history = [
                item for item in self.historical_matches
                if item.kickoff_time < historical_match.kickoff_time
                and item.league_id == historical_match.league_id
            ]
            return historical_match, history

        target_match_id = match.match_id.replace("API:", "")
        current_match = self.collector.get_match(target_match_id)

        if current_match is None:
            raise ValueError(f"Match not found: {match.match_id}")

        home_team_id = current_match.home_team_id.replace("TEAM:", "")
        away_team_id = current_match.away_team_id.replace("TEAM:", "")

        home_history = self._get_current_team_history(home_team_id, current_match.season)
        away_history = self._get_current_team_history(away_team_id, current_match.season)

        combined = self.historical_matches + home_history + away_history

        unique_matches = {
            item.match_id: item
            for item in combined
            if item.kickoff_time < current_match.kickoff_time
            and item.league_id == current_match.league_id
        }

        return current_match, list(unique_matches.values())
PYEOF

cat > src/pipeline/prediction_pipeline.py << 'PYEOF'
from typing import Any, Dict, List, Optional

from src.contracts.decision import DecisionResult
from src.contracts.enums import IntegrityStatus
from src.contracts.match import MatchData
from src.data.api_football_odds_parser import ApiFootballOddsParser
from src.decision.engine import DecisionEngine
from src.markets.value_engine import ValueEngine
from src.predictions.match_prediction_engine import MatchPrediction, MatchPredictionEngine


class PredictionPipeline:
    VERSION = "3.0.0"

    def __init__(self) -> None:
        self.decision_engine = DecisionEngine()
        self.prediction_engine = MatchPredictionEngine()

    def predict_match(self, match: MatchData) -> MatchPrediction:
        return self.prediction_engine.predict_match(match)

    def evaluate_market(
        self,
        match_id: str,
        market: str,
        probability: float,
        prediction: Optional[MatchPrediction] = None,
        confidence: Optional[float] = None,
        odds: Optional[float] = None,
        integrity_status: str = IntegrityStatus.LOW.value,
    ) -> DecisionResult:

        if confidence is None:
            confidence = abs((2 * probability) - 1)

        data_quality = prediction.data_quality if prediction else 0.0
        league_reliability = prediction.league_reliability if prediction else 0.0
        model_agreement = prediction.model_agreement if prediction else 0.0

        return self.decision_engine.evaluate(
            match_id=match_id,
            market=market,
            probability=probability,
            confidence=confidence,
            data_quality=data_quality,
            league_reliability=league_reliability,
            model_agreement=model_agreement,
            odds=odds,
            integrity_status=integrity_status,
        )

    @staticmethod
    def _market_probability(prediction: MatchPrediction, market: str, label: str) -> Optional[float]:

        if market == "MATCH_WINNER":
            mapping = {"Home": "HOME_WIN", "Draw": "DRAW", "Away": "AWAY_WIN"}
            key = mapping.get(label)
            return prediction.goal_markets.get(key) if key else None

        if market == "GOALS_OVER_UNDER":
            key = label.upper().replace(" ", "_")
            return prediction.goal_markets.get(key)

        if market == "BTTS":
            mapping = {"Yes": "BTTS_YES", "No": "BTTS_NO"}
            key = mapping.get(label)
            return prediction.goal_markets.get(key) if key else None

        if market == "SHOTS_OVER_UNDER":
            key = label.upper().replace(" ", "_")
            return prediction.shot_markets.get(key)

        return None

    def evaluate_odds(
        self,
        prediction: MatchPrediction,
        bookmakers: List[Dict[str, Any]],
        integrity_status: str = IntegrityStatus.LOW.value,
    ) -> List[DecisionResult]:

        parsed_markets = ApiFootballOddsParser.parse(bookmakers)

        decisions: List[DecisionResult] = []

        for market, odds_list in parsed_markets.items():
            for item in odds_list:
                label = item.get("label")
                odds = item.get("odd")

                if label is None or odds is None:
                    continue

                probability = self._market_probability(prediction, market, str(label))

                if probability is None:
                    continue

                decision = self.evaluate_market(
                    match_id=prediction.match_id,
                    market=f"{market}:{label}",
                    probability=probability,
                    prediction=prediction,
                    odds=float(odds),
                    integrity_status=integrity_status,
                )

                decision.bookmaker = item.get("bookmaker")

                if odds > 1.0:
                    value_result = ValueEngine.calculate(probability=probability, odds=float(odds))
                    decision.fair_odds = value_result.fair_odds
                    decision.value = value_result.value
                    decision.edge = value_result.edge

                decisions.append(decision)

        return decisions

    @staticmethod
    def select_best_odds(decisions: List[DecisionResult]) -> List[DecisionResult]:
        best: Dict[str, DecisionResult] = {}

        for decision in decisions:
            current = best.get(decision.market)

            if current is None:
                best[decision.market] = decision
                continue

            if (decision.value or 0.0) > (current.value or 0.0):
                best[decision.market] = decision

        return list(best.values())

    def rank(self, decisions: List[DecisionResult]) -> List[DecisionResult]:
        best_decisions = self.select_best_odds(decisions)
        return self.decision_engine.rank(best_decisions)
PYEOF

cat > src/calibration/calibrator.py << 'PYEOF'
from dataclasses import dataclass
from typing import List


@dataclass
class CalibrationResult:
    raw_probability: float
    calibrated_probability: float
    sample_size: int
    confidence: float


class ProbabilityCalibrator:
    """
    Leakage-safe empirical probability calibrator.
    Calibration bins are learned only from training data.
    Bin probabilities are smoothed and forced to be monotonic
    (Pool Adjacent Violators Algorithm).
    """

    VERSION = "2.1.0"

    def __init__(self, bins: int = 10, smoothing: float = 5.0) -> None:
        if bins < 2:
            raise ValueError("bins must be at least 2")
        if smoothing < 0.0:
            raise ValueError("smoothing must be non-negative")

        self.bins = bins
        self.smoothing = smoothing
        self._boundaries: List[float] = []
        self._centers: List[float] = []
        self._probabilities: List[float] = []
        self._counts: List[int] = []
        self._fitted = False

    @staticmethod
    def _pava(values: List[float]) -> List[float]:
        blocks = []

        for index, value in enumerate(values):
            blocks.append({"start": index, "end": index, "value": value})

            while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
                left = blocks[-2]
                right = blocks[-1]

                size_left = left["end"] - left["start"] + 1
                size_right = right["end"] - right["start"] + 1
                total_size = size_left + size_right

                merged_value = (left["value"] * size_left + right["value"] * size_right) / total_size

                blocks[-2] = {"start": left["start"], "end": right["end"], "value": merged_value}
                blocks.pop()

        result = [0.0] * len(values)
        for block in blocks:
            for index in range(block["start"], block["end"] + 1):
                result[index] = block["value"]

        return result

    def fit(self, probabilities: List[float], outcomes: List[int]) -> None:
        if len(probabilities) != len(outcomes):
            raise ValueError("probabilities and outcomes must have the same length")
        if not probabilities:
            raise ValueError("calibration data cannot be empty")

        pairs = []
        for probability, outcome in zip(probabilities, outcomes):
            if not 0.0 <= probability <= 1.0:
                raise ValueError("probability must be between 0 and 1")
            if outcome not in (0, 1):
                raise ValueError("outcome must be 0 or 1")
            pairs.append((float(probability), int(outcome)))

        pairs.sort(key=lambda item: item[0])

        n = len(pairs)
        actual_bins = min(self.bins, n)
        base_size = n // actual_bins
        remainder = n % actual_bins

        raw_rates, centers, counts = [], [], []
        position = 0

        for bin_index in range(actual_bins):
            size = base_size + (1 if bin_index < remainder else 0)
            current = pairs[position:position + size]
            position += size

            count = len(current)
            successes = sum(outcome for _, outcome in current)
            center = sum(probability for probability, _ in current) / count

            smoothed_rate = (successes + self.smoothing * center) / (count + self.smoothing)

            centers.append(center)
            counts.append(count)
            raw_rates.append(smoothed_rate)

        calibrated_rates = self._pava(raw_rates)

        boundaries = [
            (centers[index] + centers[index + 1]) / 2.0
            for index in range(actual_bins - 1)
        ]

        self._boundaries = boundaries
        self._centers = centers
        self._probabilities = calibrated_rates
        self._counts = counts
        self._fitted = True

    def _find_bin(self, probability: float) -> int:
        for index, boundary in enumerate(self._boundaries):
            if probability < boundary:
                return index
        return len(self._centers) - 1

    def calibrate(self, probability: float) -> CalibrationResult:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if not self._fitted:
            raise RuntimeError("calibrator must be fitted before calibration")

        index = self._find_bin(probability)
        calibrated = self._probabilities[index]
        sample_size = self._counts[index]
        confidence = min(sample_size / (sample_size + 20.0), 1.0)

        return CalibrationResult(
            raw_probability=probability,
            calibrated_probability=calibrated,
            sample_size=sample_size,
            confidence=confidence,
        )
PYEOF

cat > src/calibration/walk_forward.py << 'PYEOF'
from dataclasses import dataclass
from typing import Dict, List, Sequence

from src.calibration.calibrator import ProbabilityCalibrator


@dataclass
class MarketEvalResult:
    market: str
    n_test: int
    raw_brier: float
    calibrated_brier: float
    raw_ece: float
    calibrated_ece: float
    improvement: float


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(probabilities)


def expected_calibration_error(probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> float:
    total = len(probabilities)
    result = 0.0

    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins

        bucket = [
            (p, y) for p, y in zip(probabilities, outcomes)
            if lower <= p < upper or (index == bins - 1 and p == upper)
        ]

        if not bucket:
            continue

        mean_p = sum(p for p, _ in bucket) / len(bucket)
        mean_y = sum(y for _, y in bucket) / len(bucket)

        result += (len(bucket) / total) * abs(mean_p - mean_y)

    return result


def run_walk_forward_calibration(
    market_names: List[str],
    predictions_by_market: Dict[str, Dict[str, list]],
    train_ratio: float = 0.70,
    bins: int = 10,
    smoothing: float = 5.0,
) -> List[MarketEvalResult]:
    """
    predictions_by_market: {market: {"p": [...], "y": [...]}}, already
    sorted chronologically by the caller (point-in-time order).
    """
    results: List[MarketEvalResult] = []

    for market in market_names:
        data = predictions_by_market.get(market)

        if not data or not data["p"]:
            continue

        split = int(len(data["p"]) * train_ratio)
        train_p, train_y = data["p"][:split], data["y"][:split]
        test_p, test_y = data["p"][split:], data["y"][split:]

        if not train_p or not test_p:
            continue

        calibrator = ProbabilityCalibrator(bins=bins, smoothing=smoothing)
        calibrator.fit(train_p, train_y)

        calibrated = [calibrator.calibrate(p).calibrated_probability for p in test_p]

        results.append(
            MarketEvalResult(
                market=market,
                n_test=len(test_p),
                raw_brier=brier_score(test_p, test_y),
                calibrated_brier=brier_score(calibrated, test_y),
                raw_ece=expected_calibration_error(test_p, test_y, bins),
                calibrated_ece=expected_calibration_error(calibrated, test_y, bins),
                improvement=brier_score(test_p, test_y) - brier_score(calibrated, test_y),
            )
        )

    return results
PYEOF

cat > 17_scripts/walk_forward_calibration.py << 'PYEOF'
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
    print("Walk-Forward + Calibration Backtest (v3 -- Goals & Shots, live pipeline)")
    for league_id, season, path in DATASETS:
        run_league(league_id, season, path)

    print(f"\n{'=' * 90}")
    print("Point-in-Time: ENABLED | Future Leakage Protection: ENABLED")
PYEOF

echo "=== Step 3: Appending methodology notes to README.md ==="

cat >> README.md << 'MDEOF'

## Methodology (Phase 1)

**Model type:** Point-in-time statistical model — league-baseline
Poisson with attack/defense strength blending (not classical ML yet).
Includes:
- Dixon-Coles correction for low-score correlation (goals only)
- Bayesian shrinkage toward league-neutral for teams with little history
- Recency-weighted historical averaging (exponential decay, 180-day half-life)
- Per-league calibrated `strength_weight` (see `LEAGUE_STRENGTH_WEIGHTS`)

**Cross-check signal:** an independent ELO rating model is used to
compute `model_agreement` for the Decision Engine.

**Validation:** Walk-forward, strictly point-in-time (no future-match
leakage), train/test split for calibration evaluation.

**Calibration:** Empirical monotonic binning (PAVA), evaluated with
Brier score and Expected Calibration Error (ECE).

**Markets covered so far:** Goals (1X2, BTTS, Over/Under), Shots
(match-total Over/Under, expected shots per team — experimental,
Poisson approximation; shot counts are typically over-dispersed,
Negative Binomial is a candidate future upgrade).

**Known limitations (tracked for future rounds):**
- Decision thresholds (0.78 / 0.68 in `DecisionEngine`) are not yet
  calibrated against historical decision outcomes.
- `strength_weight` is currently shared between goals and shots;
  a shots-specific calibration is a future improvement.
- Corners and cards markets are not yet implemented (data fields
  already exist in `HistoricalMatch`).
MDEOF

echo "=== Step 4: Fixing .gitignore so 05_backtest code is not ignored ==="
if ! grep -q "!05_backtest/\*\*/\*.py" .gitignore 2>/dev/null; then
  echo "!05_backtest/**/*.py" >> .gitignore
fi

echo "=== Step 5: Compile check ==="
python3 -m py_compile $(find src 17_scripts -name "*.py" -not -path "*/__pycache__/*")

echo ""
echo "=== ALL DONE. No errors above means everything is syntactically valid. ==="
