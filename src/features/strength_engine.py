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
