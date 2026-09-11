from dataclasses import dataclass

from src.features.league_baseline import LeagueBaseline
from src.features.team_performance import TeamPerformance


@dataclass
class TeamStrength:
    team_id: str

    attack_strength: float = 0.0
    defense_strength: float = 0.0

    home_attack_strength: float = 0.0
    home_defense_strength: float = 0.0

    away_attack_strength: float = 0.0
    away_defense_strength: float = 0.0


class StrengthEngine:
    VERSION = "1.0.0"

    @staticmethod
    def _safe_ratio(value: float, baseline: float) -> float:
        if baseline <= 0:
            return 0.0
        return value / baseline

    def calculate(
        self,
        performance: TeamPerformance,
        baseline: LeagueBaseline,
    ) -> TeamStrength:

        return TeamStrength(
            team_id=performance.team_id,

            attack_strength=self._safe_ratio(
                performance.goals_for_per_match,
                baseline.goals_per_match / 2,
            ),

            defense_strength=self._safe_ratio(
                performance.goals_against_per_match,
                baseline.goals_per_match / 2,
            ),

            home_attack_strength=self._safe_ratio(
                performance.home_goals_for_per_match,
                baseline.home_goals_per_match,
            ),

            home_defense_strength=self._safe_ratio(
                performance.home_goals_against_per_match,
                baseline.away_goals_per_match,
            ),

            away_attack_strength=self._safe_ratio(
                performance.away_goals_for_per_match,
                baseline.away_goals_per_match,
            ),

            away_defense_strength=self._safe_ratio(
                performance.away_goals_against_per_match,
                baseline.home_goals_per_match,
            ),
        )
