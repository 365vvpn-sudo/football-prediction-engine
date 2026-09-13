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
