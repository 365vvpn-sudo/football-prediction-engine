from src.data.historical import HistoricalMatch
from src.features.match_features import MatchFeatures
from src.features.team_stats_engine import TeamStatsEngine


class MatchFeatureEngine:
    """
    Builds model-ready features for a match
    using historical team performance.
    """

    def __init__(self) -> None:
        self.team_stats_engine = TeamStatsEngine()

    def build(
        self,
        match: HistoricalMatch,
        home_history: list[HistoricalMatch],
        away_history: list[HistoricalMatch],
    ) -> MatchFeatures:

        home_stats = self.team_stats_engine.calculate(
            match.home_team_id,
            match.home_team_name,
            home_history,
        )

        away_stats = self.team_stats_engine.calculate(
            match.away_team_id,
            match.away_team_name,
            away_history,
        )

        return MatchFeatures(
            match_id=match.match_id,

            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,

            home_matches=home_stats.matches_played,
            away_matches=away_stats.matches_played,

            home_goals_for=(
                home_stats.home_goals_for_per_match
                if home_stats.home_goals_for_per_match is not None
                else home_stats.goals_for_per_match
            ),

            home_goals_against=(
                home_stats.home_goals_against_per_match
                if home_stats.home_goals_against_per_match is not None
                else home_stats.goals_against_per_match
            ),

            away_goals_for=(
                away_stats.away_goals_for_per_match
                if away_stats.away_goals_for_per_match is not None
                else away_stats.goals_for_per_match
            ),

            away_goals_against=(
                away_stats.away_goals_against_per_match
                if away_stats.away_goals_against_per_match is not None
                else away_stats.goals_against_per_match
            ),
        )
