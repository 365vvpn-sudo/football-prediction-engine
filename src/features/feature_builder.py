from datetime import datetime
from typing import List

from src.data.historical import HistoricalMatch
from src.features.match_features import MatchFeatures


class FeatureBuilder:
    """
    Builds point-in-time model-ready features using
    only historical matches available before the target match.
    """

    VERSION = "1.1.0"

    def build(
        self,
        match: HistoricalMatch,
        historical_matches: List[HistoricalMatch],
        recent_limit: int = 5,
    ) -> MatchFeatures:

        available_matches = [
            item
            for item in historical_matches
            if item.kickoff_time < match.kickoff_time
        ]

        home_history = [
            item
            for item in available_matches
            if (
                item.home_team_id == match.home_team_id
                or item.away_team_id == match.home_team_id
            )
        ]

        away_history = [
            item
            for item in available_matches
            if (
                item.home_team_id == match.away_team_id
                or item.away_team_id == match.away_team_id
            )
        ]

        home_recent = home_history[-recent_limit:]
        away_recent = away_history[-recent_limit:]

        def goals_for(
            item: HistoricalMatch,
            team_id: str,
        ) -> float:
            if item.home_team_id == team_id:
                return float(item.home_goals or 0)
            return float(item.away_goals or 0)

        def goals_against(
            item: HistoricalMatch,
            team_id: str,
        ) -> float:
            if item.home_team_id == team_id:
                return float(item.away_goals or 0)
            return float(item.home_goals or 0)

        home_goals_for = (
            sum(
                goals_for(item, match.home_team_id)
                for item in home_recent
            )
            / len(home_recent)
            if home_recent
            else 0.0
        )

        home_goals_against = (
            sum(
                goals_against(item, match.home_team_id)
                for item in home_recent
            )
            / len(home_recent)
            if home_recent
            else 0.0
        )

        away_goals_for = (
            sum(
                goals_for(item, match.away_team_id)
                for item in away_recent
            )
            / len(away_recent)
            if away_recent
            else 0.0
        )

        away_goals_against = (
            sum(
                goals_against(item, match.away_team_id)
                for item in away_recent
            )
            / len(away_recent)
            if away_recent
            else 0.0
        )

        return MatchFeatures(
            match_id=match.match_id,
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            home_matches=len(home_recent),
            away_matches=len(away_recent),
            home_goals_for=home_goals_for,
            home_goals_against=home_goals_against,
            away_goals_for=away_goals_for,
            away_goals_against=away_goals_against,
            feature_version=self.VERSION,
        )
