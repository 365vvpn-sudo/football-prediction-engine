from src.contracts import MatchData
from src.data.api_football_collector import ApiFootballCollector
from src.data.csv_historical_source import CsvManifestLoader
from src.data.historical import HistoricalMatch


class FeatureEngine:
    """
    Resolves point-in-time historical context needed by the
    prediction pipeline for a given match (live API match or
    an already-known historical match).

    Base historical data comes from the CSV manifest (see
    CsvManifestLoader). A match not found there (a genuinely new
    live fixture) falls back to live API-Football lookups.
    """

    VERSION = "3.0.0"

    def __init__(self) -> None:
        self.collector = ApiFootballCollector()
        self.historical_matches = CsvManifestLoader().load_all().matches

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
