import json
from datetime import datetime
from pathlib import Path
from typing import List

from src.data.historical import HistoricalMatch


class StatsBombLoader:
    """
    Loads StatsBomb match JSON files and converts them
    into the project's standard HistoricalMatch format.
    """

    SOURCE_NAME = "StatsBomb Open Data"
    VERSION = "1.0.0"

    def load_file(self, file_path: str) -> List[HistoricalMatch]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("StatsBomb match data must be a list")

        matches: List[HistoricalMatch] = []

        for item in data:
            matches.append(self._normalize_match(item))

        return matches

    def _normalize_match(self, item: dict) -> HistoricalMatch:
        home_team = item.get("home_team", {})
        away_team = item.get("away_team", {})
        competition = item.get("competition", {})
        season = item.get("season", {})

        match_date = item.get("match_date")
        kick_off = item.get("kick_off", "00:00:00")

        kickoff_time = datetime.fromisoformat(
            f"{match_date}T{kick_off.split('.')[0]}"
        )

        return HistoricalMatch(
            match_id=str(item["match_id"]),
            league_id=str(competition["competition_id"]),
            season=str(season["season_name"]),

            kickoff_time=kickoff_time,

            home_team_id=str(home_team["home_team_id"]),
            home_team_name=home_team["home_team_name"],

            away_team_id=str(away_team["away_team_id"]),
            away_team_name=away_team["away_team_name"],

            home_goals=item.get("home_score"),
            away_goals=item.get("away_score"),

            source=self.SOURCE_NAME,
        )
