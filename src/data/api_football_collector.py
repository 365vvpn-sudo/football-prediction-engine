import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict

from src.data.collector import DataCollector
from src.data.historical import HistoricalMatch


class ApiFootballCollector(DataCollector):
    BASE_URL = "https://v3.football.api-sports.io"
    SOURCE_NAME = "API-Football"
    VERSION = "1.1.0"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or self._load_key()

    @staticmethod
    def _load_key() -> str:
        value = os.getenv("API_FOOTBALL_KEY")
        if value:
            return value.strip()

        with open(".env", "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line.startswith("API_FOOTBALL_KEY="):
                    return line.split("=", 1)[1].strip()

        raise RuntimeError("API_FOOTBALL_KEY not found")

    def _request(
        self,
        endpoint: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        query = urllib.parse.urlencode(params)
        url = f"{self.BASE_URL}/{endpoint}?{query}"

        request = urllib.request.Request(
            url,
            headers={"x-apisports-key": self.api_key},
        )

        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(
                response.read().decode("utf-8")
            )

    @staticmethod
    def _to_historical_match(
        item: Dict[str, Any],
    ) -> HistoricalMatch:
        fixture = item["fixture"]
        league = item["league"]
        teams = item["teams"]
        goals = item.get("goals", {})

        kickoff_time = datetime.fromtimestamp(
            fixture["timestamp"]
        )

        return HistoricalMatch(
            match_id=f"API:{fixture['id']}",
            league_id=str(league["id"]),
            season=str(league["season"]),
            kickoff_time=kickoff_time,
            home_team_id=f"TEAM:{teams['home']['id']}",
            home_team_name=teams["home"]["name"],
            away_team_id=f"TEAM:{teams['away']['id']}",
            away_team_name=teams["away"]["name"],
            home_goals=goals.get("home"),
            away_goals=goals.get("away"),
            source=ApiFootballCollector.SOURCE_NAME,
        )

    def get_matches(
        self,
        date: str,
    ) -> list[HistoricalMatch]:
        data = self._request(
            "fixtures",
            {"date": date},
        )

        return [
            self._to_historical_match(item)
            for item in data.get("response", [])
        ]

    def get_match(
        self,
        match_id: str,
    ) -> HistoricalMatch | None:
        data = self._request(
            "fixtures",
            {"id": match_id},
        )

        response = data.get("response", [])

        if not response:
            return None

        return self._to_historical_match(response[0])


    def get_team_history(
        self,
        league_id: str,
        season: str,
        team_id: str,
    ) -> list[HistoricalMatch]:
        """
        Return all matches of a team from a league season.

        Uses league + season because the API-Football Free plan
        does not provide access to the fixtures 'last' parameter.
        """
        data = self._request(
            "fixtures",
            {
                "league": league_id,
                "season": season,
            },
        )

        team_id = str(team_id)

        return [
            self._to_historical_match(item)
            for item in data.get("response", [])
            if (
                str(item.get("teams", {}).get("home", {}).get("id")) == team_id
                or
                str(item.get("teams", {}).get("away", {}).get("id")) == team_id
            )
        ]

    def get_team(
        self,
        team_id: str,
    ) -> Dict[str, Any]:
        data = self._request(
            "teams",
            {"id": team_id},
        )

        response = data.get("response", [])

        return response[0] if response else {}
