from datetime import datetime
from typing import Optional

from src.data.dataset import HistoricalDataset
from src.data.historical import HistoricalMatch


class HistoricalDatasetManager:
    """
    Manages validated historical football datasets.

    Designed to provide point-in-time historical data
    without exposing future matches to the prediction pipeline.
    """

    VERSION = "1.0.0"

    def __init__(self, dataset: HistoricalDataset) -> None:
        self.dataset = dataset
        self._deduplicate()
        self._sort_chronologically()

    def _deduplicate(self) -> None:
        unique_matches: dict[str, HistoricalMatch] = {}

        for match in self.dataset.matches:
            unique_matches[match.match_id] = match

        self.dataset.matches = list(unique_matches.values())

    def _sort_chronologically(self) -> None:
        self.dataset.matches.sort(
            key=lambda match: match.kickoff_time
        )

    @property
    def size(self) -> int:
        return len(self.dataset.matches)

    def get_all(self) -> list[HistoricalMatch]:
        return list(self.dataset.matches)

    def get_by_league(
        self,
        league_id: str,
    ) -> list[HistoricalMatch]:

        return [
            match
            for match in self.dataset.matches
            if match.league_id == league_id
        ]

    def get_by_season(
        self,
        season: str,
    ) -> list[HistoricalMatch]:

        return [
            match
            for match in self.dataset.matches
            if match.season == season
        ]

    def get_before(
        self,
        cutoff_time: datetime,
        league_id: Optional[str] = None,
    ) -> list[HistoricalMatch]:

        matches = [
            match
            for match in self.dataset.matches
            if match.kickoff_time < cutoff_time
        ]

        if league_id is not None:
            matches = [
                match
                for match in matches
                if match.league_id == league_id
            ]

        return matches
