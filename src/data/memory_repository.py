from typing import Dict, List, Optional

from src.data.historical import HistoricalMatch
from src.data.repository import HistoricalRepository


class InMemoryHistoricalRepository(HistoricalRepository):
    """
    Temporary in-memory repository for testing.
    Will later be replaced or complemented by a database repository.
    """

    def __init__(self) -> None:
        self._matches: Dict[str, HistoricalMatch] = {}

    def save(self, match: HistoricalMatch) -> None:
        self._matches[match.match_id] = match

    def get_match(self, match_id: str) -> Optional[HistoricalMatch]:
        return self._matches.get(match_id)

    def get_team_matches(
        self,
        team_id: str,
        limit: Optional[int] = None,
    ) -> List[HistoricalMatch]:

        matches = [
            match
            for match in self._matches.values()
            if match.home_team_id == team_id
            or match.away_team_id == team_id
        ]

        matches.sort(key=lambda x: x.kickoff_time, reverse=True)

        if limit is not None:
            matches = matches[:limit]

        return matches
