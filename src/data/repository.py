from abc import ABC, abstractmethod
from typing import List, Optional

from src.data.historical import HistoricalMatch


class HistoricalRepository(ABC):
    """
    Standard interface for accessing historical match data.
    """

    @abstractmethod
    def save(self, match: HistoricalMatch) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_match(self, match_id: str) -> Optional[HistoricalMatch]:
        raise NotImplementedError

    @abstractmethod
    def get_team_matches(
        self,
        team_id: str,
        limit: Optional[int] = None,
    ) -> List[HistoricalMatch]:
        raise NotImplementedError
