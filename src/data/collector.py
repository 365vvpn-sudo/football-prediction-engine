from abc import ABC, abstractmethod
from typing import Any, Dict


class DataCollector(ABC):
    """
    Standard interface for all football data sources.
    """

    @abstractmethod
    def get_match(self, match_id: str) -> Dict[str, Any]:
        """
        Return normalized match information.
        """
        raise NotImplementedError

    @abstractmethod
    def get_team(self, team_id: str) -> Dict[str, Any]:
        """
        Return normalized team information.
        """
        raise NotImplementedError

    @abstractmethod
    def get_matches(self, date: str) -> list[Dict[str, Any]]:
        """
        Return normalized matches for a given date.
        """
        raise NotImplementedError
