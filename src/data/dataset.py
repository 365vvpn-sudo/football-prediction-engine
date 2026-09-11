from dataclasses import dataclass, field
from typing import List

from src.data.historical import HistoricalMatch


@dataclass
class HistoricalDataset:
    """
    Standard container for historical football matches.
    """

    matches: List[HistoricalMatch] = field(default_factory=list)

    source_name: str = ""
    dataset_version: str = "1.0.0"

    def add(self, match: HistoricalMatch) -> None:
        self.matches.append(match)

    @property
    def size(self) -> int:
        return len(self.matches)
