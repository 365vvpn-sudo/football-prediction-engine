from src.data.api_football_loader import ApiFootballLoader
from src.data.dataset_manager import HistoricalDatasetManager


class ApiFootballDataset:
    VERSION = "1.0.0"

    def __init__(self) -> None:
        self.loader = ApiFootballLoader()
        self.dataset = self.loader.load_all()
        self.manager = HistoricalDatasetManager(self.dataset)

    @property
    def size(self) -> int:
        return self.manager.size

    def get_all(self):
        return self.manager.get_all()

    def get_by_league(self, league_id: str):
        return self.manager.get_by_league(league_id)

    def get_by_season(self, season: str):
        return self.manager.get_by_season(season)

    def get_before(self, cutoff_time, league_id=None):
        return self.manager.get_before(
            cutoff_time,
            league_id=league_id,
        )
