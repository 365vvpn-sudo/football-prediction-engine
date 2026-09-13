import json
from pathlib import Path

from src.data.dataset import HistoricalDataset
from src.data.football_data_loader import FootballDataLoader


class CsvManifestLoader:
    """
    Loads the project's point-in-time historical dataset from CSV
    files listed in a manifest (01_data/raw/football_data/manifest.json).

    This is the single source of truth for historical matches, used
    by BOTH the live prediction pipeline (FeatureEngine) and the
    offline backtest/calibration scripts, so live predictions and
    tested performance are always backed by the same data.
    """

    VERSION = "1.0.0"

    def __init__(self, manifest_path: str = "01_data/raw/football_data/manifest.json") -> None:
        self.manifest_path = Path(manifest_path)
        self._csv_loader = FootballDataLoader()

    def load_all(self) -> HistoricalDataset:
        dataset = HistoricalDataset(source_name="CSV-Manifest", dataset_version=self.VERSION)

        if not self.manifest_path.exists():
            return dataset

        with self.manifest_path.open("r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            try:
                matches = self._csv_loader.load_file(entry["path"], entry["league_id"], entry["season"])
            except (FileNotFoundError, ValueError) as error:
                print(f"WARNING: skipping {entry.get('path')}: {error}")
                continue

            for match in matches:
                dataset.add(match)

        dataset.matches.sort(key=lambda m: m.kickoff_time)
        return dataset
