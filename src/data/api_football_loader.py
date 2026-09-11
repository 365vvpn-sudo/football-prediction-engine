import json
from datetime import datetime
from pathlib import Path

from src.data.dataset import HistoricalDataset
from src.data.historical import HistoricalMatch


class ApiFootballLoader:
    VERSION = "1.1.0"

    def __init__(
        self,
        input_dir: str = "01_data/raw/api_football",
    ) -> None:
        self.input_dir = Path(input_dir)

    def load_file(self, file_path: str) -> list[HistoricalMatch]:
        matches: list[HistoricalMatch] = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                data = json.loads(line)

                matches.append(
                    HistoricalMatch(
                        match_id=data["match_id"],
                        league_id=data["league_id"],
                        season=data["season"],
                        kickoff_time=datetime.fromisoformat(
                            data["kickoff_time"]
                        ),
                        home_team_id=data["home_team_id"],
                        home_team_name=data["home_team_name"],
                        away_team_id=data["away_team_id"],
                        away_team_name=data["away_team_name"],
                        home_goals=data.get("home_goals"),
                        away_goals=data.get("away_goals"),
                        home_xg=data.get("home_xg"),
                        away_xg=data.get("away_xg"),
                        home_shots=data.get("home_shots"),
                        away_shots=data.get("away_shots"),
                        home_shots_on_target=data.get(
                            "home_shots_on_target"
                        ),
                        away_shots_on_target=data.get(
                            "away_shots_on_target"
                        ),
                        home_corners=data.get("home_corners"),
                        away_corners=data.get("away_corners"),
                        home_cards=data.get("home_cards"),
                        away_cards=data.get("away_cards"),
                        source=data.get("source"),
                    )
                )

        return matches

    def load_all(self) -> HistoricalDataset:
        dataset = HistoricalDataset(
            source_name="API-Football",
            dataset_version=self.VERSION,
        )

        for file_path in sorted(
            self.input_dir.glob("league_*_season_*.jsonl")
        ):
            for match in self.load_file(file_path):
                dataset.add(match)

        dataset.matches.sort(
            key=lambda match: match.kickoff_time
        )

        return dataset
