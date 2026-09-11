from dataclasses import asdict
import json
from pathlib import Path

from src.data.api_football_collector import ApiFootballCollector


class ApiFootballHistoricalStore:
    VERSION = "1.1.0"

    def __init__(
        self,
        output_dir: str = "01_data/raw/api_football",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def collect_league_season(
        self,
        league_id: int,
        season: int,
    ) -> int:
        collector = ApiFootballCollector()

        data = collector._request(
            "fixtures",
            {
                "league": league_id,
                "season": season,
            },
        )

        matches = data.get("response", [])

        if not matches:
            return 0

        output_file = (
            self.output_dir
            / f"league_{league_id}_season_{season}.jsonl"
        )

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            for item in matches:
                match = collector._to_historical_match(item)

                file.write(
                    json.dumps(
                        asdict(match),
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

        return len(matches)

    def collect_date(self, date: str) -> int:
        collector = ApiFootballCollector()
        matches = collector.get_matches(date)

        if not matches:
            return 0

        output_file = self.output_dir / f"{date}.jsonl"

        with output_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            for match in matches:
                file.write(
                    json.dumps(
                        asdict(match),
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )

        return len(matches)
