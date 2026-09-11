import json
from pathlib import Path
from typing import Any, Dict

from src.data.api_football_collector import ApiFootballCollector


class ApiFootballOddsStore:
    VERSION = "1.1.0"

    def __init__(
        self,
        output_dir: str = "01_data/raw/api_football/odds",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def collect_fixture(
        self,
        fixture_id: str,
    ) -> int:
        collector = ApiFootballCollector()

        data = collector._request(
            "odds",
            {
                "fixture": fixture_id,
            },
        )

        response = data.get("response", [])

        if not response:
            return 0

        output_file = (
            self.output_dir
            / f"fixture_{fixture_id}.jsonl"
        )

        with output_file.open(
            "a",
            encoding="utf-8",
        ) as file:

            for item in response:
                record: Dict[str, Any] = {
                    "version": self.VERSION,
                    "fixture_id": fixture_id,
                    "update": item.get("update"),
                    "league": item.get("league"),
                    "fixture": item.get("fixture"),
                    "bookmakers": item.get("bookmakers", []),
                }

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        return len(response)
