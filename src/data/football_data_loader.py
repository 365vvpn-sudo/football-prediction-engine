import csv
from datetime import datetime
from pathlib import Path

from src.data.historical import HistoricalMatch


class FootballDataLoader:
    SOURCE_NAME = "Football-Data.co.uk"
    VERSION = "2.0.0"

    @staticmethod
    def _parse_date(value: str) -> datetime:
        value = value.strip()

        for date_format in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, date_format)
            except ValueError:
                continue

        raise ValueError(f"Unsupported date format: {value}")

    @staticmethod
    def _optional_int(row: dict, key: str):
        value = row.get(key, "")
        return int(value) if value not in ("", None) else None

    def load_file(
        self,
        file_path: str,
        league_id: str,
        season: str,
    ) -> list[HistoricalMatch]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        matches = []

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            required_columns = {
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTHG",
                "FTAG",
            }

            missing = required_columns - set(reader.fieldnames or [])

            if missing:
                raise ValueError(
                    f"Missing required columns: {sorted(missing)}"
                )

            for index, row in enumerate(reader, start=1):
                home_team = row["HomeTeam"].strip()
                away_team = row["AwayTeam"].strip()

                if not home_team or not away_team:
                    raise ValueError(
                        f"Missing team name at row {index}"
                    )

                if row["FTHG"] in ("", None) or row["FTAG"] in ("", None):
                    raise ValueError(
                        f"Missing final score at row {index}"
                    )

                match_date = self._parse_date(row["Date"])

                matches.append(
                    HistoricalMatch(
                        match_id=f"{league_id}-{season}-{index:03d}",
                        league_id=league_id,
                        season=season,
                        kickoff_time=match_date,
                        home_team_id=f"TEAM:{home_team}",
                        home_team_name=home_team,
                        away_team_id=f"TEAM:{away_team}",
                        away_team_name=away_team,
                        home_goals=int(row["FTHG"]),
                        away_goals=int(row["FTAG"]),
                        home_shots=self._optional_int(row, "HS"),
                        away_shots=self._optional_int(row, "AS"),
                        home_shots_on_target=self._optional_int(
                            row, "HST"
                        ),
                        away_shots_on_target=self._optional_int(
                            row, "AST"
                        ),
                        home_corners=self._optional_int(row, "HC"),
                        away_corners=self._optional_int(row, "AC"),
                        home_cards=self._optional_int(row, "HY"),
                        away_cards=self._optional_int(row, "AY"),
                        source=self.SOURCE_NAME,
                    )
                )

        return matches
