#!/usr/bin/env bash
set -e

echo "=== Step 1: Downloading historical CSVs (football-data.co.uk) ==="
mkdir -p 01_data/raw/football_data

LEAGUE_CODES=(E0 E1 SP1 D1 I1 F1)
SEASON_CODES=(1920 2021 2122 2223 2324 2425)

for season in "${SEASON_CODES[@]}"; do
  for code in "${LEAGUE_CODES[@]}"; do
    url="https://www.football-data.co.uk/mmz4281/${season}/${code}.csv"
    out="01_data/raw/football_data/${code}_${season}.csv"
    if curl -fsSL "$url" -o "$out"; then
      echo "OK   $out"
    else
      echo "SKIP $out (not available)"
      rm -f "$out"
    fi
  done
done

echo ""
echo "=== Step 2: Building manifest.json (single source of truth for historical data) ==="

python3 - << 'PYEOF'
import json
import re
from pathlib import Path

LEAGUE_MAP = {
    "E0": "ENGLAND_PREMIER_LEAGUE",
    "E1": "ENGLAND_CHAMPIONSHIP",
    "SP1": "SPAIN_LALIGA",
    "D1": "GERMANY_BUNDESLIGA",
    "I1": "ITALY_SERIE_A",
    "F1": "FRANCE_LIGUE_1",
}


def season_label(code: str) -> str:
    start = "20" + code[:2]
    end = "20" + code[2:]
    return f"{start}/{end}"


data_dir = Path("01_data/raw/football_data")
manifest = []

# Keep the two files already used in earlier backtests working.
legacy = [
    {"path": "01_data/raw/football_data/E0_2025_2026.csv", "league_id": "ENGLAND_PREMIER_LEAGUE", "season": "2025/2026"},
    {"path": "01_data/raw/laliga/laliga_2020_2021.csv", "league_id": "SPAIN_LALIGA", "season": "2020/2021"},
]
for entry in legacy:
    if Path(entry["path"]).exists():
        manifest.append(entry)

pattern = re.compile(r"^([A-Z0-9]+)_(\d{4})\.csv$")

for f in sorted(data_dir.glob("*.csv")):
    match = pattern.match(f.name)
    if not match:
        continue

    code, season_code = match.group(1), match.group(2)
    if code not in LEAGUE_MAP:
        continue

    manifest.append({
        "path": str(f),
        "league_id": LEAGUE_MAP[code],
        "season": season_label(season_code),
    })

manifest_path = data_dir / "manifest.json"
with manifest_path.open("w", encoding="utf-8") as out:
    json.dump(manifest, out, indent=2, ensure_ascii=False)

print(f"Manifest written: {manifest_path} ({len(manifest)} entries)")
PYEOF

echo ""
echo "=== Step 3: Writing csv_historical_source.py (unified historical data loader) ==="

cat > src/data/csv_historical_source.py << 'PYEOF'
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
PYEOF

echo ""
echo "=== Step 4: Rewriting features/engine.py to use the CSV manifest as base historical data ==="

cat > src/features/engine.py << 'PYEOF'
from src.contracts import MatchData
from src.data.api_football_collector import ApiFootballCollector
from src.data.csv_historical_source import CsvManifestLoader
from src.data.historical import HistoricalMatch


class FeatureEngine:
    """
    Resolves point-in-time historical context needed by the
    prediction pipeline for a given match (live API match or
    an already-known historical match).

    Base historical data comes from the CSV manifest (see
    CsvManifestLoader). A match not found there (a genuinely new
    live fixture) falls back to live API-Football lookups.
    """

    VERSION = "3.0.0"

    def __init__(self) -> None:
        self.collector = ApiFootballCollector()
        self.historical_matches = CsvManifestLoader().load_all().matches

    def _current_team_leagues(self, team_id: str) -> list[str]:
        data = self.collector._request("leagues", {"team": team_id, "current": "true"})
        return [
            str(item["league"]["id"])
            for item in data.get("response", [])
            if item.get("league", {}).get("id") is not None
        ]

    def _get_current_team_history(self, team_id: str, season: str):
        matches = []
        for league_id in self._current_team_leagues(team_id):
            matches.extend(
                self.collector.get_team_history(league_id=league_id, season=season, team_id=team_id)
            )
        return matches

    def build_v2_context(self, match: MatchData) -> tuple[HistoricalMatch, list[HistoricalMatch]]:
        historical_match = next(
            (item for item in self.historical_matches if item.match_id == match.match_id), None,
        )

        if historical_match is not None:
            history = [
                item for item in self.historical_matches
                if item.kickoff_time < historical_match.kickoff_time
                and item.league_id == historical_match.league_id
            ]
            return historical_match, history

        target_match_id = match.match_id.replace("API:", "")
        current_match = self.collector.get_match(target_match_id)

        if current_match is None:
            raise ValueError(f"Match not found: {match.match_id}")

        home_team_id = current_match.home_team_id.replace("TEAM:", "")
        away_team_id = current_match.away_team_id.replace("TEAM:", "")

        home_history = self._get_current_team_history(home_team_id, current_match.season)
        away_history = self._get_current_team_history(away_team_id, current_match.season)

        combined = self.historical_matches + home_history + away_history

        unique_matches = {
            item.match_id: item
            for item in combined
            if item.kickoff_time < current_match.kickoff_time
            and item.league_id == current_match.league_id
        }

        return current_match, list(unique_matches.values())
PYEOF

echo ""
echo "=== Step 5: Adding get_match_statistics (shots/corners/cards) to ApiFootballCollector ==="

cat > src/data/api_football_collector.py << 'PYEOF'
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, Optional

from src.data.collector import DataCollector
from src.data.historical import HistoricalMatch


class ApiFootballCollector(DataCollector):
    BASE_URL = "https://v3.football.api-sports.io"
    SOURCE_NAME = "API-Football"
    VERSION = "1.2.0"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or self._load_key()

    @staticmethod
    def _load_key() -> str:
        value = os.getenv("API_FOOTBALL_KEY")
        if value:
            return value.strip()

        with open(".env", "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line.startswith("API_FOOTBALL_KEY="):
                    return line.split("=", 1)[1].strip()

        raise RuntimeError("API_FOOTBALL_KEY not found")

    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        query = urllib.parse.urlencode(params)
        url = f"{self.BASE_URL}/{endpoint}?{query}"

        request = urllib.request.Request(url, headers={"x-apisports-key": self.api_key})

        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _to_historical_match(item: Dict[str, Any]) -> HistoricalMatch:
        fixture = item["fixture"]
        league = item["league"]
        teams = item["teams"]
        goals = item.get("goals", {})

        kickoff_time = datetime.fromtimestamp(fixture["timestamp"])

        return HistoricalMatch(
            match_id=f"API:{fixture['id']}",
            league_id=str(league["id"]),
            season=str(league["season"]),
            kickoff_time=kickoff_time,
            home_team_id=f"TEAM:{teams['home']['id']}",
            home_team_name=teams["home"]["name"],
            away_team_id=f"TEAM:{teams['away']['id']}",
            away_team_name=teams["away"]["name"],
            home_goals=goals.get("home"),
            away_goals=goals.get("away"),
            source=ApiFootballCollector.SOURCE_NAME,
        )

    def get_match_statistics(self, fixture_id: str) -> Optional[Dict[str, Optional[int]]]:
        """
        Fetches shots/corners/cards for ONE fixture via the
        `fixtures/statistics` endpoint.

        COST WARNING: this is a separate API call per fixture, on
        top of the normal fixtures call. Do NOT call this in a loop
        over many historical fixtures (e.g. inside get_team_history)
        on the API-Football free plan -- it will exhaust the daily
        quota very fast (each team history call can already return
        30+ matches). Use it selectively, e.g. to enrich a single
        upcoming match you're specifically reviewing. Bulk historical
        shot/corner/card data should keep coming from the CSV
        manifest (CsvManifestLoader), which already has it for free.
        """
        data = self._request("fixtures/statistics", {"fixture": fixture_id})
        response = data.get("response", [])

        if len(response) < 2:
            return None

        def _stat(team_stats: Dict[str, Any], stat_type: str) -> Optional[int]:
            for entry in team_stats.get("statistics", []):
                if entry.get("type") == stat_type:
                    value = entry.get("value")
                    return int(value) if value is not None else None
            return None

        home_stats, away_stats = response[0], response[1]

        return {
            "home_shots": _stat(home_stats, "Total Shots"),
            "away_shots": _stat(away_stats, "Total Shots"),
            "home_shots_on_target": _stat(home_stats, "Shots on Goal"),
            "away_shots_on_target": _stat(away_stats, "Shots on Goal"),
            "home_corners": _stat(home_stats, "Corner Kicks"),
            "away_corners": _stat(away_stats, "Corner Kicks"),
            "home_cards": _stat(home_stats, "Yellow Cards"),
            "away_cards": _stat(away_stats, "Yellow Cards"),
        }

    def get_matches(self, date: str) -> list[HistoricalMatch]:
        data = self._request("fixtures", {"date": date})
        return [self._to_historical_match(item) for item in data.get("response", [])]

    def get_match(self, match_id: str) -> HistoricalMatch | None:
        data = self._request("fixtures", {"id": match_id})
        response = data.get("response", [])

        if not response:
            return None

        return self._to_historical_match(response[0])

    def get_team_history(self, league_id: str, season: str, team_id: str) -> list[HistoricalMatch]:
        """
        Return all matches of a team from a league season.

        Uses league + season because the API-Football Free plan
        does not provide access to the fixtures 'last' parameter.
        """
        data = self._request("fixtures", {"league": league_id, "season": season})
        team_id = str(team_id)

        return [
            self._to_historical_match(item)
            for item in data.get("response", [])
            if (
                str(item.get("teams", {}).get("home", {}).get("id")) == team_id
                or str(item.get("teams", {}).get("away", {}).get("id")) == team_id
            )
        ]

    def get_team(self, team_id: str) -> Dict[str, Any]:
        data = self._request("teams", {"id": team_id})
        response = data.get("response", [])
        return response[0] if response else {}
PYEOF

echo ""
echo "=== Step 6: Updating walk_forward_calibration.py to read leagues/seasons from the manifest ==="

python3 - << 'PYEOF'
path = "17_scripts/walk_forward_calibration.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old_import_block = "from src.calibration.walk_forward import run_walk_forward_calibration\nfrom src.data.football_data_loader import FootballDataLoader"
new_import_block = (
    "import json\n"
    "from pathlib import Path\n\n"
    "from src.calibration.walk_forward import run_walk_forward_calibration\n"
    "from src.data.football_data_loader import FootballDataLoader"
)

if old_import_block not in content:
    raise SystemExit("Import block not found -- send this error back, do not edit manually yet.")

content = content.replace(old_import_block, new_import_block, 1)

old_datasets = (
    'DATASETS = [\n'
    '    ("ENGLAND_PREMIER_LEAGUE", "2025/2026", "01_data/raw/football_data/E0_2025_2026.csv"),\n'
    '    ("SPAIN_LALIGA", "2020/2021", "01_data/raw/laliga/laliga_2020_2021.csv"),\n'
    '    # Add more (league_id, season, csv_path) rows here as more\n'
    '    # historical data is collected.\n'
    ']'
)
new_datasets = (
    'def _load_datasets():\n'
    '    manifest_path = Path("01_data/raw/football_data/manifest.json")\n'
    '    if not manifest_path.exists():\n'
    '        return []\n'
    '    with manifest_path.open("r", encoding="utf-8") as f:\n'
    '        entries = json.load(f)\n'
    '    return [(e["league_id"], e["season"], e["path"]) for e in entries]\n\n\n'
    'DATASETS = _load_datasets()'
)

if old_datasets not in content:
    raise SystemExit("DATASETS block not found -- send this error back, do not edit manually yet.")

content = content.replace(old_datasets, new_datasets, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("walk_forward_calibration.py updated to read DATASETS from manifest.json")
PYEOF

echo ""
echo "=== Step 7: Compile check ==="
python3 -m py_compile $(find src 17_scripts -name "*.py" -not -path "*/__pycache__/*")

echo ""
echo "=== Step 8: Manifest summary ==="
python3 -c "
import json
from pathlib import Path
p = Path('01_data/raw/football_data/manifest.json')
if p.exists():
    entries = json.load(p.open())
    print(f'Total datasets in manifest: {len(entries)}')
    for e in entries:
        print(f\"  - {e[\\\"league_id\\\"]:28s} {e[\\\"season\\\"]:10s} {e[\\\"path\\\"]}\")
else:
    print('No manifest.json found!')
"

echo ""
echo "=== ALL DONE. No errors above means everything is syntactically valid. ==="
