from dataclasses import dataclass
from typing import List

from src.data.historical import HistoricalMatch


@dataclass
class ValidationResult:
    valid: bool
    reasons: List[str]


class DataQualityValidator:
    """
    Validates the minimum structural quality of historical matches.
    """

    VERSION = "1.0.0"

    def validate(self, match: HistoricalMatch) -> ValidationResult:
        reasons: List[str] = []

        if not match.match_id:
            reasons.append("Missing match_id")

        if not match.league_id:
            reasons.append("Missing league_id")

        if not match.season:
            reasons.append("Missing season")

        if match.kickoff_time is None:
            reasons.append("Missing kickoff_time")

        if not match.home_team_id or not match.home_team_name:
            reasons.append("Missing home team")

        if not match.away_team_id or not match.away_team_name:
            reasons.append("Missing away team")

        if match.home_team_id == match.away_team_id:
            reasons.append("Home and away teams are identical")

        if match.home_goals is None or match.away_goals is None:
            reasons.append("Missing final score")
        else:
            if match.home_goals < 0:
                reasons.append("Invalid home goals")

            if match.away_goals < 0:
                reasons.append("Invalid away goals")

        if not match.source:
            reasons.append("Missing data source")

        return ValidationResult(
            valid=len(reasons) == 0,
            reasons=reasons,
        )

    def validate_dataset(
        self,
        matches: List[HistoricalMatch],
    ) -> List[ValidationResult]:

        results = []

        seen_ids = set()

        for match in matches:
            result = self.validate(match)

            if match.match_id in seen_ids:
                result.valid = False
                result.reasons.append("Duplicate match_id")

            seen_ids.add(match.match_id)
            results.append(result)

        return results
