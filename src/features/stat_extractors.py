"""
Central definitions of how to extract a given statistic (goals,
shots, ...) from a HistoricalMatch, for a given team's perspective.

Adding a new statistic (e.g. corners, cards) later means adding
4 small functions here — no changes needed in the engines that
consume them (TeamPerformanceEngine, LeagueBaselineEngine).
"""
from src.data.historical import HistoricalMatch


def goals_for(match: HistoricalMatch, is_home: bool):
    if match.home_goals is None or match.away_goals is None:
        return None
    return float(match.home_goals if is_home else match.away_goals)


def goals_against(match: HistoricalMatch, is_home: bool):
    if match.home_goals is None or match.away_goals is None:
        return None
    return float(match.away_goals if is_home else match.home_goals)


def shots_for(match: HistoricalMatch, is_home: bool):
    if match.home_shots is None or match.away_shots is None:
        return None
    return float(match.home_shots if is_home else match.away_shots)


def shots_against(match: HistoricalMatch, is_home: bool):
    if match.home_shots is None or match.away_shots is None:
        return None
    return float(match.away_shots if is_home else match.home_shots)


def league_home_goals(match: HistoricalMatch):
    return float(match.home_goals) if match.home_goals is not None else None


def league_away_goals(match: HistoricalMatch):
    return float(match.away_goals) if match.away_goals is not None else None


def league_home_shots(match: HistoricalMatch):
    return float(match.home_shots) if match.home_shots is not None else None


def league_away_shots(match: HistoricalMatch):
    return float(match.away_shots) if match.away_shots is not None else None
