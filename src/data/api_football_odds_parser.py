from typing import Any, Dict, List


class ApiFootballOddsParser:
    VERSION = "1.0.0"

    TARGET_BETS = {
        1: "MATCH_WINNER",
        5: "GOALS_OVER_UNDER",
        8: "BTTS",
    }

    @classmethod
    def parse(
        cls,
        bookmakers: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:

        markets: Dict[str, List[Dict[str, Any]]] = {
            value: []
            for value in cls.TARGET_BETS.values()
        }

        for bookmaker in bookmakers:
            bookmaker_name = bookmaker.get("name")

            for bet in bookmaker.get("bets", []):
                bet_id = bet.get("id")

                if bet_id not in cls.TARGET_BETS:
                    continue

                market_name = cls.TARGET_BETS[bet_id]

                for value in bet.get("values", []):
                    odd = value.get("odd")

                    if odd is None:
                        continue

                    markets[market_name].append(
                        {
                            "bookmaker": bookmaker_name,
                            "bet_id": bet_id,
                            "label": value.get("value"),
                            "odd": float(odd),
                        }
                    )

        return markets
