from typing import Any, Dict


class DataNormalizer:
    """
    Converts source-specific raw data into the project's
    standard internal data format.
    """

    def normalize_match(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize raw match data.
        """
        raise NotImplementedError

    def normalize_team(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize raw team data.
        """
        raise NotImplementedError
