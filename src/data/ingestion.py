from src.data.dataset import HistoricalDataset
from src.data.quality_validator import DataQualityValidator
from src.data.statsbomb_loader import StatsBombLoader


class HistoricalIngestionPipeline:
    """
    Standard pipeline for loading, validating,
    and assembling historical football data.
    """

    VERSION = "1.0.0"

    def __init__(self) -> None:
        self.loader = StatsBombLoader()
        self.validator = DataQualityValidator()

    def ingest_statsbomb(
        self,
        file_path: str,
    ) -> HistoricalDataset:

        matches = self.loader.load_file(file_path)

        validation_results = self.validator.validate_dataset(matches)

        dataset = HistoricalDataset(
            source_name=StatsBombLoader.SOURCE_NAME,
            dataset_version=self.VERSION,
        )

        for match, result in zip(matches, validation_results):

            if not result.valid:
                reasons = ", ".join(result.reasons)
                raise ValueError(
                    f"Invalid match {match.match_id}: {reasons}"
                )

            dataset.add(match)

        return dataset
