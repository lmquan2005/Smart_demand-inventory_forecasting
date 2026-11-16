import logging
from typing import Any, Mapping

from pyspark.sql import DataFrame

from ..ingestion.M5_loader import M5Loader


class BronzePipeline:
    """
    Bronze Pipeline:
    - Load raw M5 datasets.
    - Load simulated inventory dataset.
    - Validate input DataFrames.
    - Write all datasets into Bronze Delta Lake.
    - Validate output record counts.
    """

    def __init__(self, config: Any = None):
        self.config = config or {}
        self.loader = M5Loader(config=self.config)

        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_dataframe(
        self,
        name: str,
        dataframe: DataFrame,
    ) -> None:
        """Validate that a DataFrame is not empty and has columns."""

        if dataframe is None:
            raise ValueError(
                f"{name}: DataFrame is None."
            )

        if len(dataframe.columns) == 0:
            raise ValueError(
                f"{name}: DataFrame has no columns."
            )

        self.logger.info(
            "%s schema: %s",
            name,
            dataframe.schema.simpleString(),
        )

        # Check whether the DataFrame contains records.
        if dataframe.limit(1).count() == 0:
            raise ValueError(
                f"{name}: DataFrame is empty."
            )

        self.logger.info(
            "%s validation passed.",
            name,
        )

    def _write_bronze(
        self,
        name: str,
        dataframe: DataFrame,
    ) -> None:
        """Write one DataFrame into Bronze Delta Lake."""

        write_methods = {
            "sales": self.loader.write_sales,
            "calendar": self.loader.write_calendar,
            "prices": self.loader.write_prices,
            "inventory": self.loader.write_inventory,
        }

        if name not in write_methods:
            raise ValueError(
                f"Unsupported Bronze dataset: {name}"
            )

        self.logger.info(
            "Writing %s into Bronze Delta Lake...",
            name,
        )

        write_methods[name](dataframe)

        self.logger.info(
            "%s written successfully.",
            name,
        )

    def _get_record_count(
        self,
        name: str,
        dataframe: DataFrame,
    ) -> int:
        """Count records after writing."""

        count = dataframe.count()

        self.logger.info(
            "%s: %s records.",
            name,
            f"{count:,}",
        )

        return count

    def run(self) -> dict[str, Any]:
        """Execute the complete Bronze pipeline."""

        self.logger.info("=" * 60)
        self.logger.info("STARTING BRONZE PIPELINE")
        self.logger.info("=" * 60)

        # 1. Load datasets
        self.logger.info("Step 1: Loading datasets...")

        dataframes = {
            "sales": self.loader.load_sales(),
            "calendar": self.loader.load_calendar(),
            "prices": self.loader.load_prices(),
            "inventory": self.loader.load_inventory(),
        }

        # 2. Validate input datasets
        self.logger.info("Step 2: Validating datasets...")

        for name, dataframe in dataframes.items():
            self._validate_dataframe(name, dataframe)

        # 3. Write datasets into Bronze
        self.logger.info("Step 3: Writing Bronze Delta tables...")

        for name, dataframe in dataframes.items():
            self._write_bronze(name, dataframe)

        # 4. Validate record counts
        self.logger.info("Step 4: Checking record counts...")

        record_counts = {}

        for name, dataframe in dataframes.items():
            record_counts[name] = self._get_record_count(
                name,
                dataframe,
            )

        # 5. Pipeline summary
        self.logger.info("=" * 60)
        self.logger.info("BRONZE PIPELINE COMPLETED")
        self.logger.info("=" * 60)

        for name, count in record_counts.items():
            self.logger.info(
                "%-12s : %s records",
                name,
                f"{count:,}",
            )

        return {
            "status": "success",
            "record_counts": record_counts,
            "datasets": list(dataframes.keys()),
        }


def main():
    """Run Bronze Pipeline."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    config = {
        "raw_data_path": "dataset/raw",
        "simulated_data_path": "dataset/simulated",
        "bronze_data_path": "dataset/processed/bronze",
    }

    pipeline = BronzePipeline(config=config)

    try:
        result = pipeline.run()

        print("\nBRONZE PIPELINE SUMMARY")
        print("-" * 40)
        print(f"Status: {result['status']}")

        for name, count in result["record_counts"].items():
            print(f"{name:12}: {count:,} records")

    except Exception:
        logging.exception("Bronze Pipeline failed.")
        raise


if __name__ == "__main__":
    main()
