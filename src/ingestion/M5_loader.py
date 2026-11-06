from pathlib import Path
from typing import Any, Mapping


class M5Loader:
    """Load the raw M5 sales, calendar, and sell-price CSV files with Spark."""

    _FILES = {
        "sales": "sales_train_evaluation.csv",
        "calendar": "calendar.csv",
        "prices": "sell_prices.csv",
    }

    def __init__(self, spark: Any, config: Any):
        self.spark = spark
        self.config = config
        self.raw_data_path = self._config_value(
            "raw_data_path",
            Path(__file__).resolve().parents[2] / "dataset" / "raw",
        )

    def _config_value(self, key: str, default: Any) -> Any:
        if isinstance(self.config, Mapping):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def _load_csv(self, name: str) -> Any:
        file_name = self._FILES[name]
        path = self._config_value(f"{name}_path", None)
        if path is None:
            path = str(Path(self.raw_data_path) / file_name)

        return (
            self.spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(str(path))
        )

    def load_sales(self) -> Any:
        return self._load_csv("sales")

    def load_calendar(self) -> Any:
        return self._load_csv("calendar")

    def load_prices(self) -> Any:
        return self._load_csv("prices")

    def run(self) -> dict[str, Any]:
        return {
            "sales": self.load_sales(),
            "calendar": self.load_calendar(),
            "prices": self.load_prices(),
        }