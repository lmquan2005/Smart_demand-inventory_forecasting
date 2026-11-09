from pathlib import Path
from typing import Any, Mapping
from ..common.spark_manager import get_spark

class M5Loader:
    """Load the raw M5 sales, calendar, and sell-price CSV files with Spark."""

    _FILES = {
        "sales": "sales_train_evaluation.csv",
        "calendar": "calendar.csv",
        "prices": "sell_prices.csv",
    }
    _BRONZE_DIRS = {
        "sales": "bronze_sales",
        "calendar": "bronze_calendar",
        "prices": "bronze_sell_prices",
    }

    def __init__(self, config: Any):
        self.spark = get_spark()
        self.config = config
        self.raw_data_path = self._config_value(
            "raw_data_path",
            Path(__file__).resolve().parents[2] / "dataset" / "raw",
        )
        self.bronze_data_path = self._config_value(
            "bronze_data_path",
            Path(__file__).resolve().parents[2]
            / "dataset"
            / "processed"
            / "bronze",
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

    def _write_delta(self, name: str, dataframe: Any) -> None:
        path = self._config_value(
            f"bronze_{name}_path",
            Path(self.bronze_data_path) / self._BRONZE_DIRS[name],
        )

        (
            dataframe.write
            .format("delta")
            .mode("overwrite")
            .save(str(path))
        )

    def write_sales(self, sales: Any) -> None:
        self._write_delta("sales", sales)

    def write_calendar(self, calendar: Any) -> None:
        self._write_delta("calendar", calendar)

    def write_prices(self, prices: Any) -> None:
        self._write_delta("prices", prices)

    def run(self) -> dict[str, Any]:
        dataframes = {
            "sales": self.load_sales(),
            "calendar": self.load_calendar(),
            "prices": self.load_prices(),
        }
        self.write_sales(dataframes["sales"])
        self.write_calendar(dataframes["calendar"])
        self.write_prices(dataframes["prices"])
        return dataframes